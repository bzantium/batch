"""
공통 번역 파이프라인 유틸리티 (Batch-only)

- 상태 관리 (load/save)
- 배치 API 래퍼 (upload, create, monitor, download, parse)
- 배치 파일 병렬 생성 (split_and_save_batch_files)
- 데이터셋 저장 로직 (apply_translations, save_translated_dataset)
- 공통 파이프라인 실행기 (run_batch_pipeline, handle_retry_failures, main_runner)
"""

import json
import os
import time
import sys
import argparse
import asyncio
import glob
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from datetime import datetime
from tqdm.contrib.concurrent import process_map, thread_map

# ==============================================================================
# Configuration (Common)
# ==============================================================================
MODEL = "gpt-5-mini"
MAX_COMPLETION_TOKENS = 128000
REASONING_EFFORT = "medium"  # Options: "minimal", "low", "medium", "high"
MAX_REQUESTS_PER_BATCH = 5000
CHECK_INTERVAL = 60
DEBUG_COUNT = 20
CHUNK_MAX_LENGTH = 4000  # Default chunk size for content splitting

# ==============================================================================
# STATE HELPER FUNCTIONS
# ==============================================================================

def load_state(state_file: str) -> Dict[str, Any]:
    """Load the batch job state from a JSON file."""
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
            print(f"\n✓ Found existing state file: {state_file}")
            return state
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"\nStarting a new batch run. State file not found or invalid.")
        return {}

def save_state(state_file: str, data: Dict[str, Any]) -> None:
    """Save the current batch job state to a JSON file."""
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ State saved to {state_file}")
    except IOError as e:
        print(f"  ✗ Warning: Could not save state file {state_file}. Error: {e}")

# ==============================================================================
# COMMON HELPER FUNCTIONS
# ==============================================================================

def chunk_content(text: str, max_length: int = CHUNK_MAX_LENGTH) -> List[str]:
    """
    Chunk content if it exceeds max_length.
    - Splits by double newlines ("\n\n")
    - Groups chunks until reaching target size: len(text) / ceil(len(text) / max_length)
    - If a single chunk exceeds target size, makes it its own chunk
    """
    if len(text) <= max_length:
        return [text]

    # Calculate target chunk size
    num_chunks = math.ceil(len(text) / max_length)
    target_size = len(text) / num_chunks

    # Split by double newlines
    parts = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_length = 0

    for part in parts:
        part_length = len(part)

        # If a single part exceeds target size, make it its own chunk
        if part_length > target_size:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Add this large part as its own chunk
            chunks.append(part)
        else:
            # Calculate length if we add this part (including "\n\n" separator)
            new_length = current_length + part_length
            if current_chunk:  # Add separator length if not first part
                new_length += 2

            # Check if adding this part would exceed target size
            if new_length > target_size and current_chunk:
                # Save current chunk and start new one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [part]
                current_length = part_length
            else:
                # Add to current chunk
                current_chunk.append(part)
                current_length = new_length

    # Add remaining chunk if any
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def parse_failure_log(log_file: str) -> List[str]:
    """
    Parse failure log file and extract custom IDs of failed translations.
    Returns a list of custom_ids.
    """
    print(f"Parsing failure log: {log_file}")
    failed_ids = []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("ID: "):
                    custom_id = line.replace("ID: ", "").strip()
                    failed_ids.append(custom_id)

        print(f"✓ Found {len(failed_ids)} failed translation IDs")
        return failed_ids
    except FileNotFoundError:
        print(f"✗ Error: Failure log not found at {log_file}")
        return []
    except Exception as e:
        print(f"✗ Error parsing failure log: {e}")
        return []

# ==============================================================================
# DATASET APPLY & SAVE FUNCTIONS
# ==============================================================================

def apply_translations(dataset: Dataset, translations: Dict[str, str], is_debug: bool = False) -> Dataset:
    """
    번역된 내용을 원본 데이터셋에 적용합니다.

    Args:
        dataset: 원본 데이터셋
        translations: custom_id -> 번역된 텍스트 매핑
        is_debug: True인 경우 original_content 및 original_reasoning_content를 보존합니다.
    """
    print("\nApplying translations to dataset...")
    if is_debug:
        print("  (Debug mode: Preserving original content fields)")

    translated_data = []
    for record_idx, record in enumerate(tqdm(dataset, desc="Applying translations")):
        translated_record = dict(record)

        try:
            messages = json.loads(record.get("messages", "[]"))
        except json.JSONDecodeError:
            print(f"  ⚠ Warning: Could not parse messages for record {record_idx}. Skipping.")
            messages = []

        translated_messages = []

        for msg_idx, message in enumerate(messages):
            translated_message = dict(message)

            custom_id_content = f"record_{record_idx}_msg_{msg_idx}_content"
            custom_id_reasoning = f"record_{record_idx}_msg_{msg_idx}_reasoning"

            # Helper function to merge chunked translations
            def get_merged_translation(base_custom_id: str) -> Optional[str]:
                """
                Chunking된 번역 결과를 찾아서 병합합니다.

                처리 로직:
                1. 먼저 non-chunked 번역 확인 (예: record_0_msg_0_content)
                   - 있으면 바로 반환 (chunking 안된 메시지)
                2. 없으면 chunking된 번역들을 순서대로 찾기
                   - record_0_msg_0_content_chunk_0, chunk_1, ... 형식
                   - 모든 chunk를 \n\n로 병합하여 반환
                3. 둘 다 없으면 None 반환
                """
                # 1. 먼저 non-chunked 번역 확인 (chunking 안된 메시지)
                if base_custom_id in translations:
                    return translations[base_custom_id]

                # 2. Chunking된 번역 찾기 (chunking된 메시지)
                chunk_translations = []
                chunk_idx = 0
                while True:
                    chunk_custom_id = f"{base_custom_id}_chunk_{chunk_idx}"
                    if chunk_custom_id in translations:
                        chunk_translations.append(translations[chunk_custom_id])
                        chunk_idx += 1
                    else:
                        break

                if chunk_translations:
                    # 모든 chunk를 \n\n로 병합 (원본 chunking 시 사용한 구분자와 동일)
                    return "\n\n".join(chunk_translations)

                # 3. 번역 결과가 없는 경우
                return None

            # Debug 모드: 원본 내용 보존
            merged_content = get_merged_translation(custom_id_content)
            if is_debug and merged_content is not None:
                original_content = message.get("content", "")
                if original_content:
                    translated_message["original_content"] = original_content

            if merged_content is not None:
                translated_message["content"] = merged_content

            # Debug 모드: 원본 reasoning 내용 보존
            merged_reasoning = get_merged_translation(custom_id_reasoning)
            if is_debug and merged_reasoning is not None:
                original_reasoning = message.get("reasoning_content", "")
                if original_reasoning:
                    translated_message["original_reasoning_content"] = original_reasoning

            if merged_reasoning is not None:
                translated_message["reasoning_content"] = merged_reasoning

            translated_messages.append(translated_message)

        # messages 리스트를 다시 JSON 문자열로 저장
        translated_record["messages"] = json.dumps(translated_messages, ensure_ascii=False)
        translated_data.append(translated_record)

    print(f"✓ Applied translations to {len(translated_data)} records")
    from datasets import Dataset as HFDataset
    translated_dataset = HFDataset.from_list(translated_data)
    return translated_dataset

def _extract_shard(idx: int, num_shards: int, dataset: Dataset, out_dirpath: str):
    """
    (Helper) 데이터셋의 단일 샤드를 Parquet 파일로 저장합니다.
    """
    try:
        os.makedirs(out_dirpath, exist_ok=True)
        shard = dataset.shard(num_shards=num_shards, index=idx)
        shard_filename = f"shard_{idx:09d}.zst.parquet"
        shard_filepath = os.path.join(out_dirpath, shard_filename)
        shard.to_parquet(shard_filepath, compression="zstd")
    except Exception as e:
        print(f"Error saving shard {idx}: {e}")

def save_translated_dataset(dataset: Dataset, output_dirpath: Path, original_dataset_path: str) -> None:
    """
    번역된 데이터셋을 원본 샤드 갯수와 동일하게 .zst.parquet 파일로 샤딩하여 저장합니다.
    """
    os.makedirs(output_dirpath, exist_ok=True)
    print(f"\nSaving translated dataset (as Parquet shards) to {output_dirpath}...")

    # 원본 DATASET_PATH에서 Parquet 샤드 갯수 계산
    try:
        shard_files = glob.glob(f"{original_dataset_path}/*.parquet")
        num_shards = len(shard_files)
        if num_shards == 0:
            print(f"  ✗ Warning: No source parquet files found at {original_dataset_path}. Defaulting to 1 shard.")
            num_shards = 1
        print(f"  Matching original shard count: {num_shards}")
    except Exception as e:
        print(f"  ✗ Error counting source shards: {e}. Defaulting to 1 shard.")
        num_shards = 1

    # 병렬 처리 설정 (CPU 바운드 작업인 Parquet 저장을 위해 process_map 사용)
    num_proc = max(os.cpu_count() // 4, 1)  # 최소 1개 워커 보장
    print(f"  Using {num_proc} workers for parallel saving.")

    # partial 함수 생성
    extract_shard_partial = partial(
        _extract_shard,
        num_shards=num_shards,
        dataset=dataset,
        out_dirpath=str(output_dirpath),
    )

    # 병렬 처리 실행
    try:
        process_map(
            extract_shard_partial,
            range(num_shards),
            max_workers=num_proc,
            chunksize=max(math.ceil(num_shards / num_proc), 1),
            desc="Saving shards"
        )

        print(f"\n✓ Translated dataset saved successfully to {output_dirpath}")
        print(f"  Total records: {len(dataset)}")
        print(f"  Total shards: {num_shards}")

    except Exception as e:
        print(f"\n✗ Error during parallel shard saving: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# BATCH API HELPER FUNCTIONS
# ==============================================================================

def _save_batch_file_parallel(
    batch_idx: int,
    all_requests: List[Dict],
    max_per_batch: int,
    total_requests: int,
    input_dir: Path
) -> Dict[str, Any]:
    """(Helper) 단일 배치 .jsonl 파일을 병렬(스레드)로 저장합니다."""
    start_idx = batch_idx * max_per_batch
    end_idx = min((batch_idx + 1) * max_per_batch, total_requests)
    batch_requests = all_requests[start_idx:end_idx]

    # 파일명 패딩 (e.g., batch_00001.jsonl)
    batch_file = input_dir / f"batch_{batch_idx:05d}.jsonl"

    try:
        with open(batch_file, "w", encoding="utf-8") as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        metadata = {
            "batch_idx": batch_idx,
            "batch_file": str(batch_file),
            "num_requests": len(batch_requests),
            "start_request": start_idx,
            "end_request": end_idx
        }
        return metadata
    except Exception as e:
        print(f"  ✗ Error saving batch file {batch_idx}: {e}")
        return {} # 실패 시 빈 dict 반환


def split_and_save_batch_files(
    all_batch_requests: List[Dict],
    batch_input_dir: Path,
    max_requests_per_batch: int
) -> List[Dict[str, Any]]:
    """
    (Common Function)
    전체 배치 요청 리스트를 받아 병렬(multi-threaded)로
    여러 .jsonl 파일에 분할 저장합니다.
    """
    batch_input_dir.mkdir(parents=True, exist_ok=True)

    total_requests = len(all_batch_requests)
    if total_requests == 0:
        print("  ⚠ Warning: No batch requests generated.")
        return []

    num_batches = (total_requests + max_requests_per_batch - 1) // max_requests_per_batch

    print(f"\n✓ Total translation requests: {total_requests}")
    print(f"  Splitting into {num_batches} batch file(s) (using parallel I/O)...")

    # I/O 작업이므로 스레드 워커 수를 넉넉하게 설정
    num_workers = max(min(os.cpu_count() * 4, 32), 4)

    save_partial = partial(
        _save_batch_file_parallel,
        all_requests=all_batch_requests,
        max_per_batch=max_requests_per_batch,
        total_requests=total_requests,
        input_dir=batch_input_dir
    )

    batch_files_metadata = []

    # thread_map을 사용하여 병렬 I/O 실행
    results = thread_map(
        save_partial,
        range(num_batches),
        max_workers=num_workers,
        chunksize=1, # 각 스레드가 파일 하나씩 처리
        desc="Saving batch files"
    )

    # 실패한 경우(빈 dict)를 필터링
    batch_files_metadata = [res for res in results if res]

    # 결과가 순서대로 반환되지만, 만약을 위해 batch_idx 기준으로 정렬
    batch_files_metadata.sort(key=lambda m: m["batch_idx"])

    print(f"\n✓ Created {len(batch_files_metadata)} batch files in {batch_input_dir}")
    if len(batch_files_metadata) != num_batches:
        print(f"  ✗ Warning: Expected {num_batches} files, but only {len(batch_files_metadata)} were saved successfully.")

    return batch_files_metadata

def upload_batch_file(client: OpenAI, file_path: str, verbose: bool = True) -> str:
    """배치 파일을 업로드합니다."""
    if verbose:
        print(f"\nUploading {file_path} to OpenAI...")
    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    file_id = batch_input_file.id
    if verbose:
        print(f"✓ File uploaded successfully. File ID: {file_id}")
    return file_id

async def upload_batch_file_async(client: AsyncOpenAI, file_path: str, batch_idx: int) -> tuple[int, str]:
    """배치 파일을 비동기(Executor 사용)로 업로드합니다."""
    loop = asyncio.get_event_loop()
    def _upload():
        # AsyncOpenAI 객체에서 직접 API 키를 가져와 동기 클라이언트 생성
        sync_client = OpenAI(api_key=client.api_key)
        with open(file_path, "rb") as f:
            batch_input_file = sync_client.files.create(file=f, purpose="batch")
        return batch_input_file.id

    file_id = await loop.run_in_executor(None, _upload)
    return batch_idx, file_id

def create_batch_job(client: OpenAI, file_id: str, verbose: bool = True) -> str:
    """배치 작업을 생성합니다."""
    if verbose:
        print("\nCreating batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Translation to Korean"}
    )
    batch_id = batch.id
    if verbose:
        print(f"✓ Batch job created successfully. Batch ID: {batch_id}")
        print(f"  Status: {batch.status}")
    return batch_id

def monitor_batch_job(client: OpenAI, batch_id: str, check_interval: int = 60) -> Dict[str, Any]:
    """배치 작업 완료를 모니터링합니다."""
    print(f"\nMonitoring batch job {batch_id}...")
    print(f"{check_interval}초마다 상태를 확인합니다...\n")
    start_time = time.time()
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"✗ Error retrieving batch status: {e}. Retrying in {check_interval}s...")
            time.sleep(check_interval)
            continue

        elapsed = time.time() - start_time
        status_info = {
            "status": batch.status,
            "completed": getattr(batch.request_counts, 'completed', 0),
            "failed": getattr(batch.request_counts, 'failed', 0),
            "total": getattr(batch.request_counts, 'total', 0)
        }
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status_info['status']} | "
              f"Completed: {status_info['completed']}/{status_info['total']} | "
              f"Failed: {status_info['failed']} | "
              f"Elapsed: {elapsed/60:.1f}m")

        if batch.status == "completed":
            print(f"\n✓ Batch job completed successfully!")
            print(f"  Total time: {elapsed/60:.1f} minutes")
            print(f"  Output file ID: {batch.output_file_id}")
            return {
                "status": "completed",
                "batch": batch,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id
            }
        elif batch.status in ["failed", "expired", "cancelling", "cancelled"]:
            print(f"\n✗ Batch job status: {batch.status}! Exiting monitoring.")
            return {
                "status": batch.status,
                "batch": batch,
                "error_file_id": batch.error_file_id
            }
        time.sleep(check_interval)

def download_batch_results(client: OpenAI, output_file_id: str, output_path: str, verbose: bool = True) -> None:
    """배치 결과 파일을 다운로드합니다."""
    if verbose:
        print(f"\nDownloading batch results...")
    file_response = client.files.content(output_file_id)
    with open(output_path, "wb") as f:
        f.write(file_response.content)
    if verbose:
        print(f"✓ Results saved to {output_path}")

async def download_batch_results_async(client: AsyncOpenAI, output_file_id: str, output_path: str, batch_idx: int) -> int:
    """배치 결과 파일을 비동기(Executor 사용)로 다운로드합니다."""
    loop = asyncio.get_event_loop()
    def _download():
        sync_client = OpenAI(api_key=client.api_key)
        file_response = sync_client.files.content(output_file_id)
        with open(output_path, "wb") as f:
            f.write(file_response.content)

    await loop.run_in_executor(None, _download)
    return batch_idx

def parse_batch_results(batch_output_file: str) -> (Dict[str, str], List[Dict[str, Any]]):
    """배치 결과 JSONL 파일을 파싱하여 성공/실패를 분리합니다."""
    print(f"\nParsing batch results from {batch_output_file}...")
    translations = {}
    failures = []
    try:
        with open(batch_output_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
                response = result.get("response", {})
                status_code = response.get("status_code")

                if status_code == 200:
                    try:
                        translated_text = response["body"]["choices"][0]["message"]["content"]
                        translations[custom_id] = translated_text
                    except (KeyError, IndexError, TypeError):
                        error_msg = "Status 200 but response body is malformed"
                        print(f"  ✗ Failed: {custom_id} - {error_msg}")
                        failures.append({"id": custom_id, "status": 200, "error": error_msg, "response_body": response.get("body")})
                else:
                    error_message = "Unknown error"
                    error_body = response.get("body", {})
                    try:
                        if isinstance(error_body, dict) and "error" in error_body:
                            error_message = error_body.get("error", {}).get("message", "No error message found in body")
                        else:
                            error_message = str(error_body)
                    except Exception:
                        error_message = f"Error parsing error response body: {error_body}"
                    print(f"  ✗ Failed: {custom_id} - Status {status_code} - Reason: {error_message}")
                    failures.append({"id": custom_id, "status": status_code, "error": error_message, "response_body": error_body})
    except FileNotFoundError:
        print(f"✗ Error: Batch output file not found at {batch_output_file}")
        return {}, []
    except json.JSONDecodeError:
        print(f"✗ Error: Could not parse JSON from {batch_output_file}. File might be empty or corrupt.")
        return {}, []

    print(f"✓ Parsed {len(translations)} successful translations")
    if failures:
        print(f"  ✗ Found {len(failures)} failed translations")
    return translations, failures

# ==============================================================================
# BATCH API PIPELINE (MAIN)
# ==============================================================================

async def run_batch_pipeline(
    dataset: Dataset,
    output_dir: Path,
    state_file_path: str,
    original_dataset_path: str,
    prepare_batch_input_fn: Callable,
    is_debug: bool = False,
    chunk_max_length: int = CHUNK_MAX_LENGTH,
):
    """
    Execute the full Batch API pipeline with multiple batch files and state resume.

    Args:
        dataset: The input dataset to translate.
        output_dir: The directory to save all outputs (state, inputs, outputs).
        state_file_path: Path to the .state file for resuming.
        original_dataset_path: Path to the original dataset (for shard matching).
        prepare_batch_input_fn: Dataset-specific function that takes (dataset,
                                model, reasoning_effort, chunk_max_length) and
                                returns List[Dict] of batch requests.
        is_debug: Whether to run in debug mode.
        chunk_max_length: Maximum length for content chunking.
    """
    print("\n" + "="*80)
    print(f"Running Batch API Translation to Korean")
    print(f"Model: {MODEL}, Reasoning effort: {REASONING_EFFORT}")
    print("="*80)

    state = load_state(state_file_path)
    if "is_debug" not in state:
        state["is_debug"] = is_debug

    if state.get("status") == "completed":
        translated_dir_name = state.get("translated_output_directory", "translated")
        translated_dataset_dir = output_dir / translated_dir_name
        print(f"✓ Pipeline already completed for this dataset. Final directory:")
        print(f"  {translated_dataset_dir}")
        print(f"  To re-run, delete the state file and output files in this folder.")
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    batch_input_dir = output_dir / "batch_input"
    batch_output_dir = output_dir / "batch_output"
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Prepare and Save batch input files
        batch_files_metadata = state.get("batch_files_metadata")
        if not batch_files_metadata:
            print("\n--- Step 1: Preparing and saving batch input files ---")

            # 1a. (Specific) 데이터셋별 배치 요청 생성 (CPU-bound)
            all_batch_requests = prepare_batch_input_fn(
                dataset=dataset,
                model=MODEL,
                reasoning_effort=REASONING_EFFORT,
                chunk_max_length=chunk_max_length
            )

            # 1b. (Common) 배치 파일 분할 및 병렬 저장 (I/O-bound)
            batch_files_metadata = split_and_save_batch_files(
                all_batch_requests=all_batch_requests,
                batch_input_dir=batch_input_dir,
                max_requests_per_batch=MAX_REQUESTS_PER_BATCH
            )

            state["batch_files_metadata"] = batch_files_metadata
            state["batches"] = {}
            save_state(state_file_path, state)
        else:
            print(f"\n--- Step 1: Already prepared {len(batch_files_metadata)} batch file(s) ---")

        batches_state = state.get("batches", {})

        # Step 2: Upload all batch files (in parallel)
        print(f"\n{'='*80}")
        print(f"Step 2: Uploading all batch files (parallel)")
        print(f"{'='*80}")

        to_upload = [m for m in batch_files_metadata if not batches_state.get(f"batch_{m['batch_idx']}", {}).get("file_id")]
        already_uploaded = len(batch_files_metadata) - len(to_upload)

        if already_uploaded > 0:
            print(f"Already uploaded: {already_uploaded} batch(es)")

        if to_upload:
            print(f"Uploading {len(to_upload)} batch file(s) in parallel...")
            upload_tasks = []
            for batch_meta in to_upload:
                batch_idx = batch_meta["batch_idx"]
                batch_file = batch_meta["batch_file"]
                upload_tasks.append(upload_batch_file_async(async_client, batch_file, batch_idx))

            upload_results = await asyncio.gather(*upload_tasks)

            for batch_idx, file_id in upload_results:
                batch_key = f"batch_{batch_idx}"
                batch_state = batches_state.get(batch_key, {})
                batch_state["file_id"] = file_id
                batch_state["status"] = "uploaded"
                batches_state[batch_key] = batch_state

            state["batches"] = batches_state
            save_state(state_file_path, state)
            print(f"✓ Uploaded {len(to_upload)} batch file(s) in parallel")

        # Step 3: Create all batch jobs
        print(f"\n{'='*80}")
        print(f"Step 3: Creating all batch jobs")
        print(f"{'='*80}")

        to_create = [m for m in batch_files_metadata
                     if batches_state.get(f"batch_{m['batch_idx']}", {}).get("file_id")
                     and not batches_state.get(f"batch_{m['batch_idx']}", {}).get("batch_id")]
        already_created = sum(1 for m in batch_files_metadata if batches_state.get(f"batch_{m['batch_idx']}", {}).get("batch_id"))

        if already_created > 0:
            print(f"Already created: {already_created} batch job(s)")

        if to_create:
            print(f"Creating {len(to_create)} batch job(s)...")
            for batch_meta in tqdm(to_create, desc="Creating jobs"):
                batch_idx = batch_meta["batch_idx"]
                batch_key = f"batch_{batch_idx}"
                batch_state = batches_state.get(batch_key, {})
                file_id = batch_state["file_id"]
                batch_id = create_batch_job(client, file_id, verbose=False)
                batch_state["batch_id"] = batch_id
                batch_state["status"] = "created"
                batches_state[batch_key] = batch_state
                state["batches"] = batches_state
                save_state(state_file_path, state)
            print(f"✓ Created {len(to_create)} batch job(s)")

        # Step 4: Monitor all batch jobs and download immediately when completed
        print(f"\n{'='*80}")
        print(f"Step 4: Monitoring all batch jobs (downloading completed batches immediately)")
        print(f"{'='*80}")

        pending_batches = []
        downloaded_batches = set()
        for batch_meta in batch_files_metadata:
            batch_idx = batch_meta["batch_idx"]
            batch_key = f"batch_{batch_idx}"
            batch_state = batches_state.get(batch_key, {})
            # 파일명 형식 수정 (e.g., batch_00001_output.jsonl)
            batch_output_file = batch_output_dir / f"batch_{batch_idx:05d}_output.jsonl"

            if batch_output_file.exists() or batch_state.get("status") == "downloaded":
                downloaded_batches.add(batch_idx)
                if not batch_state.get("output_file"):
                    batch_state["output_file"] = str(batch_output_file)
                    batch_state["status"] = "downloaded"
                    batches_state[batch_key] = batch_state
            elif batch_state.get("batch_id"):
                pending_batches.append((batch_idx, batch_key, batch_state["batch_id"]))

        if downloaded_batches:
            print(f"Already downloaded: {len(downloaded_batches)} batch(es)")

        if pending_batches:
            print(f"Monitoring {len(pending_batches)} batch job(s)...")
            print(f"Checking every {CHECK_INTERVAL} seconds...\n")

            start_time = time.time()
            download_tasks = {} # {batch_idx: asyncio.Task}

            while pending_batches or download_tasks:
                completed_this_round = []
                # 1. Check status of pending jobs
                for batch_idx, batch_key, batch_id in pending_batches:
                    try:
                        batch = client.batches.retrieve(batch_id)
                        batch_state = batches_state[batch_key]

                        if batch.status == "completed":
                            elapsed = time.time() - start_time
                            print(f"[{time.strftime('%H:%M:%S')}] Batch {batch_idx:05d}: COMPLETED - Starting download... (elapsed: {elapsed/60:.1f}m)")
                            batch_state["output_file_id"] = batch.output_file_id
                            batch_state["status"] = "completed"
                            batches_state[batch_key] = batch_state
                            state["batches"] = batches_state
                            save_state(state_file_path, state)

                            batch_output_file = batch_output_dir / f"batch_{batch_idx:05d}_output.jsonl"
                            # Start async download task
                            download_task = asyncio.create_task(
                                download_batch_results_async(async_client, batch.output_file_id, str(batch_output_file), batch_idx)
                            )
                            download_tasks[batch_idx] = download_task
                            completed_this_round.append((batch_idx, batch_key, batch_id))

                        elif batch.status in ["failed", "expired", "cancelling", "cancelled"]:
                            print(f"[{time.strftime('%H:%M:%S')}] Batch {batch_idx:05d}: FAILED (status: {batch.status})")
                            batch_state["status"] = "failed"
                            batch_state["error"] = batch.status
                            batches_state[batch_key] = batch_state
                            state["batches"] = batches_state
                            save_state(state_file_path, state)
                            completed_this_round.append((batch_idx, batch_key, batch_id))

                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Error checking batch {batch_idx:05d}: {e}")

                # Remove completed/failed jobs from monitoring list
                for item in completed_this_round:
                    pending_batches.remove(item)

                # 2. Check status of download tasks
                done_downloads = []
                for batch_idx, task in download_tasks.items():
                    if task.done():
                        try:
                            await task # Check for exceptions
                            batch_key = f"batch_{batch_idx}"
                            batch_state = batches_state[batch_key]
                            batch_output_file = batch_output_dir / f"batch_{batch_idx:05d}_output.jsonl"
                            batch_state["output_file"] = str(batch_output_file)
                            batch_state["status"] = "downloaded"
                            batches_state[batch_key] = batch_state
                            state["batches"] = batches_state
                            save_state(state_file_path, state)
                            elapsed = time.time() - start_time
                            print(f"[{time.strftime('%H:%M:%S')}] Batch {batch_idx:05d}: DOWNLOADED (elapsed: {elapsed/60:.1f}m)")
                            done_downloads.append(batch_idx)
                        except Exception as e:
                            print(f"[{time.strftime('%H:%M:%S')}] Error downloading batch {batch_idx:05d}: {e}")
                            done_downloads.append(batch_idx) # Remove task even if failed

                # Remove completed download tasks
                for batch_idx in done_downloads:
                    del download_tasks[batch_idx]

                # 3. Sleep if there are still tasks
                if pending_batches or download_tasks:
                    elapsed = time.time() - start_time
                    total_batches = len(batch_files_metadata)
                    completed_count = total_batches - len(pending_batches) - len(download_tasks)
                    monitoring_count = len(pending_batches)
                    downloading_count = len(download_tasks)
                    print(f"[{time.strftime('%H:%M:%S')}] Status: {completed_count}/{total_batches} done | {monitoring_count} monitoring | {downloading_count} downloading | Elapsed: {elapsed/60:.1f}m")
                    await asyncio.sleep(CHECK_INTERVAL)

            print(f"\n✓ All batch jobs completed and downloaded!")
        else:
            print(f"✓ All batches already completed and downloaded!")

        # Step 5: Aggregate and Parse all results
        print(f"\n{'='*80}")
        print("Step 5: Aggregating results from all batches")
        print(f"{'='*80}")

        all_translations = {}
        all_failures = []

        for batch_meta in batch_files_metadata:
            batch_idx = batch_meta["batch_idx"]
            batch_key = f"batch_{batch_idx}"
            batch_state = batches_state.get(batch_key, {})

            if batch_state.get("status") != "downloaded":
                print(f"  ⚠ Warning: Batch {batch_idx:05d} not completed or downloaded, skipping")
                continue

            batch_output_file = batch_state.get("output_file")
            if not batch_output_file or not Path(batch_output_file).exists():
                print(f"  ⚠ Warning: Output file for batch {batch_idx:05d} not found at {batch_output_file}, skipping")
                continue

            print(f"  Processing batch {batch_idx:05d}...")
            translations, failures = parse_batch_results(batch_output_file)
            all_translations.update(translations)
            all_failures.extend(failures)

        print(f"\n✓ Aggregated {len(all_translations)} successful translations from all batches")

        # 재시도 로직을 위해 집계된 번역 결과 저장
        all_translations_file = output_dir / "all_translations.json"
        print(f"  Saving aggregated translations to {all_translations_file}...")
        try:
            with open(all_translations_file, "w", encoding="utf-8") as f:
                json.dump(all_translations, f, ensure_ascii=False)
        except Exception as e:
            print(f"  ✗ Warning: Could not save aggregated translations file: {e}")

        # Step 6: Save failure log
        if all_failures:
            print(f"\n--- Summary of Failed Translations (Total: {len(all_failures)}) ---")
            fail_log_file = output_dir / "translation_failures.log"
            print(f"  Saving failure details to: {fail_log_file}")
            try:
                with open(fail_log_file, "w", encoding="utf-8") as f:
                    f.write(f"Total Failed Tasks: {len(all_failures)}\n")
                    f.write("="*30 + "\n")
                    for failure in all_failures:
                        f.write(f"ID: {failure['id']}\n")
                        f.write(f"Status: {failure['status']}\n")
                        f.write(f"Error: {failure['error']}\n")
                        f.write(f"Response Body: {json.dumps(failure.get('response_body'), ensure_ascii=False)}\n---\n")
            except Exception as e:
                print(f"  ✗ Warning: Could not write failure log. Error: {e}")

            for i, failure in enumerate(all_failures[:5]):
                 print(f"  - ID: {failure['id']}, Status: {failure['status']}, Error: {failure['error']}")
            if len(all_failures) > 5:
                print(f"  ... and {len(all_failures) - 5} more (see log file for details).")

        if not all_translations:
            print("\n✗ No translations were parsed from any batch. Exiting.")
            return

        # Step 7: Apply Translations
        print("\n--- Step 7: Applying translations to dataset ---")
        translated_dataset = apply_translations(dataset, all_translations, is_debug=is_debug)

        # Step 8: Save Final Translated Dataset
        print("\n--- Step 8: Saving final translated dataset (as Parquet shards) ---")

        output_shard_dir_name = "translated"
        translated_dataset_dir = output_dir / output_shard_dir_name
        save_translated_dataset(translated_dataset, translated_dataset_dir, original_dataset_path)

        state["translated_output_directory"] = output_shard_dir_name
        state["status"] = "completed"
        save_state(state_file_path, state)

        print(f"\n✓ Final translated dataset shards saved to: {translated_dataset_dir}")

    except Exception as e:
        print(f"\n\n✗✗✗ An unexpected error occurred: {e} ✗✗✗")
        print("  Current state has been saved. You can re-run the script to resume.")
        import traceback
        traceback.print_exc()
        save_state(state_file_path, state)
        sys.exit(1)


# ==============================================================================
# BATCH API PIPELINE (RETRY HELPER)
# ==============================================================================

async def _run_single_batch_job(
    client: OpenAI,
    batch_requests: List[Dict[str, Any]],
    output_dir: Path,
) -> (Dict[str, str], List[Dict[str, Any]]):
    """
    (Helper) Runs a single batch file job. Used by retry logic.
    Returns: (translations, failures)
    """
    if not batch_requests:
        print("  No valid requests to process.")
        return {}, []

    retry_input_dir = output_dir / "retry_batch_input"
    retry_input_dir.mkdir(parents=True, exist_ok=True)
    retry_output_dir = output_dir / "retry_batch_output"
    retry_output_dir.mkdir(parents=True, exist_ok=True)

    batch_file = retry_input_dir / "retry_batch_00000.jsonl"
    with open(batch_file, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"✓ Saved retry batch file: {batch_file}")

    print("\nUploading retry batch file...")
    file_id = upload_batch_file(client, str(batch_file))

    print("\nCreating retry batch job...")
    batch_id = create_batch_job(client, file_id)

    print("\nMonitoring retry batch job...")
    result = monitor_batch_job(client, batch_id, CHECK_INTERVAL)

    if result["status"] != "completed":
        print(f"\n✗ Retry batch job failed with status: {result['status']}")
        return {}, []

    output_file_id = result["output_file_id"]
    batch_output_file = retry_output_dir / "retry_batch_00000_output.jsonl"

    print("\nDownloading retry batch results...")
    download_batch_results(client, output_file_id, str(batch_output_file))

    print("\nParsing retry batch results...")
    retry_translations, retry_failures = parse_batch_results(str(batch_output_file))

    print(f"\n✓ Retry batch completed")
    print(f"  Successful: {len(retry_translations)}")
    print(f"  Still failed: {len(retry_failures)}")

    if retry_failures:
        print(f"\n  ⚠ Warning: {len(retry_failures)} translations still failed after retry")
        fail_log = output_dir / "retry_failures.log"
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write(f"Total Still Failed: {len(retry_failures)}\n")
            f.write("="*30 + "\n")
            for failure in retry_failures:
                f.write(f"ID: {failure['id']}\n")
                f.write(f"Status: {failure['status']}\n")
                f.write(f"Error: {failure['error']}\n---\n")
        print(f"  Saved retry failure log: {fail_log}")

    return retry_translations, retry_failures

# ==============================================================================
# MAIN EXECUTION (GENERIC)
# ==============================================================================

async def handle_retry_failures(
    args: argparse.Namespace,
    original_dataset_path: str,
    extract_failed_records_fn: Callable,
    prepare_retry_requests_fn: Callable
):
    """
    (Generic) --retry-failures 플래그를 처리합니다.

    Args:
        args: Parsed command-line arguments (must include --retry-failures path).
        original_dataset_path: Path to the original dataset.
        extract_failed_records_fn: Dataset-specific function to get failed record data.
        prepare_retry_requests_fn: Dataset-specific function to build retry requests.
    """
    output_dir = Path(args.retry_failures)

    if not output_dir.is_dir():
        print(f"\n✗ Error: Output folder not found: {args.retry_failures}")
        sys.exit(1)

    print("=" * 80)
    print(f"Retry Failed Translations Mode (Batch API Only)")
    print(f"Output Folder: {output_dir.resolve()}")
    print("=" * 80)

    # 1. 필수 파일 확인
    failure_log = output_dir / "translation_failures.log"
    all_translations_file = output_dir / "all_translations.json"

    if not failure_log.exists():
        print(f"\n✗ Error: Failure log not found: {failure_log}")
        sys.exit(1)
    if not all_translations_file.exists():
        print(f"\n✗ Error: Aggregated translations file not found: {all_translations_file}")
        print("  Run the main pipeline first, or check for errors in the main run.")
        sys.exit(1)

    # Load state to check if debug mode was used
    safe_dataset_name = Path(original_dataset_path).name
    safe_model_name = MODEL.replace("/", "_").replace(".", "-")
    base_filename = f"{safe_dataset_name}_{safe_model_name}"
    state_file_path = output_dir / f".{base_filename}.state"

    is_debug = False
    if state_file_path.exists():
        state = load_state(str(state_file_path))
        is_debug = state.get("is_debug", False)
        if is_debug:
            print("  ✓ Debug mode detected from state file (will preserve original content)")
    else:
        print(f"  ⚠ Warning: State file not found at {state_file_path}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 2. 원본 데이터셋 로드
    print("\nLoading original dataset...")
    dataset = load_dataset(original_dataset_path, split="train")
    print(f"✓ Loaded {len(dataset)} records")

    # 3. 기존 번역 및 실패 내역 로드
    print(f"Loading previous translations from {all_translations_file}...")
    with open(all_translations_file, "r", encoding="utf-8") as f:
        all_translations = json.load(f)
    print(f"✓ Loaded {len(all_translations)} successful translations")

    print(f"Loading failed IDs from {failure_log}...")
    failed_ids = parse_failure_log(str(failure_log))
    if not failed_ids:
        print("\n✗ No failed translations found in log. Nothing to retry.")
        return

    # 4. (Specific) 재시도 대상 레코드 추출
    failed_records = extract_failed_records_fn(dataset, failed_ids)
    if not failed_records:
        print("\n✗ Could not extract any valid failed records. Exiting.")
        return

    # 5. (Specific) 재시도 요청 생성
    print(f"\nPreparing {len(failed_records)} retry requests...")
    retry_requests = prepare_retry_requests_fn(
        failed_records=failed_records,
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        chunk_max_length=getattr(args, 'chunk_max_length', CHUNK_MAX_LENGTH)
    )
    print(f"✓ Created {len(retry_requests)} retry requests")

    # 6. 재시도 파이프라인 실행 (Single Batch Job)
    retry_translations, _ = await _run_single_batch_job(
        client=client,
        batch_requests=retry_requests,
        output_dir=output_dir,
    )

    if not retry_translations:
        print("\n✗ No successful retry translations. Exiting.")
        return

    # 7. 번역 병합
    print("\nMerging original and retry translations...")
    original_count = len(all_translations)
    all_translations.update(retry_translations)
    merged_count = len(all_translations)
    print(f"  ✓ Original: {original_count}, Retried: {len(retry_translations)}, Total: {merged_count}")

    # 8. 전체 번역본 적용
    print("\nApplying all merged translations to the dataset...")
    translated_dataset = apply_translations(dataset, all_translations, is_debug=is_debug)

    # 9. 병합된 새 Parquet 샤드 저장
    merged_dir = output_dir / "translated_merged"
    print(f"\nSaving merged dataset to {merged_dir}...")
    save_translated_dataset(translated_dataset, merged_dir, original_dataset_path)

    print("\n" + "=" * 80)
    print("Retry translations completed successfully!")
    print(f"Mode: Batch API (Retry)")
    print(f"\nFiles:")
    print(f"  - Merged output (directory): {merged_dir.name}/")
    print(f"  - Original translations: {all_translations_file.name}")
    print(f"  - Failure log: {failure_log.name}")
    print("=" * 80)
    print("\nYou can now load the merged dataset with:\n")
    print(f"from datasets import load_dataset")
    print(f"dataset = load_dataset('{merged_dir}')")

async def main_runner(
    dataset_path: str,
    prepare_batch_input_fn: Callable,
    extract_failed_records_fn: Callable,
    prepare_retry_requests_fn: Callable
):
    """
    (Generic) Main entry point.
    Parses arguments and routes to Batch pipeline.
    """

    parser = argparse.ArgumentParser(description="OpenAI Batch API Translation Pipeline")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the pipeline with only DEBUG_COUNT records for testing."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="FOLDER_PATH",
        help="Resume a batch job from the state file in this folder."
    )
    parser.add_argument(
        "--retry-failures",
        type=str,
        default=None,
        metavar="FOLDER_PATH",
        help="Retry failed translations from the specified output folder (uses Batch API)."
    )
    parser.add_argument(
        "--chunk-max-length",
        type=int,
        default=CHUNK_MAX_LENGTH,
        metavar="LENGTH",
        help=f"Maximum length for content chunking (default: {CHUNK_MAX_LENGTH})."
    )

    args = parser.parse_args()

    # 재시도 모드
    if args.retry_failures:
        await handle_retry_failures(
            args,
            dataset_path,
            extract_failed_records_fn,
            prepare_retry_requests_fn
        )
        return

    # --- 메인 파이프라인 실행 ---

    safe_dataset_name = Path(dataset_path).name
    safe_model_name = MODEL.replace("/", "_").replace(".", "-")
    base_filename = f"{safe_dataset_name}_{safe_model_name}"

    is_debug = args.debug
    output_dir = None
    state_file_path = None

    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.is_dir():
            print(f"\n✗ Error: Resume folder not found: {args.resume}")
            sys.exit(1)
        print(f"Resuming batch job from folder: {output_dir.resolve()}")

        state_file_path = output_dir / f".{base_filename}.state"
        state = load_state(str(state_file_path))
        is_debug_from_state = state.get("is_debug", False)

        if is_debug_from_state:
            print("  ✓ Resuming in DEBUG mode (loaded from state file).")
            is_debug = True
        elif args.debug:
            print("  ⚠ Warning: Resuming a non-debug run with --debug flag.")
            is_debug = True
        else:
            is_debug = False

    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if is_debug:
            timestamp = timestamp + "_debug"
        # 데이터셋 이름별로 하위 폴더 생성
        output_dir = Path(f"experiments/{safe_dataset_name}/" + timestamp)
        print(f"Starting new run. Outputs will be saved to: {output_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    state_file_path = output_dir / f".{base_filename}.state"

    print("=" * 80)
    print(f"OpenAI Batch Translation Pipeline to Korean")
    mode_str = "Batch API"
    if args.resume:
        mode_str += " (Resuming)"
    print(f"Model: {MODEL}")
    print(f"Mode: {mode_str}")
    print(f"Debug: {is_debug}")
    print(f"Output Folder: {output_dir.resolve()}")
    print(f"Source Dataset: {dataset_path}")
    print("=" * 80)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        print("  Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    print("\nLoading dataset...")
    dataset = load_dataset(
        dataset_path,
        split="train",
    )

    if is_debug:
        dataset = dataset.take(DEBUG_COUNT)

    print(f"✓ Loaded {len(dataset)} records")

    await run_batch_pipeline(
        dataset=dataset,
        output_dir=output_dir,
        state_file_path=str(state_file_path),
        original_dataset_path=dataset_path,
        prepare_batch_input_fn=prepare_batch_input_fn,
        is_debug=is_debug,
        chunk_max_length=args.chunk_max_length
    )

    print("\n" + "=" * 80)
    print("Translation pipeline completed successfully!")
    print(f"Mode: {mode_str}")
    print(f"\nAll outputs saved in folder: {output_dir.resolve()}")

    state = load_state(str(state_file_path))
    translated_dir_name = state.get("translated_output_directory", "translated")
    translated_dataset_dir_path = output_dir / translated_dir_name
    print(f"  - Translated dataset (directory): {translated_dir_name}/")

    fail_log_file = output_dir / "translation_failures.log"
    if fail_log_file.exists():
        print(f"  - Failure log: {fail_log_file.name}")

    all_translations_file = output_dir / "all_translations.json"
    if all_translations_file.exists():
        print(f"  - Aggregated translations: {all_translations_file.name}")

    print(f"  - Batch input folder: batch_input/")
    print(f"  - Batch output folder: batch_output/")
    print(f"  - State file: {Path(state_file_path).name}")

    if not args.resume:
        print(f"\n  (To resume this job if it fails, use: --resume {output_dir})")

    print(f"\n  (To retry failures, use: --retry-failures {output_dir})")
    print("\nYou can now load the translated dataset with:\n")
    print(f"from datasets import load_dataset")
    print(f"dataset = load_dataset('{translated_dataset_dir_path}', split='train')")