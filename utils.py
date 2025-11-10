"""
Common translation pipeline utilities (Batch-only)

- State management (load/save)
- Batch API wrappers (upload, create, monitor, download, parse)
- Parallel batch file creation (split_and_save_batch_files)
- Dataset saving logic (apply_translations, save_translated_dataset)
- Batch pipeline runners (run_batch_pipeline)
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

from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from datetime import datetime
from tqdm.contrib.concurrent import process_map, thread_map

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

def chunk_content(text: str, max_length: int) -> List[str]:
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

# ==============================================================================
# DATASET APPLY & SAVE FUNCTIONS
# ==============================================================================

def apply_translations(dataset: Dataset, translations: Dict[str, str], is_debug: bool = False) -> Dataset:
    """
    Applies translated content to the original dataset.

    Args:
        dataset: The original dataset.
        translations: A mapping from custom_id to translated text.
        is_debug: If True, preserves original content in 'original_content' fields.
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
                Finds and merges chunked translations.

                Logic:
                1. Check for a non-chunked translation (e.g., record_0_msg_0_content)
                2. If not found, search for chunked translations
                   (e.g., record_0_msg_0_content_chunk_0, chunk_1, ...)
                3. Merge chunks with "\n\n"
                """
                # 1. Check for non-chunked translation
                if base_custom_id in translations:
                    return translations[base_custom_id]

                # 2. Check for chunked translations
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
                    # Merge using the same separator used for chunking
                    return "\n\n".join(chunk_translations)

                # 3. No translation found
                return None

            # Apply 'content' translation
            merged_content = get_merged_translation(custom_id_content)
            if merged_content is not None:
                if is_debug:
                    original_content = message.get("content", "")
                    if original_content:
                        translated_message["original_content"] = original_content
                translated_message["content"] = merged_content

            # Apply 'reasoning_content' translation
            merged_reasoning = get_merged_translation(custom_id_reasoning)
            if merged_reasoning is not None:
                if is_debug:
                    original_reasoning = message.get("reasoning_content", "")
                    if original_reasoning:
                        translated_message["original_reasoning_content"] = original_reasoning
                translated_message["reasoning_content"] = merged_reasoning

            translated_messages.append(translated_message)

        # Save the list of messages back as a JSON string
        translated_record["messages"] = json.dumps(translated_messages, ensure_ascii=False)
        translated_data.append(translated_record)

    print(f"✓ Applied translations to {len(translated_data)} records")
    from datasets import Dataset as HFDataset
    translated_dataset = HFDataset.from_list(translated_data)
    return translated_dataset

def _extract_shard(idx: int, num_shards: int, dataset: Dataset, out_dirpath: str):
    """
    (Helper) Saves a single shard of the dataset to a Parquet file.
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
    Saves the translated dataset into sharded .zst.parquet files,
    matching the shard count of the original dataset.
    """
    os.makedirs(output_dirpath, exist_ok=True)
    print(f"\nSaving translated dataset (as Parquet shards) to {output_dirpath}...")

    # Count the number of shards in the original dataset path
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

    # Use process_map for parallel, CPU-bound Parquet saving
    num_proc = max(os.cpu_count() // 4, 1)  # Use 1/4 of CPUs, min 1
    print(f"  Using {num_proc} workers for parallel saving.")

    # Create a partial function for process_map
    extract_shard_partial = partial(
        _extract_shard,
        num_shards=num_shards,
        dataset=dataset,
        out_dirpath=str(output_dirpath),
    )

    # Run parallel saving
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
    """(Helper) Saves a single batch .jsonl file in a parallel thread."""
    start_idx = batch_idx * max_per_batch
    end_idx = min((batch_idx + 1) * max_per_batch, total_requests)
    batch_requests = all_requests[start_idx:end_idx]

    # Pad filename, e.g., batch_00001.jsonl
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
        return {} # Return empty dict on failure


def split_and_save_batch_files(
    all_batch_requests: List[Dict],
    batch_input_dir: Path,
    max_requests_per_batch: int
) -> List[Dict[str, Any]]:
    """
    (Common Function)
    Takes a list of all batch requests and saves them into
    multiple .jsonl files in parallel using multi-threading.
    """
    batch_input_dir.mkdir(parents=True, exist_ok=True)

    total_requests = len(all_batch_requests)
    if total_requests == 0:
        print("  ⚠ Warning: No batch requests generated.")
        return []

    num_batches = (total_requests + max_requests_per_batch - 1) // max_requests_per_batch

    print(f"\n✓ Total translation requests: {total_requests}")
    print(f"  Splitting into {num_batches} batch file(s) (using parallel I/O)...")

    # Use a thread pool for I/O-bound tasks
    num_workers = max(min(os.cpu_count() * 4, 32), 4)

    save_partial = partial(
        _save_batch_file_parallel,
        all_requests=all_batch_requests,
        max_per_batch=max_requests_per_batch,
        total_requests=total_requests,
        input_dir=batch_input_dir
    )

    batch_files_metadata = []

    # Run parallel I/O
    results = thread_map(
        save_partial,
        range(num_batches),
        max_workers=num_workers,
        chunksize=1, # Each thread handles one file
        desc="Saving batch files"
    )

    # Filter out any failed saves (empty dicts)
    batch_files_metadata = [res for res in results if res]

    # Sort by batch_idx to ensure order, though thread_map should preserve it
    batch_files_metadata.sort(key=lambda m: m["batch_idx"])

    print(f"\n✓ Created {len(batch_files_metadata)} batch files in {batch_input_dir}")
    if len(batch_files_metadata) != num_batches:
        print(f"  ✗ Warning: Expected {num_batches} files, but only {len(batch_files_metadata)} were saved successfully.")

    return batch_files_metadata

def upload_batch_file(client: OpenAI, file_path: str, verbose: bool = True) -> str:
    """Uploads a batch file to OpenAI."""
    if verbose:
        print(f"\nUploading {file_path} to OpenAI...")
    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    file_id = batch_input_file.id
    if verbose:
        print(f"✓ File uploaded successfully. File ID: {file_id}")
    return file_id

async def upload_batch_file_async(client: AsyncOpenAI, file_path: str, batch_idx: int) -> tuple[int, str]:
    """Uploads a batch file asynchronously using a thread pool executor."""
    loop = asyncio.get_event_loop()
    def _upload():
        # Create a sync client within the thread for thread-safety
        sync_client = OpenAI(api_key=client.api_key)
        with open(file_path, "rb") as f:
            batch_input_file = sync_client.files.create(file=f, purpose="batch")
        return batch_input_file.id

    file_id = await loop.run_in_executor(None, _upload)
    return batch_idx, file_id

def create_batch_job(client: OpenAI, file_id: str, verbose: bool = True) -> str:
    """Creates a batch job from an uploaded file ID."""
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

def monitor_batch_job(client: OpenAI, batch_id: str, check_interval: int) -> Dict[str, Any]:
    """Monitors a batch job until completion."""
    print(f"\nMonitoring batch job {batch_id}...")
    print(f"Checking status every {check_interval} seconds...\n")
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
    """Downloads batch results from an output file ID."""
    if verbose:
        print(f"\nDownloading batch results...")
    file_response = client.files.content(output_file_id)
    with open(output_path, "wb") as f:
        f.write(file_response.content)
    if verbose:
        print(f"✓ Results saved to {output_path}")

async def download_batch_results_async(client: AsyncOpenAI, output_file_id: str, output_path: str, batch_idx: int) -> int:
    """Downloads batch results asynchronously using a thread pool executor."""
    loop = asyncio.get_event_loop()
    def _download():
        sync_client = OpenAI(api_key=client.api_key)
        file_response = sync_client.files.content(output_file_id)
        with open(output_path, "wb") as f:
            f.write(file_response.content)

    await loop.run_in_executor(None, _download)
    return batch_idx

def parse_batch_results(batch_output_file: str) -> (Dict[str, str], List[Dict[str, Any]]):
    """Parses a batch output JSONL file, separating successes and failures."""
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
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    max_requests_per_batch: int,
    check_interval: int,
    chunk_max_length: int,
    is_debug: bool = False,
):
    """
    Execute the full Batch API pipeline with multiple batch files and state resume.
    """
    print("\n" + "="*80)
    print(f"Running Batch API Translation to Korean")
    print(f"Model: {model}, Reasoning effort: {reasoning_effort}")
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

            # 1a. (Specific) Generate dataset-specific batch requests (CPU-bound)
            all_batch_requests = prepare_batch_input_fn(
                dataset=dataset,
                model=model,
                reasoning_effort=reasoning_effort,
                chunk_max_length=chunk_max_length
            )

            # 1b. (Common) Split and save batch files in parallel (I/O-bound)
            batch_files_metadata = split_and_save_batch_files(
                all_batch_requests=all_batch_requests,
                batch_input_dir=batch_input_dir,
                max_requests_per_batch=max_requests_per_batch
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
            print(f"Checking every {check_interval} seconds...\n")

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
                    await asyncio.sleep(check_interval)

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

        # Save aggregated translations for potential retry
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
