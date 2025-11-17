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
import asyncio
import math
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from functools import partial

from openai import OpenAI, AsyncOpenAI
from datasets import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Shard size constants
MIB = 1024 * 1024  # 1 Mebibyte
SHARD_SIZE_MIB = 256  # Target shard size in MiB

# Worker count constants
CPU_MULTIPLIER_IO = 4  # For I/O-bound tasks
MAX_WORKERS_IO = 32  # Maximum workers for I/O tasks
MAX_WORKERS_CPU = 16  # Maximum workers for CPU tasks
MIN_WORKERS = 4  # Minimum workers
CPU_DIVIDER_CPU_BOUND = 4  # For CPU-bound tasks (use 1/4 of CPUs)
MAX_CONCURRENT_DOWNLOADS = 25  # Maximum concurrent batch downloads (prevent "too many open files")

# Batch status constants
STATUS_COMPLETED = "completed"
STATUS_DOWNLOADED = "downloaded"
STATUS_FAILED = "failed"
STATUS_UPLOADED = "uploaded"
STATUS_CREATED = "created"
STATUS_VALIDATING = "validating"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINALIZING = "finalizing"
STATUS_EXPIRED = "expired"
STATUS_CANCELLING = "cancelling"
STATUS_CANCELLED = "cancelled"

# Status groups for monitoring
PENDING_STATUSES = [STATUS_VALIDATING, STATUS_IN_PROGRESS, STATUS_FINALIZING]
FAILED_STATUSES = [STATUS_FAILED, STATUS_EXPIRED, STATUS_CANCELLED]
RETRIABLE_STATUSES = [STATUS_FINALIZING, STATUS_FAILED]

# ==============================================================================
# STATE HELPER FUNCTIONS
# ==============================================================================

# Global lock for async-safe state file access
_state_file_lock = None  # Lazy-initialized asyncio.Lock()
_state_file_lock_sync = threading.Lock()  # For sync contexts

def _get_or_create_async_lock():
    """Get or create the asyncio lock (lazy initialization)."""
    global _state_file_lock
    if _state_file_lock is None:
        _state_file_lock = asyncio.Lock()
    return _state_file_lock

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
    """
    Save the current batch job state to a JSON file (thread-safe).

    Uses a global lock to prevent concurrent writes from multiple threads/coroutines.
    Ensures proper file closure to prevent file handle leaks.
    """
    with _state_file_lock_sync:
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Ensure OS flushes to disk
            # File is automatically closed by context manager
            print(f"  ✓ State saved to {state_file}")
        except IOError as e:
            print(f"  ✗ Warning: Could not save state file {state_file}. Error: {e}")
        except Exception as e:
            print(f"  ✗ Warning: Unexpected error saving state file {state_file}. Error: {e}")

async def save_state_async(state_file: str, data: Dict[str, Any]) -> None:
    """
    Save the current batch job state to a JSON file (async-safe).

    Uses an asyncio lock to prevent concurrent writes in async contexts.
    Ensures proper file closure to prevent "too many open files" errors.
    """
    lock = _get_or_create_async_lock()
    async with lock:
        try:
            # Perform file I/O in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _save_state_sync, state_file, data)
        except IOError as e:
            print(f"  ✗ Warning: Could not save state file {state_file}. Error: {e}")
        except Exception as e:
            print(f"  ✗ Warning: Unexpected error saving state file {state_file}. Error: {e}")

def _save_state_sync(state_file: str, data: Dict[str, Any]) -> None:
    """
    Helper function for synchronous file write.
    Ensures file is properly closed and flushed to prevent file handle leaks.
    """
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Ensure OS flushes to disk
        # File is automatically closed by context manager
        print(f"  ✓ State saved to {state_file}")
    except Exception as e:
        # Ensure we don't leave the file handle open on error
        print(f"  ✗ Error in _save_state_sync: {e}")
        raise

def process_retry_batches(state_file_path: str) -> bool:
    """
    Process retry logic: Reset batches with 'finalizing' or 'failed' status.

    This allows those batches to be re-uploaded and re-executed.
    Keeps 'completed' and 'downloaded' batches as-is.

    Args:
        state_file_path: Path to the state.json file

    Returns:
        True if retry was processed, False if no state file or no batches to retry
    """
    print("\n" + "=" * 80)
    print("Processing --retry command...")
    print(f"Loading state file: {state_file_path}")

    try:
        state = load_state(state_file_path)
        original_batches_state = state.get("batches", {})

        if not original_batches_state:
            print("  No 'batches' found in state file. Nothing to retry.")
            print("=" * 80)
            return False

        new_batches_state = {}
        retried_count = 0
        kept_count = 0

        for batch_key, batch_info in original_batches_state.items():
            batch_status = batch_info.get("status", "N/A")

            # Keep batches that are already downloaded OR completed (just need downloading)
            if batch_status in [STATUS_COMPLETED, STATUS_DOWNLOADED]:
                new_batches_state[batch_key] = batch_info
                kept_count += 1
            # Retry batches with finalizing or failed status
            elif batch_status in RETRIABLE_STATUSES:
                print(f"  - Resetting batch: {batch_key} (status: {batch_status})")
                retried_count += 1
            # Also reset any other non-terminal states
            else:
                print(f"  - Resetting batch: {batch_key} (status: {batch_status})")
                retried_count += 1

        state["batches"] = new_batches_state

        # Reset the main pipeline status if it was completed
        if state.get("status") == STATUS_COMPLETED:
            print("  - Resetting main pipeline status from 'completed' to allow retry.")
            del state["status"]

        save_state(state_file_path, state)

        print(f"\n  ✓ Kept {kept_count} 'completed' or 'downloaded' batches.")
        print(f"  ✓ Reset {retried_count} batches for retry.")
        if retried_count > 0:
            print(f"    (Batches with status: {', '.join(RETRIABLE_STATUSES)} or other non-terminal states)")
        print("  ✓ State file updated. Proceeding with pipeline.")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"  ✗ Error processing retry: {e}")
        print("  Proceeding without retry state modification...")
        print("=" * 80)
        return False

# ==============================================================================
# COMMON HELPER FUNCTIONS
# ==============================================================================

def get_optimal_worker_count(task_type: str = "io") -> int:
    """
    Calculate optimal worker count based on task type.

    Args:
        task_type: "io" for I/O-bound tasks, "cpu" for CPU-bound tasks

    Returns:
        Optimal number of workers
    """
    cpu_count = os.cpu_count() or 1

    if task_type == "io":
        return max(min(cpu_count * CPU_MULTIPLIER_IO, MAX_WORKERS_IO), MIN_WORKERS)
    else:  # cpu-bound
        return max(min(cpu_count // CPU_DIVIDER_CPU_BOUND, MAX_WORKERS_CPU), 1)

def is_translation_successful(result: Dict[str, Any]) -> bool:
    """Check if a translation result is successful."""
    return result.get("status_code") == 200 and result.get("content") is not None

def get_batch_file_paths(batch_output_dir: Path, batch_idx: int) -> tuple[Path, Path]:
    """Get standardized output and error file paths for a batch."""
    batch_output_file = batch_output_dir / f"batch_{batch_idx:05d}_output.jsonl"
    batch_error_file = batch_output_dir / f"batch_{batch_idx:05d}_error.jsonl"
    return batch_output_file, batch_error_file

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

def make_body(
    model: str,
    messages: List[Dict[str, str]],
    reasoning_effort: str,
    max_completion_tokens: int
) -> Dict[str, Any]:
    """
    Create the API request body based on model type.

    Args:
        model: Model name (e.g., "gpt-5-...", "gpt-4.1-...")
        messages: List of message dicts with "role" and "content" keys
        reasoning_effort: Reasoning effort level (for gpt-5 models)
        max_completion_tokens: Maximum completion tokens

    Returns:
        Dict containing the request body for the Batch API
    """
    if "gpt-5" in model:
        return {
            "model": model,
            "messages": messages,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens
        }
    elif "gpt-4.1" in model:
        return {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens
        }
    else:
        raise ValueError(f"Model '{model}' not supported.")

# ==============================================================================
# DATASET APPLY & SAVE FUNCTIONS
# ==============================================================================

def build_expected_custom_ids_from_results(all_results: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Build expected custom_ids for each record from all_results.

    Args:
        all_results: Dict mapping custom_id -> result info (successful and failed)

    Returns:
        Dict mapping record_idx -> list of expected custom_ids for that record
    """
    record_to_custom_ids = {}

    for custom_id in all_results:  # Direct iteration over dict
        if custom_id.startswith("record_"):
            parts = custom_id.split("_", 2)  # Split into max 3 parts (optimized)
            if len(parts) >= 2:
                try:
                    record_idx = int(parts[1])
                    record_to_custom_ids.setdefault(record_idx, []).append(custom_id)
                except (ValueError, IndexError):
                    pass

    return record_to_custom_ids

def get_merged_translation(all_results: Dict[str, Dict[str, Any]], base_custom_id: str) -> Optional[str]:
    """
    Finds and merges chunked translations.

    Logic:
    1. Check for a non-chunked translation (e.g., record_0_msg_0_content)
    2. If not found, search for chunked translations
       (e.g., record_0_msg_0_content_chunk_0, chunk_1, ...)
    3. Merge chunks with "\n\n"

    Args:
        all_results: Dict of all translation results
        base_custom_id: Base custom ID without chunk suffix

    Returns:
        Merged translation or None if not found
    """
    # 1. Check for non-chunked translation
    if base_custom_id in all_results:
        result = all_results[base_custom_id]
        if is_translation_successful(result):
            return result["content"]

    # 2. Check for chunked translations
    chunk_translations = []
    chunk_idx = 0
    while True:
        chunk_custom_id = f"{base_custom_id}_chunk_{chunk_idx}"
        if chunk_custom_id in all_results:
            result = all_results[chunk_custom_id]
            if is_translation_successful(result):
                chunk_translations.append(result["content"])
                chunk_idx += 1
            else:
                break  # Failed chunk
        else:
            break

    if chunk_translations:
        # Merge using the same separator used for chunking
        return "\n\n".join(chunk_translations)

    # 3. No translation found
    return None

def _apply_translation_to_record(
    record_with_idx: tuple[int, Dict[str, Any]],
    all_results: Dict[str, Dict[str, Any]],
    expected_custom_ids: Dict[int, List[str]],
    is_debug: bool
) -> tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    (Helper) Apply translations to a single record.

    Returns:
        Tuple of (record_idx, translated_record or None, skip_info or None)
    """
    record_idx, record = record_with_idx

    # Validate: Check if all expected custom_ids for this record are successfully translated
    if record_idx in expected_custom_ids:
        expected_ids = expected_custom_ids[record_idx]
        # Check if any translation is missing or failed
        missing_ids = [
            cid for cid in expected_ids
            if cid not in all_results or not is_translation_successful(all_results[cid])
        ]

        if missing_ids:
            # Skip this record - it has missing/failed translations
            skip_info = {
                "record_idx": record_idx,
                "missing_count": len(missing_ids),
                "total_expected": len(expected_ids),
                "missing_ids": missing_ids[:5]  # Store first 5 for logging
            }
            return (record_idx, None, skip_info)

    translated_record = dict(record)

    try:
        messages = json.loads(record.get("messages", "[]"))
    except json.JSONDecodeError:
        # Skip records with invalid JSON
        skip_info = {
            "record_idx": record_idx,
            "missing_count": 0,
            "total_expected": 0,
            "missing_ids": ["JSON_PARSE_ERROR"]
        }
        return (record_idx, None, skip_info)

    translated_messages = []

    for msg_idx, message in enumerate(messages):
        translated_message = dict(message)

        custom_id_content = f"record_{record_idx}_msg_{msg_idx}_content"
        custom_id_reasoning = f"record_{record_idx}_msg_{msg_idx}_reasoning"

        # Apply 'content' translation
        merged_content = get_merged_translation(all_results, custom_id_content)
        if merged_content is not None:
            if is_debug:
                original_content = message.get("content", "")
                if original_content:
                    translated_message["original_content"] = original_content
            translated_message["content"] = merged_content

        # Apply 'reasoning_content' translation
        merged_reasoning = get_merged_translation(all_results, custom_id_reasoning)
        if merged_reasoning is not None:
            if is_debug:
                original_reasoning = message.get("reasoning_content", "")
                if original_reasoning:
                    translated_message["original_reasoning_content"] = original_reasoning
            translated_message["reasoning_content"] = merged_reasoning

        translated_messages.append(translated_message)

    # Save the list of messages back as a JSON string
    translated_record["messages"] = json.dumps(translated_messages, ensure_ascii=False)

    return (record_idx, translated_record, None)

def apply_translations(dataset: Dataset, all_results: Dict[str, Dict[str, Any]], is_debug: bool = False) -> Dataset:
    """
    Applies translated content to the original dataset (in parallel).

    Args:
        dataset: The original dataset.
        all_results: Dict mapping custom_id -> result info (successful and failed translations).
        is_debug: If True, preserves original content in 'original_content' fields.

    Note:
        Records with any missing/failed translations will be SKIPPED from the output.
        This ensures only records with complete translations are included.
    """
    print("\nApplying translations to dataset...")
    print(f"  Total records in dataset: {len(dataset)}")
    if is_debug:
        print("  (Debug mode: Preserving original content fields)")

    # Count successful translations
    successful_count = sum(1 for r in all_results.values() if is_translation_successful(r))
    print(f"  ✓ Found {successful_count} successful translations in all_results")

    # Build expected custom_ids for each record from all_results
    expected_custom_ids = build_expected_custom_ids_from_results(all_results)
    print(f"  ✓ Found expected custom_ids for {len(expected_custom_ids)} records")

    # Use thread pool for parallel processing (JSON parsing + string operations)
    num_workers = get_optimal_worker_count("io")
    print(f"  Using {num_workers} workers for parallel translation application...")

    # Create partial function with fixed parameters
    apply_partial = partial(
        _apply_translation_to_record,
        all_results=all_results,
        expected_custom_ids=expected_custom_ids,
        is_debug=is_debug
    )

    # Process all records in parallel
    results = thread_map(
        apply_partial,
        enumerate(dataset),
        max_workers=num_workers,
        chunksize=max(len(dataset) // (num_workers * 4), 1),
        desc="Applying translations"
    )

    # Collect translated records and skipped records (maintain order)
    translated_data = []
    skipped_records = []

    for record_idx, translated_record, skip_info in results:
        if translated_record is not None:
            translated_data.append(translated_record)
        elif skip_info is not None:
            skipped_records.append(skip_info)

    # Summary statistics
    total_records = len(dataset)
    successful_records = len(translated_data)
    skipped_count = len(skipped_records)
    success_rate = (successful_records / total_records * 100) if total_records > 0 else 0

    print(f"\n{'='*60}")
    print(f"Translation Application Summary:")
    print(f"{'='*60}")
    print(f"  Total records in dataset:        {total_records:>6}")
    print(f"  Successfully translated records:  {successful_records:>6} ({success_rate:.1f}%)")
    print(f"  Skipped records (incomplete):     {skipped_count:>6} ({100-success_rate:.1f}%)")
    print(f"{'='*60}")

    # Report skipped records details
    if skipped_records:
        print(f"\n⚠ Warning: {skipped_count} record(s) were skipped due to missing/failed translations")
        print(f"  First few skipped records:")
        for skip_info in skipped_records[:5]:
            print(f"    - Record {skip_info['record_idx']}: Missing {skip_info['missing_count']}/{skip_info['total_expected']} translations")
            print(f"      Example missing IDs: {skip_info['missing_ids']}")
        if len(skipped_records) > 5:
            print(f"    ... and {len(skipped_records) - 5} more skipped records")
        print(f"  (Check batch_output/*_error.jsonl files for failure details)")
    else:
        print(f"\n✓ All records have complete translations!")

    from datasets import Dataset as HFDataset
    translated_dataset = HFDataset.from_list(translated_data)
    return translated_dataset

def _extract_shard(idx: int, num_shards: int, dataset: Dataset, out_dirpath: str) -> bool:
    """
    (Helper) Saves a single shard of the dataset to a Parquet file.

    Returns:
        True if successful, False if failed
    """
    try:
        os.makedirs(out_dirpath, exist_ok=True)
        shard = dataset.shard(num_shards=num_shards, index=idx)
        shard_filename = f"shard_{idx:09d}.zst.parquet"
        shard_filepath = os.path.join(out_dirpath, shard_filename)
        shard.to_parquet(shard_filepath, compression="zstd")
        return True
    except Exception as e:
        print(f"\n  ✗ Error saving shard {idx}: {e}")
        return False

def calculate_proper_shard_count(dataset_size_in_bytes: int) -> int:
    """Calculate number of shards based on dataset size."""
    return math.ceil(dataset_size_in_bytes / (SHARD_SIZE_MIB * MIB))

def save_translated_dataset(dataset: Dataset, output_dirpath: Path, original_dataset_path: str) -> None:
    """
    Saves the translated dataset into sharded .zst.parquet files,
    with shard count calculated based on dataset size (256 MiB per shard).
    """
    os.makedirs(output_dirpath, exist_ok=True)
    print(f"\nSaving translated dataset (as Parquet shards) to {output_dirpath}...")

    # Calculate the number of shards based on dataset size
    try:
        dataset_size_in_bytes = dataset._estimate_nbytes()
        num_shards = calculate_proper_shard_count(dataset_size_in_bytes)
        print(f"  Calculated shard count: {num_shards} (dataset size: {dataset_size_in_bytes / (1024**3):.2f} GB)")
    except Exception as e:
        print(f"  ✗ Error calculating shard count: {e}. Defaulting to 1 shard.")
        num_shards = 1

    # Use thread_map instead of process_map to avoid memory issues
    # Parquet writing is I/O-bound and has internal parallelization for compression
    num_workers = get_optimal_worker_count("io")
    print(f"  Using {num_workers} workers (thread-based) for parallel saving...")

    # Create a partial function for thread_map
    extract_shard_partial = partial(
        _extract_shard,
        num_shards=num_shards,
        dataset=dataset,
        out_dirpath=str(output_dirpath),
    )

    # Run parallel saving with thread_map (more memory-efficient)
    try:
        results = thread_map(
            extract_shard_partial,
            range(num_shards),
            max_workers=num_workers,
            chunksize=1,
            desc="Saving shards"
        )

        # Check for any failures
        failed_shards = [i for i, result in enumerate(results) if result is False]

        if failed_shards:
            print(f"\n⚠ Warning: {len(failed_shards)} shard(s) failed to save")
            print(f"  Failed shard indices: {failed_shards[:10]}")
            if len(failed_shards) > 10:
                print(f"  ... and {len(failed_shards) - 10} more")

        print(f"\n✓ Translated dataset saved successfully to {output_dirpath}")
        print(f"  Total records: {len(dataset)}")
        print(f"  Total shards: {num_shards}")
        print(f"  Successful shards: {num_shards - len(failed_shards)}")

        with open(f"{output_dirpath}/stats.json", "w") as io:
            json.dump(
                {"num_rows": len(dataset)},
                io,
                indent=4,
                ensure_ascii=False,
            )

    except Exception as e:
        print(f"\n✗ Error during parallel shard saving: {e}")
        print(f"  Attempting sequential fallback...")
        import traceback
        traceback.print_exc()

        # Fallback: Try sequential saving
        try:
            print(f"\n  Retrying with sequential saving...")
            failed_count = 0
            for idx in tqdm(range(num_shards), desc="Saving shards (sequential)"):
                result = _extract_shard(idx, num_shards, dataset, str(output_dirpath))
                if result is False:
                    failed_count += 1

            if failed_count > 0:
                print(f"\n⚠ Warning: {failed_count} shard(s) failed even in sequential mode")
            else:
                print(f"\n✓ Sequential save completed successfully")

            print(f"  Total records: {len(dataset)}")
            print(f"  Total shards: {num_shards}")

            with open(f"{output_dirpath}/stats.json", "w") as io:
                json.dump(
                    {"num_rows": len(dataset)},
                    io,
                    indent=4,
                    ensure_ascii=False,
                )
        except Exception as fallback_error:
            print(f"\n✗ Sequential fallback also failed: {fallback_error}")
            traceback.print_exc()
            raise

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
    num_workers = get_optimal_worker_count("io")

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

async def upload_batch_file_async(client: AsyncOpenAI, file_path: str, batch_idx: int) -> tuple[int, str]:
    """Uploads a batch file asynchronously using a thread pool executor."""
    loop = asyncio.get_event_loop()
    def _upload():
        # Create a sync client within the thread for thread-safety
        sync_client = OpenAI(api_key=client.api_key)
        with open(file_path, "rb") as file_handle:
            batch_input_file = sync_client.files.create(file=file_handle, purpose="batch")
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
        with open(output_path, "wb") as file_handle:
            file_handle.write(file_response.content)
            file_handle.flush()  # Ensure all data is written

    await loop.run_in_executor(None, _download)
    return batch_idx

async def retrieve_batch_status_async(client: OpenAI, batch_id: str, batch_idx: int, batch_key: str) -> tuple[int, str, str, Any, Optional[Exception]]:
    """Retrieves batch status asynchronously using a thread pool executor."""
    loop = asyncio.get_event_loop()
    def _retrieve():
        return client.batches.retrieve(batch_id)

    try:
        batch = await loop.run_in_executor(None, _retrieve)
        return (batch_idx, batch_key, batch_id, batch, None)
    except Exception as e:
        return (batch_idx, batch_key, batch_id, None, e)

def parse_batch_output_file(batch_output_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses a batch output JSONL file and returns all results.

    Note: Output files only contain successful requests (status_code=200).
          Failed requests are in the error file.

    Returns:
        Dict mapping custom_id -> {status_code: 200, content: "..."}
    """
    results = {}
    if not Path(batch_output_file).exists():
        return results

    try:
        with open(batch_output_file, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                try:
                    result_line = json.loads(line)
                    custom_id = result_line.get("custom_id", "")
                    if not custom_id:
                        continue

                    response = result_line.get("response", {})

                    # Output files always have status_code=200
                    result_obj = {"status_code": 200}

                    try:
                        content = response["body"]["choices"][0]["message"]["content"]
                        result_obj["content"] = content
                    except (KeyError, IndexError, TypeError) as e:
                        # Malformed response body (rare case)
                        result_obj["content"] = None
                        result_obj["error"] = f"Malformed response body: {e}"
                        result_obj["response_body"] = response.get("body")

                    results[custom_id] = result_obj
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

    except json.JSONDecodeError:
        print(f"  ✗ Warning: Could not parse JSON from {batch_output_file}")
    except Exception as e:
        print(f"  ✗ Warning: Error reading {batch_output_file}: {e}")

    return results

def parse_batch_error_file(batch_error_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses a batch error JSONL file (failed requests).

    Note: Error files contain failed requests with status_code != 200.
          Structure is similar to output files but with error responses.

    Returns:
        Dict mapping custom_id -> {status_code: ..., error: "...", ...}
    """
    results = {}
    if not Path(batch_error_file).exists():
        return results

    try:
        with open(batch_error_file, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                try:
                    error_line = json.loads(line)
                    custom_id = error_line.get("custom_id", "")
                    if not custom_id:
                        continue

                    response = error_line.get("response", {})
                    status_code = response.get("status_code")

                    result_obj = {
                        "status_code": status_code,
                        "content": None
                    }

                    # Extract error message from response.body.error
                    error_body = response.get("body", {})
                    if isinstance(error_body, dict) and "error" in error_body:
                        error_info = error_body["error"]
                        result_obj["error"] = error_info.get("message", "Unknown error")
                        result_obj["error_type"] = error_info.get("type")
                        result_obj["error_code"] = error_info.get("code")
                    else:
                        result_obj["error"] = str(error_body) if error_body else "Unknown error"

                    results[custom_id] = result_obj
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

    except json.JSONDecodeError:
        print(f"  ✗ Warning: Could not parse JSON from {batch_error_file}")
    except Exception as e:
        print(f"  ✗ Warning: Error reading {batch_error_file}: {e}")

    return results

# ==============================================================================
# BATCH API PIPELINE (MAIN)
# ==============================================================================

def prepare_batch_input_files(
    state: Dict[str, Any],
    state_file_path: str,
    dataset: Dataset,
    batch_input_dir: Path,
    prepare_batch_input_fn: Callable,
    model: str,
    reasoning_effort: str,
    enable_chunk: bool,
    chunk_max_length: int,
    max_completion_tokens: int,
    max_requests_per_batch: int
) -> List[Dict[str, Any]]:
    """
    Step 1: Prepare and save batch input files.

    Returns:
        List of batch file metadata
    """
    batch_files_metadata = state.get("batch_files_metadata")
    if not batch_files_metadata:
        print("\n--- Step 1: Preparing and saving batch input files ---")

        # 1a. Generate dataset-specific batch requests (CPU-bound)
        all_batch_requests = prepare_batch_input_fn(
            dataset=dataset,
            model=model,
            reasoning_effort=reasoning_effort,
            enable_chunk=enable_chunk,
            chunk_max_length=chunk_max_length,
            max_completion_tokens=max_completion_tokens
        )

        # 1b. Split and save batch files in parallel (I/O-bound)
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

    return batch_files_metadata

async def upload_batch_files(
    batch_files_metadata: List[Dict[str, Any]],
    batches_state: Dict[str, Any],
    async_client: AsyncOpenAI,
    state: Dict[str, Any],
    state_file_path: str
) -> Dict[str, Any]:
    """
    Step 2: Upload all batch files in parallel.

    Returns:
        Updated batches_state
    """
    print(f"\n{'='*80}")
    print(f"Step 2: Uploading all batch files")
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
            batch_state["status"] = STATUS_UPLOADED
            batches_state[batch_key] = batch_state

        state["batches"] = batches_state
        await save_state_async(state_file_path, state)
        print(f"✓ Uploaded {len(to_upload)} batch file(s) in parallel")

    return batches_state

def create_batch_jobs(
    batch_files_metadata: List[Dict[str, Any]],
    batches_state: Dict[str, Any],
    client: OpenAI,
    state: Dict[str, Any],
    state_file_path: str
) -> Dict[str, Any]:
    """
    Step 3: Create all batch jobs.

    Returns:
        Updated batches_state
    """
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
        STATE_SAVE_INTERVAL = 30  # Save state every 30 seconds
        last_save_time = time.time()

        for idx, batch_meta in enumerate(tqdm(to_create, desc="Creating jobs"), 1):
            batch_idx = batch_meta["batch_idx"]
            batch_key = f"batch_{batch_idx}"
            batch_state = batches_state.get(batch_key, {})
            file_id = batch_state["file_id"]
            batch_id = create_batch_job(client, file_id, verbose=False)
            batch_state["batch_id"] = batch_id
            batch_state["status"] = STATUS_CREATED
            batches_state[batch_key] = batch_state
            state["batches"] = batches_state

            # Save state periodically (every 30 seconds or at the end)
            current_time = time.time()
            if (current_time - last_save_time) >= STATE_SAVE_INTERVAL or idx == len(to_create):
                save_state(state_file_path, state)
                last_save_time = current_time

        print(f"✓ Created {len(to_create)} batch job(s)")

    return batches_state

async def monitor_and_download_batches(
    batch_files_metadata: List[Dict[str, Any]],
    batches_state: Dict[str, Any],
    batch_output_dir: Path,
    client: OpenAI,
    async_client: AsyncOpenAI,
    check_interval: int,
    state: Dict[str, Any],
    state_file_path: str
) -> Dict[str, Any]:
    """
    Step 4: Monitor all batch jobs and download immediately when completed.

    Returns:
        Updated batches_state
    """
    print(f"\n{'='*80}")
    print(f"Step 4: Monitoring all batch jobs")
    print(f"{'='*80}")

    # Create a semaphore to limit concurrent downloads (prevent "too many open files")
    download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    print(f"  Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")

    pending_batches = []
    downloaded_batches = set()
    for batch_meta in batch_files_metadata:
        batch_idx = batch_meta["batch_idx"]
        batch_key = f"batch_{batch_idx}"
        batch_state = batches_state.get(batch_key, {})
        batch_output_file, batch_error_file = get_batch_file_paths(batch_output_dir, batch_idx)

        if batch_output_file.exists() or batch_state.get("status") == STATUS_DOWNLOADED:
            downloaded_batches.add(batch_idx)
            if not batch_state.get("output_file"):
                batch_state["output_file"] = str(batch_output_file)
                batch_state["error_file"] = str(batch_error_file)
                batch_state["status"] = STATUS_DOWNLOADED
                batches_state[batch_key] = batch_state
        elif batch_state.get("batch_id"):
            pending_batches.append((batch_idx, batch_key, batch_state["batch_id"]))

    if downloaded_batches:
        print(f"Already downloaded: {len(downloaded_batches)} batch(es)")

    if pending_batches:
        print(f"Monitoring {len(pending_batches)} batch job(s)...")
        print(f"Checking every {check_interval} seconds...\n")

        start_time = time.time()
        download_tasks = {}  # {batch_idx: asyncio.Task}

        # Track status counts for display
        status_counts = {
            STATUS_VALIDATING: 0,
            STATUS_IN_PROGRESS: 0,
            STATUS_FINALIZING: 0
        }

        # Track last save time for periodic state saving
        last_save_time = time.time()
        STATE_SAVE_INTERVAL = 30  # Save state every 30 seconds
        state_changed = False

        while pending_batches or download_tasks:
            completed_this_round = []
            # Reset status counts
            for key in status_counts:
                status_counts[key] = 0

            # 1. Check status of pending jobs (async gather for all batches)
            if pending_batches:
                # Create async tasks for all batch status retrievals
                retrieve_tasks = [
                    retrieve_batch_status_async(client, batch_id, batch_idx, batch_key)
                    for batch_idx, batch_key, batch_id in pending_batches
                ]

                # Gather all results concurrently
                retrieve_results = await asyncio.gather(*retrieve_tasks)

                # Process results
                for batch_idx, batch_key, batch_id, batch, error in retrieve_results:
                    if error:
                        print(f"[{time.strftime('%H:%M:%S')}] Error checking batch {batch_idx:05d}: {error}")
                        continue

                    batch_state = batches_state[batch_key]

                    # Count current status
                    if batch.status in status_counts:
                        status_counts[batch.status] += 1

                    if batch.status == STATUS_COMPLETED:
                        batch_state["output_file_id"] = batch.output_file_id
                        batch_state["error_file_id"] = batch.error_file_id
                        batch_state["status"] = STATUS_COMPLETED
                        batches_state[batch_key] = batch_state
                        state["batches"] = batches_state
                        state_changed = True

                        batch_output_file, batch_error_file = get_batch_file_paths(batch_output_dir, batch_idx)

                        # Start async download tasks for both output and error files
                        # Wrap with semaphore to limit concurrent file operations
                        async def download_with_semaphore(semaphore, output_file_id, error_file_id, output_path, error_path, idx):
                            async with semaphore:
                                tasks = []
                                if output_file_id:
                                    tasks.append(
                                        download_batch_results_async(async_client, output_file_id, output_path, idx)
                                    )
                                if error_file_id:
                                    tasks.append(
                                        download_batch_results_async(async_client, error_file_id, error_path, idx)
                                    )
                                if tasks:
                                    return await asyncio.gather(*tasks, return_exceptions=True)
                                return []

                        download_task = download_with_semaphore(
                            download_semaphore,
                            batch.output_file_id,
                            batch.error_file_id,
                            str(batch_output_file),
                            str(batch_error_file),
                            batch_idx
                        )
                        download_tasks[batch_idx] = asyncio.create_task(download_task)

                        completed_this_round.append((batch_idx, batch_key, batch_id))

                    elif batch.status in FAILED_STATUSES:
                        print(f"[{time.strftime('%H:%M:%S')}] Batch {batch_idx:05d}: FAILED (status: {batch.status})")
                        batch_state["status"] = STATUS_FAILED
                        batch_state["error"] = batch.status
                        batches_state[batch_key] = batch_state
                        state["batches"] = batches_state
                        state_changed = True
                        completed_this_round.append((batch_idx, batch_key, batch_id))

            # Remove completed/failed jobs from monitoring list
            for item in completed_this_round:
                pending_batches.remove(item)

            # 2. Check status of download tasks
            done_downloads = []
            for batch_idx, task in download_tasks.items():
                if task.done():
                    try:
                        await task  # Check for exceptions
                        batch_key = f"batch_{batch_idx}"
                        batch_state = batches_state[batch_key]
                        batch_output_file, batch_error_file = get_batch_file_paths(batch_output_dir, batch_idx)
                        batch_state["output_file"] = str(batch_output_file)
                        batch_state["error_file"] = str(batch_error_file)
                        batch_state["status"] = STATUS_DOWNLOADED
                        batches_state[batch_key] = batch_state
                        state["batches"] = batches_state
                        state_changed = True
                        done_downloads.append(batch_idx)
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Error downloading batch {batch_idx:05d}: {e}")
                        done_downloads.append(batch_idx)  # Remove task even if failed

            # Remove completed download tasks
            for batch_idx in done_downloads:
                del download_tasks[batch_idx]

            # Save state periodically (every 30 seconds) if there were changes
            current_time = time.time()
            if state_changed and (current_time - last_save_time) >= STATE_SAVE_INTERVAL:
                await save_state_async(state_file_path, state)
                last_save_time = current_time
                state_changed = False

            # 3. Sleep if there are still tasks
            if pending_batches or download_tasks:
                elapsed = time.time() - start_time
                total_batches = len(batch_files_metadata)
                completed_count = total_batches - len(pending_batches) - len(download_tasks)
                monitoring_count = len(pending_batches)
                downloading_count = len(download_tasks)

                # Build detailed status string
                status_parts = []
                if status_counts[STATUS_VALIDATING] > 0:
                    status_parts.append(f"validating: {status_counts[STATUS_VALIDATING]}")
                if status_counts[STATUS_IN_PROGRESS] > 0:
                    status_parts.append(f"in_progress: {status_counts[STATUS_IN_PROGRESS]}")
                if status_counts[STATUS_FINALIZING] > 0:
                    status_parts.append(f"finalizing: {status_counts[STATUS_FINALIZING]}")

                status_detail = f" ({', '.join(status_parts)})" if status_parts else ""

                print(f"[{time.strftime('%H:%M:%S')}] Status: {completed_count}/{total_batches} done | "
                      f"{monitoring_count} monitoring{status_detail} | "
                      f"{downloading_count} downloading | Elapsed: {elapsed/60:.1f}m")
                await asyncio.sleep(check_interval)

        # Final state save after all batches are completed
        if state_changed:
            await save_state_async(state_file_path, state)

        print(f"\n✓ All batch jobs completed and downloaded!")
    else:
        print(f"✓ All batches already completed and downloaded!")

    return batches_state

def _parse_single_batch(
    batch_meta: Dict[str, Any],
    batches_state: Dict[str, Any]
) -> tuple[int, Optional[Dict[str, Dict[str, Any]]]]:
    """
    (Helper) Parse a single batch's output and error files.

    Returns:
        Tuple of (batch_idx, results_dict or None if skipped)
    """
    batch_idx = batch_meta["batch_idx"]
    batch_key = f"batch_{batch_idx}"
    batch_state = batches_state.get(batch_key, {})

    if batch_state.get("status") != STATUS_DOWNLOADED:
        return (batch_idx, None)  # Skipped

    results = {}

    # Parse output file (contains successful API responses)
    batch_output_file = batch_state.get("output_file")
    if batch_output_file and Path(batch_output_file).exists():
        output_results = parse_batch_output_file(batch_output_file)
        results.update(output_results)

    # Parse error file (contains failed requests)
    batch_error_file = batch_state.get("error_file")
    if batch_error_file and Path(batch_error_file).exists():
        error_results = parse_batch_error_file(batch_error_file)
        if error_results:
            results.update(error_results)

    return (batch_idx, results)

def aggregate_batch_results(
    batch_files_metadata: List[Dict[str, Any]],
    batches_state: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Step 5: Aggregate and parse all results from output and error files.

    Returns:
        Dict mapping custom_id -> result info
    """
    print(f"\n{'='*80}")
    print("Step 5: Aggregating results from all batches (output + error files)")
    print(f"{'='*80}")

    # Use thread pool for parallel I/O-bound file parsing
    num_workers = get_optimal_worker_count("io")
    print(f"  Using {num_workers} workers for parallel parsing...")

    # Create partial function for thread_map
    parse_partial = partial(_parse_single_batch, batches_state=batches_state)

    # Parse all batches in parallel
    parse_results = thread_map(
        parse_partial,
        batch_files_metadata,
        max_workers=num_workers,
        chunksize=1,
        desc="Parsing batch results"
    )

    # Aggregate results and track skipped batches
    all_results = {}
    skipped_batches = []

    for batch_idx, batch_results in parse_results:
        if batch_results is None:
            skipped_batches.append(batch_idx)
        else:
            all_results.update(batch_results)

    # Report skipped batches
    if skipped_batches:
        print(f"\n⚠ Warning: {len(skipped_batches)} batch(es) not completed or downloaded, skipped:")
        print(f"  Batch IDs: {', '.join([f'{idx:05d}' for idx in skipped_batches[:5]])}")
        if len(skipped_batches) > 5:
            print(f"  ... and {len(skipped_batches) - 5} more")

    # Count successes and failures
    successful_count = sum(1 for r in all_results.values() if is_translation_successful(r))
    failed_count = len(all_results) - successful_count

    print(f"\n✓ Aggregated {len(all_results)} total results from {len(batch_files_metadata) - len(skipped_batches)} batches")
    rate = (successful_count / len(all_results) * 100) if len(all_results) > 0 else 0
    print(f"  - Successful: {successful_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Success Rate: {rate:.1f}%")

    return all_results

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
    enable_chunk: bool,
    is_debug: bool = False,
):
    """
    Execute the full Batch API pipeline with multiple batch files and state resume.

    This orchestrates the entire pipeline by calling step-specific functions:
    1. Prepare and save batch input files
    2. Upload batch files in parallel
    3. Create batch jobs
    4. Monitor and download results
    5. Aggregate results from all batches
    6. Apply translations to dataset
    7. Save final translated dataset
    """
    print("\n" + "="*80)
    print(f"Running Batch API Translation to Korean")
    print(f"Model: {model}, Reasoning effort: {reasoning_effort}")
    print("="*80)

    state = load_state(state_file_path)

    # Check if pipeline already completed
    if state.get("status") == STATUS_COMPLETED:
        translated_dir_name = state.get("translated_output_directory", "translated")
        translated_dataset_dir = output_dir / translated_dir_name
        print(f"✓ Pipeline already completed for this dataset. Final directory:")
        print(f"  {translated_dataset_dir}")
        print(f"  To re-run, delete the state file and output files in this folder.")
        return

    # Initialize OpenAI clients
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Setup directories
    batch_input_dir = output_dir / "batch_input"
    batch_output_dir = output_dir / "batch_output"
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Prepare and save batch input files
        batch_files_metadata = prepare_batch_input_files(
            state=state,
            state_file_path=state_file_path,
            dataset=dataset,
            batch_input_dir=batch_input_dir,
            prepare_batch_input_fn=prepare_batch_input_fn,
            model=model,
            reasoning_effort=reasoning_effort,
            enable_chunk=enable_chunk,
            chunk_max_length=chunk_max_length,
            max_completion_tokens=max_completion_tokens,
            max_requests_per_batch=max_requests_per_batch
        )

        batches_state = state.get("batches", {})

        # Step 2: Upload all batch files (in parallel)
        batches_state = await upload_batch_files(
            batch_files_metadata=batch_files_metadata,
            batches_state=batches_state,
            async_client=async_client,
            state=state,
            state_file_path=state_file_path
        )

        # Step 3: Create all batch jobs
        batches_state = create_batch_jobs(
            batch_files_metadata=batch_files_metadata,
            batches_state=batches_state,
            client=client,
            state=state,
            state_file_path=state_file_path
        )

        # Step 4: Monitor and download batch results
        batches_state = await monitor_and_download_batches(
            batch_files_metadata=batch_files_metadata,
            batches_state=batches_state,
            batch_output_dir=batch_output_dir,
            client=client,
            async_client=async_client,
            check_interval=check_interval,
            state=state,
            state_file_path=state_file_path
        )

        # Step 5: Aggregate results from all batches
        all_results = aggregate_batch_results(
            batch_files_metadata=batch_files_metadata,
            batches_state=batches_state
        )

        # Check if we have any successful translations
        successful_count = sum(1 for r in all_results.values() if is_translation_successful(r))
        if successful_count == 0:
            print("\n✗ No successful translations found. Exiting.")
            return

        # Step 6: Apply Translations to dataset
        print(f"\n{'='*80}")
        print("Step 6: Applying translations to dataset")
        print(f"{'='*80}")

        translated_dataset = apply_translations(
            dataset=dataset,
            all_results=all_results,
            is_debug=is_debug
        )

        # Check if any records were successfully translated
        if len(translated_dataset) == 0:
            print("\n✗ No records with complete translations. Cannot save dataset.")
            print("  All records had missing translations.")
            print("  Tip: Check batch error files in batch_output/ for failure details.")
            return

        # Step 7: Save Final Translated Dataset
        print(f"\n{'='*80}")
        print("Step 7: Saving final translated dataset (as Parquet shards)")
        print(f"{'='*80}")

        output_shard_dir_name = "translated"
        translated_dataset_dir = output_dir / output_shard_dir_name
        save_translated_dataset(translated_dataset, translated_dataset_dir, original_dataset_path)

        state["translated_output_directory"] = output_shard_dir_name
        state["status"] = STATUS_COMPLETED
        save_state(state_file_path, state)

        print(f"\n✓ Final translated dataset shards saved to: {translated_dataset_dir}")

    except Exception as e:
        print(f"\n\n✗✗✗ An unexpected error occurred: {e} ✗✗✗")
        print("  Current state has been saved. You can re-run the script to resume.")
        import traceback
        traceback.print_exc()
        save_state(state_file_path, state)
        sys.exit(1)
