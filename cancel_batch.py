"""
OpenAI Batch Job Cancellation Script (Parallel)

Accepts a .state file as an argument, reads all batch IDs
recorded in the 'batches' key, queries their status, and
cancels any jobs that are in 'validating' or 'in_progress'
status using parallel (multi-threaded) API calls.

Usage:
python cancel_batch.py --file /path/to/experiments/your_run/.dataset_model.state
"""

import argparse
import json
import os
import sys
from openai import OpenAI
from typing import Dict, Any, List
from functools import partial
from tqdm.contrib.concurrent import thread_map

# List of cancellable statuses (per OpenAI API)
CANCELLABLE_STATUSES = ["validating", "in_progress"]

def load_state(state_file: str) -> Dict[str, Any]:
    """
    Loads the .state file.
    (Similar to utils.load_state, but embedded for script independence)
    """
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
            print(f"✓ State file loaded: {state_file}")
            return state
    except FileNotFoundError:
        print(f"✗ Error: State file not found at {state_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"✗ Error: Could not parse JSON from {state_file}. File might be corrupt.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading state file: {e}")
        sys.exit(1)

def cancel_single_batch(
    batch_info: Dict[str, Any],
    client: OpenAI
) -> Dict[str, str]:
    """
    (Helper) Checks the status of a single batch job and cancels if necessary.
    This function is designed for use with thread_map.
    """
    batch_key = batch_info.get("batch_key", "unknown_key")
    batch_id = batch_info.get("batch_id")

    if not batch_id:
        return {"status": "skipped_no_id", "key": batch_key, "id": None, "reason": "No 'batch_id'"}

    try:
        # 1. Retrieve the latest status
        batch = client.batches.retrieve(batch_id)
        current_status = batch.status

        # 2. Check status and cancel if appropriate
        if current_status in CANCELLABLE_STATUSES:
            cancelled_batch = client.batches.cancel(batch_id)
            return {
                "status": "cancelled",
                "key": batch_key,
                "id": batch_id,
                "reason": f"Cancelled (was {current_status}, now {cancelled_batch.status})"
            }
        else:
            # Status is 'finalizing', 'completed', 'failed', 'expired', 'cancelling', 'cancelled'
            return {
                "status": "skipped_done",
                "key": batch_key,
                "id": batch_id,
                "reason": f"Not cancellable (status: {current_status})"
            }

    except Exception as e:
        # e.g., "NotFound" error (if job was deleted or ID is invalid)
        return {
            "status": "error",
            "key": batch_key,
            "id": batch_id,
            "reason": f"Error processing: {e}"
        }

def main():
    parser = argparse.ArgumentParser(
        description="Cancel OpenAI batch jobs from a .state file using parallel threads."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .state file containing the batch job IDs."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel threads to use for API calls."
    )

    args = parser.parse_args()

    # 1. Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        print("  Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        print(f"✗ Error initializing OpenAI client: {e}")
        sys.exit(1)

    # 2. Load state file
    print("=" * 80)
    print("OpenAI Batch Job Canceller (Parallel)")
    print("=" * 80)
    state = load_state(args.file)

    batches_state = state.get("batches")
    if not batches_state or not isinstance(batches_state, dict):
        print("✗ No 'batches' key found in state file, or it's not a dictionary. Nothing to cancel.")
        sys.exit(0)

    print(f"Found {len(batches_state)} batch entries in state file.")

    # 3. Create list of jobs to process
    batches_to_process = []
    for batch_key, batch_info in batches_state.items():
        # Add 'batch_key' so the thread helper knows the key
        batch_info['batch_key'] = batch_key
        batches_to_process.append(batch_info)

    # Pre-filter jobs that don't have a batch_id
    valid_jobs = [info for info in batches_to_process if info.get("batch_id")]
    skipped_no_id_count = len(batches_to_process) - len(valid_jobs)

    if not valid_jobs:
        print("No valid batch jobs with 'batch_id' found to process.")
        if skipped_no_id_count > 0:
            print(f"({skipped_no_id_count} entries without 'batch_id' were skipped)")
        sys.exit(0)

    print(f"Querying and attempting to cancel {len(valid_jobs)} jobs in parallel (using {args.workers} workers)...")

    # 4. Create partial function for the thread pool
    cancel_partial = partial(cancel_single_batch, client=client)

    cancelled_count = 0
    skipped_done_count = 0
    error_count = 0

    # 5. Execute in parallel using thread_map
    results = thread_map(
        cancel_partial,
        valid_jobs,
        max_workers=args.workers,
        desc="Cancelling jobs"
    )

    # 6. Aggregate results
    print("\n--- Individual Job Results ---")
    for result in results:
        status = result["status"]
        if status == "cancelled":
            cancelled_count += 1
            print(f"✓ Cancelled: {result['key']} (ID: {result['id']}) - Reason: {result['reason']}")
        elif status == "skipped_done":
            skipped_done_count += 1
            print(f"  Skipped:   {result['key']} (ID: {result['id']}) - Reason: {result['reason']}")
        elif status == "error":
            error_count += 1
            print(f"✗ Error:     {result['key']} (ID: {result['id']}) - Reason: {result['reason']}")

    # 7. Final summary
    total_skipped = skipped_no_id_count + skipped_done_count
    print("\n" + "=" * 80)
    print("Cancellation process finished.")
    print(f"  Successfully requested cancellation: {cancelled_count}")
    print(f"  Skipped (already done / no ID):  {total_skipped}")
    print(f"     (Already completed/failed: {skipped_done_count})")
    print(f"     (No 'batch_id' in state: {skipped_no_id_count})")
    print(f"  Errors (e.g., batch not found):  {error_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
