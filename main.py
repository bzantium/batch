#!/usr/bin/env python3
"""
Main entry point for the Batch Translation Pipeline.

This script provides two main commands:
1.  'run':   Run a new translation pipeline from scratch or resume an existing one.
2.  'retry': Retry failed translations from a completed or partially failed run.
"""

import argparse
import asyncio
import sys
import os
import importlib
from pathlib import Path
from datetime import datetime
from typing import Callable, List, Dict, Any

import utils  # Refactored utils
from datasets import load_dataset, Dataset

# --- Type Hinting for Pipeline Functions ---

PrepareBatchInputFn = Callable[[Dataset, str, str, int], List[Dict[str, Any]]]
ExtractFailedRecordsFn = Callable[[Dataset, List[str]], Dict[str, Dict[str, Any]]]
PrepareRetryRequestsFn = Callable[[Dict[str, Dict[str, Any]], str, str, int, int], List[Dict[str, Any]]]

# --- Pipeline Loading ---

def load_pipeline_functions(pipeline_name: str) -> (
    PrepareBatchInputFn, ExtractFailedRecordsFn, PrepareRetryRequestsFn
):
    """
    Dynamically loads the required functions from a pipeline module.

    Args:
        pipeline_name: The name of the pipeline (e.g., 'rstem', 'rtool').

    Returns:
        A tuple containing the three required functions:
        (prepare_batch_input, extract_failed_records, prepare_retry_requests)
    """
    try:
        module_path = f"pipelines.{pipeline_name}"
        pipeline_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"✗ Error: Pipeline '{pipeline_name}' not found.")
        print(f"  (Ensure 'pipelines/{pipeline_name}.py' exists)")
        print(f"  Details: {e}")
        sys.exit(1)

    # Load the required functions by their standardized names
    try:
        # [MODIFIED] Look for fixed function names, not dynamic ones
        prepare_batch_fn = getattr(pipeline_module, "prepare_batch_input")
        extract_failed_fn = getattr(pipeline_module, "extract_failed_records")
        prepare_retry_fn = getattr(pipeline_module, "prepare_retry_requests")
        return prepare_batch_fn, extract_failed_fn, prepare_retry_fn
    except AttributeError as e:
        print(f"✗ Error: Pipeline module '{module_path}' is missing a required function.")
        print(f"  One of 'prepare_batch_input', "
              f"'extract_failed_records', or "
              f"'prepare_retry_requests' is missing.")
        print(f"  Details: {e}")
        sys.exit(1)

# --- Command 1: Run Pipeline ---

async def run_pipeline(args: argparse.Namespace):
    """
    Main logic for the 'run' command.
    Adapts logic from the original 'utils.main_runner'.
    """
    safe_dataset_name = Path(args.data).name
    safe_model_name = args.model.replace("/", "_").replace(".", "-")
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
        state = utils.load_state(str(state_file_path))
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
            timestamp = f"{timestamp}_debug"
        # Create a sub-folder for the dataset name
        output_dir = Path(f"experiments/{safe_dataset_name}/" + timestamp)
        print(f"Starting new run. Outputs will be saved to: {output_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_file_path = output_dir / f".{base_filename}.state"

    print("=" * 80)
    print(f"OpenAI Batch Translation Pipeline to Korean")
    mode_str = "Batch API"
    if args.resume:
        mode_str += " (Resuming)"
    print(f"Pipeline: {args.pipeline}")
    print(f"Model: {args.model}")
    print(f"Mode: {mode_str}")
    print(f"Debug: {is_debug}")
    print(f"Output Folder: {output_dir.resolve()}")
    print(f"Source Dataset: {args.data}")
    print("=" * 80)

    # Load the specific pipeline functions
    prepare_batch_fn, _, _ = load_pipeline_functions(args.pipeline)

    print("\nLoading dataset...")
    dataset = load_dataset(args.data, split="train")

    if is_debug:
        dataset = dataset.take(args.debug_count)

    print(f"✓ Loaded {len(dataset)} records")

    # Call the main batch pipeline function from utils
    await utils.run_batch_pipeline(
        dataset=dataset,
        output_dir=output_dir,
        state_file_path=str(state_file_path),
        original_dataset_path=args.data,
        prepare_batch_input_fn=prepare_batch_fn,
        # Pass all configurable arguments
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        max_requests_per_batch=args.max_requests_per_batch,
        check_interval=args.check_interval,
        chunk_max_length=args.chunk_max_length,
        is_debug=is_debug
    )

    print("\n" + "=" * 80)
    print("Translation pipeline completed successfully!")
    print(f"Mode: {mode_str}")
    print(f"\nAll outputs saved in folder: {output_dir.resolve()}")

    state = utils.load_state(str(state_file_path))
    translated_dir_name = state.get("translated_output_directory", "translated")
    print(f"  - Translated dataset (directory): {translated_dir_name}/")

    if not args.resume:
        print(f"\n  (To resume this job if it fails, use: --resume {output_dir})")
    print(f"  (To retry failures, use: --retry-failures {output_dir})")


# --- Command 2: Retry Failures ---

async def retry_pipeline(args: argparse.Namespace):
    """
    Main logic for the 'retry' command.
    Adapts logic from the original 'utils.handle_retry_failures'.
    """
    output_dir = Path(args.output_dir)

    if not output_dir.is_dir():
        print(f"\n✗ Error: Output folder not found: {args.output_dir}")
        sys.exit(1)

    print("=" * 80)
    print(f"Retry Failed Translations Mode")
    print(f"Pipeline: {args.pipeline}")
    print(f"Output Folder: {output_dir.resolve()}")
    print(f"Source Dataset: {args.data}")
    print("=" * 80)

    # Load the specific pipeline functions
    _, extract_failed_fn, prepare_retry_fn = load_pipeline_functions(args.pipeline)

    # Call the retry handler from utils
    await utils.handle_retry_failures(
        output_dir=output_dir,
        original_dataset_path=args.data,
        extract_failed_records_fn=extract_failed_fn,
        prepare_retry_requests_fn=prepare_retry_fn,
        # Pass all configurable arguments
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        check_interval=args.check_interval,
        chunk_max_length=args.chunk_max_length
    )

# --- Main Argument Parser ---

def main():
    parser = argparse.ArgumentParser(description="Bzantium Batch Translation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Common arguments for 'run' and 'retry' ---
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--pipeline',
        type=str,
        required=True,
        choices=['rchat', 'rcode', 'rmath', 'rstem', 'rtool'],
        help="Name of the pipeline logic to use (e.g., 'rstem')."
    )
    common_parser.add_argument(
        '--data',
        type=str,
        required=True,
        help="Path to the source dataset directory (e.g., /path/to/rstem)."
    )
    common_parser.add_argument(
        '--model',
        type=str,
        default="gpt-5-mini",
        help="Model name (default: gpt-5-mini)."
    )
    common_parser.add_argument(
        '--max-completion-tokens',
        type=int,
        default=128000,
        help="Max completion tokens for API calls (default: 128000)."
    )
    common_parser.add_argument(
        '--reasoning-effort',
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for API calls (default: medium)."
    )
    common_parser.add_argument(
        '--chunk-max-length',
        type=int,
        default=4000,
        help="Maximum character length for content chunking (default: 4000)."
    )
    common_parser.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help="Interval (seconds) to check batch status (default: 60)."
    )

    # --- Parser 1: 'run' ---
    parser_run = subparsers.add_parser(
        "run",
        parents=[common_parser],
        help="Run a new translation pipeline or resume a previous one."
    )
    parser_run.add_argument(
        '--debug',
        action="store_true",
        help="Run the pipeline with a small subset of records for testing."
    )
    parser_run.add_argument(
        '--debug-count',
        type=int,
        default=20,
        help="Number of records to process in debug mode (default: 20)."
    )
    parser_run.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar="FOLDER_PATH",
        help="Resume a batch job from the state file in this folder."
    )
    parser_run.add_argument(
        '--max-requests-per-batch',
        type=int,
        default=5000,
        help="Max requests per batch file (default: 5000)."
    )
    parser_run.set_defaults(func=run_pipeline)

    # --- Parser 2: 'retry' ---
    parser_retry = subparsers.add_parser(
        "retry",
        parents=[common_parser],
        help="Retry failed translations from a previous run."
    )
    parser_retry.add_argument(
        '--output-dir',
        type=str,
        required=True,
        metavar="FOLDER_PATH",
        help="Path to the *existing* output folder (containing .log and .state files)."
    )
    parser_retry.set_defaults(func=retry_pipeline)

    # --- Execute ---
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        print("  Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Run the selected command function (run_pipeline or retry_pipeline)
    if hasattr(args, 'func'):
        asyncio.run(args.func(args))
    else:
        parser.print_help(sys.stderr)

if __name__ == "__main__":
    main()
