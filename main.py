#!/usr/bin/env python3
"""
Main entry point for the Batch Translation Pipeline.

This script runs or resumes a translation pipeline.
"""

import argparse
import asyncio
import sys
import os
import json
import importlib
from pathlib import Path
from datetime import datetime
from typing import Callable, List, Dict, Any

import utils  # Refactored utils
from datasets import load_dataset, Dataset

# --- Type Hinting for Pipeline Functions ---

PrepareBatchInputFn = Callable[[Dataset, str, str, bool, int, int], List[Dict[str, Any]]]

# --- Pipeline Loading ---

def load_pipeline_functions(pipeline_name: str) -> PrepareBatchInputFn:
    """
    Dynamically loads the required functions from a pipeline module.

    Args:
        pipeline_name: The name of the pipeline (e.g., 'rstem', 'rtool').

    Returns:
        The 'prepare_batch_input' function from the specified pipeline.
    """
    try:
        module_path = f"pipelines.{pipeline_name}"
        pipeline_module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"✗ Error: Pipeline '{pipeline_name}' not found.")
        print(f"  (Ensure 'pipelines/{pipeline_name}.py' exists)")
        print(f"  Details: {e}")
        sys.exit(1)

    # Load the required function by its standardized name
    try:
        prepare_batch_fn = getattr(pipeline_module, "prepare_batch_input")
        return prepare_batch_fn
    except AttributeError as e:
        print(f"✗ Error: Pipeline module '{module_path}' is missing a required function.")
        print(f"  'prepare_batch_input' is missing.")
        print(f"  Details: {e}")
        sys.exit(1)

# --- Run Pipeline Logic ---

async def run_pipeline(args: argparse.Namespace):
    """
    Main logic for running or resuming the pipeline.
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

        args_file_path = output_dir / "args.json"
        if args_file_path.exists():
            try:
                with open(args_file_path, 'r', encoding='utf-8') as f:
                    saved_args = json.load(f)
                is_debug = saved_args.get("debug", False)
                print(f"  ✓ Loaded arguments from {args_file_path}")
                print("  ✓ Resuming in DEBUG mode (loaded from args.json)." if is_debug else "  ✓ Resuming in normal mode (loaded from args.json).")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load args.json: {e}. Defaulting to non-debug mode.")
                is_debug = False
        else:
            print(f"  ⚠ Warning: args.json not found. Defaulting to non-debug mode.")
            is_debug = False

        # Allow command-line --debug flag to override
        if args.debug:
            print("  ⚠ Overriding with --debug flag from command line.")
            is_debug = True

    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if is_debug:
            timestamp = f"{timestamp}_debug"
        # Create a sub-folder for the dataset name
        output_dir = Path(f"experiments/{safe_dataset_name}/" + timestamp)
        print(f"Starting new run. Outputs will be saved to: {output_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_file_path = output_dir / "state.json"

    # Save arguments to args.json for new runs
    if not args.resume:
        args_file_path = output_dir / "args.json"
        args_dict = vars(args)
        try:
            with open(args_file_path, 'w', encoding='utf-8') as f:
                json.dump(args_dict, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved arguments to {args_file_path}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not save args.json: {e}")

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
    prepare_batch_fn = load_pipeline_functions(args.pipeline)

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
        enable_chunk=args.enable_chunk,
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


# --- Main Argument Parser ---

def main():
    parser = argparse.ArgumentParser(
        description="Bzantium Batch Translation Pipeline. Runs or resumes a translation job."
    )

    # --- Pipeline Arguments ---
    parser.add_argument(
        '--pipeline',
        type=str,
        required=True,
        choices=['rchat', 'rcode', 'rmath', 'rstem', 'rtool'],
        help="Name of the pipeline logic to use (e.g., 'rstem')."
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help="Path to the source dataset directory (e.g., /path/to/rstem)."
    )

    # --- Model & API Arguments ---
    parser.add_argument(
        '--model',
        type=str,
        default="gpt-5-mini",
        help="Model name (default: gpt-5-mini)."
    )
    parser.add_argument(
        '--max-completion-tokens',
        type=int,
        default=128000,
        help="Max completion tokens for API calls (default: 128000)."
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        default="low",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for API calls (default: low)."
    )

    # --- Batch Job Arguments ---
    parser.add_argument(
        '--disable-chunk',
        action="store_false",
        dest="enable_chunk",
        default=True,
        help="Disable content chunking. By default, chunking is enabled when text exceeds chunk-max-length."
    )
    parser.add_argument(
        '--chunk-max-length',
        type=int,
        default=3000,
        help="Maximum character length for content chunking (default: 3000). Chunking is enabled by default."
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=180,
        help="Interval (seconds) to check batch status (default: 180)."
    )
    parser.add_argument(
        '--max-requests-per-batch',
        type=int,
        default=5000,
        help="Max requests per batch file (default: 5000)."
    )

    # --- Execution Mode Arguments ---
    parser.add_argument(
        '--debug',
        action="store_true",
        help="Run the pipeline with a small subset of records for testing."
    )
    parser.add_argument(
        '--debug-count',
        type=int,
        default=20,
        help="Number of records to process in debug mode (default: 20)."
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar="FOLDER_PATH",
        help="Resume a batch job from the state file in this folder."
    )

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

    # Directly run the pipeline logic
    asyncio.run(run_pipeline(args))

if __name__ == "__main__":
    main()
