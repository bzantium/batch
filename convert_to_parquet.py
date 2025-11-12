import argparse
import os
import math
import json
from datasets import load_dataset
from tqdm.contrib.concurrent import process_map
from functools import partial


def _extract_shard(idx: int, num_shards: int, dataset, output_path: str):
    """
    (Helper) Saves a single shard of the dataset to a Parquet file.
    """
    try:
        shard = dataset.shard(num_shards=num_shards, index=idx)
        shard_filename = f"shard_{idx:09d}.zst.parquet"
        shard_filepath = os.path.join(output_path, shard_filename)
        shard.to_parquet(shard_filepath, compression="zstd")
    except Exception as e:
        print(f"Error saving shard {idx}: {e}")

def calculate_proper_shard_count(dataset_size_in_bytes):
    MiB = 1024 * 1024 # 1 Mebibyte = 1024 * 1024 bytes
    return math.ceil(dataset_size_in_bytes / (256 * MiB))


def save_dataset_to_parquet(dataset, output_path: str):
    num_proc = max(os.cpu_count() // 4, 1)
    num_shards = calculate_proper_shard_count(dataset._estimate_nbytes())
    os.makedirs(output_path, exist_ok=True)

    extract_shard = partial(
        _extract_shard,
        num_shards=num_shards,
        dataset=dataset,
        output_path=output_path,
    )
    process_map(
        extract_shard,
        range(num_shards),
        max_workers=num_proc,
        chunksize=max(math.ceil(num_shards / num_proc), 1)
    )

    with open(f"{output_path}/stats.json", "w") as io:
        json.dump(
            {"num_rows": dataset.num_rows},
            io,
            indent=4,
            ensure_ascii=False,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.input_path, split="train")
    save_dataset_to_parquet(dataset, args.output_path)
