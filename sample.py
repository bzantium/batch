from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()

for dir in ["rcode-0528", "rmath-0528", "rstem", "rchat"]:
    print(f"Sampling {dir}...")
    # ds = load_dataset(f"/data/ib-huawei-nas-lmt_980/datasets/Kanana-2-Post-Training-Dataset/{dir}", split="train")
    ds = load_dataset(f"/mnt/kakao/lmt/datasets/Kanana-2-Post-Training-Dataset/{dir}", split="train", num_proc=16)
    sampled_ds = ds.shuffle(seed=42).take(int(len(ds) * 0.2))
    print(f"Sampled {len(sampled_ds)} records from total {len(ds)} records")
    sampled_ds.to_json(f"sample/{dir}/sample.jsonl", lines=True, num_proc=16)
    print(f"Saved to sample/{dir}/sample.jsonl")