import json
import argparse
from datasets import load_dataset
from convert_to_parquet import save_dataset_to_parquet


# ds = load_dataset("experiments/rtool/2025-11-10_19-27-00_debug/translated", split='train')
# regex = r"^<tool_call>\n[\s\S]*?\n</tool_call>$"

# for row in ds:
#     messages = json.loads(row['messages'])
#     for msg in messages:
#         if msg['role'] == 'assistant':
#             if re.match(regex, msg['content']):
#                 pass


def filter_error_messages(example):
    messages = json.loads(example['messages'])
    for msg in messages:
        if msg['role'] == 'assistant':
            if msg['content'] and '<tool_call>' in msg['content']:
                return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset(args.input_path, split="train")
    ds = ds.filter(filter_error_messages)
    import ipdb; ipdb.set_trace()
    save_dataset_to_parquet(ds, args.output_path)