"""
RChat Dataset Translation Pipeline (Specific Logic)

- Defines RChat-specific logic (generic prompt preserving math/code)
- Injected into the main pipeline runner.
"""

import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm

# Import common utils (relative to main.py)
import utils

# ==============================================================================
# 1. RChat Specific Prompt
# ==============================================================================

def create_rchat_translation_prompt(text: str) -> str:
    """
    (RChat Generic) Translation prompt for 'content' and 'reasoning_content'.
    Preserves all technical elements: LaTeX, code, markdown, etc.
    """

    prompt = f"""You are an expert translation engine. Your task is to translate the given text into Korean.
This text may contain general conversation, as well as **Mathematics (LaTeX)** or **Code (Programming)** related content.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone like "합니다", "입니다", or "습니다".

Translation Rules (Preserve the following as-is):
1.  **LaTeX and Formulas:** Perfectly preserve all LaTeX syntax (e.g., $...$, $$...$$, \\frac{{}}{{}}, \\sqrt{{}}).
2.  **Code Blocks and Inline Code:** Perfectly preserve all code snippets (```...```) and inline code (`...`).
3.  **Identifiers and Variables:** Keep all technical/math identifiers (e.g., function names `my_func`, variable names `user_id`, `x`, `n_samples`, class names `MyClass`, JSON keys) in English.
4.  **Technical Terms and Paths:** Do not alter technical terms (API, SDK, JSON, SQL), file paths (`/path/to/file.py`), or URLs (`https://...`).
5.  **Formatting:** Preserve all formatting, including line breaks, markdown (e.g., `**bold**`, `*`, `1.`), and whitespace.
6.  **Chinese Characters (Translate):** Translate all Chinese characters (Hanja) into Korean or English based on the context and rules above.

Input Example 1:
How do I use the `get_user(user_id)` function?
Translation Example 1:
`get_user(user_id)` 함수는 어떻게 사용하나요?

Input Example 2:
Calculate the value of $x^2$ where `x = 5`.
Translation Example 2:
`x = 5`일 때 $x^2$의 값을 계산하세요.

이제 아래 입력 텍스트를 한국어로 번역하세요.

입력 텍스트:
{text}
---
"""
    return prompt

# ==============================================================================
# 2. RChat Batch Input Preparation (Injected Function)
# ==============================================================================

def prepare_batch_input(
    dataset: Dataset,
    model: str,
    reasoning_effort: str,
    chunk_max_length: int
) -> List[Dict[str, Any]]:
    """
    (RChat Specific) Creates the list of batch requests.
    - RChat does not use 'metadata' or 'tools'.
    - Uses the same generic prompt for 'content' and 'reasoning_content'.
    """
    print(f"Preparing RChat batch requests for {len(dataset)} records...")
    print(f"  Model: {model}, Reasoning effort: {reasoning_effort}")

    all_batch_requests = []
    total_messages = 0

    for record_idx, record in enumerate(tqdm(dataset, desc="Processing records")):

        try:
            messages = json.loads(record.get("messages", "[]"))
        except json.JSONDecodeError:
            messages = []

        for msg_idx, message in enumerate(messages):
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content")

            # 1. 'content' translation request
            if content and content.strip():
                content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

                for chunk_idx, chunk in enumerate(content_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_content"
                    if len(content_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    prompt_content = create_rchat_translation_prompt(chunk)

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt_content}],
                            "reasoning_effort": reasoning_effort
                            # max_completion_tokens is set by the retry helper, not here
                        }
                    }
                    all_batch_requests.append(batch_request)
                    total_messages += 1

            # 2. 'reasoning_content' translation request
            if reasoning_content and reasoning_content.strip():
                reasoning_chunks = utils.chunk_content(reasoning_content, max_length=chunk_max_length)

                for chunk_idx, chunk in enumerate(reasoning_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_reasoning"
                    if len(reasoning_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    prompt_content = create_rchat_translation_prompt(chunk)

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt_content}],
                            "reasoning_effort": reasoning_effort
                        }
                    }
                    all_batch_requests.append(batch_request)
                    total_messages += 1

    print(f"\n✓ Total records processed: {len(dataset)}")
    print(f"  Total fields to translate (content + reasoning): {total_messages}")

    return all_batch_requests

# ==============================================================================
# 3. RChat Failed Record Extraction (Injected Function)
# ==============================================================================

def extract_failed_records(dataset: Dataset, failed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    (RChat Specific) Extracts only the 'content' body needed for retries.
    """
    print(f"Extracting RChat failed records from dataset...")
    failed_records = {}

    for custom_id in failed_ids:
        # custom_id format: "record_{record_idx}_msg_{msg_idx}_{content_type}"
        # or chunked: "record_{record_idx}_msg_{msg_idx}_{content_type}_chunk_{chunk_idx}"
        try:
            parts = custom_id.split("_")
            record_idx = int(parts[1])
            msg_idx = int(parts[3])
            content_type = parts[4] # "content" or "reasoning"

            if record_idx < len(dataset):
                record = dataset[record_idx]

                try:
                    messages = json.loads(record.get("messages", "[]"))
                except json.JSONDecodeError:
                    messages = []

                if msg_idx < len(messages):
                    message = messages[msg_idx]

                    content_to_translate = ""
                    if content_type == "content":
                        content_to_translate = message.get("content", "")
                    elif content_type == "reasoning":
                        content_to_translate = message.get("reasoning_content", "")
                    else:
                        print(f"  ⚠ Warning: Unknown content_type '{content_type}' in custom_id {custom_id}")
                        continue

                    # RChat retries only need the original content
                    failed_records[custom_id] = {
                        "content": content_to_translate
                    }
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠ Warning: Could not parse custom_id {custom_id}: {e}")
            continue

    print(f"✓ Extracted {len(failed_records)} records for retry")
    return failed_records

# ==============================================================================
# 4. RChat Retry Request Preparation (Injected Function)
# ==============================================================================

def prepare_retry_requests(
    failed_records: Dict[str, Dict[str, Any]],
    model: str,
    reasoning_effort: str,
    chunk_max_length: int,
    max_completion_tokens: int
) -> List[Dict[str, Any]]:
    """
    (RChat Specific) Prepares the list of batch requests for retrying failures.
    - Uses the same generic RChat prompt.
    """
    batch_requests = []

    for custom_id, record_info in failed_records.items():
        content = record_info.get("content", "")

        if not content or not content.strip():
            continue

        # Get base custom_id if it was chunked
        # e.g., "record_0_msg_0_content_chunk_0" -> "record_0_msg_0_content"
        base_custom_id = custom_id
        if "_chunk_" in custom_id:
            base_custom_id = custom_id.rsplit("_chunk_", 1)[0]

        # Re-chunk the content
        content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_custom_id = base_custom_id
            if len(content_chunks) > 1:
                chunk_custom_id += f"_chunk_{chunk_idx}"

            prompt_content = create_rchat_translation_prompt(chunk)

            batch_request = {
                "custom_id": chunk_custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_content}],
                    "max_completion_tokens": max_completion_tokens,
                    "reasoning_effort": reasoning_effort
                }
            }
            batch_requests.append(batch_request)

    return batch_requests
