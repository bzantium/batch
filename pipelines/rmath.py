"""
RMath Dataset Translation Pipeline (Specific Logic)

- Defines RMath-specific logic (strict LaTeX/math preservation prompt)
- Injected into the main pipeline runner.
"""

import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm

import utils

# ==============================================================================
# 1. RMath Specific Prompt
# ==============================================================================

def create_rmath_translation_prompt(text: str) -> str:
    """
    (RMath Strict) Translation prompt for 'content' and 'reasoning_content'.
    Strictly preserves LaTeX, formulas, variables, and code.
    """

    prompt = f"""You are an expert translation engine. Your task is to translate the given text into Korean.
This text may contain Mathematics (Math) related content.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone like "합니다", "입니다", or "습니다".

Translation Rules (Preserve the following as-is):
1.  **LaTeX and Formulas:** Perfectly preserve all LaTeX syntax (e.g., $...$, $$...$$, \\frac{{}}{{}}, \\sqrt{{}}).
2.  **Variables and Symbols:** Keep all math variables (e.g., `x`, `y`, `n_samples`), symbols, and equations in English.
3.  **Code and Numbers:** Do not alter code snippets (```...```), inline code (`...`), numbers (123, 0.5), or units of measurement (kg, m/s).
4.  **Formatting:** Preserve all formatting, including line breaks, markdown (e.g., `**bold**`), and whitespace.
5.  **Chinese Characters (Translate):** Translate all Chinese characters (Hanja) into Korean or English based on the context and rules above.

Input Example 1:
What is the value of $x^2$ when `x = 5`?
Translation Example 1:
`x = 5`일 때 $x^2$의 값은 무엇인가요?

Input Example 2:
Calculate the value of $x^2$ where `x = 5`.
Translation Example 2:
`x = 5`일 때 $x^2$의 값을 계산하세요.

이제 아래 입력 텍스트를 한국어로 번역하세요.

입력 텍스트:
{text}
"""
    return prompt

# ==============================================================================
# 2. RMath Batch Input Preparation (Injected Function)
# ==============================================================================

def prepare_batch_input(
    dataset: Dataset,
    model: str,
    reasoning_effort: str,
    chunk_max_length: int
) -> List[Dict[str, Any]]:
    """
    (RMath Specific) Creates the list of batch requests.
    - RMath does not use 'metadata' or 'tools'.
    - Uses the same strict math prompt for 'content' and 'reasoning_content'.
    """
    print(f"Preparing RMath batch requests for {len(dataset)} records...")
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

                    prompt_content = create_rmath_translation_prompt(chunk)
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

            # 2. 'reasoning_content' translation request
            if reasoning_content and reasoning_content.strip():
                reasoning_chunks = utils.chunk_content(reasoning_content, max_length=chunk_max_length)
                for chunk_idx, chunk in enumerate(reasoning_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_reasoning"
                    if len(reasoning_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    prompt_content = create_rmath_translation_prompt(chunk)
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
# 3. RMath Failed Record Extraction (Injected Function)
# ==============================================================================

def extract_failed_records(dataset: Dataset, failed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    (RMath Specific) Extracts only the 'content' body needed for retries.
    """
    print(f"Extracting RMath failed records from dataset...")
    failed_records = {}

    for custom_id in failed_ids:
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

                    failed_records[custom_id] = {
                        "content": content_to_translate
                    }
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠ Warning: Could not parse custom_id {custom_id}: {e}")
            continue

    print(f"✓ Extracted {len(failed_records)} records for retry")
    return failed_records

# ==============================================================================
# 4. RMath Retry Request Preparation (Injected Function)
# ==============================================================================

def prepare_retry_requests(
    failed_records: Dict[str, Dict[str, Any]],
    model: str,
    reasoning_effort: str,
    chunk_max_length: int,
    max_completion_tokens: int
) -> List[Dict[str, Any]]:
    """
    (RMath Specific) Prepares the list of batch requests for retrying failures.
    - Uses the same strict RMath prompt.
    """
    batch_requests = []

    for custom_id, record_info in failed_records.items():
        content = record_info.get("content", "")
        if not content or not content.strip():
            continue

        base_custom_id = custom_id
        if "_chunk_" in custom_id:
            base_custom_id = custom_id.rsplit("_chunk_", 1)[0]

        content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_custom_id = base_custom_id
            if len(content_chunks) > 1:
                chunk_custom_id += f"_chunk_{chunk_idx}"

            prompt_content = create_rmath_translation_prompt(chunk)
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
