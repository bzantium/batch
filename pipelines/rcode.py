"""
RCode Dataset Translation Pipeline (Specific Logic)

- Defines RCode-specific logic (strict code/variable preservation prompt)
- Injected into the main pipeline runner.
"""

import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm

import utils

# ==============================================================================
# 1. RCode Specific Prompt
# ==============================================================================

def create_rcode_translation_prompt(text: str) -> str:
    """
    (RCode Strict) Translation prompt for 'content' and 'reasoning_content'.
    Strictly preserves code, variables, function names, and technical terms.
    """

    prompt = f"""You are an expert translation engine. Your task is to translate the given text into Korean.
This text may contain Code (Programming) related content.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone like "합니다", "입니다", "됩니다", or "습니다".

Translation Rules (Preserve the following as-is):
1.  **Code Blocks and Inline Code:** Perfectly preserve all code snippets (```...```) and inline code (`...`).
2.  **Identifiers:** Keep all technical identifiers (e.g., function names `my_func`, variable names `user_id`, class names `MyClass`, JSON keys `{{"key": "value"}}`, etc.) in English.
3.  **Technical Terms:** Keep proper technical terms like API, SDK, JSON, XML, Docker, Kubernetes, React, SQL, etc., in English.
4.  **Paths and URLs:** Do not alter file paths (`/path/to/file.py`), URLs (`https://...`), or API endpoints (`/v1/users`).
5.  **Formatting:** Preserve all formatting, including line breaks, markdown (e.g., `**bold**`), and whitespace.

Input Example 1:
How do I use the `get_user(user_id)` function?
Translation Example 1:
`get_user(user_id)` 함수는 어떻게 사용하나요?

Input Example 2:
The function `get_user(user_id)` retrieves user data from the `/api/v1/users` endpoint.
Translation Example 2:
`get_user(user_id)` 함수는 `/api/v1/users` 엔드포인트에서 사용자 데이터를 가져옵니다.

이제 아래 입력 텍스트를 한국어로 번역하세요.

입력 텍스트:
{text}
"""
    return prompt

# ==============================================================================
# 2. RCode Batch Input Preparation (Injected Function)
# ==============================================================================

def prepare_batch_input(
    dataset: Dataset,
    model: str,
    reasoning_effort: str,
    chunk_max_length: int
) -> List[Dict[str, Any]]:
    """
    (RCode Specific) Creates the list of batch requests.
    - RCode does not use 'metadata' or 'tools'.
    - Uses the same strict code prompt for 'content' and 'reasoning_content'.
    """
    print(f"Preparing RCode batch requests for {len(dataset)} records...")
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

                    prompt_content = create_rcode_translation_prompt(chunk)
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

                    prompt_content = create_rcode_translation_prompt(chunk)
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
