"""
RCode 데이터셋 번역 파이프라인 (translate_rcode.py)

- 'utils.py'의 공통 파이프라인 실행기를 사용합니다.
- RCode 고유의 로직 (코드/변수명 보존 프롬프트)을 정의하여 주입합니다.
- RMath와 마찬가지로 tools, tool_calls, metadata 로직이 없습니다.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm

# 공통 유틸리티 및 설정 임포트
import utils

# ==============================================================================
# RCode Specific Configuration
# ==============================================================================
# RCode 데이터셋 경로 설정
utils.DATASET_PATH = "/data/ib-a100-cluster-a-pri-lmalign_942/personal/ryan/project/data-api/sample/rcode-0528"
DATASET_PATH = utils.DATASET_PATH # main_runner에 전달하기 위함

# ==============================================================================
# RCode SPECIFIC FUNCTIONS
# ==============================================================================

# 1. RCode용 프롬프트 (코드/변수 보존)
# ==============================================================================

def create_rcode_translation_prompt(text: str) -> str:
    """
    (RCode 엄격) 번역 프롬프트: 'content' 및 'reasoning_content'용.
    코드, 변수, 함수명, 기술 용어 등을 엄격하게 보존합니다.
    """

    prompt = f"""You are an expert translation engine. Your task is to translate the given text into Korean.
This text may contain Code (Programming) related content.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone, ending sentences with '합니다' or '습니다'.

Translation Rules (Preserve the following as-is):
1.  **Code Blocks and Inline Code:** Perfectly preserve all code snippets (```...```) and inline code (`...`).
2.  **Identifiers:** Keep all technical identifiers (e.g., function names `my_func`, variable names `user_id`, class names `MyClass`, JSON keys `{{"key": "value"}}`, etc.) in English.
3.  **Technical Terms:** Keep proper technical terms like API, SDK, JSON, XML, Docker, Kubernetes, React, SQL, etc., in English.
4.  **Paths and URLs:** Do not alter file paths (`/path/to/file.py`), URLs (`https://...`), or API endpoints (`/v1/users`).
5.  **Formatting:** Preserve all formatting, including line breaks, markdown (e.g., `**bold**`), and whitespace.
6.  **Chinese Characters (Translate):** Translate all Chinese characters (Hanja) into Korean or English based on the context and rules above.

입력 예시 1:
How do I use the `get_user(user_id)` function?
번역 예시 1:
`get_user(user_id)` 함수는 어떻게 사용하나요?

입력 예시 2:
The function `get_user(user_id)` retrieves user data from the `/api/v1/users` endpoint.
번역 예시 2:
`get_user(user_id)` 함수는 `/api/v1/users` 엔드포인트에서 사용자 데이터를 가져옵니다.

이제 아래 입력 텍스트를 한국어로 번역하세요.

입력 텍스트:
{text}
"""
    return prompt

# ==============================================================================
# 2. RCode용 배치 입력 생성 (utils.run_batch_pipeline에 주입될 함수)
# ==============================================================================

def prepare_rcode_batch_input(
    dataset: Dataset,
    model: str,
    reasoning_effort: str,
    chunk_max_length: int = utils.CHUNK_MAX_LENGTH
) -> List[Dict[str, Any]]:
    """
    (RCode Specific) Batch API 입력을 위한 *요청 리스트*를 생성합니다.
    (파일 분할 및 저장은 utils.py에서 처리)

    - RCode는 'metadata'나 'tools'를 사용하지 않습니다.
    - 'user'와 'assistant' 역할만 존재합니다.
    - 'content'와 'reasoning_content' 모두 동일한 엄격한 코드용 프롬프트를 사용합니다.
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

            # 1. 'content' 번역 요청
            if content and content.strip():
                # Chunk content if too long
                content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

                for chunk_idx, chunk in enumerate(content_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_content"
                    if len(content_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    # RCode는 항상 엄격한 코드용 프롬프트 사용
                    prompt_content = create_rcode_translation_prompt(chunk)

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt_content}],
                            "max_completion_tokens": utils.MAX_COMPLETION_TOKENS,
                            "reasoning_effort": reasoning_effort
                        }
                    }
                    all_batch_requests.append(batch_request)
                    total_messages += 1

            # 2. 'reasoning_content' 번역 요청
            if reasoning_content and reasoning_content.strip():
                # Chunk reasoning_content if too long
                reasoning_chunks = utils.chunk_content(reasoning_content, max_length=chunk_max_length)

                for chunk_idx, chunk in enumerate(reasoning_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_reasoning"
                    if len(reasoning_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    # RCode는 reasoning도 항상 엄격한 코드용 프롬프트 사용
                    prompt_content = create_rcode_translation_prompt(chunk)

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt_content}],
                            "max_completion_tokens": utils.MAX_COMPLETION_TOKENS,
                            "reasoning_effort": reasoning_effort
                        }
                    }
                    all_batch_requests.append(batch_request)
                    total_messages += 1

    print(f"\n✓ Total records processed: {len(dataset)}")
    print(f"  Total fields to translate (content + reasoning): {total_messages}")

    return all_batch_requests


# ==============================================================================
# 3. RCode용 실패 레코드 추출 (utils.handle_retry_failures에 주입될 함수)
# ==============================================================================

def extract_rcode_failed_records(dataset: Dataset, failed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    (RCode Specific) Extract records and messages that correspond to failed custom IDs.
    - 재시도에 필요한 'content' 본문만 추출합니다.
    """
    print(f"Extracting RCode failed records from dataset...")
    failed_records = {}

    for custom_id in failed_ids:
        # custom_id 파싱: "record_{record_idx}_msg_{msg_idx}_{content_type}"
        # 또는 chunking된 경우: "record_{record_idx}_msg_{msg_idx}_{content_type}_chunk_{chunk_idx}"
        try:
            parts = custom_id.split("_")
            record_idx = int(parts[1])
            msg_idx = int(parts[3])
            content_type = parts[4] # "content" or "reasoning"

            # chunking 여부 확인 (chunking된 경우 parts[5] == "chunk")
            is_chunked = len(parts) > 5 and parts[5] == "chunk"

            if record_idx < len(dataset):
                record = dataset[record_idx]

                # RCode는 metadata/tools_json이 필요 없음

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

                    # RCode 재시도는 content 본문만 필요
                    # chunking된 경우에도 전체 content를 가져와서 재시도 시 다시 chunking합니다
                    failed_records[custom_id] = {
                        "content": content_to_translate
                    }
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠ Warning: Could not parse custom_id {custom_id}: {e}")
            continue

    print(f"✓ Extracted {len(failed_records)} records for retry")
    return failed_records


# ==============================================================================
# 4. RCode용 재시도 요청 생성 (utils.handle_retry_failures에 주입될 함수)
# ==============================================================================

def prepare_rcode_retry_requests(
    failed_records: Dict[str, Dict[str, Any]],
    model: str,
    reasoning_effort: str,
    chunk_max_length: int = utils.CHUNK_MAX_LENGTH
) -> List[Dict[str, Any]]:
    """
    (RCode Specific) Prepares the list of batch requests for retrying failures.
    - RCode는 항상 동일한 엄격한 프롬프트를 사용합니다.
    """
    batch_requests = []

    for custom_id, record_info in failed_records.items():
        content = record_info.get("content", "")

        if not content or not content.strip():
            continue

        # 원래 custom_id가 chunking된 경우 base custom_id 추출
        # 예: "record_0_msg_0_content_chunk_0" -> "record_0_msg_0_content"
        base_custom_id = custom_id
        if "_chunk_" in custom_id:
            base_custom_id = custom_id.rsplit("_chunk_", 1)[0]

        # Chunk content if too long
        content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_custom_id = base_custom_id
            if len(content_chunks) > 1:
                chunk_custom_id += f"_chunk_{chunk_idx}"

            # RCode는 항상 엄격한 코드용 프롬프트 사용
            prompt_content = create_rcode_translation_prompt(chunk)

            batch_request = {
                "custom_id": chunk_custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_content}],
                    "max_completion_tokens": utils.MAX_COMPLETION_TOKENS,
                    "reasoning_effort": reasoning_effort
                }
            }
            batch_requests.append(batch_request)

    return batch_requests

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # utils.main_runner에 RCode 고유의 설정과 함수들을 주입합니다.
    asyncio.run(utils.main_runner(
        dataset_path=DATASET_PATH,
        prepare_batch_input_fn=prepare_rcode_batch_input,
        extract_failed_records_fn=extract_rcode_failed_records,
        prepare_retry_requests_fn=prepare_rcode_retry_requests
    ))
