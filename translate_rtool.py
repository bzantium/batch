"""
RTool 데이터셋 번역 파이프라인 (translate_rtool.py)

- 'utils.py'의 공통 파이프라인 실행기를 사용합니다.
- RTool 고유의 로직 (tools_json, 프롬프트 선택)을 정의하여 주입합니다.
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
# RTool Specific Configuration
# ==============================================================================
# 덮어쓰거나 새로 정의할 RTool 고유의 설정
# (utils.py의 기본값을 사용하려면 이 섹션을 비워도 됩니다)
utils.DATASET_PATH = "/data/ib-huawei-nas-lmt_980/datasets/Kanana-2-Post-Training-Dataset/rtool"
DATASET_PATH = utils.DATASET_PATH # main_runner에 전달하기 위함

# ==============================================================================
# RTool SPECIFIC FUNCTIONS
# ==============================================================================

# 1. RTool용 프롬프트 (tools_json을 인자로 받음)
# ==============================================================================

def create_translation_prompt(text: str, tools_json: Optional[str] = None) -> str:
    """
    (엄격) 번역 프롬프트: 사용자 입력, 어시스턴트의 Tool Reasoning용.
    기술 용어, 함수명, 인자 등을 엄격하게 보존합니다.
    """

    prompt = f"""You are an expert translation engine. Your task is to translate the given text into Korean.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone, ending sentences with '합니다' or '습니다'.

Translation Rules (Preserve the following as-is):
1.  All technical identifiers (e.g., function names, variable names, class names) must remain in English.
2.  All function arguments and parameters (e.g., `vin_number`, `user_id`) must remain in English.
3.  All technical acronyms and API-related terms (e.g., VIN, PPSR, DMV, API) must remain in English.
4.  Code snippets, JSON structures, and technical formats must not be altered.
5.  Units of measurement (e.g., 287m, 0.5m/pixel) must be preserved in their original format.
6.  All formatting, including line breaks, markdown (e.g., `![Heightmap](...)`), and whitespace, must be preserved.
7.  **Chinese Characters (Translate):** Translate all Chinese characters (Hanja) into Korean.
"""

    if tools_json:
        prompt += f"""
Reference: Pay close attention to the `function.name` and `function.parameters` in this JSON.
---
Tool JSON:
{tools_json}
---
"""

    prompt += f"""

입력 예시:
Here are the results for `ppsr_lookup_by_vin(vin='123')`.
번역 예시:
`ppsr_lookup_by_vin(vin='123')`에 대한 결과입니다.

이제 아래 입력 텍스트를 한국어로 번역하세요.

입력 텍스트:
{text}
"""
    return prompt

def create_flexible_translation_prompt(text: str) -> str:
    """
    (유연) 번역 프롬프트: 어시스턴트의 최종 답변(Tool-call이 없는)용.
    마크다운, 서식, 특정 개체(URL, 숫자)는 유지하되, 모든 자연어는 유연하게 번역합니다.
    """

    prompt = f"""You are an expert translation engine. Your task is to translate the given text into Korean.
This text is a final response to the user, often a summary or a list.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone, ending sentences with '합니다' or '습니다'.

Translation Rules:
1.  **Translate All Natural Language:** This is the most important rule. All descriptive text, headers, labels, and descriptions (e.g., "Final Pricing Analysis", "Sale Price", "Eligibility", "Global STEM Education Innovation Challenge", "Platform", "Deadline") **must** be translated into Korean.
2.  **Preserve Formatting:** Keep all line breaks, whitespace, and markdown (e.g., `**bold**`, `*italic*`, `![links](...)`, list bullets `*`, `-`, `1.`) identical to the original.
3.  **Preserve Specific Entities:** Do not translate or alter the following:
    - Numbers (e.g., 2025, $10k, 30%, 0.5m/pixel)
    - Prices ($4.23)
    - URLs and paths (https://...)
    - Specific brand/platform proper nouns (e.g., "Challenge.gov", "FoodInnovate", "Kaggle", "AWS", "Michelin-starred")
    - Coordinates (33.49°N, -112.05°W)
4.  Combine these rules. For example, `**Deadline**: April 30, 2025` should become `**마감일**: April 30, 2025`.
5.  **Chinese Characters (Translate):** Translate all Chinese characters (Hanja) into Korean or English based on the context and rules above.

입력 예시:
1. **Global STEM Education Innovation Challenge**
   - **Platform**: Challenge.gov
   - **Deadline**: April 30, 2025
번역 예시:
1. **글로벌 STEM 교육 혁신 챌린지**
   - **플랫폼**: Challenge.gov
   - **마감일**: April 30, 2025

이제 아래 입력 텍스트를 한국어로 번역하세요.

입력 텍스트:
{text}
"""
    return prompt

# ==============================================================================
# 2. RTool용 배치 입력 생성 (utils.run_batch_pipeline에 주입될 함수)
# ==============================================================================

def prepare_rtool_batch_input(
    dataset: Dataset,
    model: str,
    reasoning_effort: str,
    chunk_max_length: int = utils.CHUNK_MAX_LENGTH
) -> List[Dict[str, Any]]:
    """
    (RTool Specific) Batch API 입력을 위한 *요청 리스트*를 생성합니다.
    (파일 분할 및 저장은 utils.py에서 처리)

    - 'metadata'에서 'tools_json'을 추출합니다.
    - 'role' 및 'tool_calls' 유무에 따라 다른 프롬프트를 사용합니다.
    """
    print(f"Preparing RTool batch requests for {len(dataset)} records...")
    print(f"  Model: {model}, Reasoning effort: {reasoning_effort}")

    all_batch_requests = []
    total_messages = 0

    for record_idx, record in enumerate(tqdm(dataset, desc="Processing records")):

        # (RTool-specific) record의 metadata(JSON string)에서 tools 추출
        tools_json_str = None
        metadata_str = record.get("metadata")
        if metadata_str:
            try:
                metadata_dict = json.loads(metadata_str)
                tools_metadata = metadata_dict.get("tools")
                if tools_metadata:
                    tools_json_str = json.dumps(tools_metadata, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError) as e:
                if record_idx < 5: # 처음 몇 개만 경고 출력
                    print(f"  ⚠ Warning: Could not parse metadata for record {record_idx}: {e}")
                pass

        try:
            messages = json.loads(record.get("messages", "[]"))
        except json.JSONDecodeError:
            messages = []

        for msg_idx, message in enumerate(messages):
            content = message.get("content", "")
            role = message.get("role", "")
            reasoning_content = message.get("reasoning_content")
            tool_calls = message.get("tool_calls") # (RTool-specific)

            # 1. 'content' 번역 요청
            if content and content.strip() and role != "tool":
                # Chunk content if too long
                content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

                for chunk_idx, chunk in enumerate(content_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_content"
                    if len(content_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    # (RTool-specific) content 번역 시 프롬프트 선택
                    if role == "assistant" and not tool_calls:
                        # Case 1: Assistant의 최종 답변 (tool_calls 없음) -> 유연한 프롬프트
                        prompt_content = create_flexible_translation_prompt(chunk)
                    else:
                        # Case 2: User의 요청 또는 Assistant의 tool_call 중간 답변 -> 엄격한 프롬프트
                        prompt_content = create_translation_prompt(
                            chunk, tools_json=tools_json_str
                        )

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

                    # reasoning_content는 항상 엄격한 프롬프트 사용
                    prompt_content = create_translation_prompt(
                        chunk, tools_json=tools_json_str
                    )

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
# 3. RTool용 실패 레코드 추출 (utils.handle_retry_failures에 주입될 함수)
# ==============================================================================

def extract_rtool_failed_records(dataset: Dataset, failed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    (RTool Specific) Extract records and messages that correspond to failed custom IDs.
    - 'metadata'에서 'tools_json_str'을 추출하여 포함합니다.
    - 재시도 시 프롬프트 선택을 위해 'role', 'tool_calls'를 포함합니다.
    """
    print(f"Extracting RTool failed records from dataset...")
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

                # (RTool-specific) record의 metadata(JSON string)에서 tools 추출
                tools_json_str = None
                metadata_str = record.get("metadata")
                if metadata_str:
                    try:
                        metadata_dict = json.loads(metadata_str)
                        tools_metadata = metadata_dict.get("tools")
                        if tools_metadata:
                            tools_json_str = json.dumps(tools_metadata, indent=2, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError) as e:
                         print(f"  ⚠ Warning: Could not parse metadata for record {record_idx} (retry extraction): {e}")

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

                    # (RTool-specific) 재시도 로직을 위해 role 및 tool_calls 저장
                    role = message.get("role", "")
                    tool_calls = message.get("tool_calls")

                    failed_records[custom_id] = {
                        "record_idx": record_idx,
                        "msg_idx": msg_idx,
                        "content": content_to_translate,
                        "role": role,
                        "tool_calls": tool_calls, # 재시도 시 프롬프트 선택용
                        "tools_json_str": tools_json_str # 재시도 시 프롬프트 주입용
                    }
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠ Warning: Could not parse custom_id {custom_id}: {e}")
            continue

    print(f"✓ Extracted {len(failed_records)} records for retry")
    return failed_records


# ==============================================================================
# 4. RTool용 재시도 요청 생성 (utils.handle_retry_failures에 주입될 함수)
# ==============================================================================

def prepare_rtool_retry_requests(
    failed_records: Dict[str, Dict[str, Any]],
    model: str,
    reasoning_effort: str,
    chunk_max_length: int = utils.CHUNK_MAX_LENGTH
) -> List[Dict[str, Any]]:
    """
    (RTool Specific) Prepares the list of batch requests for retrying failures.
    - Uses RTool-specific prompt selection logic based on 'role' and 'tool_calls'.
    """
    batch_requests = []

    for custom_id, record_info in failed_records.items():
        content = record_info["content"]
        role = record_info.get("role", "")
        tool_calls = record_info.get("tool_calls") # (RTool-specific)
        tools_json_str = record_info.get("tools_json_str")

        if not content or not content.strip() or role == "tool":
            continue

        # 원래 custom_id가 chunking된 경우 base custom_id 추출
        # 예: "record_0_msg_0_content_chunk_0" -> "record_0_msg_0_content"
        base_custom_id = custom_id
        if "_chunk_" in custom_id:
            base_custom_id = custom_id.rsplit("_chunk_", 1)[0]

        # Chunk content if too long
        content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

        # (RTool-specific) 재시도 시 프롬프트 선택
        is_reasoning = base_custom_id.endswith("_reasoning")

        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_custom_id = base_custom_id
            if len(content_chunks) > 1:
                chunk_custom_id += f"_chunk_{chunk_idx}"

            if is_reasoning:
                # Case 1: reasoning_content 재시도 -> 항상 엄격한 프롬프트
                prompt_content = create_translation_prompt(
                    chunk, tools_json=tools_json_str
                )
            elif role == "assistant" and not tool_calls:
                # Case 2: Assistant의 최종 답변 (tool_calls 없음) -> 유연한 프롬프트
                prompt_content = create_flexible_translation_prompt(chunk)
            else:
                # Case 3: User의 요청 또는 기타 -> 엄격한 프롬프트
                prompt_content = create_translation_prompt(
                    chunk, tools_json=tools_json_str
                )

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
    # utils.main_runner에 RTool 고유의 설정과 함수들을 주입합니다.
    asyncio.run(utils.main_runner(
        dataset_path=DATASET_PATH,
        prepare_batch_input_fn=prepare_rtool_batch_input,
        extract_failed_records_fn=extract_rtool_failed_records,
        prepare_retry_requests_fn=prepare_rtool_retry_requests
    ))