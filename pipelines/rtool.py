"""
RTool Dataset Translation Pipeline (Specific Logic)

- Defines RTool-specific logic (tools_json, prompt selection)
- Injected into the main pipeline runner.
"""

import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from tqdm import tqdm

import utils

# ==============================================================================
# 1. RTool Specific Prompts
# ==============================================================================

def create_translation_prompt_with_tool_calls(
    tools_json: Optional[str] = None
) -> str:
    """
    (Strict) Translation prompt: For user input and assistant's tool reasoning.
    Strictly preserves technical terms, function names, and arguments.
    Returns system_prompt string.
    """

    system_prompt = """You are an expert translation engine. Your task is to translate the given text into Korean.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone like "합니다", "입니다", "됩니다", or "습니다".

Translation Rules (Preserve the following as-is):
1.  All technical identifiers (e.g., function names, variable names, class names) must remain in English.
2.  All function arguments and parameters (e.g., `vin_number`, `user_id`) must remain in English.
3.  All technical acronyms and API-related terms (e.g., VIN, PPSR, DMV, API) must remain in English.
4.  Code snippets, JSON structures, and technical formats must not be altered.
5.  Units of measurement (e.g., 287m, 0.5m/pixel) must be preserved in their original format.
6.  All formatting, including line breaks, markdown (e.g., `![Heightmap](...)`), and whitespace, must be preserved."""

    if tools_json:
        system_prompt += f"""

Reference: Pay close attention to the `function.name` and `function.parameters` in this JSON.
---
Tool JSON:
{tools_json}
---"""

    system_prompt += """

Input Example:
Here are the results for `ppsr_lookup_by_vin(vin='123')`.
Translation Example:
`ppsr_lookup_by_vin(vin='123')`에 대한 결과입니다.
"""

    return system_prompt

def create_translation_prompt_without_tool_calls() -> str:
    """
    (Flexible) Translation prompt: For assistant's final answer (no tool-call).
    Translates natural language flexibly but preserves formatting and entities.
    Returns system_prompt string.
    """

    system_prompt = """You are an expert translation engine. Your task is to translate the given text into Korean.

Output Rules (Required):
1.  Return **only** the translated Korean text.
2.  Do **not** repeat the original English text.
3.  Do **not** include preambles, explanations, or labels like "Translation:".
4.  The response must start *immediately* with the first translated word.
5.  **Korean Tone:** Use a formal, polite tone like "합니다", "입니다", "됩니다", or "습니다".

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

Input Example:
1. **Global STEM Education Innovation Challenge**
   - **Platform**: Challenge.gov
   - **Deadline**: April 30, 2025
Translation Example:
1. **글로벌 STEM 교육 혁신 챌린지**
   - **플랫폼**: Challenge.gov
   - **마감일**: April 30, 2025
"""

    return system_prompt

# ==============================================================================
# 2. RTool Batch Input Preparation (Injected Function)
# ==============================================================================

def prepare_batch_input(
    dataset: Dataset,
    model: str,
    reasoning_effort: str,
    chunk_max_length: int
) -> List[Dict[str, Any]]:
    """
    (RTool Specific) Creates the list of batch requests.
    - Extracts 'tools_json' from 'metadata'.
    - Selects different prompts based on 'role' and 'tool_calls'.
    """
    print(f"Preparing RTool batch requests for {len(dataset)} records...")
    print(f"  Model: {model}, Reasoning effort: {reasoning_effort}")

    all_batch_requests = []
    total_messages = 0

    for record_idx, record in enumerate(tqdm(dataset, desc="Processing records")):

        # (RTool-specific) Extract tools from metadata JSON string
        tools_json_str = None
        metadata_str = record.get("metadata")
        if metadata_str:
            try:
                metadata_dict = json.loads(metadata_str)
                tools_metadata = metadata_dict.get("tools")
                if tools_metadata:
                    tools_json_str = json.dumps(tools_metadata, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError) as e:
                if record_idx < 5: # Show warning only for first few
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

            # 1. 'content' translation request
            if content and content.strip() and role != "tool":
                content_chunks = utils.chunk_content(content, max_length=chunk_max_length)

                for chunk_idx, chunk in enumerate(content_chunks):
                    custom_id = f"record_{record_idx}_msg_{msg_idx}_content"
                    if len(content_chunks) > 1:
                        custom_id += f"_chunk_{chunk_idx}"

                    # (RTool-specific) Prompt selection for content
                    if role == "assistant" and not tool_calls:
                        # Case 1: Assistant's final answer (no tool_calls) -> Flexible prompt
                        system_prompt = create_translation_prompt_without_tool_calls()
                    else:
                        # Case 2: User request or Assistant tool_call reasoning -> Strict prompt
                        system_prompt = create_translation_prompt_with_tool_calls(
                            tools_json=tools_json_str
                        )

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": chunk}
                            ],
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

                    # reasoning_content always uses the strict prompt
                    system_prompt = create_translation_prompt_with_tool_calls(
                        tools_json=tools_json_str
                    )

                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": chunk}
                            ],
                            "reasoning_effort": reasoning_effort
                        }
                    }
                    all_batch_requests.append(batch_request)
                    total_messages += 1

    print(f"\n✓ Total records processed: {len(dataset)}")
    print(f"  Total fields to translate (content + reasoning): {total_messages}")
    return all_batch_requests
