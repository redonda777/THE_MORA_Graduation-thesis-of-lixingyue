import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# ===================== 路径与配置 =====================
# 一共28549条

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "sentence_edit_distance_with_lcs_v4.0_0406.json"
OUTPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "llm_edit_distance_0406" / "0331_27000-28550_sentence_edit_distance_llm.json"
FAILED_LOG_PATH = ROOT_DIR / "connect_edit_distance" / "llm_edit_distance_0406" / "0331_27000-28550_sentence_edit_distance_llm_failed.jsonl"

# 通义千问 API 配置（请替换成你自己的 key，或用环境变量）
DASHSCOPE_API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"
MODEL_NAME = "qwen-turbo-latest"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 批量处理控制
START_INDEX = 0
END_INDEX = 1000
SLEEP_BETWEEN_CALLS = 0.3

# 额度节省策略
MIN_EDIT_DISTANCE_FOR_LLM = 2
MAX_EDIT_DISTANCE_FOR_LLM = None
ENABLE_LLM_RESULT_CACHE = True
MAX_TOKENS = 450

# 运行期缓存
LLM_RESULT_CACHE: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}


# ===================== 调用与 Prompt =====================

def build_realign_prompt(
    chapter: int,
    sentence: int,
    original_text: str,
    modified_text: str,
    dp_edit_distance: int,
    dp_operations: List[Dict[str, Any]],
) -> str:
    """
    构造让大模型“重排操作序列”的 Prompt。
    【修复】位置规则改为：每一步都基于当前最新字符串，0-based 动态位置
    【新增】# 占位符严格规则
    """
    dp_ops_str = json.dumps(dp_operations, ensure_ascii=False, indent=2)

    prompt = f"""你是古籍校勘专家，正在处理《道德经》的不同版本异文。

【任务】
现在给你一条句子对，以及一组由算法自动生成的字符级操作序列。
请你在 **不改变总编辑距离（步数）的前提下**，重新设计一组更符合中文语感的操作序列。

【基本定义】
- 我们在字符级别操作，允许三种操作：
  1. replace：把当前字符串某个位置的字符替换成另一个
  2. insert：在当前字符串某个位置之前插入一个字符
  3. delete：删除当前字符串某个位置的字符
- 每个操作代价为 1，总编辑距离 = 操作步数。

【重要！位置规则（必须严格遵守）】
- 所有 position 都是 **0-based 动态下标**
- 每一步操作的 position 都基于 **执行完上一步之后的最新字符串**
- 例如：
  原字符串：A B C D
  删除位置 0 → 变成 B C D
  下一步操作的 position 就基于 B C D

【# 占位符严格规则（必须遵守）】
- 句子中的 “#” 表示 **一个字** 的占位符
- 每一个 “#” 都默认代表 **一个独立的未知字符**
- **即使两个句子在相同位置都出现 “#”，也不能假设它们是同一个字或可直接对齐**
- 必须通过 replace / delete / insert 显式处理，不能自动匹配

【必须满足的硬性约束】
1. 输出的 "edit_distance" 必须等于 "operations" 的长度。
2. 按顺序执行你给出的操作后，最终字符串必须 **完全等于** modified_text。
3. 操作类型只能是："replace" / "insert" / "delete"。

【操作格式】
- replace:
{{
  "type": "replace",
  "original_char": "<被替换的字（必须和当前字符串一致）>",
  "target_char": "<替换后的字>",
  "position": int
}}
- insert:
{{
  "type": "insert",
  "char": "<插入的字>",
  "position": int
}}
- delete:
{{
  "type": "delete",
  "char": "<被删除的字（必须和当前字符串一致）>",
  "position": int
}}

【语感要求】
- 尽量把对应的字直接 replace
- 尽量删除多余虚词
- 不要绕路操作

【补充规则】
- % 表示顺序被打乱，忽略 % 之后再处理

【本次输入】
章节：{chapter}
句子：{sentence}
原始文本：{original_text}
目标文本：{modified_text}
算法编辑距离：{dp_edit_distance}

【输出要求】
只输出 JSON，不要任何多余文字！
{{
  "edit_distance": int,
  "operations": []
}}
"""
    return prompt


def call_qwen(prompt: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "result_format": "json",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": MAX_TOKENS,
        },
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        content = data["output"]["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected API response format: {json.dumps(data, ensure_ascii=False)}")

    try:
        return json.loads(content)
    except Exception:
        raise RuntimeError(f"Model content is not valid JSON: {content}")


# ===================== 操作执行（基于“当前下标”） =====================

def apply_operations_dynamic_pos(src: str, operations: List[Dict[str, Any]]) -> str:
    """
    【修复版】
    支持 动态当前下标（每一步基于最新字符串）
    模型能轻松理解，几乎不会出错
    """
    current = list(src)

    for op in operations:
        op_type = op["type"]
        pos = op["position"]

        # 越界保护
        if pos < 0:
            pos = 0
        if op_type in ["replace", "delete"] and pos >= len(current):
            raise ValueError(f"位置越界：{pos}，当前字符串长度：{len(current)}")

        if op_type == "replace":
            current[pos] = op["target_char"]

        elif op_type == "delete":
            del current[pos]

        elif op_type == "insert":
            current.insert(pos, op["char"])

    return "".join(current)


# ===================== 单条处理逻辑 =====================

def build_passthrough_result(item: Dict[str, Any], dp_operations: List[Dict[str, Any]], dp_edit_distance: int) -> Dict[str, Any]:
    return {
        **item,
        "operations_dp": dp_operations,
        "operations_llm": dp_operations,
        "edit_distance_llm": dp_edit_distance,
    }


def process_one_item(item: Dict[str, Any]) -> Dict[str, Any]:
    chapter = item["chapter_number"]
    sentence = item["sentence_number"]
    original_text = item["original_text"]
    modified_text = item["modified_text"]
    dp_edit_distance = item["edit_distance"]
    dp_operations = item.get("operations", [])

    if dp_edit_distance == 0:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    if dp_edit_distance < MIN_EDIT_DISTANCE_FOR_LLM:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    if MAX_EDIT_DISTANCE_FOR_LLM is not None and dp_edit_distance > MAX_EDIT_DISTANCE_FOR_LLM:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    if not dp_operations:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    cache_key = (
        original_text,
        modified_text,
        dp_edit_distance,
        json.dumps(dp_operations, ensure_ascii=False, sort_keys=True),
    )
    if ENABLE_LLM_RESULT_CACHE and cache_key in LLM_RESULT_CACHE:
        cached = LLM_RESULT_CACHE[cache_key]
        return {
            **item,
            "operations_dp": dp_operations,
            "operations_llm": cached["operations_llm"],
            "edit_distance_llm": cached["edit_distance_llm"],
        }

    prompt = build_realign_prompt(
        chapter,
        sentence,
        original_text,
        modified_text,
        dp_edit_distance,
        dp_operations,
    )

    result = call_qwen(prompt)

    edit_distance_llm = result.get("edit_distance")
    ops_llm = result.get("operations", [])

    if not isinstance(edit_distance_llm, int):
        raise ValueError("edit_distance must be an integer in LLM result")
    if not isinstance(ops_llm, list):
        raise ValueError("operations must be a list in LLM result")

    if edit_distance_llm != len(ops_llm):
        raise ValueError(
            f"LLM edit_distance({edit_distance_llm}) != len(operations)({len(ops_llm)})"
        )

    # 验证：使用动态下标执行操作
    final_text = apply_operations_dynamic_pos(original_text, ops_llm)
    if final_text != modified_text:
        raise ValueError(
            f"LLM operations do not transform original_text to modified_text.\n"
            f"Expected: {modified_text}\nGot:      {final_text}"
        )

    output = {
        **item,
        "operations_dp": dp_operations,
        "operations_llm": ops_llm,
        "edit_distance_llm": edit_distance_llm,
    }
    if ENABLE_LLM_RESULT_CACHE:
        LLM_RESULT_CACHE[cache_key] = {
            "operations_llm": ops_llm,
            "edit_distance_llm": edit_distance_llm,
        }
    return output


# ===================== 批量主流程 =====================

def main() -> None:
    print(f"加载输入文件: {INPUT_JSON_PATH}")
    with INPUT_JSON_PATH.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total = len(data)
    start = max(0, START_INDEX)
    end = END_INDEX if END_INDEX is not None else total
    end = min(end, total)

    print(f"总记录数: {total}，本次处理范围: [{start}, {end})")

    results: List[Dict[str, Any]] = []
    failed_count = 0

    failed_log_f = FAILED_LOG_PATH.open("w", encoding="utf-8")

    for idx in range(start, end):
        item = data[idx]
        key_info = f"(index={idx}, chapter={item.get('chapter_number')}, sentence={item.get('sentence_number')})"
        try:
            print(f"处理 {key_info} ...", flush=True)
            new_item = process_one_item(item)
            results.append(new_item)
        except Exception as e:
            failed_count += 1
            err = {
                "index": idx,
                "error": str(e),
                "item": item,
            }
            failed_log_f.write(json.dumps(err, ensure_ascii=False) + "\n")
            print(f"  -> 失败: {e}", flush=True)
        time.sleep(SLEEP_BETWEEN_CALLS)

    failed_log_f.close()

    print(f"写出结果到: {OUTPUT_JSON_PATH}")
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("处理完成。")
    print(f"- 成功记录数: {len(results)}")
    print(f"- 失败记录数: {failed_count}")
    print(f"- 失败详情日志: {FAILED_LOG_PATH}")


if __name__ == "__main__":
    main()