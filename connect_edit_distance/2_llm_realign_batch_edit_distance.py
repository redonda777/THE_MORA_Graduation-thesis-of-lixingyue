import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# ===================== 路径与配置 =====================
#一共28549条

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "sentence_edit_distance_with_lcs_v4.0_0406.json"
OUTPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" /"llm_edit_distance_0406"/ "0331_27000-28550_sentence_edit_distance_llm.json"
FAILED_LOG_PATH = ROOT_DIR / "connect_edit_distance"/"llm_edit_distance_0406" / "0331_27000-28550_sentence_edit_distance_llm_failed.jsonl"

# 通义千问 API 配置（请替换成你自己的 key，或用环境变量）
DASHSCOPE_API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"
MODEL_NAME = "qwen-turbo-latest"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 批量处理控制（可以根据需要调整 / 传参）
START_INDEX =   27000    # 从第几条记录开始（含）
END_INDEX=  28550  # 处理到第几条记录（不含）；None 表示到最后
SLEEP_BETWEEN_CALLS = 0.3     # 每次调用之间的间隔秒数，防止限流

# 额度节省策略（按需调整）
MIN_EDIT_DISTANCE_FOR_LLM = 2   # 小于该值时直接沿用 DP，不调用 LLM
MAX_EDIT_DISTANCE_FOR_LLM = None  # 大于该值时直接沿用 DP；None 表示不限制
ENABLE_LLM_RESULT_CACHE = True    # 对重复输入复用同一条 LLM 结果
MAX_TOKENS = 450                  # 限制模型输出长度，减少 token 消耗

# 运行期缓存（key: 输入特征，value: {"edit_distance_llm": int, "operations_llm": list}）
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
    """
    dp_ops_str = json.dumps(dp_operations, ensure_ascii=False, indent=2)

    prompt = f"""你是古籍校勘专家，正在处理《道德经》的不同版本异文。

【任务】
现在给你一条句子对，以及一组由算法（编辑距离动态规划）自动生成的字符级操作序列。
这些操作在“总步数”上是最少的，但在中文语法和语义上不一定是最自然的。
请你在 **不改变总编辑距离（步数）的前提下**，重新设计一组更符合中文语感的操作序列。

【基本定义】
- 我们在字符级别操作，允许三种操作：
  1. replace：把原句中的一个字符替换成另一个字符
  2. insert：在原句的某个位置插入一个字符
  3. delete：删除原句中的一个字符
- 每个操作的代价为 1，总编辑距离 = 操作步数。
- 位置 position 采用 0-based，下标始终相对于“原始字符串”的索引。
  从 original_text 出发，按顺序依次执行 operations：
  - 第一条操作的 position 在 original_text 上取索引；
  - 第二条操作的 position 在“执行完第一条操作后的字符串”上对应原始下标的位置；
  - 以此类推。
  在实现时，我们会根据 position 维护一份“原始下标→当前字符串位置”的映射，
  因此对于 delete / replace，你只需要给出原始下标即可。

【需要满足的硬性约束】
1. 你的输出中，"edit_distance" 必须等于 "operations" 的长度。
2. 如果从 original_text 出发，按你给出的 operations 依次执行，最终字符串必须完全等于 modified_text。
3. 操作类型只能是："replace" / "insert" / "delete"。
4. 每条操作的字段格式：
   - replace:
     {{
       "type": "replace",
       "original_char": "<原来的字>",
       "target_char": "<替换后的字>",
       "position": <int, 0-based，相对于原始字符串的下标>
     }}
   - insert:
     {{
       "type": "insert",
       "char": "<要插入的字>",
       "position": <int, 0-based，相对于原始字符串的下标；
                   表示在所有原始下标 >= position 的第一个字符之前插入。
                   如果不存在原始下标 >= position 的字符，则在末尾插入。>
     }}
   - delete:
     {{
       "type": "delete",
       "char": "<被删除的字>",
       "position": <int, 0-based，相对于原始字符串的下标>
     }}

【语感要求】
- 在总步数不变的前提下，尽量让每一步操作在中文语法与语义上更自然：
  - 尽量把明显对应的一对字（如“智”和“知”）视为一次 replace，而不是“删一个字 + 把别的字替成它”。
  - 对于明显多余的虚词或语尾助词（如“也”），更倾向于 delete。
  - 不要为了“凑步数”而做绕远路的替换。

【补充的信息】
- 句子中的#代表这个字丢失，按照一个字符处理，可以插入也可以删除也可以是替换。
- 句子中的%表示句子的顺序被打乱，忽略%之后进行计算。

【本次输入】
章节：{chapter}
句号：{sentence}
original_text：{original_text}
modified_text：{modified_text}

算法给出的编辑距离（步数）：{dp_edit_distance}
算法给出的操作序列（仅供参考，可能不自然）：
{dp_ops_str}

【输出要求】
请你只输出一个 JSON，对象结构如下（不要有任何多余文字）：
{{
  "edit_distance": <int>,
  "operations": [
    {{
      "type": "replace" | "insert" | "delete",
      ...
    }},
    ...
  ]
}}
"""
    return prompt


def call_qwen(prompt: str) -> Dict[str, Any]:
    """
    调用通义千问 API，返回解析后的 JSON 结果。
    期望模型的最终 content 是一个 JSON 字符串。
    """
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


# ===================== 操作执行（基于“原始下标”） =====================

def apply_operations_with_original_pos(src: str, operations: List[Dict[str, Any]]) -> str:
    """
    从 src 出发，依次执行 operations，返回最终字符串。

    约定：
    - replace/delete 的 position 为“相对于原始字符串的下标”；
    - insert 的 position 表示：
      在所有“原始下标 >= position 的第一个字符”之前插入；
      如果不存在这样的字符，则在末尾插入。
    为了支持这个语义，这里内部维护一份 (当前字符, 原始下标/None) 的序列。
    """
    seq: List[Tuple[str, int | None]] = [(ch, idx) for idx, ch in enumerate(src)]

    def find_index_by_original_pos(original_pos: int) -> int:
        for i, (_ch, orig_idx) in enumerate(seq):
            if orig_idx == original_pos:
                return i
        raise ValueError(f"Cannot find original position {original_pos} in current sequence")

    for op in operations:
        op_type = op.get("type")
        pos = op["position"]

        if op_type == "replace":
            target_char = op["target_char"]
            idx = find_index_by_original_pos(pos)
            _ch, orig_idx = seq[idx]
            seq[idx] = (target_char, orig_idx)

        elif op_type == "delete":
            idx = find_index_by_original_pos(pos)
            del seq[idx]

        elif op_type == "insert":
            char = op["char"]
            insert_idx = None
            for i, (_ch, orig_idx) in enumerate(seq):
                if orig_idx is not None and orig_idx >= pos:
                    insert_idx = i
                    break
            if insert_idx is None:
                insert_idx = len(seq)
            seq.insert(insert_idx, (char, None))

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    return "".join(ch for ch, _orig in seq)


# ===================== 单条处理逻辑 =====================

def build_passthrough_result(item: Dict[str, Any], dp_operations: List[Dict[str, Any]], dp_edit_distance: int) -> Dict[str, Any]:
    """
    不调用 LLM 时的统一输出结构。
    """
    return {
        **item,
        "operations_dp": dp_operations,
        "operations_llm": dp_operations,
        "edit_distance_llm": dp_edit_distance,
    }


def process_one_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单条 sentence_edit_distance 记录调用 LLM 重排操作，
    返回包含 dp 和 llm 两套操作的结果。
    """
    chapter = item["chapter_number"]
    sentence = item["sentence_number"]
    original_text = item["original_text"]
    modified_text = item["modified_text"]
    dp_edit_distance = item["edit_distance"]
    dp_operations = item.get("operations", [])

    # edit_distance == 0 不调用 LLM，但保持和非零样本一致的输出字段结构
    if dp_edit_distance == 0:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    # 节流：低编辑距离样本直接使用 DP 结果，通常收益小、却消耗额度
    if dp_edit_distance < MIN_EDIT_DISTANCE_FOR_LLM:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    # 节流：可选上限，超大编辑距离样本也可直接跳过 LLM
    if MAX_EDIT_DISTANCE_FOR_LLM is not None and dp_edit_distance > MAX_EDIT_DISTANCE_FOR_LLM:
        return build_passthrough_result(item, dp_operations, dp_edit_distance)

    # 防御：若无可重排操作，直接透传
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

    # 确认执行后能得到目标句
    final_text = apply_operations_with_original_pos(original_text, ops_llm)
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

    # 失败日志采用 jsonl 形式，方便查看/重试
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