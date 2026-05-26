import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# ===================== 路径与配置 =====================
# 一共28549条

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "sentence_edit_distance_with_lcs_new_0410.json"
OUTPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "llm_edit_distance_0410"/ "formal_26000-28550_sentence_edit_distance_llm_v2.json"
FAILED_LOG_PATH = ROOT_DIR / "connect_edit_distance" / "llm_edit_distance_0410"/"formal_26000-28550_sentence_edit_distance_llm_failed_v2.jsonl"

# 通义千问 API 配置（请替换成你自己的 key，或用环境变量）
DASHSCOPE_API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"
MODEL_NAME = "qwen-turbo-latest"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 批量处理控制
START_INDEX = 26000
END_INDEX = 28550
SLEEP_BETWEEN_CALLS = 0.3

# 额度节省策略
MIN_EDIT_DISTANCE_FOR_LLM = 2
MAX_EDIT_DISTANCE_FOR_LLM = None
ENABLE_LLM_RESULT_CACHE = True
MAX_TOKENS = 650

# 重试策略
ENABLE_RETRY_ON_VALIDATION_FAILURE = True
MAX_RETRY_TIMES = 1
FALLBACK_TO_BASELINE_ON_FAILURE = True

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
    位置规则：每一步都基于当前最新字符串，0-based 动态位置。
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
- replace/delete 的 position 必须满足：0 <= position < 当前字符串长度
- insert 的 position 必须满足：0 <= position <= 当前字符串长度

【占位符规则（必须遵守）】
- 句子中的 “#” 与 “□” 都表示 **一个字** 的占位符
- 每一个占位符都默认代表 **独立的未知字符**
- 即使两个句子在相同位置都出现占位符，也不能假设可自动对齐
- 必须通过 replace / delete / insert 显式处理

【必须满足的硬性约束】
1. 输出的 "edit_distance" 必须等于 "operations" 的长度。
2. 按顺序执行你给出的操作后，最终字符串必须 **完全等于** modified_text。
3. 操作类型只能是："replace" / "insert" / "delete"。
4. 对于 replace/delete，若给出了 original_char/char，必须与该步当前字符串对应位置字符一致。

【操作格式】
- replace:
{{
  "type": "replace",
  "original_char": "<被替换的字（应与当前字符串一致）>",
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
  "char": "<被删除的字（应与当前字符串一致）>",
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
算法参考操作（可能不自然，仅供参考）：
{dp_ops_str}

【输出要求】
只输出 JSON，不要任何多余文字！
{{
  "edit_distance": int,
  "operations": []
}}
"""
    return prompt


def build_retry_prompt(
    chapter: int,
    sentence: int,
    original_text: str,
    modified_text: str,
    dp_edit_distance: int,
    previous_operations: List[Dict[str, Any]],
    validation_error: str,
) -> str:
    """
    二次纠错 Prompt：将失败原因回灌给模型，仅修正操作位置/局部步骤，保证可执行并可达目标。
    """
    prev_ops_str = json.dumps(previous_operations, ensure_ascii=False, indent=2)

    prompt = f"""你是古籍校勘专家，正在修复一组失败的编辑操作。

上一版 operations 执行失败，请你在不改变总步数的前提下，修复它。

【失败信息】
{validation_error}

【硬性要求】
1. edit_distance 必须等于 operations 长度，且必须等于 {dp_edit_distance}。
2. 位置采用 0-based 动态下标（每一步基于最新字符串）。
3. replace/delete: 0 <= position < 当前字符串长度。
4. insert: 0 <= position <= 当前字符串长度。
5. 最终必须把 original_text 精确变成 modified_text。

【占位符规则】
- “#”与“□”都按单字占位符处理，不能自动对齐，必须显式操作。

【本次输入】
章节：{chapter}
句子：{sentence}
original_text：{original_text}
modified_text：{modified_text}
固定步数：{dp_edit_distance}

上一版失败 operations：
{prev_ops_str}

【输出要求】
只输出 JSON，不要任何多余文字：
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
        "input": {"messages": [{"role": "user", "content": prompt}]},
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


# ===================== 基础编辑距离（动态下标） =====================

def compute_edit_distance_with_ops_dynamic(src: str, tgt: str) -> Tuple[int, List[Dict[str, Any]]]:
    """
    计算 src -> tgt 的最小编辑距离，并返回“动态下标”语义下可直接执行的操作序列。
    """
    m, n = len(src), len(tgt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            replace_cost = 0 if src[i - 1] == tgt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + replace_cost,
            )

    ops_rev: List[Dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src[i - 1] == tgt[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops_rev.append(
                {
                    "type": "replace",
                    "original_char": src[i - 1],
                    "target_char": tgt[j - 1],
                    "position": i - 1,
                }
            )
            i -= 1
            j -= 1
            continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops_rev.append({"type": "delete", "char": src[i - 1], "position": i - 1})
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops_rev.append({"type": "insert", "char": tgt[j - 1], "position": i})
            j -= 1
            continue

        raise RuntimeError(f"Unexpected backtrace state: i={i}, j={j}")

    ops_rev.reverse()
    return dp[m][n], ops_rev


def _normalize_model_op(op: Dict[str, Any]) -> Dict[str, Any]:
    op_type = op.get("type")
    if op_type not in {"replace", "insert", "delete"}:
        raise ValueError(f"Unknown operation type: {op_type}")
    if "position" not in op:
        raise ValueError("operation missing position")

    try:
        pos = int(op["position"])
    except Exception:
        raise ValueError(f"position must be int-like, got: {op.get('position')}")

    fixed = dict(op)
    fixed["position"] = pos

    if op_type == "insert" and "char" not in fixed:
        raise ValueError(f"insert operation missing char: {json.dumps(op, ensure_ascii=False)}")
    if op_type == "replace" and "target_char" not in fixed:
        raise ValueError(f"replace operation missing target_char: {json.dumps(op, ensure_ascii=False)}")

    return fixed


def apply_operations_dynamic_pos_strict(
    src: str,
    operations: List[Dict[str, Any]],
    strict_char_check: bool = False,
) -> str:
    """
    严格执行动态下标操作。出错时返回更可诊断的信息：
    - 步号、操作内容、当前长度、当前字符串快照。
    """
    current = list(src)

    for step_no, raw_op in enumerate(operations, start=1):
        op = _normalize_model_op(raw_op)
        op_type = op["type"]
        pos = op["position"]

        if pos < 0:
            raise ValueError(
                f"[step={step_no}] 位置小于0: {pos}; len={len(current)}; op={json.dumps(op, ensure_ascii=False)}; "
                f"cur={''.join(current)}"
            )

        if op_type in {"replace", "delete"} and pos >= len(current):
            raise ValueError(
                f"[step={step_no}] 位置越界: {pos}; len={len(current)}; op={json.dumps(op, ensure_ascii=False)}; "
                f"cur={''.join(current)}"
            )

        if op_type == "insert" and pos > len(current):
            raise ValueError(
                f"[step={step_no}] insert位置越界: {pos}; len={len(current)}; op={json.dumps(op, ensure_ascii=False)}; "
                f"cur={''.join(current)}"
            )

        if op_type == "replace":
            expected = op.get("original_char")
            if strict_char_check and expected is not None and current[pos] != expected:
                raise ValueError(
                    f"[step={step_no}] replace字符不匹配: pos={pos}, expected={expected}, got={current[pos]}; "
                    f"op={json.dumps(op, ensure_ascii=False)}; cur={''.join(current)}"
                )
            current[pos] = op["target_char"]

        elif op_type == "delete":
            expected = op.get("char")
            if strict_char_check and expected is not None and current[pos] != expected:
                raise ValueError(
                    f"[step={step_no}] delete字符不匹配: pos={pos}, expected={expected}, got={current[pos]}; "
                    f"op={json.dumps(op, ensure_ascii=False)}; cur={''.join(current)}"
                )
            del current[pos]

        elif op_type == "insert":
            current.insert(pos, op["char"])

    return "".join(current)


def apply_operations_original_pos_and_convert_to_dynamic(
    src: str,
    operations: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    将“原始下标语义”操作执行在 src 上，同时转换为等价的“动态下标语义”操作。
    用于兼容模型偶发回退到旧语义的情况。
    """
    seq: List[Tuple[str, int | None]] = [(ch, idx) for idx, ch in enumerate(src)]
    converted: List[Dict[str, Any]] = []

    def find_index_by_original_pos(original_pos: int) -> int:
        for i, (_ch, orig_idx) in enumerate(seq):
            if orig_idx == original_pos:
                return i
        raise ValueError(f"Cannot find original position {original_pos} in current sequence")

    for step_no, raw_op in enumerate(operations, start=1):
        op = _normalize_model_op(raw_op)
        op_type = op["type"]
        pos = op["position"]

        if pos < 0:
            raise ValueError(
                f"[step={step_no}] 原始下标小于0: {pos}; op={json.dumps(op, ensure_ascii=False)}"
            )

        if op_type == "replace":
            idx = find_index_by_original_pos(pos)
            cur_ch, orig_idx = seq[idx]
            converted.append(
                {
                    "type": "replace",
                    "original_char": cur_ch,
                    "target_char": op["target_char"],
                    "position": idx,
                }
            )
            seq[idx] = (op["target_char"], orig_idx)

        elif op_type == "delete":
            idx = find_index_by_original_pos(pos)
            cur_ch, _orig_idx = seq[idx]
            converted.append({"type": "delete", "char": cur_ch, "position": idx})
            del seq[idx]

        elif op_type == "insert":
            insert_idx = None
            for i, (_ch, orig_idx) in enumerate(seq):
                if orig_idx is not None and orig_idx >= pos:
                    insert_idx = i
                    break
            if insert_idx is None:
                insert_idx = len(seq)
            converted.append({"type": "insert", "char": op["char"], "position": insert_idx})
            seq.insert(insert_idx, (op["char"], None))

    return "".join(ch for ch, _ in seq), converted


def validate_ops_or_raise(src: str, tgt: str, ops: List[Dict[str, Any]], expected_ed: int) -> List[Dict[str, Any]]:
    if expected_ed != len(ops):
        raise ValueError(f"LLM edit_distance({expected_ed}) != len(operations)({len(ops)})")

    normalized_ops = [_normalize_model_op(op) for op in ops]

    # 1) 优先按动态下标语义验证（字符字段仅作参考，不做硬约束）
    dynamic_error = None
    try:
        final_text = apply_operations_dynamic_pos_strict(src, normalized_ops, strict_char_check=False)
        if final_text == tgt:
            return normalized_ops
    except Exception as e:
        dynamic_error = str(e)
        final_text = None

    # 2) 动态失败后，尝试按原始下标语义执行并转换成动态下标
    try:
        final_text_orig, converted_ops = apply_operations_original_pos_and_convert_to_dynamic(src, normalized_ops)
        if final_text_orig == tgt:
            return converted_ops
        raise ValueError(
            "LLM operations do not transform original_text to modified_text under both semantics.\n"
            f"Expected: {tgt}\n"
            f"Got(dynamic): {final_text}\n"
            f"Got(original_pos): {final_text_orig}"
        )
    except Exception as e:
        if dynamic_error:
            raise ValueError(f"{dynamic_error}\n[original_pos_fallback_error] {e}")
        raise


# ===================== 单条处理逻辑 =====================

def build_passthrough_result(
    item: Dict[str, Any],
    dp_operations: List[Dict[str, Any]],
    dp_edit_distance: int,
    dp_source: str,
) -> Dict[str, Any]:
    return {
        **item,
        "operations_dp": dp_operations,
        "operations_dp_source": dp_source,
        "operations_llm": dp_operations,
        "edit_distance_llm": dp_edit_distance,
        "llm_result_source": "passthrough",
    }


def get_valid_baseline_ops(item: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], str]:
    """
    先尝试使用 item 自带 operations；若不可执行或不达标，则重算一个可执行基线。
    返回：(distance, operations, source_tag)
    """
    src = item["original_text"]
    tgt = item["modified_text"]
    raw_ed = int(item["edit_distance"])
    raw_ops = item.get("operations", [])

    try:
        normalized = validate_ops_or_raise(src, tgt, raw_ops, raw_ed)
        return raw_ed, normalized, "input"
    except Exception:
        new_ed, new_ops = compute_edit_distance_with_ops_dynamic(src, tgt)
        normalized = validate_ops_or_raise(src, tgt, new_ops, new_ed)
        return new_ed, normalized, "recomputed_dynamic_dp"


def run_llm_once(
    chapter: int,
    sentence: int,
    original_text: str,
    modified_text: str,
    baseline_ed: int,
    baseline_ops: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    prompt = build_realign_prompt(
        chapter,
        sentence,
        original_text,
        modified_text,
        baseline_ed,
        baseline_ops,
    )
    result = call_qwen(prompt)
    edit_distance_llm = result.get("edit_distance")
    ops_llm = result.get("operations", [])
    if not isinstance(edit_distance_llm, int):
        raise ValueError("edit_distance must be an integer in LLM result")
    if not isinstance(ops_llm, list):
        raise ValueError("operations must be a list in LLM result")
    return edit_distance_llm, ops_llm


def run_llm_retry_once(
    chapter: int,
    sentence: int,
    original_text: str,
    modified_text: str,
    fixed_ed: int,
    previous_ops: List[Dict[str, Any]],
    validation_error: str,
) -> Tuple[int, List[Dict[str, Any]]]:
    retry_prompt = build_retry_prompt(
        chapter,
        sentence,
        original_text,
        modified_text,
        fixed_ed,
        previous_ops,
        validation_error,
    )
    result = call_qwen(retry_prompt)
    edit_distance_llm = result.get("edit_distance")
    ops_llm = result.get("operations", [])
    if not isinstance(edit_distance_llm, int):
        raise ValueError("edit_distance must be an integer in retry LLM result")
    if not isinstance(ops_llm, list):
        raise ValueError("operations must be a list in retry LLM result")
    return edit_distance_llm, ops_llm


def process_one_item(item: Dict[str, Any]) -> Dict[str, Any]:
    chapter = item["chapter_number"]
    sentence = item["sentence_number"]
    original_text = item["original_text"]
    modified_text = item["modified_text"]

    # 先拿到“可执行且可达目标”的 baseline，避免坏样本带偏模型
    baseline_ed, baseline_ops, baseline_source = get_valid_baseline_ops(item)

    if baseline_ed == 0:
        return build_passthrough_result(item, baseline_ops, baseline_ed, baseline_source)

    if baseline_ed < MIN_EDIT_DISTANCE_FOR_LLM:
        return build_passthrough_result(item, baseline_ops, baseline_ed, baseline_source)

    if MAX_EDIT_DISTANCE_FOR_LLM is not None and baseline_ed > MAX_EDIT_DISTANCE_FOR_LLM:
        return build_passthrough_result(item, baseline_ops, baseline_ed, baseline_source)

    if not baseline_ops:
        return build_passthrough_result(item, baseline_ops, baseline_ed, baseline_source)

    cache_key = (
        original_text,
        modified_text,
        baseline_ed,
        json.dumps(baseline_ops, ensure_ascii=False, sort_keys=True),
    )
    if ENABLE_LLM_RESULT_CACHE and cache_key in LLM_RESULT_CACHE:
        cached = LLM_RESULT_CACHE[cache_key]
        return {
            **item,
            "operations_dp": baseline_ops,
            "operations_dp_source": baseline_source,
            "operations_llm": cached["operations_llm"],
            "edit_distance_llm": cached["edit_distance_llm"],
        }

    # 第一次 LLM
    ed_llm, ops_llm = run_llm_once(
        chapter,
        sentence,
        original_text,
        modified_text,
        baseline_ed,
        baseline_ops,
    )

    last_error = None
    for attempt in range(MAX_RETRY_TIMES + 1):
        try:
            normalized_ops_llm = validate_ops_or_raise(original_text, modified_text, ops_llm, ed_llm)
            output = {
                **item,
                "operations_dp": baseline_ops,
                "operations_dp_source": baseline_source,
                "operations_llm": normalized_ops_llm,
                "edit_distance_llm": ed_llm,
                "llm_result_source": "llm",
            }
            if ENABLE_LLM_RESULT_CACHE:
                LLM_RESULT_CACHE[cache_key] = {
                    "operations_llm": normalized_ops_llm,
                    "edit_distance_llm": ed_llm,
                }
            return output
        except Exception as e:
            last_error = str(e)
            if not ENABLE_RETRY_ON_VALIDATION_FAILURE or attempt >= MAX_RETRY_TIMES:
                break

            # 仅纠错一次：传递失败细节，让模型修正位置/局部步骤
            ed_llm, ops_llm = run_llm_retry_once(
                chapter,
                sentence,
                original_text,
                modified_text,
                baseline_ed,
                ops_llm,
                last_error,
            )

    if FALLBACK_TO_BASELINE_ON_FAILURE:
        return {
            **item,
            "operations_dp": baseline_ops,
            "operations_dp_source": baseline_source,
            "operations_llm": baseline_ops,
            "edit_distance_llm": baseline_ed,
            "llm_result_source": "baseline_fallback_after_llm_failure",
            "llm_failure_reason": last_error or "Unknown validation error",
        }

    raise ValueError(last_error or "Unknown validation error")


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
    REPORT_EVERY_N = 100

    FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    failed_log_f = FAILED_LOG_PATH.open("w", encoding="utf-8")

    for idx in range(start, end):
        item = data[idx]
        key_info = f"(index={idx}, chapter={item.get('chapter_number')}, sentence={item.get('sentence_number')})"
        try:
            print(f"处理 {key_info} ...", flush=True)
            new_item = process_one_item(item)
            results.append(new_item)
            done = len(results)

            if done % REPORT_EVERY_N == 0:
                source_counter: Dict[str, int] = {}
                fallback_reason_counter: Dict[str, int] = {}

                for r in results:
                    src_tag = str(r.get("llm_result_source", "unknown"))
                    source_counter[src_tag] = source_counter.get(src_tag, 0) + 1
                    if src_tag == "baseline_fallback_after_llm_failure":
                        reason = str(r.get("llm_failure_reason", "unknown"))
                        fallback_reason_counter[reason] = fallback_reason_counter.get(reason, 0) + 1

                source_parts = [f"{k}={v}" for k, v in sorted(source_counter.items(), key=lambda kv: kv[0])]
                top_reasons = sorted(
                    fallback_reason_counter.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:3]
                reason_parts = [f"{k}:{v}" for k, v in top_reasons]
                reason_msg = "; ".join(reason_parts) if reason_parts else "None"

                print(
                    f"[report] success_done={done}, failed={failed_count}, "
                    f"source_stats: {', '.join(source_parts)}, "
                    f"fallback_top3: {reason_msg}",
                    flush=True,
                )
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
