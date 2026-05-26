import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import requests

# ===================== 路径与配置 =====================

ROOT_DIR = Path(__file__).resolve().parent.parent

# 合并后的失败案例（每行一个 JSON：{"index": int, "error": str, "item": {...}}）
FAILED_INPUT_JSONL = ROOT_DIR / "connect_edit_distance" / "all_llm_failed.jsonl"

# 输出：每行一个 JSON：{"index": int, "fixed": {...(成功样本同款结构)...}}
FIXED_OUTPUT_JSONL = ROOT_DIR / "connect_edit_distance" / "all_llm_failed_fixed.jsonl"

# 仍失败的条目（便于二次处理）
STILL_FAILED_JSONL = ROOT_DIR / "connect_edit_distance" / "all_llm_failed_still_failed.jsonl"

# 通义千问 API 配置（建议改为环境变量或自行替换）
DASHSCOPE_API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"
MODEL_NAME = "qwen-turbo-latest"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 处理控制
SLEEP_BETWEEN_CALLS = 0.3
MAX_TOKENS = 512
LOG_EVERY_N = 1

# 重跑失败样本时的清洗：去掉 % 与空白字符（保留 #）
WHITESPACE_RE = re.compile(r"\s+")


# ===================== 工具：清洗与映射 =====================

def clean_and_map(raw: str) -> Tuple[str, List[int], List[int]]:
    """
    清洗 raw（去掉 % 与所有空白字符），同时构造映射：
    - clean: 清洗后的字符串
    - clean_to_raw: clean 下标 -> raw 下标（单调递增）
    - raw_to_clean: raw 下标 -> clean 下标（被清洗掉的字符记为 -1）
    """
    clean_chars: List[str] = []
    clean_to_raw: List[int] = []
    raw_to_clean: List[int] = [-1] * len(raw)

    for i, ch in enumerate(raw):
        if ch == "%":
            continue
        if ch.isspace():
            continue
        clean_to_raw.append(i)
        raw_to_clean[i] = len(clean_chars)
        clean_chars.append(ch)

    return "".join(clean_chars), clean_to_raw, raw_to_clean


def map_ops_clean_to_raw(
    ops_clean: List[Dict[str, Any]],
    clean_to_raw: List[int],
    raw_len: int,
) -> List[Dict[str, Any]]:
    """
    将基于 clean“原始下标语义”的操作序列，映射为 raw“原始下标语义”的操作序列。
    约定：
    - replace/delete: position 是 clean 的原始下标 -> 映射为 raw 的原始下标
    - insert: position 表示插在第一个 clean 原始下标 >= position 的字符之前；
      映射时用该 clean 位置对应的 raw 原始下标作为锚点；若 position >= len(clean)，按末尾处理。
    """
    mapped: List[Dict[str, Any]] = []
    clean_len = len(clean_to_raw)

    for op in ops_clean:
        op_type = op.get("type")
        pos_clean = int(op["position"])

        if op_type in {"replace", "delete"}:
            if pos_clean < 0 or pos_clean >= clean_len:
                raise ValueError(f"Clean position out of range: {pos_clean} (len={clean_len})")
            pos_raw = clean_to_raw[pos_clean]
            mapped.append({**op, "position": pos_raw})

        elif op_type == "insert":
            if pos_clean < 0:
                raise ValueError(f"Clean insert position must be >=0, got {pos_clean}")
            if pos_clean >= clean_len:
                pos_raw = raw_len
            else:
                pos_raw = clean_to_raw[pos_clean]
            mapped.append({**op, "position": pos_raw})
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    return mapped


# ===================== LLM 调用与 Prompt =====================

def build_realign_prompt(
    chapter: int,
    sentence: int,
    clean_original_text: str,
    clean_modified_text: str,
    dp_edit_distance: int,
    dp_operations: List[Dict[str, Any]],
) -> str:
    """
    让 LLM 在 clean 文本上重排操作序列（position 语义以 clean 的“原始下标”为准）。
    """
    dp_ops_str = json.dumps(dp_operations, ensure_ascii=False, indent=2)

    return f"""你是古籍校勘专家，正在处理《道德经》的不同版本异文。

【任务】
现在给你一条句子对，以及一组由算法（编辑距离动态规划）自动生成的字符级操作序列。
这些操作在“总步数”上是最少的，但在中文语法和语义上不一定是最自然的。
请你在 **不改变总编辑距离（步数）的前提下**，重新设计一组更符合中文语感的操作序列。

【重要说明（请严格遵守）】
1) 本次输入文本已经做过清洗：已去掉“%”与所有空白字符；你无需考虑它们的存在。
2) 句子中的“#”表示“一个字”的占位符：
   - 每一个“#”都默认代表一个独立的未知字符；
   - 即使两个句子在相同位置都出现“#”，也不能假设它们是同一个字或可直接对齐。

【基本定义】
- 我们在字符级别操作，允许三种操作：
  1. replace：把原句中的一个字符替换成另一个字符
  2. insert：在原句的某个位置插入一个字符
  3. delete：删除原句中的一个字符
- 每个操作的代价为 1，总编辑距离 = 操作步数。
- position 采用 0-based，且始终相对于 clean_original_text 的“原始下标”语义：
  从 clean_original_text 出发，按顺序依次执行 operations：
  - replace/delete 的 position 为 clean_original_text 的原始下标；
  - insert 的 position 表示：在所有原始下标 >= position 的第一个字符之前插入；
    如果不存在这样的字符，则在末尾插入。

【需要满足的硬性约束】
1. 你的输出中，"edit_distance" 必须等于 "operations" 的长度。
2. 如果从 clean_original_text 出发，按你给出的 operations 依次执行，最终字符串必须完全等于 clean_modified_text。
3. 操作类型只能是："replace" / "insert" / "delete"。
4. 每条操作的字段格式：
   - replace:
     {{
       "type": "replace",
       "original_char": "<原来的字>",
       "target_char": "<替换后的字>",
       "position": <int, 0-based，相对于 clean_original_text 的原始下标>
     }}
   - insert:
     {{
       "type": "insert",
       "char": "<要插入的字>",
       "position": <int, 0-based，相对于 clean_original_text 的原始下标>
     }}
   - delete:
     {{
       "type": "delete",
       "char": "<被删除的字>",
       "position": <int, 0-based，相对于 clean_original_text 的原始下标>
     }}

【本次输入】
章节：{chapter}
句号：{sentence}
clean_original_text：{clean_original_text}
clean_modified_text：{clean_modified_text}

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
        raise RuntimeError(
            f"Unexpected API response format: {json.dumps(data, ensure_ascii=False)}"
        )

    try:
        return json.loads(content)
    except Exception:
        raise RuntimeError(f"Model content is not valid JSON: {content}")


# ===================== 执行与校验（原始下标语义） =====================

def apply_operations_with_original_pos(src: str, operations: List[Dict[str, Any]]) -> str:
    """
    从 src 出发，依次执行 operations，返回最终字符串。
    position 语义同主脚本：replace/delete 为“原始下标”，insert 为“在第一个原始下标>=pos前插入”。
    """
    seq: List[Tuple[str, int | None]] = [(ch, idx) for idx, ch in enumerate(src)]

    def find_index_by_original_pos(original_pos: int) -> int:
        for i, (_ch, orig_idx) in enumerate(seq):
            if orig_idx == original_pos:
                return i
        raise ValueError(f"Cannot find original position {original_pos} in current sequence")

    for op in operations:
        op_type = op.get("type")
        pos = int(op["position"])

        if op_type == "replace":
            idx = find_index_by_original_pos(pos)
            _ch, orig_idx = seq[idx]
            seq[idx] = (op["target_char"], orig_idx)
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


# ===================== 单条失败修复 =====================

def build_passthrough_fixed(item: Dict[str, Any]) -> Dict[str, Any]:
    dp_ops = item.get("operations", [])
    dp_ed = int(item.get("edit_distance", 0))
    return {
        **item,
        "operations_dp": dp_ops,
        "operations_llm": dp_ops,
        "edit_distance_llm": dp_ed,
    }


def classify_error(error: str) -> str:
    if error.startswith("Model content is not valid JSON"):
        return "invalid_json"
    if "Read timed out" in error:
        return "timeout"
    if error.startswith("Cannot find original position"):
        return "position_not_found"
    if error.startswith("LLM edit_distance"):
        return "distance_len_mismatch"
    if error.startswith("LLM operations do not transform"):
        return "transform_mismatch"
    return "other"


def fix_one_failed_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入 record: {"index": int, "error": str, "item": {...}}
    输出 {"index": int, "fixed": {...成功样本同款结构...}}
    """
    idx = int(record["index"])
    error = str(record.get("error", ""))
    item = record["item"]

    err_type = classify_error(error)

    # invalid_json：按你的策略直接回退 DP
    if err_type == "invalid_json":
        return {"index": idx, "fixed": build_passthrough_fixed(item)}

    # 其余类型：重跑（对文本先清洗：去 % 与空白；保留 #）
    raw_original = str(item["original_text"])
    raw_modified = str(item["modified_text"])
    clean_original, clean_to_raw_src, _raw_to_clean_src = clean_and_map(raw_original)
    clean_modified, _clean_to_raw_tgt, _raw_to_clean_tgt = clean_and_map(raw_modified)

    dp_edit_distance = int(item["edit_distance"])
    dp_operations = item.get("operations", [])

    prompt = build_realign_prompt(
        int(item["chapter_number"]),
        int(item["sentence_number"]),
        clean_original,
        clean_modified,
        dp_edit_distance,
        dp_operations,
    )

    result = call_qwen(prompt)
    ed_llm = result.get("edit_distance")
    ops_clean = result.get("operations", [])

    if not isinstance(ed_llm, int):
        raise ValueError("edit_distance must be an integer in LLM result")
    if not isinstance(ops_clean, list):
        raise ValueError("operations must be a list in LLM result")
    if ed_llm != len(ops_clean):
        raise ValueError(f"LLM edit_distance({ed_llm}) != len(operations)({len(ops_clean)})")

    # 先在 clean 语义下校验一次（更符合“忽略 %/空白”）
    clean_final = apply_operations_with_original_pos(clean_original, ops_clean)
    if clean_final != clean_modified:
        raise ValueError(
            "LLM operations do not transform original_text to modified_text.\n"
            f"Expected: {clean_modified}\nGot:      {clean_final}"
        )

    # 将 clean 的 positions 映射回 raw positions
    ops_raw = map_ops_clean_to_raw(ops_clean, clean_to_raw_src, raw_len=len(raw_original))

    # 额外再校验一次：在 raw 上执行并比较“清洗后”的结果，避免 %/空白干扰
    raw_final = apply_operations_with_original_pos(raw_original, ops_raw)
    raw_final_clean, _a, _b = clean_and_map(raw_final)
    if raw_final_clean != clean_modified:
        raise ValueError(
            "Mapped operations do not match cleaned modified_text.\n"
            f"Expected(clean): {clean_modified}\nGot(clean):      {raw_final_clean}"
        )

    fixed_item = {
        **item,
        "operations_dp": dp_operations,
        "operations_llm": ops_raw,
        "edit_distance_llm": ed_llm,
        # 便于追溯：保留清洗文本（不影响你后续合并）
        "clean_original_text": clean_original,
        "clean_modified_text": clean_modified,
    }
    return {"index": idx, "fixed": fixed_item}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {e}")


def load_processed_indices(*paths: Path) -> Set[int]:
    """
    读取已输出的 fixed / still_failed 文件，返回已处理过的 index 集合。
    用于断点续跑：再次运行时跳过这些 index，并以 append 模式继续写入。
    """
    processed: Set[int] = set()
    for p in paths:
        if not p.exists():
            continue
        try:
            for obj in iter_jsonl(p):
                if "index" in obj:
                    try:
                        processed.add(int(obj["index"]))
                    except Exception:
                        continue
        except Exception:
            # 如果旧文件末尾可能被中断写坏，允许继续跑：尽量加载到能解析的部分即可。
            # 这里选择保守：一旦解析报错，停止读取该文件，避免整段任务中断。
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        if "index" in obj:
                            processed.add(int(obj["index"]))
                    except Exception:
                        break
    return processed


def main() -> None:
    print(f"读取失败集合: {FAILED_INPUT_JSONL}")
    if not FAILED_INPUT_JSONL.exists():
        raise FileNotFoundError(f"找不到输入文件: {FAILED_INPUT_JSONL}")

    processed = load_processed_indices(FIXED_OUTPUT_JSONL, STILL_FAILED_JSONL)
    if processed:
        print(
            f"断点续跑: 已处理 index 数={len(processed)}，将跳过并继续追加写入。",
            flush=True,
        )

    total = 0  # 本次扫描到的总条数（含跳过）
    skipped = 0
    fixed_count = 0
    still_failed = 0

    FIXED_OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    STILL_FAILED_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：用追加模式写入
    with FIXED_OUTPUT_JSONL.open("a", encoding="utf-8", newline="\n") as out_fixed, STILL_FAILED_JSONL.open(
        "a", encoding="utf-8", newline="\n"
    ) as out_failed:
        start_t = time.time()
        for rec in iter_jsonl(FAILED_INPUT_JSONL):
            total += 1
            rec_index = rec.get("index")
            try:
                rec_index_int = int(rec_index)
            except Exception:
                rec_index_int = None

            if rec_index_int is not None and rec_index_int in processed:
                skipped += 1
                # 跳过不算“新处理”，但也可以打印出来方便观察断点续跑是否生效
                print(
                    f"跳过 index={rec_index_int} | scan={total} skipped={skipped} "
                    f"fixed={fixed_count} still_failed={still_failed}",
                    flush=True,
                )
                continue

            try:
                err_type = classify_error(str(rec.get("error", "")))
                fixed = fix_one_failed_record(rec)
                out_fixed.write(json.dumps(fixed, ensure_ascii=False) + "\n")
                fixed_count += 1
                outcome = "fixed"
            except Exception as e:
                still_failed += 1
                out_failed.write(
                    json.dumps(
                        {
                            "index": rec.get("index"),
                            "error": str(e),
                            "orig_error": rec.get("error"),
                            "item": rec.get("item"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                outcome = "still_failed"
            if rec_index_int is not None:
                processed.add(rec_index_int)

            done = fixed_count + still_failed
            # 每处理一条就打印一条日志（LOG_EVERY_N=1）
            if LOG_EVERY_N and done % LOG_EVERY_N == 0:
                elapsed = max(0.001, time.time() - start_t)
                rate = done / elapsed
                idx_str = str(rec_index_int) if rec_index_int is not None else str(rec_index)
                print(
                    f"处理 index={idx_str} | type={err_type} | outcome={outcome} | "
                    f"scan={total} skipped={skipped} done={done} "
                    f"(fixed={fixed_count}, still_failed={still_failed}) | "
                    f"rate≈{rate:.2f}/s",
                    flush=True,
                )

            time.sleep(SLEEP_BETWEEN_CALLS)

    print("完成。")
    print(f"- 本次扫描总数(含跳过): {total}")
    print(f"- 本次跳过(已处理): {skipped}")
    print(f"- 本次新增 fixed: {fixed_count} -> {FIXED_OUTPUT_JSONL}")
    print(f"- 本次新增 still_failed: {still_failed} -> {STILL_FAILED_JSONL}")


if __name__ == "__main__":
    main()

