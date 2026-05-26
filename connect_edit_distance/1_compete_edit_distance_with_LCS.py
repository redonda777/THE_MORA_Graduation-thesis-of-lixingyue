import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

'''
在compete_edit_distance.py的基础上，使用 LCS 算法计算编辑距离，并回溯字符级操作。
使用 LCS 算法计算编辑距离，并回溯字符级操作。但是【目前无效】
仅允许 delete / insert / replace，代价均为 1。
输入参数：mora_v1.2_1228.json
输出参数：sentence_edit_distance_with_lcs.json
'''


ROOT_DIR = Path(__file__).resolve().parent.parent
#DEFAULT_INPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "mora_v1.2_1228.json"
DEFAULT_INPUT_JSON_PATH = ROOT_DIR / "contribute" / "mora_v4.1_0406.json"
#FALLBACK_INPUT_JSON_PATH = ROOT_DIR / "contribute" / "mora_v1.2_1228.json"
FALLBACK_INPUT_JSON_PATH = ROOT_DIR / "contribute" / "mora_v4.1_0406.json"
OUTPUT_JSON_PATH = (
    Path(__file__).resolve().parent / "sentence_edit_distance_with_lcs_v4.1_0406.json"
)


def collect_sentence_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """递归遍历整棵树，收集所有 type == 'sentence' 的节点。"""
    sentences: List[Dict[str, Any]] = []

    if node.get("type") == "sentence":
        sentences.append(
            {
                "chapter_number": node.get("chapter_number"),
                "sentence_number": node.get("sentence_number"),
                "version": node.get("version"),
                "text": node.get("text", ""),
            }
        )

    for child in node.get("children", []) or []:
        sentences.extend(collect_sentence_nodes(child))

    return sentences


def group_by_chapter_sentence(
    sentences: List[Dict[str, Any]]
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """按 (chapter_number, sentence_number) 对句子进行分组。"""
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for s in sentences:
        chap = s.get("chapter_number")
        sent = s.get("sentence_number")
        if chap is None or sent is None:
            continue
        key = (int(chap), int(sent))
        grouped.setdefault(key, []).append(s)
    return grouped


def compute_edit_distance_with_ops(
    src: str, tgt: str
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    使用动态规划计算 src -> tgt 的最小编辑距离，并回溯字符级操作。
    仅允许 delete / insert / replace，代价均为 1。
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
                dp[i - 1][j] + 1,  # delete
                dp[i][j - 1] + 1,  # insert
                dp[i - 1][j - 1] + replace_cost,  # replace or match
            )

    ops: List[Dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src[i - 1] == tgt[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(
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
            ops.append({"type": "delete", "char": src[i - 1], "position": i - 1})
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append({"type": "insert", "char": tgt[j - 1], "position": i})
            j -= 1
            continue

        raise RuntimeError(f"Unexpected backtrace state: i={i}, j={j}")

    ops.reverse()
    return dp[m][n], ops


def build_lcs_anchors(src: str, tgt: str) -> List[Tuple[int, int]]:
    """
    构建 LCS 锚点列表 (src_idx, tgt_idx)。
    回溯时优先 diagonal 匹配，有助于保留连续相同字符的对齐。
    """
    m, n = len(src), len(tgt)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src[i - 1] == tgt[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])

    anchors_rev: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if src[i - 1] == tgt[j - 1] and lcs[i][j] == lcs[i - 1][j - 1] + 1:
            anchors_rev.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif lcs[i - 1][j] > lcs[i][j - 1]:
            i -= 1
        elif lcs[i - 1][j] < lcs[i][j - 1]:
            j -= 1
        else:
            # 长度相同的并列路径，优先向上走可减少 tgt 侧跳动，通常更稳定。
            i -= 1

    anchors_rev.reverse()
    return anchors_rev


def compute_edit_distance_with_lcs_alignment(
    src: str, tgt: str
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    先用 LCS 建立锚点，再对锚点之间的片段做 DP 编辑距离回溯，减少语义错位。
    """
    anchors = build_lcs_anchors(src, tgt)
    all_ops: List[Dict[str, Any]] = []
    total_distance = 0

    prev_src = 0
    prev_tgt = 0
    current_global_pos = 0

    for anchor_src, anchor_tgt in anchors:
        seg_src = src[prev_src:anchor_src]
        seg_tgt = tgt[prev_tgt:anchor_tgt]
        seg_dist, seg_ops = compute_edit_distance_with_ops(seg_src, seg_tgt)
        total_distance += seg_dist

        for op in seg_ops:
            fixed_op = dict(op)
            fixed_op["position"] = current_global_pos + int(op["position"])
            all_ops.append(fixed_op)

        # 该片段在变换后长度为 len(seg_tgt)，后续位置应从其末尾继续。
        current_global_pos += len(seg_tgt)

        # 跳过锚点处 match 字符（无操作，但位置向后推进 1）
        current_global_pos += 1
        prev_src = anchor_src + 1
        prev_tgt = anchor_tgt + 1

    tail_src = src[prev_src:]
    tail_tgt = tgt[prev_tgt:]
    tail_dist, tail_ops = compute_edit_distance_with_ops(tail_src, tail_tgt)
    total_distance += tail_dist

    for op in tail_ops:
        fixed_op = dict(op)
        fixed_op["position"] = current_global_pos + int(op["position"])
        all_ops.append(fixed_op)

    return total_distance, all_ops


def resolve_input_path() -> Path:
    """优先读取 connect_edit_distance 目录下输入，不存在则回退到 contribute。"""
    if DEFAULT_INPUT_JSON_PATH.exists():
        return DEFAULT_INPUT_JSON_PATH
    if FALLBACK_INPUT_JSON_PATH.exists():
        return FALLBACK_INPUT_JSON_PATH
    raise FileNotFoundError(
        f"Input JSON not found: {DEFAULT_INPUT_JSON_PATH} or {FALLBACK_INPUT_JSON_PATH}"
    )


def main() -> None:
    input_path = resolve_input_path()
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = collect_sentence_nodes(data)
    grouped = group_by_chapter_sentence(sentences)
    results: List[Dict[str, Any]] = []

    for (chapter_number, sentence_number), nodes in grouped.items():
        nodes_sorted = sorted(nodes, key=lambda x: str(x.get("version", "")))
        count = len(nodes_sorted)
        if count < 2:
            continue

        for i in range(count):
            for j in range(i + 1, count):
                a = nodes_sorted[i]
                b = nodes_sorted[j]
                version_a = a.get("version")
                version_b = b.get("version")
                if not version_a or not version_b or version_a == version_b:
                    continue

                text_a = a.get("text", "")
                text_b = b.get("text", "")

                dist, ops = compute_edit_distance_with_lcs_alignment(text_a, text_b)
                results.append(
                    {
                        "chapter_number": chapter_number,
                        "sentence_number": sentence_number,
                        "original_text_version": version_a,
                        "original_text": text_a,
                        "modified_text_version": version_b,
                        "modified_text": text_b,
                        "edit_distance": dist,
                        "operations": ops,
                    }
                )

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
