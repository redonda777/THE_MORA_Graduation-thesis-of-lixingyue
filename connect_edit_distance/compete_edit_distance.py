import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON_PATH = ROOT_DIR / "connect_edit_distance" / "mora_v1.2_1228.json"
OUTPUT_JSON_PATH = Path(__file__).resolve().parent / "sentence_edit_distance.json"


def collect_sentence_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """递归遍历整棵树，收集所有 type == 'sentence' 的节点。"""
    sentences: List[Dict[str, Any]] = []

    if node.get("type") == "sentence":
        # 只保留我们关心的字段，避免输出过大
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
    使用动态规划计算基础编辑距离，并回溯出从 src 转换到 tgt 的字符级操作序列。

    操作代价均为 1，允许 delete / insert / replace。
    位置索引使用 0-based，下标相对“当前源串”的位置：
      - replace: {"type": "replace", "original_char": src_char, "target_char": tgt_char, "position": i}
      - insert: {"type": "insert", "char": tgt_char, "position": i}
      - delete: {"type": "delete", "char": src_char, "position": i}
    """
    m, n = len(src), len(tgt)
    # dp[i][j] 表示 src[:i] -> tgt[:j] 的最小编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src[i - 1] == tgt[j - 1]:
                cost_replace = 0
            else:
                cost_replace = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # delete
                dp[i][j - 1] + 1,          # insert
                dp[i - 1][j - 1] + cost_replace,  # replace / match
            )

    distance = dp[m][n]

    # 回溯获取操作序列
    ops: List[Dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        # 同时还有字符可用时，优先考虑替换/匹配
        if i > 0 and j > 0:
            if dp[i][j] == dp[i - 1][j - 1] and src[i - 1] == tgt[j - 1]:
                # 匹配，无操作
                i -= 1
                j -= 1
                continue

            # 替换
            if dp[i][j] == dp[i - 1][j - 1] + 1:
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

        # 删除
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(
                {
                    "type": "delete",
                    "char": src[i - 1],
                    "position": i - 1,
                }
            )
            i -= 1
            continue

        # 插入
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # 插入位置使用 i（相当于在当前 src[:i] 的尾部插入）
            ops.append(
                {
                    "type": "insert",
                    "char": tgt[j - 1],
                    "position": i,
                }
            )
            j -= 1
            continue

        # 理论上不会走到这里，如果走到说明上面判断有问题
        raise RuntimeError(f"Unexpected state in backtrace: i={i}, j={j}")

    ops.reverse()
    return distance, ops


def main() -> None:
    with INPUT_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = collect_sentence_nodes(data)
    grouped = group_by_chapter_sentence(sentences)

    results: List[Dict[str, Any]] = []

    for (chapter_number, sentence_number), nodes in grouped.items():
        # 找出同一 (chapter, sentence) 下，不同 version 的组合
        # 先按 version 名字排序，再两两组合（i < j）
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

                dist, ops = compute_edit_distance_with_ops(text_a, text_b)

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

