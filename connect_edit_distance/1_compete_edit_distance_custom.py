import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_JSON_PATH = ROOT_DIR / "contribute" / "mora_v4.1_0406.json"
FALLBACK_INPUT_JSON_PATH = ROOT_DIR / "contribute" / "mora_v4.1_0406.json"
OUTPUT_JSON_PATH = (
    Path(__file__).resolve().parent / "sentence_edit_distance_custom_v6.1_0407.json"
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
    sentences: List[Dict[str, Any]],
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


def edit_distance(s1: str, s2: str) -> Tuple[float, List[Dict[str, Any]]]:
    """
    计算两个字符串的原始编辑距离和操作路径
    省略、缺失等情况不应该在此出现
    """
    if s1 == "" or s2 == "":
        raise ValueError("s1或s2字符串不能为空")

    if "……" in s1 or "……" in s2:
        raise ValueError("s1或s2字符串包含省略号")

    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    operations: List[List[Tuple[str, ...] | None]] = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = float(i)
        if i > 0:
            operations[i][0] = ("删除", s1[i - 1])
    for j in range(n + 1):
        dp[0][j] = float(j)
        if j > 0:
            operations[0][j] = ("插入", s2[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                if s2[j - 1] != "#":
                    operations[i][j] = ("匹配", s1[i - 1])
                else:
                    operations[i][j] = ("替换", s1[i - 1], s2[j - 1])
            else:
                replace_cost = dp[i - 1][j - 1] + 1
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1

                min_cost = min(replace_cost, delete_cost, insert_cost)
                dp[i][j] = min_cost

                if min_cost == replace_cost:
                    operations[i][j] = ("替换", s1[i - 1], s2[j - 1])
                elif min_cost == delete_cost:
                    operations[i][j] = ("删除", s1[i - 1])
                else:
                    operations[i][j] = ("插入", s2[j - 1])

    operations_out: List[Dict[str, Any]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            op = operations[i][j]
            if op is None:
                raise RuntimeError(f"回溯失败: i={i}, j={j}")

            if op[0] == "匹配":
                i -= 1
                j -= 1
            elif op[0] == "替换":
                operations_out.append(
                    {
                        "type": "replace",
                        "original_char": s1[i - 1],
                        "target_char": s2[j - 1],
                        "position": i - 1,
                    }
                )
                i -= 1
                j -= 1
            elif op[0] == "删除":
                operations_out.append(
                    {"type": "delete", "char": s1[i - 1], "position": i - 1}
                )
                i -= 1
            else:
                operations_out.append(
                    {"type": "insert", "char": s2[j - 1], "position": i}
                )
                j -= 1
        elif i > 0:
            op = operations[i][j]
            if op is None:
                raise RuntimeError(f"回溯失败: i={i}, j={j}")
            operations_out.append({"type": "delete", "char": s1[i - 1], "position": i - 1})
            i -= 1
        else:
            op = operations[i][j]
            if op is None:
                raise RuntimeError(f"回溯失败: i={i}, j={j}")
            operations_out.append({"type": "insert", "char": s2[j - 1], "position": i})
            j -= 1

    operations_out.reverse()
    return dp[m][n], operations_out


def resolve_input_path() -> Path:
    """优先读取默认输入，不存在则回退。"""
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
                try:
                    dist, ops = edit_distance(text_a, text_b)
                except ValueError:
                    continue

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
