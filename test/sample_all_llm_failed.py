import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List


def load_jsonl(path: Path, encoding: str) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding=encoding, newline="\n") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON 解析失败: {path} 第 {line_no} 行") from exc
            records.append(obj)
    return records


def count_errors(records: List[dict]) -> Counter:
    counter: Counter = Counter()
    for r in records:
        error_value = r.get("error", "__MISSING_ERROR__")
        counter[str(error_value)] += 1
    return counter


def print_error_stats(counter: Counter) -> None:
    print("error 类型统计（按数量降序）:")
    for error_text, cnt in counter.most_common():
        print(f"- {cnt:>6} | {error_text}")
    print(f"\n总记录数: {sum(counter.values())}")
    print(f"error 类型数: {len(counter)}")


def sample_by_target_error_keyword(
    records: List[dict], keyword: str, target_count: int, seed: int, ignore_case: bool
) -> List[dict]:
    if target_count <= 0:
        raise ValueError("target-count 必须是正整数")

    if not keyword.strip():
        raise ValueError("target-error-keyword 不能为空")

    key = keyword.lower() if ignore_case else keyword
    matched: List[dict] = []
    for r in records:
        err_text = str(r.get("error", "__MISSING_ERROR__"))
        probe = err_text.lower() if ignore_case else err_text
        if key in probe:
            matched.append(r)

    if not matched:
        raise ValueError(f"没有找到包含关键字的 error: {keyword}")
    if target_count > len(matched):
        raise ValueError(
            f"target-count({target_count}) 大于匹配到的总数量({len(matched)})"
        )

    rng = random.Random(seed)
    return rng.sample(matched, target_count)


def write_jsonl(path: Path, records: List[dict], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 all_llm_failed.jsonl 的 error 类型数量，并按指定 error 随机抽样输出。"
    )
    parser.add_argument(
        "--input-file",
        default=r"connect_edit_distance/all_llm_failed.jsonl",
        help="输入 jsonl 文件路径",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="抽样结果输出路径（jsonl）",
    )
    parser.add_argument(
        "--target-error-keyword",
        nargs="+",
        required=True,
        help="要匹配的 error 关键字（模糊匹配，支持不加引号输入多个词）",
    )
    parser.add_argument("--target-count", type=int, required=True, help="该 error 的抽样数量")
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="关键字匹配时忽略大小写",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--encoding", default="utf-8", help="文件编码")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    records = load_jsonl(input_file, args.encoding)
    counter = count_errors(records)
    print_error_stats(counter)

    keyword = " ".join(args.target_error_keyword).strip()
    sampled = sample_by_target_error_keyword(
        records=records,
        keyword=keyword,
        target_count=args.target_count,
        seed=args.seed,
        ignore_case=args.ignore_case,
    )

    write_jsonl(output_file, sampled, args.encoding)
    print(
        f"\n已从包含关键字 '{keyword}' 的 error 中抽样 {len(sampled)} 条，输出到: {output_file}"
    )


if __name__ == "__main__":
    main()

