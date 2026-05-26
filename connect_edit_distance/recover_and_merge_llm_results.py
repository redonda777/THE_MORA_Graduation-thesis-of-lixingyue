import argparse
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


RANGE_RE = re.compile(
    r"^formal_(\d+)-(\d+)_sentence_edit_distance_llm(_failed)?_v2\.(json|jsonl)$"
)


@dataclass
class RangeFiles:
    start: int
    end: int
    success_path: Path | None = None
    failed_path: Path | None = None


def load_realign_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("realign_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载算法脚本: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "process_one_item"):
        raise AttributeError("算法脚本中未找到 process_one_item(item) 函数")
    return module


def discover_range_files(data_dir: Path) -> List[RangeFiles]:
    range_map: Dict[Tuple[int, int], RangeFiles] = {}

    for file_path in sorted(data_dir.glob("formal_*_sentence_edit_distance_llm*_v2.json*")):
        m = RANGE_RE.match(file_path.name)
        if not m:
            continue

        start = int(m.group(1))
        end = int(m.group(2))
        failed_flag = m.group(3)
        ext = m.group(4)

        key = (start, end)
        if key not in range_map:
            range_map[key] = RangeFiles(start=start, end=end)

        item = range_map[key]
        if failed_flag:
            if ext != "jsonl":
                raise ValueError(f"失败文件应为 jsonl: {file_path}")
            item.failed_path = file_path
        else:
            if ext != "json":
                raise ValueError(f"成功文件应为 json: {file_path}")
            item.success_path = file_path

    ranges = sorted(range_map.values(), key=lambda x: x.start)
    if not ranges:
        raise FileNotFoundError(f"未在目录中发现可识别范围文件: {data_dir}")
    return ranges


def read_success_items(success_path: Path) -> List[Dict[str, Any]]:
    with success_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"成功文件不是 JSON 数组: {success_path}")
    return data


def read_jsonl_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                raise RuntimeError(f"解析失败日志失败: {path} line={line_no}, error={e}")
            if not isinstance(obj, dict):
                raise ValueError(f"失败日志行不是对象: {path} line={line_no}")
            rows.append(obj)
    return rows


def recover_failed_items(
    failed_rows: List[Dict[str, Any]],
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_retries: int,
    retry_sleep: float,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    recovered: Dict[int, Dict[str, Any]] = {}
    still_failed: List[Dict[str, Any]] = []

    for row_no, row in enumerate(failed_rows, start=1):
        if "index" not in row or "item" not in row:
            raise KeyError(f"失败日志缺少 index/item 字段: row_no={row_no}")
        idx = int(row["index"])
        item = row["item"]
        if not isinstance(item, dict):
            raise ValueError(f"失败日志 item 不是对象: row_no={row_no}, index={idx}")
        if idx in recovered:
            raise ValueError(f"失败日志中存在重复 index: {idx}")

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                recovered[idx] = process_fn(item)
                last_error = ""
                break
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries and retry_sleep > 0:
                    time.sleep(retry_sleep)
        if last_error:
            still_failed.append(
                {
                    "index": idx,
                    "error": last_error,
                    "item": item,
                }
            )

    return recovered, still_failed


def build_source_signature(item: Dict[str, Any]) -> str:
    core = {
        "chapter_number": item.get("chapter_number"),
        "sentence_number": item.get("sentence_number"),
        "original_text_version": item.get("original_text_version"),
        "original_text": item.get("original_text"),
        "modified_text_version": item.get("modified_text_version"),
        "modified_text": item.get("modified_text"),
        "edit_distance": item.get("edit_distance"),
        "operations": item.get("operations"),
    }
    return json.dumps(core, ensure_ascii=False, sort_keys=True)


def infer_and_recover_missing_indices(
    start: int,
    end: int,
    source_data: List[Dict[str, Any]],
    success_items: List[Dict[str, Any]],
    recovered_map: Dict[int, Dict[str, Any]],
    process_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_retries: int,
    retry_sleep: float,
    skip_indices: set[int],
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    recovered_plus = dict(recovered_map)
    extra_failed: List[Dict[str, Any]] = []

    total_count = end - start
    known_missing = len(recovered_plus)
    expected_success = total_count - known_missing
    if len(success_items) == expected_success:
        return recovered_plus, extra_failed

    if len(success_items) > expected_success:
        raise ValueError(
            f"成功条目多于理论值: range=[{start},{end}), success={len(success_items)}, "
            f"expected_success={expected_success}"
        )

    success_signatures = [build_source_signature(x) for x in success_items]
    success_cursor = 0
    inferred_missing_indices: List[int] = []

    for idx in range(start, end):
        if idx in recovered_plus:
            continue

        if success_cursor >= len(success_signatures):
            inferred_missing_indices.append(idx)
            continue

        src_sig = build_source_signature(source_data[idx])
        if src_sig == success_signatures[success_cursor]:
            success_cursor += 1
        else:
            inferred_missing_indices.append(idx)

    if success_cursor != len(success_signatures):
        raise RuntimeError(
            f"无法完成 success 对齐: range=[{start},{end}), "
            f"matched={success_cursor}, success_total={len(success_signatures)}"
        )

    for idx in inferred_missing_indices:
        if idx in skip_indices:
            continue

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                recovered_plus[idx] = process_fn(source_data[idx])
                last_error = ""
                break
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries and retry_sleep > 0:
                    time.sleep(retry_sleep)
        if last_error:
            extra_failed.append(
                {
                    "index": idx,
                    "error": last_error,
                    "item": source_data[idx],
                }
            )

    return recovered_plus, extra_failed


def merge_one_range(
    start: int,
    end: int,
    success_items: List[Dict[str, Any]],
    recovered_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    total_count = end - start
    if total_count <= 0:
        raise ValueError(f"非法范围: [{start}, {end})")

    recovered_count = len(recovered_map)
    expected_success_count = total_count - recovered_count
    if expected_success_count < 0:
        raise ValueError(f"恢复条目超出范围总数: range=[{start},{end}), recovered={recovered_count}")
    if len(success_items) != expected_success_count:
        raise ValueError(
            f"成功条目数量不匹配: range=[{start},{end}), "
            f"success={len(success_items)}, expected={expected_success_count}, recovered={recovered_count}"
        )

    merged: List[Dict[str, Any]] = []
    success_cursor = 0
    for idx in range(start, end):
        if idx in recovered_map:
            merged.append(recovered_map[idx])
        else:
            if success_cursor >= len(success_items):
                raise RuntimeError(f"成功条目不足: idx={idx}, range=[{start},{end})")
            merged.append(success_items[success_cursor])
            success_cursor += 1

    if success_cursor != len(success_items):
        raise RuntimeError(
            f"成功条目未完全消费: used={success_cursor}, total={len(success_items)}, range=[{start},{end})"
        )
    if len(merged) != total_count:
        raise RuntimeError(
            f"合并长度异常: got={len(merged)}, expected={total_count}, range=[{start},{end})"
        )

    return merged


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="重处理 failed 样本并与 success 按 index 顺序合并为全量结果。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "llm_edit_distance_0410",
        help="包含 formal_*_v2.json / formal_*_failed_v2.jsonl 的目录",
    )
    parser.add_argument(
        "--algo-script",
        type=Path,
        default=Path(__file__).resolve().parent / "2_llm_realign_edit_distance_new_0407_v2.py",
        help="包含 process_one_item(item) 的算法脚本路径",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent
        / "llm_edit_distance_0410"
        / "formal_all_sentence_edit_distance_llm_v2_merged.json",
        help="全量合并输出文件",
    )
    parser.add_argument(
        "--recovered-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "llm_edit_distance_0410" / "recovered_failed_logs",
        help="失败重处理过程输出目录（每个区间会写 recovered/still_failed 文件）",
    )
    parser.add_argument(
        "--source-input-json",
        type=Path,
        default=Path(__file__).resolve().parent / "sentence_edit_distance_with_lcs_new_0410.json",
        help="原始输入总文件（用于在 failed 日志缺失时自动推断并补齐缺失 index）",
    )
    parser.add_argument(
        "--retry-times",
        type=int,
        default=2,
        help="失败样本重处理的额外重试次数（不含首次）",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=0.5,
        help="重试间隔秒数",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    algo_script = args.algo_script.resolve()
    output_json = args.output_json.resolve()
    recovered_dir = args.recovered_dir.resolve()
    source_input_json = args.source_input_json.resolve()
    retry_times = max(0, int(args.retry_times))
    retry_sleep = max(0.0, float(args.retry_sleep))

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    if not algo_script.exists():
        raise FileNotFoundError(f"算法脚本不存在: {algo_script}")
    if not source_input_json.exists():
        raise FileNotFoundError(f"原始输入文件不存在: {source_input_json}")

    module = load_realign_module(algo_script)
    process_fn = module.process_one_item
    with source_input_json.open("r", encoding="utf-8") as f:
        source_data: List[Dict[str, Any]] = json.load(f)

    ranges = discover_range_files(data_dir)
    ranges = [r for r in ranges if r.success_path is not None]
    if not ranges:
        raise RuntimeError("没有发现成功文件，无法进行回填合并。")

    merged_all: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    for r in ranges:
        assert r.success_path is not None
        range_name = f"{r.start}-{r.end}"
        print(f"\n[range {range_name}]")
        print(f"  success: {r.success_path.name}")
        print(f"  failed : {r.failed_path.name if r.failed_path else '(none)'}")

        success_items = read_success_items(r.success_path)
        failed_rows = read_jsonl_lines(r.failed_path) if r.failed_path else []

        failed_indices = {int(x["index"]) for x in failed_rows if "index" in x}
        recovered_map, still_failed = recover_failed_items(
            failed_rows=failed_rows,
            process_fn=process_fn,
            max_retries=retry_times,
            retry_sleep=retry_sleep,
        )
        recovered_map, inferred_failed = infer_and_recover_missing_indices(
            start=r.start,
            end=r.end,
            source_data=source_data,
            success_items=success_items,
            recovered_map=recovered_map,
            process_fn=process_fn,
            max_retries=retry_times,
            retry_sleep=retry_sleep,
            skip_indices=failed_indices,
        )
        if inferred_failed:
            still_failed.extend(inferred_failed)

        recovered_rows = [
            {"index": idx, "result": recovered_map[idx]} for idx in sorted(recovered_map.keys())
        ]
        recovered_path = recovered_dir / f"formal_{range_name}_recovered_v2.jsonl"
        still_failed_path = recovered_dir / f"formal_{range_name}_still_failed_v2.jsonl"
        write_jsonl(recovered_path, recovered_rows)
        write_jsonl(still_failed_path, still_failed)

        if still_failed:
            raise RuntimeError(
                f"范围 [{r.start}, {r.end}) 仍有失败样本 {len(still_failed)} 条，"
                f"详情见: {still_failed_path}"
            )

        merged_range = merge_one_range(
            start=r.start,
            end=r.end,
            success_items=success_items,
            recovered_map=recovered_map,
        )
        merged_all.extend(merged_range)

        report.append(
            {
                "range": [r.start, r.end],
                "success_count": len(success_items),
                "failed_count": len(failed_rows),
                "recovered_count": len(recovered_map),
                "still_failed_count": len(still_failed),
                "merged_count": len(merged_range),
                "recovered_file": str(recovered_path),
                "still_failed_file": str(still_failed_path),
            }
        )

    write_json(output_json, merged_all)
    report_path = output_json.with_name(output_json.stem + "_report.json")
    write_json(report_path, report)

    print("\n处理完成")
    print(f"- 合并总条数: {len(merged_all)}")
    print(f"- 合并输出: {output_json}")
    print(f"- 统计报告: {report_path}")


if __name__ == "__main__":
    main()
