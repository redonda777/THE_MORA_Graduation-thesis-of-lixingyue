from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CONTROL_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "total_formal_all_sentence_adjusted_distance_aggressive_llm.json"
OUTPUT_PATH = CONTROL_DIR / "standard_edit_count_sentence_edges.json"
SUMMARY_PATH = CONTROL_DIR / "standard_edit_count_input_summary.json"


def text_length(value: Any) -> int:
    if value is None:
        return 0
    return len(str(value))


def get_edit_count(edge: dict[str, Any]) -> float:
    relation_summary = edge.get("relation_summary")
    if isinstance(relation_summary, dict) and "edit_count" in relation_summary:
        return float(relation_summary["edit_count"])
    if "edit_count" in edge:
        return float(edge["edit_count"])
    raise KeyError("edit_count")


def main() -> int:
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        raw_edges = json.load(f)
    if not isinstance(raw_edges, list):
        raise ValueError(f"Expected a list in {SOURCE_PATH}")

    transformed_edges: list[dict[str, Any]] = []
    layer_counts: Counter[str] = Counter()
    invalid_count = 0
    clamped_count = 0
    missing_edit_count = 0

    for edge in raw_edges:
        if not isinstance(edge, dict):
            invalid_count += 1
            continue
        try:
            edit_count = get_edit_count(edge)
        except (KeyError, TypeError, ValueError):
            missing_edit_count += 1
            continue

        original_length = text_length(edge.get("original_text"))
        modified_length = text_length(edge.get("modified_text"))
        denominator = max(original_length, modified_length, 1)
        normalized = edit_count / denominator
        if not math.isfinite(normalized):
            invalid_count += 1
            continue

        clamped = min(max(normalized, 0.0), 1.0)
        if abs(clamped - normalized) > 1e-12:
            clamped_count += 1

        if clamped <= 1e-12:
            layer = "L0"
        elif clamped <= 0.1:
            layer = "L1"
        elif clamped <= 0.15:
            layer = "L2"
        else:
            layer = "L3"
        layer_counts[layer] += 1

        output_edge = dict(edge)
        output_edge["adjusted_normalized_distance"] = edge.get("normalized_distance")
        output_edge["standard_edit_count"] = edit_count
        output_edge["standard_edit_denominator"] = denominator
        output_edge["standard_edit_normalized_distance"] = clamped
        output_edge["normalized_distance"] = clamped
        transformed_edges.append(output_edge)

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(transformed_edges, f, ensure_ascii=False)

    summary = {
        "source_path": str(SOURCE_PATH),
        "output_path": str(OUTPUT_PATH),
        "normalization": "relation_summary.edit_count / max(len(original_text), len(modified_text), 1)",
        "raw_edges": len(raw_edges),
        "written_edges": len(transformed_edges),
        "invalid_count": invalid_count,
        "missing_edit_count": missing_edit_count,
        "clamped_count": clamped_count,
        "distance_layer_counts_before_vgae_pruning": dict(sorted(layer_counts.items())),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
