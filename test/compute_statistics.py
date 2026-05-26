"""
统计三个字段的描述性统计指标：中位数、平均值、最大值、最小值、标准差、下四分位数、上四分位数

输入参数：
  --input:  JSON文件路径，默认 ../adjus_distance/total_formal_all_sentence_adjusted_distance_aggressive_llm.json
  --output-dir: 输出目录，默认 ../adjus_distance/out_data_tosee
  --edit-count-field: edit_count的来源字段，默认 operations_scored（取其长度）
  --distance-field: adjusted_edit_distance字段名，默认 adjusted_edit_distance
  --normalized-field: normalized_distance字段名，默认 normalized_distance

输出：
  在指定目录下生成三个字段的统计结果文件
"""

import json
import argparse
import os
import statistics

def compute_stats(values, field_name):
    if not values:
        return {}
    return {
        "field": field_name,
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "min": min(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        "Q1": statistics.quantiles(values, n=4)[0],
        "Q3": statistics.quantiles(values, n=4)[2],
    }

def main():
    parser = argparse.ArgumentParser(description="计算JSON中指定字段的描述性统计")
    parser.add_argument("--input", default="../adjus_distance/total_formal_all_sentence_adjusted_distance_aggressive_llm.json")
    parser.add_argument("--output-dir", default="../adjus_distance/out_data_tosee")
    parser.add_argument("--edit-count-field", default="operations_scored")
    parser.add_argument("--distance-field", default="adjusted_edit_distance")
    parser.add_argument("--normalized-field", default="normalized_distance")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    edit_counts = [len(item.get(args.edit_count_field, [])) for item in data]
    distances = [item.get(args.distance_field) for item in data if item.get(args.distance_field) is not None]
    normalizeds = [item.get(args.normalized_field) for item in data if item.get(args.normalized_field) is not None]

    results = [
        compute_stats(edit_counts, "edit_count"),
        compute_stats(distances, args.distance_field),
        compute_stats(normalizeds, args.normalized_field),
    ]

    output_path = os.path.join(args.output_dir, "statistics_summary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Human-readable output
    print(f"输出文件: {output_path}")
    for r in results:
        print(f"\n{r['field']}:")
        print(f"  中位数(median): {r['median']}")
        print(f"  平均值(mean): {r['mean']}")
        print(f"  最大值(max): {r['max']}")
        print(f"  最小值(min): {r['min']}")
        print(f"  标准差(stdev): {r['stdev']}")
        print(f"  下四分位数(Q1): {r['Q1']}")
        print(f"  上四分位数(Q3): {r['Q3']}")

if __name__ == "__main__":
    main()