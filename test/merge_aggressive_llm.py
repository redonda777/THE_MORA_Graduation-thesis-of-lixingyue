import json
import os
import re
from glob import glob

input_dir = r"D:\The_Mora\adjus_distance"
output_dir = r"D:\The_Mora\test"
pattern = "formal_all_sentence_adjusted_distance_aggressive_llm_*.json"

files = glob(os.path.join(input_dir, pattern))

def extract_range(path):
    basename = os.path.basename(path)
    nums = re.findall(r'\d+', basename)
    if len(nums) >= 2:
        return int(nums[0])
    return float('inf')

files.sort(key=extract_range)

total = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        total.extend(data)

output_path = os.path.join(output_dir, "total_formal_all_sentence_adjusted_distance_aggressive_llm.json")
with open(output_path, 'w', encoding='utf-8') as fp:
    json.dump(total, fp, ensure_ascii=False, indent=2)

print(f"Merged {len(files)} files, total {len(total)} records")
print(f"Output: {output_path}")