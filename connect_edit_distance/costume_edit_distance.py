import json
import requests  # 用于调用通义千问API
from typing import List, Dict

# ===================== 核心配置（你需要修改这部分） =====================
# 通义千问API配置（去阿里云获取：https://dashscope.console.aliyun.com/）
DASHSCOPE_API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"
# 你的原始数据文件路径（JSON格式，每行一条你之前的操作结果）
INPUT_FILE_PATH = r"D:\The_Mora\connect_edit_distance\output_exaple.json"
# 处理后结果保存路径
OUTPUT_FILE_PATH = r"processed_data_with_custom_cost.json"

# ===================== 核心函数 =====================
def get_qianfan_response(prompt: str) -> Dict:
    """
    调用通义千问API，获取字符操作的关系类型和定制代价
    :param prompt: 拼接好的Prompt字符串
    :return: 解析后的字典（relation_type, reason, custom_cost）
    """
    # 通义千问API调用参数
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",  # 基础版，也可用qwen-plus/qwen-max
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "result_format": "json",  # 强制返回JSON
            "temperature": 0.1,  # 低随机性，保证判断稳定
            "top_p": 0.9,
            "max_tokens": 500
        }
    }

    try:
        # 发送请求
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # 抛出HTTP错误
        # 解析响应
        result = response.json()
        if "output" in result and "choices" in result["output"]:
            content = result["output"]["choices"][0]["message"]["content"].strip()
            return json.loads(content)  # 转为字典返回
        else:
            raise Exception(f"API返回格式异常: {result}")
    except Exception as e:
        print(f"调用通义千问失败: {str(e)}")
        # 异常兜底：默认归为其他，代价1
        return {
            "relation_type": "其他",
            "reason": f"调用失败: {str(e)}",
            "custom_cost": 1.0
        }

def build_prompt(chapter: int, sentence: int, original_text: str, 
                 modified_text: str, original_char: str, target_char: str) -> str:
    """
    构建通义千问的Prompt（通用模板，无示例干扰）
    """
    prompt_template = """你是古籍校勘专家，处理《道德经》版本异文。
请对下面的字符替换进行分类，并输出严格JSON，无多余内容。

【上下文】
章节：{chapter}
句子：{sentence}
原句：{original_text}
目标句：{modified_text}

【待判断操作】
{original_char} → {target_char}

【类型只能选】
繁简转换、通假字、补字、误抄、同义替换、其他

【代价规则】
繁简转换：0
同义替换：0
通假字：0.5
补字：0.2
误抄：2
其他：1

【输出格式】
{{
  "relation_type": "类型",
  "reason": "理由",
  "custom_cost": 代价
}}"""
    return prompt_template.format(
        chapter=chapter,
        sentence=sentence,
        original_text=original_text,
        modified_text=modified_text,
        original_char=original_char,
        target_char=target_char
    )

def process_all_data(input_path: str, output_path: str) -> None:
    """
    批量处理所有数据：遍历每条数据→处理每个操作→计算定制距离→保存结果
    """
    # 读取原始数据
    with open(input_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)  # 直接读整个数组 JSON,不需要按行读取
    processed_data = []
    # 遍历每条数据
    for idx, item in enumerate(original_data, 1):
        print(f"正在处理第{idx}条数据（章节{item['chapter_number']}，句子{item['sentence_number']}）...")
        # 提取基础信息
        chapter = item["chapter_number"]
        sentence = item["sentence_number"]
        original_text = item["original_text"]
        modified_text = item["modified_text"]
        
        # 处理每个字符操作
        processed_operations = []
        total_custom_cost = 0.0
        for op in item["operations"]:
            op_type = op.get("type", "")

            if op_type == "replace":
                # 替换操作：调用通义千问，根据上下文和字符对得到关系类型与代价
                original_char = op["original_char"]
                target_char = op["target_char"]
                prompt = build_prompt(
                    chapter,
                    sentence,
                    original_text,
                    modified_text,
                    original_char,
                    target_char,
                )
                qianfan_result = get_qianfan_response(prompt)
                processed_op = {**op, **qianfan_result}
            elif op_type == "insert":
                # 插入操作：视为“补字”，使用固定代价 0.2
                processed_op = {
                    **op,
                    "relation_type": "补字",
                    "reason": "根据规则：插入视为补字，固定代价0.2",
                    "custom_cost": 0.2,
                }
            elif op_type == "delete":
                # 删除操作：视为“误抄”，使用固定代价 2
                processed_op = {
                    **op,
                    "relation_type": "误抄",
                    "reason": "根据规则：删除视为误抄，固定代价2",
                    "custom_cost": 2.0,
                }
            else:
                # 未知类型：归为“其他”，代价1
                processed_op = {
                    **op,
                    "relation_type": "其他",
                    "reason": f"未知操作类型: {op_type}，按其他处理",
                    "custom_cost": 1.0,
                }

            processed_operations.append(processed_op)
            total_custom_cost += processed_op["custom_cost"]
        
        # 构建最终数据
        final_item = {
            **item,  # 保留原始所有字段
            "custom_edit_distance": round(total_custom_cost, 2),  # 定制编辑距离（保留2位小数）
            "operations": processed_operations  # 替换为带代价的操作
        }
        processed_data.append(final_item)
    
    # 保存处理后的数据：作为一个完整的 JSON 数组输出，便于后续整体解析
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"所有数据处理完成！结果已保存至：{output_path}")

# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 运行批量处理
    process_all_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)