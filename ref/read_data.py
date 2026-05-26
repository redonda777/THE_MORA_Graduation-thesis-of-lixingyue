# todo: @zhangyingge
# 读取数据
# 需要处理特殊字符比如无法识别的字符，删去标点符号
import pandas as pd
import re
import os
import json
from datetime import datetime

# 全局变量用于记录方括号内容
square_bracket_patterns = set()


# 方括号预处理器
class BracketPreprocessor:
    def __init__(self, bracket_patterns_set=None):
        self.mapping = {}
        self.reverse_mapping = {}
        if bracket_patterns_set:
            self.load_mapping_from_set(bracket_patterns_set)

    def load_mapping_from_set(self, bracket_patterns_set):
        """直接从square_bracket_patterns集合加载映射"""
        bracket_patterns = list(bracket_patterns_set)

        # 使用私有区字符创建映射 (U+E000 开始)
        start_code = 0xE000
        for i, pattern in enumerate(bracket_patterns):
            replacement_char = chr(start_code + i)
            self.mapping[pattern] = replacement_char
            self.reverse_mapping[replacement_char] = pattern

        print(f"从集合加载了 {len(self.mapping)} 个方括号模式映射")

        # 打印前10个映射用于调试
        print("前10个方括号映射:")
        for i, (pattern, char) in enumerate(list(self.mapping.items())[:10]):
            print(f"  {i + 1}. {pattern} -> {char} (U+{ord(char):04X})")

    def preprocess_text(self, text):
        """将文本中的方括号替换为特殊字符"""
        if not text or pd.isna(text):
            return text

        text = str(text)

        # 按长度降序排序，确保先匹配长的模式（避免部分匹配）
        sorted_patterns = sorted(self.mapping.keys(), key=len, reverse=True)

        for pattern in sorted_patterns:
            replacement = self.mapping[pattern]
            text = text.replace(pattern, replacement)

        return text

    def postprocess_text(self, text):
        """将特殊字符恢复为方括号（用于显示结果）"""
        if not text:
            return text

        for char, pattern in self.reverse_mapping.items():
            text = text.replace(char, pattern)

        return text


# 全局预处理器实例（稍后在process_laozi_file中初始化）
bracket_preprocessor = None


def save_bracket_patterns_to_json(patterns_set):
    """
    将方括号模式保存为JSON文件

    参数:
        patterns_set: 方括号模式的集合
    """

    filename = f"square_bracket_patterns.json"

    # 转换为列表并排序以便更好的可读性
    patterns_list = sorted(list(patterns_set))

    # 构建JSON数据结构
    json_data = {
        "metadata": {
            "generated_time": datetime.now().isoformat(),
            "total_patterns": len(patterns_list),
            "description": "方括号模式收集结果"
        },
        "patterns": patterns_list,
        "mapping_info": {
            "total_mappings": len(bracket_preprocessor.mapping) if bracket_preprocessor else 0,
            "mapping_range": "U+E000 to U+E0FF (Private Use Area)"
        }
    }

    # 如果预处理器已初始化，添加映射信息
    if bracket_preprocessor:
        json_data["character_mappings"] = [
            {
                "original": pattern,
                "replacement": char,
                "unicode": f"U+{ord(char):04X}"
            }
            for pattern, char in bracket_preprocessor.mapping.items()
        ]

    # 保存为JSON文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"方括号模式已保存到: {file_path}")
        print(f"总共保存了 {len(patterns_list)} 个方括号模式")
        return file_path
    except Exception as e:
        print(f"保存方括号模式到JSON时出错: {e}")
        return None


def clean_text(text):
    global square_bracket_patterns, bracket_preprocessor

    if pd.isna(text):
        return text

    text = str(text)

    # 首先收集方括号模式
    bracket_matches = re.findall(r'\[[^\]]*\]', text)
    for pattern in bracket_matches:
        square_bracket_patterns.add(pattern)

    # 预处理方括号（如果预处理器已初始化）
    if bracket_preprocessor:
        text = bracket_preprocessor.preprocess_text(text)

    # 其他字符替换
    text = text.replace('□', '#').replace('*', '#')

    # 处理括号替换规则 - 更新汉字范围
    text = re.sub(r'([\u3400-\u4dbf\u4e00-\u9fff])(（([\u3400-\u4dbf\u4e00-\u9fff])）)', r'\3', text)
    text = re.sub(r'([\u3400-\u4dbf\u4e00-\u9fff])(\(([\u3400-\u4dbf\u4e00-\u9fff])\))', r'\3', text)

    # 若〔 〕里只有一个字，则删除
    text = re.sub(r'〔([^〕]{1})〕', '', text)

    # 删除特殊标记
    text = re.sub(r'SPC_ORD|SPC_INFO', '', text)

    # 删除带数字的括号内容
    text = re.sub(r'（[^）]*\d[^）]*）', '', text)
    text = re.sub(r'\([^)]*\d[^)]*\)', '', text)

    # 删除包含多个字的括号
    text = re.sub(r'（[^）]{2,}）', '', text)
    text = re.sub(r'\([^)]{2,}\)', '', text)

    # 删除数字
    text = re.sub(r'\d+', '', text)

    # 删除○
    text = text.replace('○', '')

    # 删除`及其后边的字符
    if '`' in text:
        text = text.split('`')[0]

    # 处理〔 〕里多个字的情况
    text = re.sub(r'〔([^〕]{2,})〕', r'\1', text)

    # 更安全的字符过滤方法
    allowed_chars = []
    for char in text:
        # 保留汉字 - 更新范围
        if '\u3400' <= char <= '\u4dbf' or '\u4e00' <= char <= '\u9fff':
            allowed_chars.append(char)
        # 保留字母
        elif 'a' <= char.lower() <= 'z':
            allowed_chars.append(char)
        # 保留特定标点符号
        elif char in {'=', '#', '%', '…', '~'}:
            allowed_chars.append(char)
        # 保留私有区字符 (方括号替换后的字符)
        elif bracket_preprocessor and '\uE000' <= char <= '\uE0FF':
            allowed_chars.append(char)
        # 其他字符不保留

    result = ''.join(allowed_chars)

    # 找到所有等号的位置
    equals_positions = [i for i, char in enumerate(result) if char == '=']

    # 从后往前处理，这样索引不会因为替换而改变
    for pos in reversed(equals_positions):
        result = result[:pos] + result[pos - 1] + result[pos + 1:]

    # 特别处理：如果结果为空或只包含特殊字符，返回空字符串
    if not result.strip():
        return ""

    return result


def process_laozi_file(save_json=True):
    global square_bracket_patterns, bracket_preprocessor

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义文件名
    filename = '老子文本比对-自校验.xlsx'
    file_path = os.path.join(current_dir, filename)

    if not os.path.exists(file_path):
        print(f"未找到文件: {filename}")
        return None

    try:
        df = pd.read_excel(file_path, sheet_name='Sheet4')

        # 获取所有文本列（从C列到N列）
        text_columns = df.columns[2:]

        # 第一遍：收集所有方括号模式
        print("正在收集方括号模式...")
        for col in text_columns:
            for text in df[col]:
                if pd.notna(text):
                    # 使用原始clean_text逻辑收集方括号模式
                    text_str = str(text)
                    bracket_matches = re.findall(r'\[[^\]]*\]', text_str)
                    for pattern in bracket_matches:
                        square_bracket_patterns.add(pattern)

        print(f"收集到 {len(square_bracket_patterns)} 种方括号模式")

        # 保存方括号模式到JSON文件
        if save_json and square_bracket_patterns:
            json_filename = save_bracket_patterns_to_json(square_bracket_patterns)

        # 初始化预处理器
        bracket_preprocessor = BracketPreprocessor(square_bracket_patterns)

        # 第二遍：使用预处理器清理文本
        print("正在清理文本...")
        for col in text_columns:
            df[col] = df[col].apply(clean_text)

        output_path = os.path.join(current_dir, 'CleanData.xlsx')
        df.to_excel(output_path, index=False)

        # 检查清理后的文本中是否还有未处理的方括号
        remaining_brackets = set()
        for col in text_columns:
            for text in df[col]:
                if pd.notna(text):
                    bracket_matches = re.findall(r'\[[^\]]*\]', str(text))
                    for pattern in bracket_matches:
                        remaining_brackets.add(pattern)

        if remaining_brackets:
            print(f"警告：清理后仍有 {len(remaining_brackets)} 种方括号模式未处理")
            print("未处理的方括号模式:")
            for pattern in list(remaining_brackets)[:10]:  # 只显示前10个
                print(f"  {pattern}")

        print(f"数据处理完成，结果保存到: {output_path}")
        return df

    except Exception as e:
        print(f"处理数据时出错: {e}")
        return None


if __name__ == "__main__":
    result = process_laozi_file(save_json=True)