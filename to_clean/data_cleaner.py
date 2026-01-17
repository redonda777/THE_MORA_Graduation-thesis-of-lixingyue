"""
数据清洗脚本
功能：清洗Excel表格中C到N列的数据，去除句子开头和末尾的标点符号
"""

import pandas as pd
import os
import re
from pathlib import Path

# ==================== 全局配置 ====================
# 要清洗的标点符号（可根据需要修改）
# 注意：反斜杠字符（\）需要特殊处理，这里使用 chr(92) 明确表示
PUNCTUATION_TO_REMOVE = '=，。、；：！？""''（）%【】《》〈〉「」『』〔〕｛｝〖〗·…—•~' + '\\'

# 特殊内容列表：如果单元格内容完全匹配这些内容，则清空单元格
# 支持两种匹配方式：
# 1. 普通字符串匹配：直接写字符串，如 '特殊标记'
# 2. 正则表达式匹配：写正则表达式字符串，如 '^\\d+$' 表示纯数字，'^[，。]+$' 表示仅包含标点
# 注意：正则表达式中的反斜杠需要转义，如 '^\\d+$' 表示匹配纯数字
# 示例：
# SPECIAL_CONTENT_TO_CLEAR = ['特殊标记', '^\\d+$', '^[，。、；：！？""''（）%【】《》〈〉「」『』〔〕｛｝〖〗·…—•\\\\]+$']
SPECIAL_CONTENT_TO_CLEAR = [
    'SPC_ORD','^\\d+$','\\(\\d+\\)','SPC_INFO',
    # 匹配含有特殊汉字的单元格（可根据需要修改特殊汉字列表）
    # 特殊汉字列表：第、项、条、款、注、说明、提示、标记、序号、编号等
    # 如果单元格中包含这些汉字中的任意一个，则会被清空
    #'.*[第项条款注说明提示标记序号编号].*',  # 包含特殊汉字的单元格会被清空
    # 匹配同时包含数字和文字的单元格（数字和文字一起出现）
    # 例如："第1条"、"项目123"、"abc123"等包含数字和文字（汉字或英文字母）的单元格会被清空
    # 使用Unicode范围匹配汉字：\u4e00-\u9fff 表示常用汉字范围
    '.*\\d.*[a-zA-Z\u4e00-\u9fff].*|.*[a-zA-Z\u4e00-\u9fff].*\\d.*',  # 同时包含数字和文字的单元格会被清空
    # 在这里添加需要清空的特殊内容
    # 示例（取消注释以启用）：
    # '特殊标记',  # 完全匹配"特殊标记"的单元格会被清空
    # '^\\d+$',  # 纯数字的单元格会被清空（如 "123"）
    # '^[，。、；：！？""''%【】《》〈〉「」『』〔〕｛｝〖〗·…—•\\\\]+$',  # 仅包含标点符号的单元格会被清空
]

# Excel文件路径（请修改为你的文件路径）
INPUT_FILE = r'D:\The_Mora\to_clean\1227_mora_v1.8.xlsx'  # 输入文件路径
OUTPUT_FILE = r'D:\The_Mora\to_clean\1227_mora_v1.9.xlsx'  # 输出文件路径

# ==================== 清洗函数 ====================

def clean_text_initial_and_final(text):
    """
    清洗文本，去除开头和末尾的标点符号
    
    参数:
        text: 要清洗的文本（字符串或数字）
    
    返回:
        清洗后的文本
    """
    # 处理空值
    if pd.isna(text):
        return text
    
    # 转换为字符串
    text = str(text)
    
    # 处理空字符串
    if not text:
        return text
    
    # 去除开头的标点符号
    while text and text[0] in PUNCTUATION_TO_REMOVE:
        text = text[1:]
    
    # 去除末尾的标点符号
    while text and text[-1] in PUNCTUATION_TO_REMOVE:
        text = text[:-1]
    
    return text


def check_and_clear_special_content(text):
    """
    检查文本是否为特殊内容，如果是则返回空值（清空单元格）
    
    参数:
        text: 要检查的文本（字符串或数字）
    
    返回:
        如果是特殊内容则返回 None（清空单元格），否则返回原文本
    """
    # 处理空值
    if pd.isna(text):
        return text
    
    # 转换为字符串并去除首尾空格
    text_str = str(text).strip()
    
    # 处理空字符串
    if not text_str:
        return text
    
    # 如果没有配置特殊内容列表，直接返回原文本
    if not SPECIAL_CONTENT_TO_CLEAR:
        return text
    
    # 检查是否匹配特殊内容
    for pattern in SPECIAL_CONTENT_TO_CLEAR:
        if not isinstance(pattern, str):
            pattern = str(pattern)
        
        # 尝试作为正则表达式匹配（如果包含正则表达式特殊字符）
        # 如果正则表达式匹配失败，则作为普通字符串匹配
        try:
            if re.match(pattern + '$', text_str):  # 完全匹配
                return None  # 匹配成功，清空单元格
        except re.error:
            # 正则表达式错误，当作普通字符串匹配
            if text_str == pattern:
                return None
    
    return text  # 不是特殊内容，返回原文本


def clean_excel_data(input_file, output_file):
    """
    清洗Excel文件中的数据
    
    参数:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径
    """
    # 标准化路径处理
    input_file = str(Path(input_file).resolve())
    output_file = str(Path(output_file).resolve())
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 '{input_file}'")
        return
    
    # 读取Excel文件
    print(f"正在读取文件: {input_file}")
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 创建数据副本（不修改原数据）
    df_cleaned = df.copy()
    
    # C列到N列对应的列索引（C=2, D=3, ..., N=13）
    # 在pandas中，列索引从0开始，但我们需要通过列名来访问
    # 获取列名列表
    columns = df.columns.tolist()
    
    # 如果列数少于14列（A到N），则提示错误
    if len(columns) < 14:
        print(f"警告：文件列数不足14列（当前{len(columns)}列），将只处理存在的列")
    
    # 确定要处理的列范围（C到N列，索引2-13）
    start_col_idx = 2  # C列
    end_col_idx = min(13, len(columns) - 1)  # N列，但不超过实际列数
    
    # 计算列字母标识
    def get_column_letter(idx):
        """将列索引转换为Excel列字母（A, B, C, ...）"""
        result = ""
        idx += 1  # 转换为1-based
        while idx > 0:
            idx -= 1
            result = chr(ord('A') + (idx % 26)) + result
            idx //= 26
        return result
    
    start_col_letter = get_column_letter(start_col_idx)
    end_col_letter = get_column_letter(end_col_idx)
    
    print(f"正在清洗第 {start_col_letter} 列到第 {end_col_letter} 列的数据...")
    print(f"要清洗的标点符号: {PUNCTUATION_TO_REMOVE}")
    if SPECIAL_CONTENT_TO_CLEAR:
        print(f"特殊内容清空规则: {SPECIAL_CONTENT_TO_CLEAR}")
    else:
        print("未配置特殊内容清空规则")
    
    # 统计清洗的单元格数量
    cleaned_count = 0
    
    # 遍历C到N列
    for col_idx in range(start_col_idx, end_col_idx + 1):
        col_name = columns[col_idx]
        col_letter = get_column_letter(col_idx)
        print(f"正在处理列: {col_name} (第{col_letter}列)")
        
        # 遍历该列的所有单元格
        for row_idx in range(len(df)):
            original_value = df.iloc[row_idx, col_idx]
            
            # 只处理不为空的单元格
            if pd.notna(original_value):
                # 先检查是否为特殊内容，如果是则清空
                checked_value = check_and_clear_special_content(original_value)
                
                # 如果被识别为特殊内容并清空
                if checked_value is None:
                    df_cleaned.iloc[row_idx, col_idx] = None
                    cleaned_count += 1
                else:
                    # 如果不是特殊内容，则进行标点符号清洗
                    cleaned_value = clean_text_initial_and_final(str(checked_value))
                    
                    # 如果清洗后的值与原值不同，则更新
                    if cleaned_value != str(original_value):
                        df_cleaned.iloc[row_idx, col_idx] = cleaned_value
                        cleaned_count += 1
    
    # 保存清洗后的数据到新文件
    print(f"\n正在保存清洗后的数据到: {output_file}")
    try:
        # 确保输出目录存在
        output_path = Path(output_file)
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"已创建输出目录: {output_dir}")
        
        # 使用Path对象确保路径正确
        df_cleaned.to_excel(str(output_path), index=False, engine='openpyxl')
        print(f"✓ 清洗完成！")
        print(f"✓ 共清洗了 {cleaned_count} 个单元格")
        print(f"✓ 结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        import traceback
        traceback.print_exc()


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("=" * 50)
    print("Excel数据清洗脚本")
    print("=" * 50)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"清洗标点: {PUNCTUATION_TO_REMOVE}")
    if SPECIAL_CONTENT_TO_CLEAR:
        print(f"特殊内容清空规则: {SPECIAL_CONTENT_TO_CLEAR}")
    print("=" * 50)
    print()
    
    clean_excel_data(INPUT_FILE, OUTPUT_FILE)

