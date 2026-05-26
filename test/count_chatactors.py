import pandas as pd
import re
from collections import Counter

def extract_non_chinese_from_excel(file_path, output_file="1227_mora_v1.9非中文内容统计.xlsx"):
    """
    提取Excel中所有非中文字符，统计数量并排序
    :param file_path: Excel文件路径（如：data.xlsx）
    :param output_file: 输出结果文件名
    """
    # 1. 读取Excel所有工作表
    all_non_chinese = []
    
    # 读取Excel文件（兼容.xls和.xlsx）
    excel_file = pd.ExcelFile(file_path)
    
    # 遍历每个工作表
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)  # 全部转为字符串
        
        # 遍历每个单元格
        for col in df.columns:
            for cell_value in df[col]:
                if pd.isna(cell_value):  # 跳过空值
                    continue
                
                # 正则匹配：提取 非中文字符
                # 中文范围：\u4e00-\u9fa5，排除中文，保留其他所有字符
                non_chinese_chars = re.findall(r'[^\u4e00-\u9fa5]', cell_value)
                
                # 过滤空字符串，添加到总列表
                for char in non_chinese_chars:
                    if char.strip() != '' or char == ' ':  # 保留空格，过滤纯空
                        all_non_chinese.append(char)

    # 2. 统计出现次数
    char_count = Counter(all_non_chinese)
    
    # 3. 按数量降序排序
    sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)

    # 4. 输出控制台
    print("="*50)
    print("Excel非中文字符统计结果（按数量降序）")
    print("="*50)
    print(f"{'字符':<6} {'出现次数':<10}")
    print("-"*50)
    for char, count in sorted_chars:
        # 特殊字符显示处理
        display_char = char if char != ' ' else '[空格]'
        print(f"{repr(display_char):<8} {count:<10}")

    # 5. 保存到Excel
    result_df = pd.DataFrame(sorted_chars, columns=['非中文字符', '出现次数'])
    result_df.to_excel(output_file, index=False, engine='openpyxl')
    
    print("\n✅ 结果已保存到：", output_file)
    print(f"✅ 共统计到 {len(char_count)} 种非中文字符")

# ====================== 使用方法 ======================
if __name__ == "__main__":
    # 把这里改成你的Excel文件路径！！！
    EXCEL_FILE = r"D:\The_Mora\to_clean\1227_mora_v1.9.xlsx"  
    
    try:
        extract_non_chinese_from_excel(EXCEL_FILE)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {EXCEL_FILE}，请检查路径是否正确！")
    except Exception as e:
        print(f"❌ 运行出错：{str(e)}")