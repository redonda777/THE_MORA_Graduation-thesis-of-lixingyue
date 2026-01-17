"""
Excel转CSV脚本
功能：将Excel文件转换为CSV格式
支持单文件转换、批量转换、指定工作表等功能
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path


def excel_to_csv(input_file, output_file=None, sheet_name=None, encoding='utf-8-sig', index=False):
    """
    将Excel文件转换为CSV文件
    
    参数:
        input_file: 输入Excel文件路径
        output_file: 输出CSV文件路径（如果为None，则自动生成）
        sheet_name: 工作表名称（如果为None，则转换第一个工作表或所有工作表）
        encoding: CSV文件编码（默认utf-8-sig，支持Excel打开）
        index: 是否包含行索引（默认False）
    
    返回:
        成功转换的文件列表
    """
    input_path = Path(input_file)
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误：找不到输入文件 '{input_file}'")
        return []
    
    # 如果没有指定输出文件，自动生成
    if output_file is None:
        output_file = input_path.with_suffix('.csv')
    else:
        output_file = Path(output_file)
    
    print(f"正在读取文件: {input_path}")
    
    try:
        # 读取Excel文件的所有工作表名称
        excel_file = pd.ExcelFile(input_path, engine='openpyxl')
        sheet_names = excel_file.sheet_names
        
        print(f"发现 {len(sheet_names)} 个工作表: {sheet_names}")
        
        converted_files = []
        
        # 如果指定了工作表名称
        if sheet_name is not None:
            if sheet_name not in sheet_names:
                print(f"错误：工作表 '{sheet_name}' 不存在")
                return []
            
            # 读取指定的工作表
            df = pd.read_excel(input_path, sheet_name=sheet_name, engine='openpyxl')
            
            # 生成输出文件名（如果指定了工作表，在文件名中包含工作表名）
            if len(sheet_names) > 1:
                output_path = output_file.parent / f"{output_file.stem}_{sheet_name}{output_file.suffix}"
            else:
                output_path = output_file
            
            # 保存为CSV
            df.to_csv(output_path, index=index, encoding=encoding)
            print(f"✓ 已转换: {input_path.name} -> {output_path.name}")
            print(f"  - 工作表: {sheet_name}")
            print(f"  - 行数: {len(df)}, 列数: {len(df.columns)}")
            converted_files.append(str(output_path))
        
        # 如果没有指定工作表，转换所有工作表
        else:
            if len(sheet_names) == 1:
                # 只有一个工作表，直接转换
                df = pd.read_excel(input_path, engine='openpyxl')
                df.to_csv(output_file, index=index, encoding=encoding)
                print(f"✓ 已转换: {input_path.name} -> {output_file.name}")
                print(f"  - 行数: {len(df)}, 列数: {len(df.columns)}")
                converted_files.append(str(output_file))
            else:
                # 多个工作表，为每个工作表生成一个CSV文件
                for sheet in sheet_names:
                    df = pd.read_excel(input_path, sheet_name=sheet, engine='openpyxl')
                    output_path = output_file.parent / f"{output_file.stem}_{sheet}{output_file.suffix}"
                    df.to_csv(output_path, index=index, encoding=encoding)
                    print(f"✓ 已转换: {input_path.name} -> {output_path.name}")
                    print(f"  - 工作表: {sheet}")
                    print(f"  - 行数: {len(df)}, 列数: {len(df.columns)}")
                    converted_files.append(str(output_path))
        
        return converted_files
    
    except Exception as e:
        print(f"转换文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return []


def batch_convert(input_dir, output_dir=None, pattern='*.xlsx', sheet_name=None, encoding='utf-8-sig', index=False):
    """
    批量转换目录中的Excel文件为CSV
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（如果为None，则在输入目录中创建csv文件夹）
        pattern: 文件匹配模式（默认*.xlsx）
        sheet_name: 工作表名称（如果为None，则转换所有工作表）
        encoding: CSV文件编码
        index: 是否包含行索引
    
    返回:
        成功转换的文件数量
    """
    input_path = Path(input_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"错误：输入目录 '{input_dir}' 不存在或不是目录")
        return 0
    
    # 确定输出目录
    if output_dir is None:
        output_path = input_path / 'csv'
    else:
        output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有匹配的Excel文件
    excel_files = list(input_path.glob(pattern))
    
    if not excel_files:
        print(f"在目录 '{input_dir}' 中未找到匹配 '{pattern}' 的文件")
        return 0
    
    print(f"找到 {len(excel_files)} 个Excel文件")
    print(f"输出目录: {output_path}")
    print("=" * 50)
    
    total_converted = 0
    
    for excel_file in excel_files:
        output_file = output_path / f"{excel_file.stem}.csv"
        converted = excel_to_csv(excel_file, output_file, sheet_name, encoding, index)
        total_converted += len(converted)
        print()
    
    print("=" * 50)
    print(f"批量转换完成！共转换 {total_converted} 个文件")
    
    return total_converted


def main():
    """主函数：解析命令行参数并执行转换"""
    parser = argparse.ArgumentParser(
        description='将Excel文件转换为CSV格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python excel2csv.py input.xlsx
  
  # 指定输出文件
  python excel2csv.py input.xlsx -o output.csv
  
  # 转换指定工作表
  python excel2csv.py input.xlsx -s Sheet1
  
  # 批量转换目录中的所有Excel文件
  python excel2csv.py -d ./excel_files -o ./csv_files
  
  # 使用GBK编码（适合中文Windows系统）
  python excel2csv.py input.xlsx -e gbk
        """
    )
    
    parser.add_argument('input', nargs='?', help='输入Excel文件路径或目录路径')
    parser.add_argument('-o', '--output', help='输出CSV文件路径或目录路径')
    parser.add_argument('-s', '--sheet', help='指定要转换的工作表名称（默认转换所有工作表）')
    parser.add_argument('-d', '--directory', action='store_true', help='批量转换模式：将输入路径视为目录')
    parser.add_argument('-p', '--pattern', default='*.xlsx', help='批量转换时的文件匹配模式（默认: *.xlsx）')
    parser.add_argument('-e', '--encoding', default='utf-8-sig', help='CSV文件编码（默认: utf-8-sig）')
    parser.add_argument('-i', '--index', action='store_true', help='在CSV中包含行索引')
    
    args = parser.parse_args()
    
    # 如果没有提供输入参数，显示帮助信息
    if args.input is None:
        parser.print_help()
        return
    
    input_path = Path(args.input)
    
    # 批量转换模式
    if args.directory or input_path.is_dir():
        batch_convert(
            args.input,
            args.output,
            args.pattern,
            args.sheet,
            args.encoding,
            args.index
        )
    # 单文件转换模式
    else:
        excel_to_csv(
            args.input,
            args.output,
            args.sheet,
            args.encoding,
            args.index
        )


if __name__ == '__main__':
    main()

