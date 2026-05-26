import pandas as pd
import json
import os
from typing import Dict, List, Any

"""
将Excel文件转换为JSON文件
包括：
1. 加载Excel文件数据
2. 创建树状结构层级
3. 将树状结构保存为JSON文件
启动口令：python excel2json.py
input:Excel文件路径 example:D:\The_Mora\to_clean\0331_mora_v3.1.xlsx
output:JSON文件路径 example:mora_v2.0_0331.json
"""

def load_excel_data(file_path: str) -> pd.DataFrame:
    """
    加载Excel文件数据
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        加载后的DataFrame数据
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 数据验证：检查必要的列是否存在
        required_columns = ['seg', 'ln']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Excel文件缺少必要的列：{col}")
        
        print(f"✅ 成功加载Excel文件：{file_path}")
        print(f"📊 数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"📋 包含版本：{[col for col in df.columns if col not in required_columns]}")
        print(f"📖 包含章节：{sorted(df['seg'].unique())[:5]}...（共{len(df['seg'].unique())}个章节）")
        
        return df
    
    except Exception as e:
        print(f"❌ 加载Excel文件失败：{str(e)}")
        raise

def create_tree_hierarchy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    创建树状结构层级
    
    Args:
        df: 包含Excel数据的DataFrame
        
    Returns:
        树状结构字典（root为根节点）
    """
    # 初始化根节点
    tree: Dict[str, Any] = {
        "name": "root",
        "type": "root",
        "description": "树状图根节点",
        "children": []
    }
    
    # 获取所有书本版本（排除seg和ln列）
    versions: List[str] = [col for col in df.columns if col not in ['seg', 'ln']]
    print(f"\n🔨 开始构建树状结构，共处理 {len(versions)} 个版本")
    
    # 为每个版本创建一级节点
    for version_idx, version in enumerate(versions, 1):
        # 版本节点基础信息
        version_node: Dict[str, Any] = {
            "name": version,
            "type": "version",
            "description": f"书本版本：{version}",
            "index": version_idx,
            "children": []
        }
        
        # 获取该版本下的所有有效章节（去重并排序）
        valid_chapters = sorted(df[df[version].notna()]['seg'].unique())
        
        # 为每个章节创建二级节点
        for chapter in valid_chapters:
            # 筛选当前章节的所有数据
            chapter_data = df[(df['seg'] == chapter) & (df[version].notna())].copy()
            
            # 章节节点基础信息
            chapter_node: Dict[str, Any] = {
                "name": f"Chapter {chapter}",
                "type": "chapter",
                "description": f"章节：第{chapter + 1}章（编号：{chapter}）",
                "chapter_number": int(chapter),
                "sentence_count": len(chapter_data),
                "children": []
            }
            
            # 为每个句子创建三级节点
            for _, row in chapter_data.iterrows():
                sentence_number = int(row['ln'])
                sentence_text = str(row[version]).strip()
                
                # 句子节点信息（包含文本属性）
                sentence_node: Dict[str, Any] = {
                    "name": f"Sentence {sentence_number}",
                    "type": "sentence",
                    "description": f"章节{chapter}中的第{sentence_number + 1}句",
                    "chapter_number": int(chapter),
                    "sentence_number": sentence_number,
                    "text": sentence_text,  # 核心属性：句子文本内容
                    "version": version
                }
                
                chapter_node["children"].append(sentence_node)
            
            # 将章节节点添加到版本节点
            if chapter_node["children"]:  # 只添加有句子的章节
                version_node["children"].append(chapter_node)
        
        # 将版本节点添加到根节点（只添加有章节的版本）
        if version_node["children"]:
            tree["children"].append(version_node)
            print(f"✅ 完成版本 {version} 的构建，包含 {len(version_node['children'])} 个章节")
    
    print(f"\n🏆 树状结构构建完成！")
    print(f"📊 最终结构：1个根节点 → {len(tree['children'])} 个版本节点 → 共 {sum(len(v['children']) for v in tree['children'])} 个章节节点")
    
    return tree

def save_tree_to_json(tree: Dict[str, Any], output_path: str = "tree_structure.json") -> None:
    """
    将树状结构保存为JSON文件
    
    Args:
        tree: 树状结构字典
        output_path: 输出JSON文件路径
    """
    try:
        # 保存JSON文件（使用UTF-8编码，便于中文显示）
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tree, f, ensure_ascii=False, indent=2, sort_keys=False)
        
        # 验证文件保存结果
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # 转换为KB
            print(f"\n💾 JSON文件已保存：{output_path}")
            print(f"📦 文件大小：{file_size:.2f} KB")
            
            # 输出结构统计信息
            version_count = len(tree['children'])
            chapter_count = sum(len(v['children']) for v in tree['children'])
            sentence_count = sum(sum(len(c['children']) for c in v['children']) for v in tree['children'])
            
            print(f"\n📈 结构统计：")
            print(f"   - 根节点：1个")
            print(f"   - 版本节点：{version_count}个")
            print(f"   - 章节节点：{chapter_count}个")
            print(f"   - 句子节点：{sentence_count}个")
            
        else:
            raise Exception("文件保存后未找到")
            
    except Exception as e:
        print(f"❌ 保存JSON文件失败：{str(e)}")
        raise

def main():
    """
    主函数：Excel转树状结构JSON的完整流程
    """
    print("=" * 60)
    print("          Excel转树状结构JSON工具          ")
    print("=" * 60)
    
    # 1. 配置参数（可根据实际情况修改）
    EXCEL_FILE_PATH = r"D:\The_Mora\to_clean\0406_mora_v4.1.xlsx"  # 输入Excel文件路径
    OUTPUT_JSON_PATH = r"D:\The_Mora\contribute\mora_v4.1_0406.json"  # 输出JSON文件路径
    
    try:
        # 2. 加载Excel数据
        df = load_excel_data(EXCEL_FILE_PATH)
        
        # 3. 构建树状结构
        tree_structure = create_tree_hierarchy(df)
        
        # 4. 保存为JSON文件
        save_tree_to_json(tree_structure, OUTPUT_JSON_PATH)
        
        print("\n" + "=" * 60)
        print("              处理完成！              ")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        print("请检查：1. Excel文件路径是否正确 2. 文件格式是否正常 3. 必要列是否存在")

if __name__ == "__main__":
    main()