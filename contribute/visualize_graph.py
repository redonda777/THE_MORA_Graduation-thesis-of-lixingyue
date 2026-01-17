#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的图可视化脚本
可以单独运行此脚本来生成可视化图表
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from build_tree_graph import (
    build_graph_from_tree,
    visualize_tree_structure,
    visualize_version_subtree,
    visualize_statistics,
    get_subgraph_by_depth
)
import json

def main():
    """主函数 - 生成所有可视化图表"""
    # JSON文件路径
    json_file = os.path.join(os.path.dirname(__file__), 'mora_v1.2_1228.json')
    
    if not os.path.exists(json_file):
        print(f"错误: 找不到文件 {json_file}")
        return
    
    print("=" * 60)
    print("图可视化工具")
    print("=" * 60)
    print(f"\n正在读取JSON文件: {json_file}")
    
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        print("✓ 成功读取JSON文件")
    except Exception as e:
        print(f"错误: 读取JSON文件失败 - {e}")
        return
    
    # 构建图
    print("\n正在构建networkx图...")
    try:
        graph = build_graph_from_tree(tree_data)
        print("✓ 成功构建图")
        print(f"  节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    except Exception as e:
        print(f"错误: 构建图失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 生成所有可视化
    print("\n" + "=" * 60)
    print("开始生成可视化图表...")
    print("=" * 60)
    
    # 1. 树结构可视化（只显示版本层级）
    print("\n[1/3] 生成树结构可视化...")
    visualize_tree_structure(graph, max_nodes=500)
    
    # 2. 统计图表
    print("\n[2/3] 生成统计图表...")
    visualize_statistics(graph)
    
    # 3. 为每个版本生成子树可视化（可选，只生成前3个版本）
    print("\n[3/3] 生成版本子树可视化...")
    root_nodes = [n for n, d in graph.in_degree() if d == 0]
    if root_nodes:
        root = root_nodes[0]
        version_nodes = [n for n in graph.successors(root)][:3]  # 只生成前3个版本
        
        for version_node in version_nodes:
            version_data = graph.nodes[version_node]
            version_name = version_data.get('name', 'unknown')
            print(f"  正在生成版本 '{version_name}' 的子树...")
            visualize_version_subtree(graph, version_name, max_chapters=5)
    
    print("\n" + "=" * 60)
    print("✓ 所有可视化图表已生成完成！")
    print(f"✓ 文件保存在: {os.path.join(os.path.dirname(__file__), 'graph_exports')}")
    print("=" * 60)
    
    return graph

if __name__ == '__main__':
    graph = main()

