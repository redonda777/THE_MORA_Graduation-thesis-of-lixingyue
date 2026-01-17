#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 networkx 构建树状图的脚本
处理 mora_v1.2_1228.json 嵌套树状结构
"""
import json
import networkx as nx
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置输出编码为UTF-8（Windows系统）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def build_graph_from_tree(data, parent_id=None, graph=None, node_counter=None):
    """
    递归构建 networkx 图
    
    Args:
        data: 当前节点的数据字典
        parent_id: 父节点的ID
        graph: networkx 有向图对象
        node_counter: 节点计数器字典，用于生成唯一ID
    
    Returns:
        graph: 构建好的图对象
    """
    if graph is None:
        graph = nx.DiGraph()
        node_counter = defaultdict(int)
    
    # 生成当前节点的唯一ID
    node_name = data.get('name', 'unnamed')
    node_type = data.get('type', 'unknown')
    
    # 根据节点类型生成唯一ID
    if node_type == 'root':
        node_id = 'root'
    elif node_type == 'version':
        node_id = f"version_{node_name}"
    elif node_type == 'chapter':
        version = parent_id.split('_')[-1] if parent_id else 'unknown'
        chapter_num = data.get('chapter_number', node_counter['chapter'])
        node_id = f"chapter_{version}_{chapter_num}"
        node_counter['chapter'] += 1
    elif node_type == 'sentence':
        version = parent_id.split('_')[1] if parent_id and '_' in parent_id else 'unknown'
        chapter_num = data.get('chapter_number', 0)
        sentence_num = data.get('sentence_number', node_counter['sentence'])
        node_id = f"sentence_{version}_{chapter_num}_{sentence_num}"
        node_counter['sentence'] += 1
    else:
        node_id = f"{node_type}_{node_counter[node_type]}"
        node_counter[node_type] += 1
    
    # 准备节点属性（排除children字段）
    node_attrs = {k: v for k, v in data.items() if k != 'children'}
    node_attrs['node_id'] = node_id
    
    # 添加节点
    graph.add_node(node_id, **node_attrs)
    
    # 如果有父节点，添加边
    if parent_id is not None:
        graph.add_edge(parent_id, node_id)
    
    # 递归处理子节点
    if 'children' in data and isinstance(data['children'], list):
        for child in data['children']:
            build_graph_from_tree(child, node_id, graph, node_counter)
    
    return graph

def analyze_tree(graph):
    """分析树的结构和属性"""
    print("=" * 60)
    print("树状图分析报告")
    print("=" * 60)
    
    # 基本统计
    print(f"\n[基本信息]")
    print(f"  节点总数: {graph.number_of_nodes()}")
    print(f"  边总数: {graph.number_of_edges()}")
    print(f"  是否是有向无环图(DAG): {nx.is_directed_acyclic_graph(graph)}")
    print(f"  是否是弱连通图: {nx.is_weakly_connected(graph)}")
    
    # 根节点
    root_nodes = [n for n, d in graph.in_degree() if d == 0]
    print(f"\n[根节点]")
    print(f"  根节点数量: {len(root_nodes)}")
    if root_nodes:
        root = root_nodes[0]
        print(f"  根节点ID: {root}")
        root_data = graph.nodes[root]
        print(f"  根节点属性: {dict(root_data)}")
        print(f"  根节点的子节点数: {graph.out_degree(root)}")
    
    # 叶子节点
    leaf_nodes = [n for n, d in graph.out_degree() if d == 0]
    print(f"\n[叶子节点]")
    print(f"  叶子节点数量: {len(leaf_nodes)}")
    if leaf_nodes:
        print(f"  前10个叶子节点: {leaf_nodes[:10]}")
        # 显示一个叶子节点的示例
        if leaf_nodes:
            leaf = leaf_nodes[0]
            leaf_data = graph.nodes[leaf]
            print(f"  示例叶子节点 ({leaf}): {dict(leaf_data)}")
    
    # 按类型统计节点
    print(f"\n[节点类型统计]")
    type_count = defaultdict(int)
    for node, data in graph.nodes.data():
        node_type = data.get('type', 'unknown')
        type_count[node_type] += 1
    
    for node_type, count in sorted(type_count.items()):
        print(f"  {node_type}: {count} 个")
    
    # 版本节点统计
    version_nodes = [n for n, d in graph.nodes.data() if d.get('type') == 'version']
    print(f"\n[版本节点]")
    print(f"  版本数量: {len(version_nodes)}")
    if version_nodes:
        print(f"  版本列表（前10个）:")
        for i, version_node in enumerate(version_nodes[:10]):
            version_data = graph.nodes[version_node]
            version_name = version_data.get('name', 'unknown')
            children_count = graph.out_degree(version_node)
            print(f"    {i+1}. {version_name} (ID: {version_node}, 子节点数: {children_count})")
    
    # 章节节点统计（按版本分组）
    print(f"\n[章节节点统计（按版本）]")
    version_chapters = defaultdict(int)
    for node, data in graph.nodes.data():
        if data.get('type') == 'chapter':
            # 从节点ID中提取版本信息
            node_id = data.get('node_id', node)
            if '_' in node_id:
                parts = node_id.split('_')
                if len(parts) >= 3:
                    version = parts[1]
                    version_chapters[version] += 1
    
    for version, count in sorted(version_chapters.items()):
        print(f"  {version}: {count} 个章节")
    
    # 句子节点统计
    sentence_nodes = [n for n, d in graph.nodes.data() if d.get('type') == 'sentence']
    print(f"\n[句子节点]")
    print(f"  句子总数: {len(sentence_nodes)}")
    
    # 树的深度
    if root_nodes:
        root = root_nodes[0]
        try:
            # 计算从根节点到所有节点的最长路径
            longest_path = 0
            for node in graph.nodes():
                if node != root:
                    try:
                        path_length = nx.shortest_path_length(graph, root, node)
                        longest_path = max(longest_path, path_length)
                    except nx.NetworkXNoPath:
                        pass
            print(f"\n[树的结构]")
            print(f"  树的深度（最长路径长度）: {longest_path}")
        except Exception as e:
            print(f"  无法计算树深度: {e}")
    
    # 显示一些示例边
    print(f"\n[边的示例（前10条）]")
    for i, (source, target) in enumerate(graph.edges()):
        if i < 10:
            source_type = graph.nodes[source].get('type', 'unknown')
            target_type = graph.nodes[target].get('type', 'unknown')
            print(f"  {source} ({source_type}) -> {target} ({target_type})")
        else:
            print(f"  ... (共 {graph.number_of_edges()} 条边)")
            break
    
    print("\n" + "=" * 60)

def export_graph(graph, output_dir=None):
    """导出图到文件"""
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    output_dir = os.path.join(output_dir, 'graph_exports')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[导出图文件]")
    
    # 导出为GraphML格式（支持节点属性）
    graphml_file = os.path.join(output_dir, 'mora_tree.graphml')
    try:
        nx.write_graphml(graph, graphml_file, encoding='utf-8')
        print(f"  ✓ GraphML格式: {graphml_file}")
    except Exception as e:
        print(f"  ✗ GraphML导出失败: {e}")
    
    # 导出为GML格式
    gml_file = os.path.join(output_dir, 'mora_tree.gml')
    try:
        nx.write_gml(graph, gml_file)
        print(f"  ✓ GML格式: {gml_file}")
    except Exception as e:
        print(f"  ✗ GML导出失败: {e}")
    
    # 导出为JSON格式（节点-链接格式）
    json_file = os.path.join(output_dir, 'mora_tree_node_link.json')
    try:
        node_link_data = nx.node_link_data(graph)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(node_link_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ JSON格式（节点-链接）: {json_file}")
    except Exception as e:
        print(f"  ✗ JSON导出失败: {e}")

def query_node(graph, node_id):
    """查询特定节点的信息"""
    if node_id not in graph:
        print(f"错误: 节点 '{node_id}' 不存在")
        return None
    
    node_data = graph.nodes[node_id]
    print(f"\n[节点查询: {node_id}]")
    print(f"  节点属性: {dict(node_data)}")
    
    # 父节点
    predecessors = list(graph.predecessors(node_id))
    if predecessors:
        print(f"  父节点: {predecessors}")
    
    # 子节点
    successors = list(graph.successors(node_id))
    if successors:
        print(f"  子节点数量: {len(successors)}")
        print(f"  前10个子节点: {successors[:10]}")
    
    return node_data

def find_nodes_by_type(graph, node_type):
    """根据类型查找所有节点"""
    nodes = [n for n, d in graph.nodes.data() if d.get('type') == node_type]
    return nodes

def find_nodes_by_version(graph, version_name):
    """查找特定版本的所有节点"""
    nodes = []
    for node, data in graph.nodes.data():
        if data.get('version') == version_name or \
           (data.get('type') == 'version' and data.get('name') == version_name) or \
           (node.startswith(f'version_{version_name}') or 
            node.startswith(f'chapter_{version_name}') or 
            node.startswith(f'sentence_{version_name}')):
            nodes.append(node)
    return nodes

def get_subgraph_by_depth(graph, root_node, max_depth=None, include_types=None):
    """
    获取指定深度的子图
    
    Args:
        graph: 原始图
        root_node: 根节点ID
        max_depth: 最大深度（None表示不限制）
        include_types: 包含的节点类型列表（None表示包含所有类型）
    
    Returns:
        子图
    """
    if root_node not in graph:
        return None
    
    subgraph_nodes = {root_node}
    
    # BFS遍历，收集节点
    queue = [(root_node, 0)]  # (node, depth)
    
    while queue:
        current_node, depth = queue.pop(0)
        
        if max_depth is not None and depth >= max_depth:
            continue
        
        for successor in graph.successors(current_node):
            node_type = graph.nodes[successor].get('type', 'unknown')
            if include_types is None or node_type in include_types:
                subgraph_nodes.add(successor)
                queue.append((successor, depth + 1))
    
    return graph.subgraph(subgraph_nodes)

def visualize_tree_structure(graph, output_dir=None, max_nodes=500):
    """
    可视化树结构（分层显示，不包含句子节点）
    
    Args:
        graph: networkx图对象
        output_dir: 输出目录
        max_nodes: 最大节点数，超过则只显示结构概览
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'graph_exports')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[生成可视化图表]")
    
    # 1. 只显示 root -> version -> chapter 的结构
    root_nodes = [n for n, d in graph.in_degree() if d == 0]
    if not root_nodes:
        print("  错误: 找不到根节点")
        return
    
    root = root_nodes[0]
    subgraph = get_subgraph_by_depth(graph, root, max_depth=2, include_types=['root', 'version', 'chapter'])
    
    if subgraph.number_of_nodes() > max_nodes:
        print(f"  警告: 节点数过多 ({subgraph.number_of_nodes()})，只显示版本层级")
        subgraph = get_subgraph_by_depth(graph, root, max_depth=1, include_types=['root', 'version'])
    
    # 使用层次布局
    try:
        pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
    except:
        # 如果没有graphviz，使用spring布局
        try:
            pos = nx.spring_layout(subgraph, k=2, iterations=50)
        except:
            pos = nx.planar_layout(subgraph)
    
    # 创建图形
    plt.figure(figsize=(16, 12))
    
    # 按类型设置颜色和大小
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get('type', 'unknown')
        node_name = subgraph.nodes[node].get('name', node)
        
        # 设置颜色
        if node_type == 'root':
            node_colors.append('#FF6B6B')  # 红色
            node_sizes.append(1000)
        elif node_type == 'version':
            node_colors.append('#4ECDC4')  # 青色
            node_sizes.append(500)
        elif node_type == 'chapter':
            node_colors.append('#95E1D3')  # 浅青色
            node_sizes.append(200)
        else:
            node_colors.append('#F38181')  # 粉色
            node_sizes.append(100)
        
        # 设置标签（只显示名称，避免过长）
        if len(node_name) > 15:
            node_labels[node] = node_name[:12] + '...'
        else:
            node_labels[node] = node_name
    
    # 绘制边
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, arrows=True, arrowsize=15, 
                          edge_color='gray', width=1.5, connectionstyle='arc3,rad=0.1')
    
    # 绘制节点
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8)
    
    # 绘制标签
    nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=8, font_weight='bold')
    
    plt.title(f'树状图结构可视化\n(节点数: {subgraph.number_of_nodes()}, 边数: {subgraph.number_of_edges()})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'tree_structure.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 结构可视化: {output_file}")
    plt.close()

def visualize_version_subtree(graph, version_name, output_dir=None, max_chapters=10):
    """
    可视化特定版本的子树（包含章节和部分句子）
    
    Args:
        graph: networkx图对象
        version_name: 版本名称（如 'hj'）
        output_dir: 输出目录
        max_chapters: 最多显示的章节数
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'graph_exports')
    os.makedirs(output_dir, exist_ok=True)
    
    version_node = f"version_{version_name}"
    if version_node not in graph:
        print(f"  错误: 找不到版本节点 '{version_node}'")
        return
    
    # 获取版本的所有子节点（章节）
    chapters = list(graph.successors(version_node))
    
    if len(chapters) > max_chapters:
        chapters = chapters[:max_chapters]
        print(f"  提示: 只显示前 {max_chapters} 个章节")
    
    # 构建子图：版本 -> 章节 -> 前几个句子
    subgraph_nodes = {version_node}
    for chapter in chapters:
        subgraph_nodes.add(chapter)
        # 每个章节只取前3个句子
        sentences = list(graph.successors(chapter))[:3]
        subgraph_nodes.update(sentences)
    
    subgraph = graph.subgraph(subgraph_nodes)
    
    # 使用层次布局
    try:
        pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
    except:
        try:
            pos = nx.spring_layout(subgraph, k=1.5, iterations=50)
        except:
            pos = nx.planar_layout(subgraph)
    
    # 创建图形
    plt.figure(figsize=(20, 14))
    
    # 按类型设置颜色和大小
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node in subgraph.nodes():
        node_type = subgraph.nodes[node].get('type', 'unknown')
        node_name = subgraph.nodes[node].get('name', node)
        
        if node_type == 'version':
            node_colors.append('#FF6B6B')
            node_sizes.append(1500)
            node_labels[node] = f"版本: {node_name}"
        elif node_type == 'chapter':
            chapter_num = subgraph.nodes[node].get('chapter_number', '?')
            node_colors.append('#4ECDC4')
            node_sizes.append(600)
            node_labels[node] = f"第{chapter_num}章"
        elif node_type == 'sentence':
            text = subgraph.nodes[node].get('text', '')
            node_colors.append('#95E1D3')
            node_sizes.append(300)
            if len(text) > 10:
                node_labels[node] = text[:8] + '...'
            else:
                node_labels[node] = text
        else:
            node_colors.append('#F38181')
            node_sizes.append(200)
            node_labels[node] = node_name[:10]
    
    # 绘制边
    nx.draw_networkx_edges(subgraph, pos, alpha=0.4, arrows=True, arrowsize=12,
                          edge_color='gray', width=1.2, connectionstyle='arc3,rad=0.1')
    
    # 绘制节点
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                          node_size=node_sizes, alpha=0.8)
    
    # 绘制标签
    nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=7, font_weight='bold')
    
    plt.title(f'版本 "{version_name}" 的子树结构\n(节点数: {subgraph.number_of_nodes()})',
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'version_{version_name}_subtree.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 版本子树可视化: {output_file}")
    plt.close()

def visualize_statistics(graph, output_dir=None):
    """
    生成统计图表
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'graph_exports')
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计各版本的章节数
    version_chapters = defaultdict(int)
    for node, data in graph.nodes.data():
        if data.get('type') == 'chapter':
            node_id = data.get('node_id', node)
            if '_' in node_id:
                parts = node_id.split('_')
                if len(parts) >= 3:
                    version = parts[1]
                    version_chapters[version] += 1
    
    # 绘制柱状图
    if version_chapters:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 版本章节数统计
        versions = sorted(version_chapters.keys())
        chapter_counts = [version_chapters[v] for v in versions]
        
        bars = ax1.bar(versions, chapter_counts, color='#4ECDC4', alpha=0.8)
        ax1.set_xlabel('版本', fontsize=12, fontweight='bold')
        ax1.set_ylabel('章节数', fontsize=12, fontweight='bold')
        ax1.set_title('各版本的章节数统计', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 节点类型统计
        type_count = defaultdict(int)
        for node, data in graph.nodes.data():
            node_type = data.get('type', 'unknown')
            type_count[node_type] += 1
        
        types = list(type_count.keys())
        counts = [type_count[t] for t in types]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
        
        ax2.pie(counts, labels=types, autopct='%1.1f%%', startangle=90,
               colors=colors[:len(types)], textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('节点类型分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'graph_statistics.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ 统计图表: {output_file}")
        plt.close()

def main():
    """主函数"""
    # JSON文件路径
    json_file = os.path.join(os.path.dirname(__file__), 'mora_v1.2_1228.json')
    
    if not os.path.exists(json_file):
        print(f"错误: 找不到文件 {json_file}")
        return
    
    print(f"正在读取JSON文件: {json_file}")
    
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
    except Exception as e:
        print(f"错误: 构建图失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 分析树
    analyze_tree(graph)
    
    # 导出图到文件
    export_graph(graph)
    
    # 生成可视化
    visualize_tree_structure(graph)
    visualize_statistics(graph)
    # 可视化一个版本的子树作为示例
    visualize_version_subtree(graph, 'hj', max_chapters=5)
    
    # 示例查询
    print("\n[示例查询]")
    query_node(graph, 'root')
    query_node(graph, 'version_hj')
    
    print("\n✓ 处理完成！")
    print("\n提示: 可以使用以下函数进行更多操作：")
    print("  - query_node(graph, 'node_id'): 查询特定节点")
    print("  - find_nodes_by_type(graph, 'type'): 按类型查找节点")
    print("  - find_nodes_by_version(graph, 'version_name'): 按版本查找节点")
    print("  - export_graph(graph): 导出图到文件")
    print("  - visualize_tree_structure(graph): 可视化树结构")
    print("  - visualize_version_subtree(graph, 'version_name'): 可视化版本子树")
    print("  - visualize_statistics(graph): 生成统计图表")
    
    return graph

if __name__ == '__main__':
    graph = main()

