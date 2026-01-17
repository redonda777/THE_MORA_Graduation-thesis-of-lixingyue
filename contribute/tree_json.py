#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import networkx as nx
import sys

# 设置输出编码为UTF-8（Windows系统）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. 读取树的JSON文件并解析
with open(r"D:\The_Mora\contribute\mora.json", "r", encoding="utf-8") as f:
    tree_json = json.load(f)

# 2. 手动构建有向树图
# 关键：直接创建有向图，避免无向图转换时产生双向边
tree = nx.DiGraph()

# 添加所有节点（带属性）
for node in tree_json["nodes"]:
    node_id = node["id"]
    node_attrs = {k: v for k, v in node.items() if k != "id"}
    tree.add_node(node_id, **node_attrs)

# 添加所有有向边
for link in tree_json["links"]:
    tree.add_edge(link["source"], link["target"])

# ========== 树的专属验证（必做，确认是合法的树） ==========
print("[OK] 是否是无环图：", nx.is_directed_acyclic_graph(tree))  # 树必须是无环的
print("[OK] 是否是连通图：", nx.is_weakly_connected(tree))       # 树必须是连通的
print("[OK] 节点数：", tree.number_of_nodes(), "边数：", tree.number_of_edges())

# ========== 树的常用高频操作（直接用NetworkX的树API） ==========
print("\n[INFO] 树的所有节点（前10个，带属性）：")
for i, (node, data) in enumerate(tree.nodes.data()):
    if i < 10:
        print(f"  {node}: {data}")
    else:
        print(f"  ... (共{tree.number_of_nodes()}个节点)")
        break

print("\n[INFO] 树的所有边（前10条）：")
for i, edge in enumerate(tree.edges):
    if i < 10:
        print(f"  {edge}")
    else:
        print(f"  ... (共{tree.number_of_edges()}条边)")
        break

root_nodes = [n for n, d in tree.in_degree() if d == 0]
print("\n[INFO] 根节点（入度为0的节点）：", root_nodes)

leaf_nodes = [n for n, d in tree.out_degree() if d == 0]
print(f"[INFO] 叶子节点（出度为0的节点）：共{len(leaf_nodes)}个（前10个：{leaf_nodes[:10]}）")

if root_nodes:
    root = root_nodes[0]
    print(f"\n[INFO] 根节点 '{root}' 的所有子节点（前10个）：", list(tree.successors(root))[:10])