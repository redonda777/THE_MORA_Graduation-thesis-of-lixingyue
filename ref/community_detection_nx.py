import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from networkx.algorithms.community.centrality import girvan_newman
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class GirvanNewmanNX:
    def __init__(self, G: nx.Graph):
        """
        基于networkx内置Girvan-Newman算法的层次化聚类
        :param G: 输入图（需包含节点向量属性'vector'）
        """
        self.original_graph = G.copy()
        self.node_vectors = nx.get_node_attributes(G, 'vector')  # 节点向量
        self.hierarchy = []  # 存储层次信息：[(层次, 社区列表), ...]
        self.edge_removal_history = []  # 记录每次移除的边

        # 验证节点向量
        for node in G.nodes:
            if node not in self.node_vectors:
                raise ValueError(f"节点 {node} 缺少向量属性 'vector'")

    def build_type_graph_edge_by_vector_cosine(self, node_type: str = None, threshold: float = 0.3):
        """
        基于节点向量的余弦距离构建特点类型节点并设置边权重（距离越小，权重越大），并可根据节点类型筛选节点

        参数:
            node_type: 要筛选的节点类型，默认为None（使用所有节点）
            threshold: 当node_type不为None时的相似度阈值，超过该值的节点对将被连接（范围0~1）

        返回:
            带权重的图（包含所有节点或筛选后的节点）
        """
        # 如果指定了节点类型，则基于该类型筛选节点并构建新图
        if node_type is not None:
            # 提取所有指定类型的节点
            target_nodes = []
            target_node_vectors = {}

            for node in self.original_graph.nodes:
                # 检查节点类型是否为指定类型
                if self.original_graph.nodes[node].get('type') == node_type:
                    # 检查是否包含向量属性
                    if 'vector' not in self.original_graph.nodes[node]:
                        raise ValueError(f"{node_type}节点 {node} 缺少向量属性 'vector'")
                    target_nodes.append(node)
                    target_node_vectors[node] = self.original_graph.nodes[node]['vector']

            if len(target_nodes) < 2:
                raise ValueError(f"{node_type}节点数量不足（至少需要2个节点才能构建相似度图）")

            # 计算所有指定类型节点对的余弦相似度
            # 转换向量为numpy数组（按节点顺序排列）
            nodes_order = target_nodes
            vectors = np.array([target_node_vectors[node] for node in nodes_order])

            # 计算余弦相似度矩阵（形状：n_nodes × n_nodes）
            similarity_matrix = cosine_similarity(vectors)

            # 构建新图
            G = nx.Graph()

            # 添加指定类型的节点（保留原始节点的所有属性）
            for node in target_nodes:
                G.add_node(node, **self.original_graph.nodes[node])  # 复制所有属性

            # 添加相似度边（仅保留超过阈值的连接）
            n = len(nodes_order)
            for i in range(n):
                for j in range(i + 1, n):  # 避免重复计算（无向图）
                    node_i = nodes_order[i]
                    node_j = nodes_order[j]
                    similarity = similarity_matrix[i][j]

                    if similarity > threshold:  # 这里可以考虑变成保留相似度最高的比率
                        # 添加边，权重为相似度值
                        # todo: 这里weight值的大小影响结果，可自行调整
                        G.add_edge(
                            node_i,
                            node_j,
                            similarity=round(similarity, 4),  # 保留4位小数
                            relation="similar",
                            weight=similarity  # 添加weight属性用于Girvan-Newman算法
                        )
        else:
            # 原始逻辑：使用所有节点，设置边权重
            G = self.original_graph.copy()
            nodes = list(G.nodes)
            vectors = np.array([self.node_vectors[node] for node in nodes])
            dist_matrix = cosine_distances(vectors)  # 余弦距离矩阵（0~2）
            node_idx = {node: i for i, node in enumerate(nodes)}

            # 为所有可能的节点对添加权重（包括非邻接节点）
            for u in nodes:
                for v in nodes:
                    if u != v:
                        i, j = node_idx[u], node_idx[v]
                        distance = dist_matrix[i][j]
                        weight = 1.0 / (1.0 + distance)  # 权重与距离成反比
                        if G.has_edge(u, v):
                            G[u][v]['weight'] = weight
                        else:
                            G.add_edge(u, v, weight=weight)  # 非邻接节点也添加带权重的边

        return G

    def run(self, max_levels: int = 10, node_type: str = None, threshold: float = 0.3) -> List[List[Set]]:
        """
        运行Girvan-Newman算法（调用networkx内置函数）
        :param max_levels: 最大层次数
        :param node_type: 要筛选的节点类型，默认为None（使用所有节点）
        :param threshold: 当node_type不为None时的相似度阈值
        :return: 层次化社区列表
        """
        # 构建带权重的图（基于节点向量余弦距离）
        weighted_graph = self.build_type_graph_edge_by_vector_cosine(node_type, threshold)

        # 调用networkx的girvan_newman迭代器
        # 注意：内置函数默认用边介数，这里通过权重影响最短路径计算
        gn_generator = girvan_newman(
            weighted_graph,
            weight='weight'  # 使用边权重计算介数（权重高的边更可能在最短路径中）
        )

        # 记录初始状态（所有节点为一个社区）
        initial_community = [set(weighted_graph.nodes)]
        self.hierarchy.append((0, initial_community))

        # 迭代获取层次化社区划分
        level = 1
        prev_communities = initial_community
        for communities in gn_generator:
            if level > max_levels:
                break
            # 转换为集合列表（便于存储和比较）
            community_sets = [set(comm) for comm in communities]
            self.hierarchy.append((level, community_sets))

            # 记录当前层次移除的边（与上一层次对比，找消失的边）
            prev_edges = set(weighted_graph.edges)
            # 临时图：移除边后，连通分量为当前社区
            temp_graph = weighted_graph.copy()
            # 移除导致社区分裂的边（通过连通分量差异推断）
            # 简化处理：计算上一层次与当前层次的边差异
            # 更精确的方式是跟踪内置函数移除的边，但networkx不直接提供，这里用近似
            if level == 1:
                removed_edges = []
            else:
                # 找到上一层次存在而当前层次不存在的边（近似）
                # 注意：这是简化方式，实际移除的是介数最高的边
                current_edges = set()
                for comm in community_sets:
                    current_edges.update(nx.complete_graph(comm).edges)  # 社区内部边
                removed_edges = list(prev_edges - current_edges)

            self.edge_removal_history.append({
                'level': level,
                'removed_edges': removed_edges
            })

            prev_communities = community_sets
            level += 1

        return [communities for _, communities in self.hierarchy]

    def get_hierarchy_info(self) -> List[Dict]:
        """返回层次化聚类的详细信息"""
        hierarchy_info = []
        
        for level, communities in self.hierarchy:
            # 为每个社区创建包含节点ID和text属性的字典列表
            enriched_communities = []
            for comm in communities:
                # 创建社区内节点的详细信息列表
                community_nodes = []
                for node_id in comm:
                    # 尝试获取节点的text属性，如果不存在则设为None
                    node_text = self.original_graph.nodes[node_id].get('text', None)
                    community_nodes.append({
                        'id': node_id,
                        'text': node_text
                    })
                enriched_communities.append(community_nodes)
            
            # 获取当前层次移除的边
            removed_edges = next(
                (info['removed_edges'] for info in self.edge_removal_history if info['level'] == level),
                None
            )
            
            hierarchy_info.append({
                'level': level,
                'num_communities': len(communities),
                'communities': enriched_communities,
                'removed_edges': removed_edges
            })
        
        return hierarchy_info

    def visualize(self, level: int = None, figsize: Tuple[int, int] = (10, 8)):
        """可视化指定层次的聚类结果"""
        if not self.hierarchy:
            raise RuntimeError("请先运行run()方法")

        # 选择层次
        if level is None:
            level = len(self.hierarchy) - 1
        if level < 0 or level >= len(self.hierarchy):
            raise ValueError(f"有效层次范围: 0~{len(self.hierarchy) - 1}")

        _, communities = self.hierarchy[level]
        print(f"可视化层次 {level}：{len(communities)} 个社区")

        # 节点颜色（按社区分配）
        color_map = plt.cm.get_cmap('tab10', len(communities))
        node_colors = []
        for node in self.original_graph.nodes:
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(color_map(i))
                    break

        # 绘图
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.original_graph, seed=42)
        # 绘制节点
        nx.draw_networkx_nodes(
            self.original_graph, pos,
            node_color=node_colors, node_size=300, alpha=0.8
        )
        # 绘制原始边（灰色）
        nx.draw_networkx_edges(
            self.original_graph, pos,
            edgelist=self.original_graph.edges,
            width=0.5, alpha=0.3, color='gray'
        )
        # 绘制已移除的边（红色虚线）
        removed_edges = []
        for info in self.edge_removal_history:
            if info['level'] <= level:
                removed_edges.extend(info['removed_edges'])
        nx.draw_networkx_edges(
            self.original_graph, pos,
            edgelist=removed_edges,
            width=1, alpha=0.7, color='red', style='dashed'
        )
        # 节点标签 - 包含ID和text属性
        node_labels = {}
        for node in self.original_graph.nodes:
            node_text = self.original_graph.nodes[node].get('text', '')
            if node_text:
                node_labels[node] = f"{node}\n{node_text[:20]}..." if len(node_text) > 20 else f"{node}\n{node_text}"
            else:
                node_labels[node] = node
        
        nx.draw_networkx_labels(self.original_graph, pos, labels=node_labels, font_size=8)

        plt.title(f"Girvan-Newman 层次聚类（层次 {level}）")
        plt.axis('off')
        plt.show()



# ------------------------------
# 示例使用
# ------------------------------
# todo: @chenxi 测试正确性
if __name__ == "__main__":
    # 1. 构建带节点向量的示例图
    def generate_vectors(n: int, dim: int = 8) -> Dict:
        """生成两组差异明显的节点向量（模拟两个社区）"""
        np.random.seed(42)
        vectors = {}
        # 第一组：均值为0的正态分布
        for i in range(n // 2):
            vectors[f"node{i}"] = np.random.randn(dim) * 0.5
        # 第二组：均值为2的正态分布（与第一组差异大）
        for i in range(n // 2, n):
            vectors[f"node{i}"] = np.random.randn(dim) * 0.5 + 2.0
        return vectors


    # 创建12个节点的图
    n_nodes = 12
    G = nx.Graph()
    G.add_nodes_from([f"node{i}" for i in range(n_nodes)])
    # 设置节点向量
    node_vectors = generate_vectors(n_nodes)
    nx.set_node_attributes(G, node_vectors, 'vector')
    # 设置节点text属性
    node_texts = {f"node{i}": f"示例文本 {i+1}" for i in range(n_nodes)}
    nx.set_node_attributes(G, node_texts, 'text')
    # 添加一些初始边（增强社区结构）
    G.add_edges_from([(f"node{i}", f"node{j}") for i in range(6) for j in range(i + 1, 6) if i != j])
    G.add_edges_from([(f"node{i}", f"node{j}") for i in range(6, 12) for j in range(i + 1, 12) if i != j])
    G.add_edges_from([("node2", "node7"), ("node3", "node8")])  # 跨社区边

    # 2. 运行Girvan-Newman算法（调用networkx内置函数）
    gn = GirvanNewmanNX(G)
    hierarchies = gn.run(max_levels=5)  # 最多5层

    # 3. 打印层次信息
    print("层次化聚类信息：")
    for info in gn.get_hierarchy_info():
        print(f"\n层次 {info['level']}:")
        print(f"  社区数量: {info['num_communities']}")
        print(f"  移除的边: {info['removed_edges']}")
        for i, comm in enumerate(info['communities']):
            print(f"  社区 {i}:")
            for node_info in comm:
                text_display = node_info['text'] or "无文本"
                print(f"    - {node_info['id']}: {text_display}")

    # 4. 可视化
    gn.visualize(level=0)  # 初始状态（1个社区）
    gn.visualize(level=2)  # 第2层（预计分裂为2个主要社区）
    gn.visualize(level=-1)  # 最后一层