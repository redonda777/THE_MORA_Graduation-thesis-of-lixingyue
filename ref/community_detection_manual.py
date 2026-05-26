import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Set


class GirvanNewmanWithVectors:
    def __init__(self, G: nx.Graph):
        """
        初始化Girvan-Newman聚类器
        :param G: 输入图（需包含节点向量属性，键为'vector'）
        """
        self.original_graph = G.copy()  # 原始图
        self.current_graph = G.copy()  # 动态更新的图（移除边后）
        self.node_vectors = nx.get_node_attributes(G, 'vector')  # 节点向量字典
        self.hierarchy = []  # 存储层次聚类信息：[(划分步骤, 社区列表), ...]
        self.edge_betweenness_history = []  # 记录每次移除的边及其介数

        # 验证节点向量是否存在
        for node in G.nodes:
            if node not in self.node_vectors:
                raise ValueError(f"节点 {node} 缺少向量属性 'vector'")

    def _compute_edge_weights(self) -> Dict[Tuple, float]:
        """
        基于节点向量的余弦距离计算边权重（距离越小，权重越大）
        对非邻接节点也计算潜在连接强度（用于介数计算）
        """
        edge_weights = {}
        nodes = list(self.current_graph.nodes)

        # 计算所有节点对的余弦距离（距离范围：0~2）
        vectors = np.array([self.node_vectors[node] for node in nodes])
        dist_matrix = cosine_distances(vectors)  # 余弦距离矩阵

        # 构建节点索引映射
        node_idx = {node: i for i, node in enumerate(nodes)}

        # 计算权重：权重 = 1 / (1 + 距离)，使距离越小权重越大
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j:  # 避免重复计算
                    dist = dist_matrix[i][j]
                    weight = 1.0 / (1.0 + dist)  # 权重范围：0.5~1（距离0~2时）
                    edge_weights[(u, v)] = weight
                    edge_weights[(v, u)] = weight  # 无向图对称

        return edge_weights

    def _compute_edge_betweenness(self) -> Dict[Tuple, float]:
        """
        基于节点向量权重计算边介数（考虑所有可能路径，而非仅现有边）
        边介数：经过该边的最短路径数量占总最短路径数量的比例
        """
        edge_weights = self._compute_edge_weights()
        betweenness = defaultdict(float)
        nodes = list(self.current_graph.nodes)

        # 对每个节点对计算最短路径，累计边介数
        for s in nodes:
            for t in nodes:
                if s == t:
                    continue

                # Dijkstra算法计算带权重的最短路径（权重越大，路径成本越低）
                try:
                    # 注意：这里用权重的倒数作为成本（权重高→成本低）
                    path = nx.dijkstra_path(
                        self.current_graph,
                        source=s,
                        target=t,
                        weight=lambda u, v, d: 1.0 / edge_weights[(u, v)]
                    )
                except nx.NetworkXNoPath:
                    continue  # 若两节点无路径，跳过

                # 累加路径中所有边的介数
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge = tuple(sorted((u, v)))  # 无向边排序
                    betweenness[edge] += 1.0

        # 归一化：除以总节点对数量
        total_pairs = len(nodes) * (len(nodes) - 1)
        if total_pairs > 0:
            for edge in betweenness:
                betweenness[edge] /= total_pairs

        return betweenness

    def run(self, max_levels: int = 10) -> List[List[Set]]:
        """
        运行Girvan-Newman算法进行层次化聚类
        :param max_levels: 最大聚类层次数（避免过度计算）
        :return: 层次化社区列表，每个元素是一个社区划分
        """
        self.hierarchy = []
        self.edge_betweenness_history = []

        # 初始状态：所有节点为一个社区
        initial_community = [set(self.current_graph.nodes)]
        self.hierarchy.append((0, initial_community))

        level = 1
        while level <= max_levels and self.current_graph.number_of_edges() > 0:
            # 1. 计算当前图的边介数
            edge_betweenness = self._compute_edge_betweenness()
            if not edge_betweenness:
                break  # 无有效边，停止

            # 2. 找到介数最高的边
            max_betweenness = max(edge_betweenness.values())
            edges_to_remove = [edge for edge, b in edge_betweenness.items() if b == max_betweenness]

            # 3. 移除介数最高的边（可能多条）
            for u, v in edges_to_remove:
                if self.current_graph.has_edge(u, v):
                    self.current_graph.remove_edge(u, v)

            # 4. 记录移除的边及其介数
            self.edge_betweenness_history.append({
                'level': level,
                'edges_removed': edges_to_remove,
                'max_betweenness': max_betweenness
            })

            # 5. 检测当前社区划分（连通分量）
            communities = [set(cc) for cc in nx.connected_components(self.current_graph)]
            self.hierarchy.append((level, communities))

            level += 1

        return [communities for _, communities in self.hierarchy]

    def get_hierarchy_info(self) -> List[Dict]:
        """返回包含层次信息的详细记录（步骤、社区、移除的边）"""
        hierarchy_info = []
        for level, communities in self.hierarchy:
            # 找到该层次对应的边移除记录
            edge_info = next(
                (info for info in self.edge_betweenness_history if info['level'] == level),
                None
            )
            hierarchy_info.append({
                'level': level,
                'communities': communities,
                'num_communities': len(communities),
                'edges_removed': edge_info['edges_removed'] if edge_info else None,
                'max_betweenness': edge_info['max_betweenness'] if edge_info else None
            })
        return hierarchy_info

    def visualize_hierarchy(self, level: int = None, figsize: Tuple[int, int] = (10, 8)):
        """
        可视化指定层次的聚类结果
        :param level: 层次编号（None则可视化最后一层）
        """
        if not self.hierarchy:
            raise RuntimeError("请先运行run()方法生成层次结构")

        # 选择要可视化的层次
        if level is None:
            level = len(self.hierarchy) - 1
        if level < 0 or level >= len(self.hierarchy):
            raise ValueError(f"层次编号无效，有效范围0~{len(self.hierarchy) - 1}")

        _, communities = self.hierarchy[level]
        print(f"可视化层次 {level}：{len(communities)} 个社区")

        # 为不同社区分配颜色
        color_map = plt.cm.get_cmap('tab10', len(communities))
        node_colors = []
        for node in self.original_graph.nodes:
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(color_map(i))
                    break

        # 绘制原始图结构，用颜色区分社区
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.original_graph, seed=42)  # 固定布局
        nx.draw_networkx_nodes(
            self.original_graph,
            pos,
            node_color=node_colors,
            node_size=300,
            alpha=0.8
        )
        nx.draw_networkx_edges(
            self.original_graph,
            pos,
            edgelist=self.original_graph.edges,
            width=0.5,
            alpha=0.3,
            edge_color='gray'
        )
        nx.draw_networkx_labels(self.original_graph, pos, font_size=8)

        # 标注已移除的边（红色虚线）
        removed_edges = []
        for info in self.edge_betweenness_history:
            if info['level'] <= level:
                removed_edges.extend(info['edges_removed'])
        nx.draw_networkx_edges(
            self.original_graph,
            pos,
            edgelist=removed_edges,
            width=1,
            alpha=0.7,
            edge_color='red',
            style='dashed'
        )

        plt.title(f"Girvan-Newman层次聚类（层次 {level}，{len(communities)} 个社区）")
        plt.axis('off')
        plt.show()


# ------------------------------
# 示例使用
# ------------------------------
if __name__ == "__main__":
    # 1. 构建示例图（带节点向量）
    def generate_random_vectors(n_nodes: int, dim: int = 10) -> Dict[str, np.ndarray]:
        """生成随机节点向量（模拟实际场景中的节点嵌入）"""
        return {
            f"node{i}": np.random.randn(dim)  # 随机正态分布向量
            for i in range(n_nodes)
        }


    # 创建10个节点的图，节点向量维度为5
    G = nx.Graph()
    nodes = [f"node{i}" for i in range(10)]
    G.add_nodes_from(nodes)
    node_vectors = generate_random_vectors(10, dim=5)
    nx.set_node_attributes(G, node_vectors, 'vector')

    # 添加一些初始边（可选，算法会动态移除）
    G.add_edges_from(combinations(nodes[:5], 2))  # 前5个节点内部密集连接
    G.add_edges_from(combinations(nodes[5:], 2))  # 后5个节点内部密集连接
    G.add_edges_from([(nodes[2], nodes[6]), (nodes[3], nodes[7])])  # 跨组连接

    # 2. 运行Girvan-Newman层次聚类
    gn = GirvanNewmanWithVectors(G)
    hierarchies = gn.run(max_levels=5)  # 最多生成5层

    # 3. 打印层次信息
    hierarchy_info = gn.get_hierarchy_info()
    for info in hierarchy_info:
        print(f"\n层次 {info['level']}:")
        print(f"  社区数量: {info['num_communities']}")
        print(f"  移除的边: {info['edges_removed']}")
        print(f"  最大介数: {info['max_betweenness']:.4f}")
        for i, comm in enumerate(info['communities']):
            print(f"  社区 {i}: {sorted(comm)}")

    # 4. 可视化不同层次的聚类结果
    gn.visualize_hierarchy(level=0)  # 初始状态（1个社区）
    gn.visualize_hierarchy(level=2)  # 第2层
    gn.visualize_hierarchy(level=-1)  # 最后一层