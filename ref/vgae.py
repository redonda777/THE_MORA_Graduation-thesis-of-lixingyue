import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE, GAE
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, negative_sampling
import networkx as nx
import numpy as np
import time
import os
import matplotlib
import warnings

import utils
from plot import visualize_version_comparison, visualize_overall_similarity, visualize_chapter_similarity, \
    visualize_chapter_comparison, compare_clustering_results
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
# 使用Agg后端，只保存不显示
matplotlib.use('Agg')
try:
    font_path = '/local/share/fonts/SimHei.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print(f"使用指定字体: {font_prop.get_name()}")
except Exception as e:
    print(f"无法加载指定字体: {e}")
    # 回退到默认字体列表
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


class VGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)  # 均值
        self.conv_logstd = GCNConv(hidden_channels, out_channels)  # 对数方差

    def forward(self, x, edge_index, edge_weight=None):
        # 修改forward方法，添加edge_weight参数
        if edge_weight is None:
            # 如果没有边权重，使用默认的全1权重
            x = self.conv1(x, edge_index).relu()
            mu = self.conv_mu(x, edge_index)
            logstd = self.conv_logstd(x, edge_index)
        else:
            # 使用提供的边权重
            x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
            mu = self.conv_mu(x, edge_index, edge_weight=edge_weight)
            logstd = self.conv_logstd(x, edge_index, edge_weight=edge_weight)
        return mu, logstd


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            # 如果没有边权重，使用默认的全1权重
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            # 使用提供的边权重
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # GAE直接返回潜在表示，不需要返回均值和对数方差
        return x


class WeightedGAE(GAE):
    def __init__(self, encoder):
        super(WeightedGAE, self).__init__(encoder)

    def recon_loss(self, z, pos_edge_index, edge_weight=None, neg_edge_index=None, l1_reg_lambda=0.01):
        # 计算所有边的预测分数
        # pos_pred = self.decoder(z, pos_edge_index)

        # 自己实现解码器逻辑，而不是使用self.decoder
        def decode(z, edge_index):
            # 确保z和edge_index在同一设备上
            edge_index = edge_index.to(z.device)
            # 内积解码器：计算两个节点嵌入的点积
            return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        # 计算所有边的预测分数
        pos_pred = decode(z, pos_edge_index)
        # 应用sigmoid激活函数将结果映射到[0,1]区间
        pos_pred = torch.sigmoid(pos_pred)
        # 如果没有提供负边索引，使用负采样生成
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index, num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1), method='sparse')

        # neg_pred = self.decoder(z, neg_edge_index)
        neg_pred = decode(z, neg_edge_index)
        # 应用sigmoid激活函数
        neg_pred = torch.sigmoid(neg_pred)

        # 计算负边损失
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()

        # 如果有权重，使用权重对正边损失进行加权
        if edge_weight is not None:
            # 确保边权重的形状与正边预测一致
            if edge_weight.dim() > 1:
                edge_weight = edge_weight.squeeze()

            # 计算加权的正边损失
            weighted_pos_loss = -torch.log(pos_pred + 1e-15) * edge_weight
            weighted_pos_loss = weighted_pos_loss.mean()

            # 使用加权的正边损失替换原始正边损失
            loss = weighted_pos_loss + neg_loss
        else:
            # 没有权重时，计算标准的正边损失
            pos_loss = -torch.log(pos_pred + 1e-15).mean()
            loss = pos_loss + neg_loss

        # 添加L1正则项以确保稀疏性
        if l1_reg_lambda > 0:
            # 计算嵌入向量z的L1范数（绝对值之和）
            l1_reg = torch.norm(z, p=1)
            # 将L1正则项添加到总损失中
            loss += l1_reg_lambda * l1_reg

        return loss


class WeightedVGAE(VGAE):
    """支持边权重的VGAE模型"""

    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    def recon_loss(self, z, edge_index, edge_weight=None, pos_weight=None, neg_edge_index=None):
        """计算考虑边权重的重构损失"""
        # 计算边的存在概率
        probs = self.decoder.forward_all(z)

        # 如果没有提供边权重，使用默认的全1权重
        if edge_weight is None:
            return super().recon_loss(z, edge_index, pos_weight=pos_weight)

        # 获取正样本边的索引
        pos_edge_index = edge_index
        pos_probs = probs[pos_edge_index[0], pos_edge_index[1]]

        # 确保edge_weight形状与loss一致
        if edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze()

        # 对于负样本，我们使用与原方法相同的策略
        # 但对正样本的损失进行加权
        loss = (-torch.log(pos_probs + 1e-15) * edge_weight).mean()

        # 添加负样本损失
        if pos_weight is None:
            pos_weight = max(1, (z.size(0) * z.size(0) - edge_index.size(1)) / edge_index.size(1))
            # 优先使用传入的neg_edge_index参数，如果没有提供则进行负采样
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(edge_index, z.size(0))
        neg_probs = probs[neg_edge_index[0], neg_edge_index[1]]
        loss += (-torch.log(1 - neg_probs + 1e-15) * pos_weight).mean()

        return loss


# 修改train_model函数，使其支持边权重
def train_model(data, neg_edge_index=None, epochs=200, model_type='gae'):
    """训练图自编码器来学习节点嵌入，支持边权重和模型类型选择
    
    参数:
        data: PyTorch Geometric数据对象
        neg_edge_index: 负样本边索引
        epochs: 训练轮数
        model_type: 模型类型，可选'gae'或'vgae'
    
    返回:
        node_embeddings: 学习到的节点嵌入
        node_features: 更新后的节点特征
        model: 训练好的模型
    """
    in_channels = data.x.size(1) if hasattr(data, 'x') and data.x is not None else 1
    hidden_channels = 64
    out_channels = 32  # 节点嵌入的维度

    # 优化器参数
    parameters = []

    # 根据model_type选择模型
    if model_type.lower() == 'vgae':
        print("使用变分图自编码器(VGAE)模型")
        encoder = VGCNEncoder(in_channels, hidden_channels, out_channels)
        model = WeightedVGAE(encoder)
        # 为VGAE设置KL散度相关参数
        kl_weight = 1.0 / data.num_nodes  # 初始KL散度权重
        kl_weight_schedule = 0.01  # KL权重递增率
    elif model_type.lower() == 'gae':
        print("使用图自编码器(GAE)模型")
        encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        model = WeightedGAE(encoder)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请选择'gae'或'vgae'")

    # 添加模型参数
    parameters.extend(list(model.parameters()))

    # 优化器 - 添加节点特征作为需要优化的参数
    if hasattr(data, 'x') and data.x is not None and data.x.requires_grad:
        parameters.append(data.x)

    # 优化器
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        assert data.edge_weight is not None, "未使用边权重信息进行训练"

        # 编码获取潜在表示
        if model_type.lower() == 'vgae':
            # VGAE的编码返回均值和对数方差
            z = model.encode(data.x, data.edge_index, data.edge_weight)
            # 动态调整KL散度权重
            current_kl_weight = min(kl_weight + epoch * kl_weight_schedule, 1.0 / data.num_nodes * 5)
            loss = model.recon_loss(z, data.edge_index, data.edge_weight,
                                    neg_edge_index=neg_edge_index) + current_kl_weight * model.kl_loss()
        else:
            # GAE的编码直接返回潜在表示
            z = model.encode(data.x, data.edge_index, data.edge_weight)
            loss = model.recon_loss(z, data.edge_index, data.edge_weight, neg_edge_index=neg_edge_index)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss.item():.4f}')

    # 获取节点嵌入
    model.eval()
    with torch.no_grad():
        if model_type.lower() == 'vgae':
            z = model.encode(data.x, data.edge_index, data.edge_weight)
            node_embeddings = z
        else:
            node_embeddings = model.encode(data.x, data.edge_index, data.edge_weight)

    return node_embeddings, data.x.cpu().detach().numpy(), model


def graph_clean_copy(G: nx.Graph):
    # 创建清理后的图副本
    G_clean = nx.Graph()

    # 复制所有节点和属性（移除手动设置feature的逻辑）
    for node in G.nodes():
        if len(G.nodes[node]) == 0:
            raise ValueError(f"节点 {node} 缺少属性")
        node_data = {}
        for key, value in G.nodes[node].items():
            if key == 'has_nonempty_statements':
                continue
            if isinstance(value, (np.ndarray, np.generic)):
                if value.size == 1:
                    node_data[key] = value.item()
                else:
                    node_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                node_data[key] = value
            else:
                node_data[key] = value

        # 保留其他必要属性的默认值设置
        if 'type' not in node_data:
            node_data['type'] = 'unknown'
        if 'version' not in node_data:
            node_data['version'] = 'unknown'
        if 'seg' not in node_data and 'chapter' in node_data:
            node_data['seg'] = node_data['chapter']
            # 新增：检查并设置seg属性（缺失时使用text属性值）
        if 'seg' not in node_data:
            node_data['seg'] = -1  # 默认为-1
        else:
            node_data['seg'] = int(node_data['seg'])
        assert type(node_data['seg']) == int, (f"章节号 {node_data['seg']} 不是整数类型", type(node_data['seg']))

        if 'ln' not in node_data:
            node_data['ln'] = -1

        G_clean.add_node(node, **node_data)

    # 复制所有边和属性，并提取边权重
    node_list = set(G_clean.nodes())
    original_node_list = set(G.nodes())
    edge_weights = {}
    similarity_edges = 0
    for u, v in G.edges():
        assert u in original_node_list and v in original_node_list
        if u not in node_list or v not in node_list:
            continue
        edge_data = {}
        assert 'relation' in G[u][v], "没有关系属性"
        if G[u][v]['relation'] != 'similar':
            edge_data['similarity'] = 1.
        else:
            assert 'similarity' in G[u][v]
            edge_data['similarity'] = G[u][v]['similarity']
            similarity_edges += 1
        # 保存边权重到字典，同时保存正向和反向边
        weight = edge_data['similarity']
        edge_weights[(u, v)] = weight
        edge_weights[(v, u)] = weight
        G_clean.add_edge(u, v, **edge_data)
    print(f"相似度边数: {similarity_edges}")
    assert similarity_edges > 0, "没有相似度边"

    return G_clean, edge_weights


def extract_edge_weights(G: nx.Graph, data: Data, node_list, edge_weights):
    # 创建节点名到节点ID的映射（这是修复的关键）
    # node_list包含原始节点名，顺序与PyG中的节点ID对应
    node_id_to_name = {i: node for i, node in enumerate(node_list)}

    print("构建边到权重的映射...")

    # 获取NetworkX图中的所有边及其对应的索引
    edge_list = list(G.edges())
    print(f"NetworkX图中的边数: {len(edge_list)}")

    if hasattr(data, 'edge_index') and data.edge_index is not None:
        edge_count = data.edge_index.size(1)
        print(f"PyG数据中的边索引数: {edge_count}")

        # 根据edge_index中的每条边查找对应的权重
        pyg_edge_weights = []
        missing_count = 0

        for i in range(edge_count):
            # 获取整数ID
            u_id, v_id = data.edge_index[:, i].tolist()

            # 将整数ID转换回原始节点名
            u_name = node_id_to_name[u_id]
            v_name = node_id_to_name[v_id]

            # 查找权重，如果不存在则使用默认值1.0
            if (u_name, v_name) in edge_weights:
                weight = edge_weights[(u_name, v_name)]
                pyg_edge_weights.append(weight)
            else:
                # 如果找不到对应的边，使用默认权重1.0
                pyg_edge_weights.append(1.0)
                missing_count += 1
                # 只打印前几个缺失的边以避免输出过多
                if missing_count <= 5:
                    print(f"警告: 边 ({u_name}, {v_name}) 在原始图中未找到，使用默认权重")

        if missing_count > 0:
            print(f"总共有 {missing_count} 条边在原始图中未找到")

        # 设置边权重
        edge_weight = torch.tensor(pyg_edge_weights, dtype=torch.float)
        print(f"最终设置的边权重形状: {edge_weight.shape}")
        # 新增：统计边权重
        if len(pyg_edge_weights) > 0:
            weights_np = np.array(pyg_edge_weights)
            print("边权重统计信息:")
            print(f"  最大值: {weights_np.max():.6f}")
            print(f"  最小值: {weights_np.min():.6f}")
            print(f"  平均值: {weights_np.mean():.6f}")
            print(f"  中位数: {np.median(weights_np):.6f}")
            print(f"  标准差: {weights_np.std():.6f}")
            print(f"  权重范围: [{weights_np.min():.6f}, {weights_np.max():.6f}]")
            # 可以根据需要添加更多统计量，如四分位数等
    else:
        # 如果没有边索引，需要特殊处理（但这种情况应该很少见）
        # 为原始无向图的每条边创建权重列表
        simple_edge_weights = []
        for u, v in G.edges():
            simple_edge_weights.append(edge_weights[(u, v)])
        edge_weight = torch.tensor(simple_edge_weights, dtype=torch.float)
        print(f"边权重形状: {edge_weight.shape}")
    return edge_weight


# 修改generate_negative_edges函数以处理同章节的statement节点对
def generate_negative_edges(data, G, node_list, negative_sampling_ratio):
    """生成负样本边索引"""
    neg_edge_index = None
    existing_edges = set()
    # 生成neg_edge_index，包含所有chapter类型节点
    print("生成负样本边索引...")
    # 获取所有chapter类型节点的ID
    chapter_node_ids = []
    non_chapter_node_ids = []
    statement_nodes_by_chapter = {}
    statement_nodes_by_chapter_ln = {}  # 新增：按seg和ln属性分组
    chapter_nodes_by_seg = {}  # 新增：按seg属性对chapter节点进行分组

    for i, node in enumerate(node_list):
        node_data = G.nodes[node]
        if node_data.get('type') == 'chapter' or node_data.get('type') == 'version':
            chapter_node_ids.append(i)
            # 新增：如果是chapter类型节点，从node ID中解析seg属性
            if node_data.get('type') == 'chapter':
                seg = node_data.get('seg', -1)
                if seg != -1:
                    if seg not in chapter_nodes_by_seg:
                        chapter_nodes_by_seg[seg] = []
                    chapter_nodes_by_seg[seg].append(i)
        else:
            non_chapter_node_ids.append(i)

        # 收集statement类型节点并按chapter分组
        if node_data.get('type') == 'statement':
            chapter_seg = node_data.get('seg', -1)
            line_number = node_data.get('ln', -1)  # 获取ln属性
            if chapter_seg != -1:
                if chapter_seg not in statement_nodes_by_chapter:
                    statement_nodes_by_chapter[chapter_seg] = []
                statement_nodes_by_chapter[chapter_seg].append(i)

                # 按seg和ln属性分组
                if line_number != -1:
                    key = (chapter_seg, line_number)
                    if key not in statement_nodes_by_chapter_ln:
                        statement_nodes_by_chapter_ln[key] = []
                    statement_nodes_by_chapter_ln[key].append(i)

    # 构建当前存在的边集合（用于过滤）
    for i in range(data.edge_index.size(1)):
        u, v = data.edge_index[:, i].tolist()
        existing_edges.add((u, v))

    neg_edges = []
    # 1. 处理chapter类型节点之间的负样本
    if len(chapter_node_ids) > 0:
        print(f"找到 {len(chapter_node_ids)} 个chapter类型节点")

        # 生成所有chapter节点之间的可能边对
        chapter_pairs = []
        for seg, nodes_in_seg in chapter_nodes_by_seg.items():
            # 对每个seg组内的节点生成所有可能的边对
            for i in range(len(nodes_in_seg)):
                for j in range(i + 1, len(nodes_in_seg)):
                    u = nodes_in_seg[i]
                    v = nodes_in_seg[j]
                    chapter_pairs.append((u, v))

        # 过滤出不存在的边作为负样本
        for u, v in chapter_pairs:
            if (u, v) not in existing_edges:
                neg_edges.append((u, v))
    else:
        print("警告: 未找到chapter类型节点")
    print(f"找到 chapter类型节点之间的负样本边: {len(neg_edges)} 个负样本边")

    # 计算目标负样本数量
    target_neg_count = int(data.edge_index.size(1) * negative_sampling_ratio)

    # 2. 新增：优先处理seg和ln都相同的statement节点对
    seg_ln_neg_count = 0
    if statement_nodes_by_chapter_ln:
        print(f"找到 {len(statement_nodes_by_chapter_ln)} 个包含statement节点的(seg, ln)组合")

        # 对每个(seg, ln)组合中的statement节点生成负样本
        for (seg, ln), statement_ids in statement_nodes_by_chapter_ln.items():
            if len(statement_ids) < 2:  # 至少需要两个节点
                continue

            # 生成该组合中所有statement节点之间的可能边对
            for i in range(len(statement_ids)):
                for j in range(i + 1, len(statement_ids)):
                    u = statement_ids[i]
                    v = statement_ids[j]

                    # 检查这两个节点之间是否没有连边
                    if (u, v) not in existing_edges and (v, u) not in existing_edges:
                        neg_edges.append((u, v))
                        seg_ln_neg_count += 1

    print(f"找到 seg和ln都相同的statement节点之间的负样本边: {seg_ln_neg_count} 个负样本边")

    # 3. 处理同属一个chapter但没有连边的statement节点对
    statement_candidates = []
    if statement_nodes_by_chapter:
        print(f"找到 {len(statement_nodes_by_chapter)} 个包含statement节点的章节")

        # 对每个章节中的statement节点生成负样本
        for chapter_seg, statement_ids in statement_nodes_by_chapter.items():
            if len(statement_ids) < 2:  # 章节中至少需要两个statement节点
                continue

            # 生成该章节中所有statement节点之间的可能边对
            for i in range(len(statement_ids)):
                for j in range(i + 1, len(statement_ids)):
                    u = statement_ids[i]
                    v = statement_ids[j]

                    # 检查这两个节点之间是否没有连边
                    if (u, v) not in existing_edges and (v, u) not in existing_edges:
                        statement_candidates.append((u, v))
    # 计算还需要多少负样本边
    remaining_needed = max(0, target_neg_count - len(neg_edges))
    if remaining_needed > 0 and len(statement_candidates) > 0:
        # 从候选集中随机采样所需数量的负样本
        # 如果候选数量少于所需数量，则全部添加
        num_to_sample = min(remaining_needed, len(statement_candidates))

        # 使用numpy进行随机采样
        if len(statement_candidates) > num_to_sample:
            sampled_indices = np.random.choice(len(statement_candidates), num_to_sample, replace=False)
            sampled_edges = [statement_candidates[i] for i in sampled_indices]
        else:
            sampled_edges = statement_candidates

        neg_edges.extend(sampled_edges)
        print(f"从statement候选集中采样的负样本边数: {len(sampled_edges)}")

    # 4. 处理非chapter类型节点之间的负样本
    if len(neg_edges) < target_neg_count and len(non_chapter_node_ids) > 0:
        print(f"找到 {len(non_chapter_node_ids)} 个非chapter类型节点")

        # 计算需要采样的非chapter负样本数量
        # 总负样本数量 = 正样本数量 * negative_sampling_ratio
        # 非chapter负样本数量 = 总负样本数量 - chapter负样本数量 - statement负样本数量
        non_chapter_target = max(0, target_neg_count - len(neg_edges))

        if non_chapter_target > 0:
            # 生成非chapter节点之间的可能边对（限制总数量以避免计算量过大）
            non_chapter_candidates = []
            assert len(existing_edges) > 0, "当前图中不存在任何边"
            while len(non_chapter_candidates) < non_chapter_target:
                u = np.random.choice(non_chapter_node_ids)
                v = np.random.choice(non_chapter_node_ids)
                if u != v and (u, v) not in existing_edges and (u, v) not in non_chapter_candidates:
                    non_chapter_candidates.append((u, v))

            neg_edges.extend(non_chapter_candidates)
            print(f"采样的非chapter类型负样本边数: {len(non_chapter_candidates)}")

    # 打印统计信息
    print(f"总共生成的负样本边数: {len(neg_edges)}")

    if len(neg_edges) > 0:
        # 转换为PyG的边索引格式
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
        print(f"生成的负样本边索引形状: {neg_edge_index.shape}")
    else:
        print("警告: 没有找到可用的负样本边")
    return neg_edge_index


def prepare_pyg_data(G, feature_dim, negative_sampling_ratio=1.):
    """准备PyTorch Geometric数据"""
    print("准备PyG数据...")

    # 创建清理后的图副本
    G_clean, edge_weights = graph_clean_copy(G)

    # 通过create_node_features获取节点特征（核心修改）
    node_features_dict = create_node_features(G_clean, feature_dim)

    # 构建特征矩阵
    node_list = list(G_clean.nodes())
    features = [node_features_dict[node] for node in node_list]

    # 构建PyG数据
    data = from_networkx(G_clean)
    data.x = torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float,
                          requires_grad=True)  # 使用create_node_features生成的特征
    print(f"节点特征形状: {data.x.shape}")

    # 添加边权重
    if edge_weights:
        data.edge_weight = extract_edge_weights(G_clean, data, node_list, edge_weights)
    else:
        print("警告: 没有找到边权重信息")

    # 处理边索引（保持不变）
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.size(1) == 0:
        print("警告: 图中没有边信息，添加自环边")
        num_nodes = data.num_nodes
        edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
        data.edge_index = edge_index

    neg_edge_index = generate_negative_edges(data, G_clean, node_list, negative_sampling_ratio)

    print(f"边索引形状: {data.edge_index.shape}")
    return data, G_clean, neg_edge_index


def create_node_features(G, feature_dim):
    """根据图构建代码创建统一的节点特征 - 使用随机向量初始化（移除元特征）"""
    print(f"创建节点特征 (随机向量初始化, 维度={feature_dim})...")

    # 创建节点特征映射
    node_features = {}

    for node_id, _ in G.nodes(data=True):
        # 生成64维随机特征向量（范围[0,1)）
        feature = np.random.normal(loc=0.0, scale=0.1, size=feature_dim)
        node_features[node_id] = feature

    print(f"随机特征维度: {feature_dim} (每个节点)")
    return node_features


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='图神经网络向量模型训练和相似度分析')
    parser.add_argument('--model_type', type=str, default='gae', choices=['gae', 'vgae'],
                        help='模型类型: gae(图自编码器) 或 vgae(变分图自编码器)')
    parser.add_argument('--feature_type', type=str, default='embedding', choices=['embedding', 'hidden'],
                        help='特征类型: embedding(节点嵌入) 或 hidden(隐藏层特征)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--negative_sample_ratio', type=float, default=1.0,
                        help='负采样比例，相对于正样本的倍数')
    parser.add_argument('--pkl_file', type=str,
                        help='pkl文件路径')
    parser.add_argument('--feature_dim', type=int, default=64, help='节点特征维度 (默认: 64)')
    parser.add_argument('--result_dir', type=str, default='plots', help='结果保存目录')
    args = parser.parse_args()
    # 打印用户设置的参数
    print(
        f"使用参数: 模型类型={args.model_type}, 训练轮数={args.epochs}, 负采样比例={args.negative_sample_ratio}, 节点特征维度={args.feature_dim}")
    # 设置随机种子，确保实验可重复性
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("=== 多版本图神经网络分层相似度分析 ===")

    # 创建基于参数的结果文件夹名称
    #result_dir = f"similarity_plots_model_{args.model_type}_epochs_{args.epochs}_neg_{args.negative_sample_ratio}_featuredim_{args.feature_dim}"
    result_dir = args.result_dir

    data_file = args.pkl_file
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        G = data['graph']
    print(f"成功加载图数据: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    print(f"\n=== 图结构信息 ===")
    print(f"节点总数: {G.number_of_nodes()}")
    print(f"边总数: {G.number_of_edges()}")

    node_types = [data.get('type', 'unknown') for _, data in G.nodes(data=True)]
    type_count = {}
    for node_type in node_types:
        type_count[node_type] = type_count.get(node_type, 0) + 1
    print(f"节点类型分布: {type_count}")

    # 打印版本信息
    all_versions = utils.get_all_versions(G)
    print(f"检测到的版本: {all_versions}")

    # 统计包含similarity属性的边的数量
    similarity_edges_count = 0
    for u, v, edge_data in G.edges(data=True):
        if 'similarity' in edge_data:
            similarity_edges_count += 1
    print(f"包含similarity属性的边数量: {similarity_edges_count}")

    data, G_clean, neg_edge_index = prepare_pyg_data(G, args.feature_dim,
                                                     negative_sampling_ratio=args.negative_sample_ratio)

    print("\n=== 开始训练图神经网络模型 ===")
    try:
        node_hidden_vectors, node_features, model = train_model(data, neg_edge_index, epochs=args.epochs, model_type=args.model_type)
        print(f"学习到的节点特征形状: {node_features.shape}")
        # 将嵌入向量添加到图节点属性中
        node_list = list(G_clean.nodes())
        for i, node_id in enumerate(node_list):
            # 将PyTorch张量转换为NumPy数组并添加到节点属性
            # G_clean.nodes[node_id]['embedding'] = node_embeddings[i].cpu().detach().numpy()
            if args.feature_type == 'embedding':
                G_clean.nodes[node_id]['embedding'] = node_features[i]
            elif args.feature_type == 'hidden':
                G_clean.nodes[node_id]['embedding'] = node_hidden_vectors[i]



        print("\n" + "=" * 60)
        print("开始生成相似度可视化")
        print("=" * 60)

        # 收集总体版本聚类结果
        overall_linkage, overall_versions_order = visualize_version_comparison(node_features, G_clean, result_dir)
        overall_clustering = (overall_linkage, overall_versions_order)

        # 生成总体相似度图表
        # visualize_overall_similarity(node_embeddings, G_clean)
        visualize_overall_similarity(node_features, G_clean, result_dir)

        print("\n=== 生成所有章节的版本相似度热力图 ===")

        # 获取所有唯一的章节号（seg值）
        chapter_set = set()
        for node_id, node_data in G_clean.nodes(data=True):
            if node_data.get('type') == 'chapter':
                seg = node_data.get('seg', -1)
                assert type(seg) == int, (f"章节号 {seg} 不是整数类型", type(seg))
                chapter_set.add(seg)  # 确保seg为整数

        # 排序章节号
        all_chapters = sorted(chapter_set)
        print(f"检测到 {len(all_chapters)} 个章节: {all_chapters}")

        # 收集各章节的聚类结果
        chapter_clusters = {}
        # 遍历章节生成热力图
        for chapter in all_chapters:
            visualize_chapter_similarity(node_features, G_clean, chapter, result_dir)
            chapter_linkage, chapter_versions_order = visualize_chapter_comparison(node_features, G_clean, chapter,
                                                                                   result_dir)
            chapter_clusters[chapter] = (chapter_linkage, chapter_versions_order)
        # 比较每个章节的聚类和总的版本聚类结果
        compare_clustering_results(chapter_clusters, overall_clustering, result_dir)
        # 使用save_to_pickle保存包含嵌入的图
        print("\n=== 保存包含嵌入特征的图数据 ===")
        # 构建符合save_to_pickle要求的数据结构
        graph_data = {
            'graph': G_clean,
            'versions': all_versions,
            'graph_info': {
                'node_count': G_clean.number_of_nodes(),
                'edge_count': G_clean.number_of_edges(),
                'node_types': type_count
            }
        }
        # 调用save_to_pickle保存
        pickle.dump(graph_data, open("graph_with_embeddings.pkl", "wb"))


    except Exception as e:
        print(f"模型训练或可视化时出错: {e}")
        import traceback
        traceback.print_exc()
    print("\n=== 分析完成！ ===")
    print("总结:")
    print("🎯 版本比较：多角度版本间相似度综合分析")
    print("📊 总体相似度：版本间的整体相似性热力图")
    print(f"💾 所有图表已保存到: {os.path.join(os.getcwd(), result_dir)}")


if __name__ == "__main__":
    main()
