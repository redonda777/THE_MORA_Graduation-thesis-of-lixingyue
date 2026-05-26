import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def get_all_versions(G):
    """获取图中所有版本 - 根据build_graph的结构修改"""
    versions = set()
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('type', '')
        if node_type == 'version':
            # 版本节点，使用text属性
            version_name = node_data.get('text', '')
            if version_name:
                versions.add(version_name)
        elif node_type in ['chapter', 'statement']:
            # 章节和语句节点，使用version属性
            version = node_data.get('version', '')
            if version:
                versions.add(version)
    return sorted(list(versions))



def get_chapter_embeddings_by_version(embeddings, G, chapter_num):
    """获取指定章节的所有版本嵌入，包括缺失版本用零向量填充 - 根据build_graph的结构修改"""
    # 获取所有版本
    all_versions = get_all_versions(G)

    # 提取章节节点信息
    node_list = list(G.nodes())
    node_embeddings_by_version = {}

    for i, node_id in enumerate(node_list):
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', 'unknown')
        node_version = node_data.get('version', 'unknown')
        node_chapter = node_data.get('seg', -1)

        # 如果是章节节点且章节号匹配
        if node_type == 'chapter' and node_chapter == chapter_num:
            if node_version not in node_embeddings_by_version:
                node_embeddings_by_version[node_version] = []
            node_embeddings_by_version[node_version].append(embeddings[i])

    # 为每个版本计算平均嵌入
    version_embeddings = {}
    for version, emb_list in node_embeddings_by_version.items():
        if emb_list:
            # 如果有多个节点，取平均
            avg_embedding = np.mean(emb_list, axis=0)
            version_embeddings[version] = avg_embedding

    # 创建完整的版本嵌入矩阵
    embedding_dim = embeddings.shape[1]
    complete_embeddings = []
    available_versions = []

    for version in all_versions:
        if version in version_embeddings:
            complete_embeddings.append(version_embeddings[version])
            available_versions.append(version)
        else:
            # 对于缺失的版本，使用零向量
            complete_embeddings.append(np.zeros(embedding_dim))
            available_versions.append(version)

    return np.array(complete_embeddings), all_versions, available_versions

def calculate_similarity_matrix(embeddings):
    """计算嵌入之间的余弦相似度矩阵"""
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings

    similarity_matrix = cosine_similarity(embeddings_np)

    # 将相似度从[-1,1]映射到[0,1]以便可视化
    similarity_matrix = (similarity_matrix + 1) / 2

    return similarity_matrix

