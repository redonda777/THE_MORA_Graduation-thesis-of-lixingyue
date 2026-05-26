import os
import time

import numpy as np

from utils import get_chapter_embeddings_by_version, calculate_similarity_matrix
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import leaves_list
from scipy.stats import spearmanr


def visualize_chapter_similarity(embeddings, G, chapter_num, result_dir):
    """生成指定章节的版本相似度热力图"""
    print(f"\n=== 生成章节 {chapter_num} 的版本相似度热力图 ===")

    # 获取该章节的所有版本嵌入
    chapter_embeddings, all_versions, available_versions = get_chapter_embeddings_by_version(embeddings, G, chapter_num)

    if len(all_versions) < 2:
        print(f"章节 {chapter_num} 的版本数量不足，无法生成相似度热力图")
        return

    # 计算相似度矩阵
    similarity_matrix = calculate_similarity_matrix(chapter_embeddings)

    # 创建热力图
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)

    # 绘制热力图
    # im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1)
    im = ax.imshow(similarity_matrix, cmap='gray_r', aspect='auto', vmin=0.3, vmax=1)

    # 设置坐标轴标签
    ax.set_xticks(range(len(all_versions)))
    ax.set_yticks(range(len(all_versions)))
    ax.set_xticklabels(all_versions, rotation=45, fontsize=10)
    ax.set_yticklabels(all_versions, fontsize=10)

    # 添加标题
    ax.set_title(f'章节 {chapter_num} 版本相似度热力图', fontsize=14, fontweight='bold')

    # 在热力图上显示数值
    for i in range(len(all_versions)):
        for j in range(len(all_versions)):
            # 对缺失版本（零向量）的相似度标记为N/A
            if chapter_embeddings[i].sum() == 0 or chapter_embeddings[j].sum() == 0:
                text = "N/A"
                color = "gray"
            else:
                text = f'{similarity_matrix[i, j]:.3f}'
                color = "white" if similarity_matrix[i, j] < 0.6 else "black"

            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    # 添加颜色条
    plt.colorbar(im, ax=ax, label='相似度')

    # 调整布局并保存
    plt.tight_layout()
    save_plot(fig, f'chapter_{chapter_num}_similarity.png', result_dir, delay=0.3)


# 在visualize_chapter_similarity函数后添加新函数
def visualize_chapter_comparison(embeddings, G, chapter_num, result_dir):
    """生成指定章节的完整版本比较图，包含多种可视化方式"""
    print(f"\n=== 生成章节 {chapter_num} 的完整版本比较图 ===")

    # 获取该章节的所有版本嵌入
    chapter_embeddings, all_versions, available_versions = get_chapter_embeddings_by_version(embeddings, G, chapter_num)

    if len(all_versions) < 2:
        print(f"章节 {chapter_num} 的版本数量不足，无法生成完整比较图")
        return

    # 计算相似度矩阵
    similarity_matrix = calculate_similarity_matrix(chapter_embeddings)

    # 创建章节比较图 - 多个子图展示
    fig = plt.figure(figsize=(20, 15))

    # 子图1：版本相似度热力图
    ax1 = plt.subplot(2, 3, 1)
    #im1 = ax1.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1)
    im1 = ax1.imshow(similarity_matrix, cmap='gray_r', aspect='auto', vmin=0.3, vmax=1)
    ax1.set_xticks(range(len(all_versions)))
    ax1.set_yticks(range(len(all_versions)))
    ax1.set_xticklabels([f'{v}' for v in all_versions], rotation=45, fontsize=10)
    ax1.set_yticklabels([f'{v}' for v in all_versions], fontsize=10)
    ax1.set_title(f'章节 {chapter_num} 版本相似度热力图', fontsize=12, fontweight='bold')

    # 在热力图上显示数值
    for i in range(len(all_versions)):
        for j in range(len(all_versions)):
            # 对缺失版本（零向量）的相似度标记为N/A
            if chapter_embeddings[i].sum() == 0 or chapter_embeddings[j].sum() == 0:
                text = "N/A"
                color = "gray"
            else:
                text = f'{similarity_matrix[i, j]:.3f}'
                color = "white" if similarity_matrix[i, j] < 0.6 else "black"

            ax1.text(j, i, text, ha="center", va="center", color=color, fontsize=8, fontweight='bold')

    plt.colorbar(im1, ax=ax1, label='相似度')

    # 子图2：版本平均相似度条形图
    ax2 = plt.subplot(2, 3, 2)
    version_avg_similarities = []
    for i in range(len(all_versions)):
        # 跳过缺失版本
        if chapter_embeddings[i].sum() == 0:
            version_avg_similarities.append(0)
            continue

        # 计算每个版本与其他版本的平均相似度（排除自身和缺失版本）
        other_indices = [j for j in range(len(all_versions)) if j != i and chapter_embeddings[j].sum() != 0]
        if other_indices:
            avg_sim = np.mean([similarity_matrix[i, j] for j in other_indices])
            version_avg_similarities.append(avg_sim)
        else:
            version_avg_similarities.append(0)

    bars = ax2.bar(range(len(all_versions)), version_avg_similarities,
                   color=plt.cm.viridis(np.linspace(0, 1, len(all_versions))))
    ax2.set_xlabel('版本')
    ax2.set_ylabel('平均相似度')
    ax2.set_title(f'章节 {chapter_num} 各版本平均相似度', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(all_versions)))
    ax2.set_xticklabels([f'{v}' for v in all_versions], rotation=45, fontsize=9)
    ax2.set_ylim(0, 1.05)

    # 在条形上添加数值
    for i, bar in enumerate(bars):
        if version_avg_similarities[i] > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # 子图3：版本相似度网络图
    ax3 = plt.subplot(2, 3, 3)

    # 创建版本网络 - 只包含有效版本
    valid_indices = [i for i in range(len(all_versions)) if chapter_embeddings[i].sum() != 0]
    valid_versions = [all_versions[i] for i in valid_indices]
    valid_embeddings = [chapter_embeddings[i] for i in valid_indices]
    n_valid = len(valid_versions)

    if n_valid >= 2:
        pos = {}
        radius = 5
        for i in range(n_valid):
            angle = 2 * np.pi * i / n_valid
            pos[i] = (radius * np.cos(angle), radius * np.sin(angle))

        # 绘制连接线
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                # 获取原始索引
                orig_i = valid_indices[i]
                orig_j = valid_indices[j]
                sim = similarity_matrix[orig_i, orig_j]
                linewidth = 1 + sim * 8  # 线宽根据相似度调整
                alpha = 0.3 + sim * 0.7  # 透明度根据相似度调整
                color = plt.cm.RdYlGn(sim)  # 颜色根据相似度调整
                ax3.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                         color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')

                # 在连线中点显示相似度
                mid_x = (pos[i][0] + pos[j][0]) / 2
                mid_y = (pos[i][1] + pos[j][1]) / 2
                ax3.text(mid_x, mid_y, f'{sim:.2f}', fontsize=7, ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # 绘制版本节点
        for i in range(n_valid):
            ax3.scatter(pos[i][0], pos[i][1], s=300, color=plt.cm.Set1(i / n_valid),
                        alpha=0.8, edgecolors='black', linewidth=2)
            ax3.text(pos[i][0], pos[i][1], f'{valid_versions[i]}', fontsize=9,
                     ha='center', va='center', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        ax3.set_xlim(-radius * 1.2, radius * 1.2)
        ax3.set_ylim(-radius * 1.2, radius * 1.2)
        ax3.set_aspect('equal')
        ax3.axis('off')
    ax3.set_title(f'章节 {chapter_num} 版本相似度网络图', fontsize=12, fontweight='bold')

    # 子图4：版本聚类热图
    ax4 = plt.subplot(2, 3, 4)

    # 对有效版本进行聚类排序
    if n_valid >= 2:
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        # 创建有效版本的相似度矩阵
        valid_similarity_matrix = np.zeros((n_valid, n_valid))
        for i in range(n_valid):
            for j in range(n_valid):
                valid_similarity_matrix[i, j] = similarity_matrix[valid_indices[i], valid_indices[j]]

        linkage_matrix = linkage(valid_similarity_matrix, method='average')
        dendrogram(linkage_matrix, ax=ax4, labels=[f'{v}' for v in valid_versions])
    ax4.set_title(f'章节 {chapter_num} 版本聚类树状图', fontsize=12, fontweight='bold')
    ax4.set_ylabel('距离')

    # 子图5：重新排序后的热力图
    ax5 = plt.subplot(2, 3, 5)

    if n_valid >= 2:
        reordered_indices = leaves_list(linkage_matrix)
        reordered_valid_indices = [valid_indices[i] for i in reordered_indices]
        reordered_matrix = similarity_matrix[reordered_valid_indices][:, reordered_valid_indices]
        reordered_versions = [all_versions[i] for i in reordered_valid_indices]

        im5 = ax5.imshow(reordered_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1)
        ax5.set_xticks(range(len(reordered_versions)))
        ax5.set_yticks(range(len(reordered_versions)))
        ax5.set_xticklabels([f'{v}' for v in reordered_versions], rotation=45, fontsize=10)
        ax5.set_yticklabels([f'{v}' for v in reordered_versions], fontsize=10)

        # 在热力图上显示数值
        for i in range(len(reordered_versions)):
            for j in range(len(reordered_versions)):
                color = "white" if reordered_matrix[i, j] < 0.6 else "black"
                ax5.text(j, i, f'{reordered_matrix[i, j]:.3f}',
                         ha="center", va="center", color=color, fontsize=8, fontweight='bold')

        plt.colorbar(im5, ax=ax5, label='相似度')
    ax5.set_title(f'章节 {chapter_num} 聚类排序后相似度热力图', fontsize=12, fontweight='bold')

    # 子图6：版本相似度分布
    ax6 = plt.subplot(2, 3, 6)

    # 提取所有非对角线的有效相似度值
    all_similarities = []
    for i in range(len(all_versions)):
        if chapter_embeddings[i].sum() == 0:
            continue
        for j in range(i + 1, len(all_versions)):
            if chapter_embeddings[j].sum() == 0:
                continue
            all_similarities.append(similarity_matrix[i, j])

    if all_similarities:
        # 绘制分布直方图
        n, bins, patches = ax6.hist(all_similarities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.set_xlabel('相似度')
        ax6.set_ylabel('频次')
        ax6.set_title(f'章节 {chapter_num} 版本间相似度分布', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0.3, 1.0)

        # 添加统计线
        mean_sim = np.mean(all_similarities)
        median_sim = np.median(all_similarities)
        ax6.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_sim:.3f}')
        ax6.axvline(median_sim, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_sim:.3f}')
        ax6.legend()

    plt.tight_layout()
    plt.suptitle(f'章节 {chapter_num} 多版本间相似度综合分析', fontsize=16, fontweight='bold', y=0.98)

    # 安全保存图表
    save_plot(fig, f'chapter_{chapter_num}_comparison_analysis.png', result_dir, delay=0.5)

    # 打印版本比较统计
    print(f"\n=== 章节 {chapter_num} 版本间相似度比较分析 ===")
    print(f"总版本数: {len(all_versions)}")
    print(f"有效版本数: {n_valid}")

    if all_similarities:
        print(f"所有版本间平均相似度: {np.mean(all_similarities):.4f}")
        print(f"相似度标准差: {np.std(all_similarities):.4f}")

        # 找到最高和最低相似度的版本对
        max_sim_idx = np.argmax(all_similarities)
        min_sim_idx = np.argmin(all_similarities)

        # 计算版本对索引
        pairs = []
        for i in range(len(all_versions)):
            if chapter_embeddings[i].sum() == 0:
                continue
            for j in range(i + 1, len(all_versions)):
                if chapter_embeddings[j].sum() == 0:
                    continue
                pairs.append((i, j))

        max_i, max_j = pairs[max_sim_idx]
        min_i, min_j = pairs[min_sim_idx]

        print(f"最高相似度版本对: {all_versions[max_i]} vs {all_versions[max_j]}: {np.max(all_similarities):.4f}")
        print(f"最低相似度版本对: {all_versions[min_i]} vs {all_versions[min_j]}: {np.min(all_similarities):.4f}")

    # 打印各版本平均相似度排名
    if any(version_avg_similarities):
        print(f"\n各版本平均相似度排名:")
        ranked_versions = sorted(zip(all_versions, version_avg_similarities),
                                 key=lambda x: x[1], reverse=True)
        for i, (version, avg_sim) in enumerate(ranked_versions, 1):
            if avg_sim > 0:
                print(f"  {i}. {version}: {avg_sim:.4f}")
    return linkage_matrix, reordered_versions


def visualize_version_comparison(embeddings, G, result_dir):
    """生成不同版本间相似度的比较图 - 根据build_graph的结构修改"""
    print("\n=== 生成版本间相似度比较图 ===")

    # 提取版本节点信息
    node_list = list(G.nodes())
    node_types = []
    node_versions = []

    for node_id in node_list:
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', 'unknown')
        node_types.append(node_type)

        # 根据节点类型获取版本信息
        if node_type == 'version':
            # 版本节点，使用text属性
            node_versions.append(node_data.get('text', 'unknown'))
        else:
            # 其他节点，使用version属性
            node_versions.append(node_data.get('version', 'unknown'))

    # 只关注version节点
    version_indices = [i for i, t in enumerate(node_types) if t == 'version']

    if len(version_indices) < 2:
        print("版本节点数量不足，无法进行版本相似度分析")
        return

    # 提取版本节点的嵌入
    version_embeddings = embeddings[version_indices]
    version_names = [node_versions[i] for i in version_indices]

    # 计算相似度矩阵
    similarity_matrix = calculate_similarity_matrix(version_embeddings)

    # 创建版本比较图 - 多个子图展示
    fig = plt.figure(figsize=(20, 15))

    # 子图1：版本相似度热力图
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1)
    ax1.set_xticks(range(len(version_names)))
    ax1.set_yticks(range(len(version_names)))
    ax1.set_xticklabels([f'{v}' for v in version_names], rotation=45, fontsize=10)
    ax1.set_yticklabels([f'{v}' for v in version_names], fontsize=10)
    ax1.set_title('版本间相似度热力图', fontsize=12, fontweight='bold')

    # 在热力图上显示数值
    for i in range(len(version_names)):
        for j in range(len(version_names)):
            color = "white" if similarity_matrix[i, j] < 0.6 else "black"
            ax1.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                     ha="center", va="center", color=color, fontsize=8, fontweight='bold')

    plt.colorbar(im1, ax=ax1, label='相似度')

    # 子图2：版本平均相似度条形图
    ax2 = plt.subplot(2, 3, 2)
    version_avg_similarities = []
    for i in range(len(version_names)):
        # 计算每个版本与其他版本的平均相似度（排除自身）
        other_indices = [j for j in range(len(version_names)) if j != i]
        avg_sim = np.mean([similarity_matrix[i, j] for j in other_indices])
        version_avg_similarities.append(avg_sim)

    bars = ax2.bar(range(len(version_names)), version_avg_similarities,
                   color=plt.cm.viridis(np.linspace(0, 1, len(version_names))))
    ax2.set_xlabel('版本')
    ax2.set_ylabel('平均相似度')
    ax2.set_title('各版本平均相似度', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(version_names)))
    ax2.set_xticklabels([f'{v}' for v in version_names], rotation=45, fontsize=9)

    # 在条形上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # 子图3：版本相似度网络图
    ax3 = plt.subplot(2, 3, 3)

    # 创建版本网络
    pos = {}
    n_versions = len(version_names)
    radius = 5
    for i in range(n_versions):
        angle = 2 * np.pi * i / n_versions
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))

    # 绘制连接线
    for i in range(n_versions):
        for j in range(i + 1, n_versions):
            sim = similarity_matrix[i, j]
            linewidth = 1 + sim * 8  # 线宽根据相似度调整
            alpha = 0.3 + sim * 0.7  # 透明度根据相似度调整
            color = plt.cm.RdYlGn(sim)  # 颜色根据相似度调整
            ax3.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                     color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')

            # 在连线中点显示相似度
            mid_x = (pos[i][0] + pos[j][0]) / 2
            mid_y = (pos[i][1] + pos[j][1]) / 2
            ax3.text(mid_x, mid_y, f'{sim:.2f}', fontsize=7, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # 绘制版本节点
    for i in range(n_versions):
        ax3.scatter(pos[i][0], pos[i][1], s=300, color=plt.cm.Set1(i / n_versions),
                    alpha=0.8, edgecolors='black', linewidth=2)
        ax3.text(pos[i][0], pos[i][1], f'{version_names[i]}', fontsize=9,
                 ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    ax3.set_xlim(-radius * 1.2, radius * 1.2)
    ax3.set_ylim(-radius * 1.2, radius * 1.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('版本相似度网络图', fontsize=12, fontweight='bold')

    # 子图4：版本聚类热图
    ax4 = plt.subplot(2, 3, 4)

    # 对版本进行聚类排序
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    linkage_matrix = linkage(similarity_matrix, method='average')
    dendrogram(linkage_matrix, ax=ax4, labels=[f'{v}' for v in version_names])
    ax4.set_title('版本聚类树状图', fontsize=12, fontweight='bold')
    ax4.set_ylabel('距离')
    # 设置y轴范围为0.2到0.8
    ax4.set_ylim(0.2, 0.8)

    # 重新排序相似度矩阵
    reordered_indices = leaves_list(linkage_matrix)
    reordered_matrix = similarity_matrix[reordered_indices][:, reordered_indices]
    reordered_versions = [version_names[i] for i in reordered_indices]

    # 子图5：重新排序后的热力图
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(reordered_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1)
    ax5.set_xticks(range(len(reordered_versions)))
    ax5.set_yticks(range(len(reordered_versions)))
    ax5.set_xticklabels([f'{v}' for v in reordered_versions], rotation=45, fontsize=10)
    ax5.set_yticklabels([f'{v}' for v in reordered_versions], fontsize=10)
    ax5.set_title('聚类排序后相似度热力图', fontsize=12, fontweight='bold')

    # 在热力图上显示数值
    for i in range(len(reordered_versions)):
        for j in range(len(reordered_versions)):
            color = "white" if reordered_matrix[i, j] < 0.6 else "black"
            ax5.text(j, i, f'{reordered_matrix[i, j]:.3f}',
                     ha="center", va="center", color=color, fontsize=8, fontweight='bold')

    plt.colorbar(im5, ax=ax5, label='相似度')

    # 子图6：版本相似度分布
    ax6 = plt.subplot(2, 3, 6)

    # 提取所有非对角线的相似度值
    all_similarities = []
    for i in range(len(version_names)):
        for j in range(i + 1, len(version_names)):
            all_similarities.append(similarity_matrix[i, j])

    # 绘制分布直方图
    n, bins, patches = ax6.hist(all_similarities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax6.set_xlabel('相似度')
    ax6.set_ylabel('频次')
    ax6.set_title('版本间相似度分布', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 添加统计线
    mean_sim = np.mean(all_similarities)
    median_sim = np.median(all_similarities)
    ax6.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_sim:.3f}')
    ax6.axvline(median_sim, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_sim:.3f}')
    ax6.legend()

    plt.tight_layout()
    plt.suptitle('多版本间相似度综合分析', fontsize=16, fontweight='bold', y=0.98)

    # 安全保存图表
    save_plot(fig, 'version_comparison_analysis.png', result_dir, delay=0.5)

    # 打印版本比较统计
    print(f"\n=== 版本间相似度比较分析 ===")
    print(f"总版本数: {len(version_names)}")
    print(f"所有版本间平均相似度: {np.mean(all_similarities):.4f}")
    print(f"相似度标准差: {np.std(all_similarities):.4f}")

    # 找到最高和最低相似度的版本对
    max_sim_idx = np.argmax(all_similarities)
    min_sim_idx = np.argmin(all_similarities)

    # 计算版本对索引
    pairs = []
    for i in range(len(version_names)):
        for j in range(i + 1, len(version_names)):
            pairs.append((i, j))

    max_i, max_j = pairs[max_sim_idx]
    min_i, min_j = pairs[min_sim_idx]

    print(f"最高相似度版本对: {version_names[max_i]} vs {version_names[max_j]}: {np.max(all_similarities):.4f}")
    print(f"最低相似度版本对: {version_names[min_i]} vs {version_names[min_j]}: {np.min(all_similarities):.4f}")

    # 打印各版本平均相似度排名
    print(f"\n各版本平均相似度排名:")
    ranked_versions = sorted(zip(version_names, version_avg_similarities),
                             key=lambda x: x[1], reverse=True)
    for i, (version, avg_sim) in enumerate(ranked_versions, 1):
        print(f"  {i}. {version}: {avg_sim:.4f}")
    return linkage_matrix, reordered_versions


def visualize_overall_similarity(embeddings, G, result_dir):
    """生成不同版本的总体相似度图表 - 根据build_graph的结构修改"""
    print("\n=== 生成总体版本相似度图表 ===")

    # 提取版本节点信息
    node_list = list(G.nodes())
    node_types = []
    node_versions = []

    for node_id in node_list:
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', 'unknown')
        node_types.append(node_type)

        # 根据节点类型获取版本信息
        if node_type == 'version':
            node_versions.append(node_data.get('text', 'unknown'))
        else:
            node_versions.append(node_data.get('version', 'unknown'))

    # 只关注version节点
    version_indices = [i for i, t in enumerate(node_types) if t == 'version']

    if len(version_indices) < 2:
        print("版本节点数量不足，无法进行版本相似度分析")
        return

    # 提取版本节点的嵌入
    version_embeddings = embeddings[version_indices]
    version_names = [node_versions[i] for i in version_indices]

    # 计算相似度矩阵
    similarity_matrix = calculate_similarity_matrix(version_embeddings)

    # 创建总体相似度图表
    fig = plt.figure(figsize=(12, 10))

    # 绘制热力图
    im = plt.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1)

    # 设置坐标轴标签
    plt.xticks(range(len(version_names)), [f'{v}' for v in version_names], rotation=45, fontsize=12)
    plt.yticks(range(len(version_names)), [f'{v}' for v in version_names], fontsize=12)

    # 在热力图上显示数值
    for i in range(len(version_names)):
        for j in range(len(version_names)):
            color = "white" if similarity_matrix[i, j] < 0.6 else "black"
            plt.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                     ha="center", va="center", color=color, fontsize=11, fontweight='bold')

    plt.title('多版本总体相似度矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(im, label='相似度')

    # 添加网格线
    plt.grid(False)
    for i in range(len(version_names) + 1):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5)
        plt.axvline(i - 0.5, color='gray', linewidth=0.5)

    plt.tight_layout()

    # 安全保存图表
    save_plot(fig, 'overall_similarity.png', result_dir, delay=0.3)

    # 打印版本相似度统计
    print(f"\n=== 版本相似度详细分析 ===")
    for i in range(len(version_names)):
        for j in range(i + 1, len(version_names)):
            sim = similarity_matrix[i, j]
            similarity_level = "极高" if sim > 0.8 else "较高" if sim > 0.6 else "中等" if sim > 0.4 else "较低"
            print(f"🔸 {version_names[i]} vs {version_names[j]}: {sim:.4f} ({similarity_level}相似)")


# 添加比较聚类结果的函数
def compare_clustering_results(chapter_clusters, overall_clustering, result_dir):
    """比较每个章节的聚类结果与总体版本聚类结果的差异

    参数:
        chapter_clusters: 字典，键为章节号，值为(linkage_matrix, reordered_versions)元组
        overall_clustering: (linkage_matrix, reordered_versions)元组，表示总体版本聚类结果
    """
    print("\n" + "=" * 60)
    print("章节聚类与总体版本聚类结果比较分析")
    print("=" * 60)

    if not overall_clustering or overall_clustering[0] is None:
        print("警告: 总体版本聚类结果不可用")
        return

    # 获取总体聚类的版本顺序
    _, overall_versions_order = overall_clustering

    # 创建总体版本顺序的排名字典
    version_rank_overall = {version: rank for rank, version in enumerate(overall_versions_order)}
    # 存储每个章节的Spearman系数结果
    chapter_correlations = []
    # 对每个章节的聚类结果进行比较
    for chapter_num, (linkage_matrix, chapter_versions_order) in chapter_clusters.items():
        if linkage_matrix is None or chapter_versions_order is None:
            print(f"章节 {chapter_num}: 聚类结果不可用")
            continue

        # 创建章节版本顺序的排名字典
        version_rank_chapter = {version: rank for rank, version in enumerate(chapter_versions_order)}

        # 找出在总体聚类和章节聚类中都存在的版本
        common_versions = list(set(overall_versions_order) & set(chapter_versions_order))

        if len(common_versions) < 2:
            print(f"章节 {chapter_num}: 共同版本数量不足，无法进行比较")
            continue

        # 为共同版本创建排名列表
        overall_ranks = [version_rank_overall[v] for v in common_versions]
        chapter_ranks = [version_rank_chapter[v] for v in common_versions]

        # 计算Spearman相关系数作为相似度指标
        correlation, p_value = spearmanr(overall_ranks, chapter_ranks)
        # 存储结果用于后续排序和保存
        chapter_correlations.append((chapter_num, correlation, p_value, len(common_versions)))
        # 解释相关性强度
        if correlation > 0.7:
            similarity_level = "极高相似度"
        elif correlation > 0.5:
            similarity_level = "较高相似度"
        elif correlation > 0.3:
            similarity_level = "中等相似度"
        else:
            similarity_level = "较低相似度"

        print(f"章节 {chapter_num} 与总体聚类比较结果:")
        print(f"  Spearman相关系数: {correlation:.4f} ({similarity_level})")
        print(f"  p值: {p_value:.4f}")
        print(f"  共同版本数: {len(common_versions)}")
    # 如果提供了保存目录，将结果按Spearman系数升序排列并保存到文件
    if result_dir:
        # 按Spearman系数升序排列
        chapter_correlations.sort(key=lambda x: x[1])

        # 创建保存目录（如果不存在）
        os.makedirs(result_dir, exist_ok=True)

        # 写入文件
        output_path = os.path.join(result_dir, 'clustering_comparison_results.txt')

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + '\n')
                f.write("章节聚类与总体版本聚类结果比较分析（按Spearman系数升序排列）\n")
                f.write("=" * 80 + '\n')
                f.write("{:<10} {:<15} {:<15} {:<15}\n".format("章节号", "Spearman系数", "p值", "共同版本数"))
                f.write("-" * 80 + '\n')

                for chapter_num, correlation, p_value, common_count in chapter_correlations:
                    # 解释相关性强度
                    if correlation > 0.7:
                        similarity_level = "极高相似度"
                    elif correlation > 0.5:
                        similarity_level = "较高相似度"
                    elif correlation > 0.3:
                        similarity_level = "中等相似度"
                    else:
                        similarity_level = "较低相似度"

                    f.write("{:<10} {:<15.4f} {:<15.4f} {:<15} {}\n".format(
                        chapter_num, correlation, p_value, common_count, similarity_level))

                f.write("=" * 80 + '\n')

            print(f"聚类比较结果已保存至: {output_path}")

        except Exception as e:
            print(f"保存聚类比较结果时出错: {str(e)}")

def save_plot(fig, filename, result_dir='similarity_plots', delay=0.3):
    """保存图表"""
    # 创建保存目录
    save_dir = os.path.join(os.getcwd(), result_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        # 保存图表
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已安全保存: {filepath}")

    except Exception as e:
        print(f"❌ 保存图表时出错: {e}")
        return None

    # 关闭图表以释放内存
    plt.close(fig)

    # 添加延迟
    time.sleep(delay)
    return filepath