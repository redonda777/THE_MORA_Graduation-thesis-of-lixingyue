# 第五阶段：Girvan-Newman 社区发现与亲缘分析

本目录用于在第四阶段 VGAE 输出基础上运行社区发现。脚本把两个任务分开：

1. 版本两两相似度排序：回答“谁和谁第一相似、第二相似”。
2. Girvan-Newman 社区发现：回答“哪些版本聚成同一组”。

## 运行方式

在 `mora` 环境和 `D:\The_Mora\vgae` 目录下运行：

```powershell
conda activate mora
cd D:\The_Mora\vgae
python .\gvnm\community_detection.py
```

如果只想快速生成版本排序和版本社区，不跑章节级分析：

```powershell
python .\gvnm\community_detection.py --skip-chapter
```

## 主要输出

默认输出目录：

```text
D:\The_Mora\vgae\gvnm\output
```

核心文件：

- `version_pair_similarity_ranking.csv`
  - 版本两两相似度排序，按 similarity 从高到低排列。
- `version_communities.json`
  - 版本级 Girvan-Newman 社区。
- `version_community_membership.csv`
  - 每个版本所属社区。
- `chapter_affinity_edges.csv`
  - 每一章内部版本对的聚合亲缘边。
- `chapter_communities.json`
  - 每一章内部 12 个版本的社区划分。
- `chapter_community_membership.csv`
  - 每章每个版本所属社区。
- `book_affinity_matrix.csv`
  - 由章节/句子相似边汇总出的书籍级亲缘矩阵。
- `community_detection_summary.json`
  - 本次运行摘要和前 20 个版本相似对。

## 带权 Girvan-Newman 说明

NetworkX 的带权最短路通常把 `weight` 当作距离/代价，而版本矩阵里的是相似度。因此脚本中保留两种边属性：

- `similarity` / `weight`：用于 modularity 和结果解释。
- `distance = 1 - similarity`：用于带权 edge betweenness。

Girvan-Newman 删除的是带权边介数最高的边，边介数计算使用 `distance`。

## 推荐入口

当前 VGAE 结果更适合从综合最佳版本开始：

```text
version_similarity_matrix.csv
```

如果要尝试句子最佳版本矩阵，可以运行：

```powershell
python .\gvnm\community_detection.py `
  --version-matrix .\sentence_best_version_similarity_matrix.csv `
  --output-dir .\gvnm\output_sentence_best
```

## 常用参数

- `--version-threshold 0.6`
  - 版本图保留的最低相似度。
- `--version-top-k 3`
  - 每个版本至少保留 top-k 近邻边。
- `--chapter-threshold 0.35`
  - 每章内部版本图的最低聚合相似度。
- `--chapter-top-k 4`
  - 每章每个版本至少保留 top-k 近邻边。
- `--chapter-aggregation mean`
  - 章节内句子相似边聚合方式，可选 `mean`、`max`、`top3_mean`。
- `--sentence-similarity-transform linear`
  - 将 `normalized_distance` 转成相似度的方式。默认 `linear` 使用 `1 - distance`，更适合排序；`exp` 使用 `exp(-distance / tau)`，会更强调完全相同句子。
