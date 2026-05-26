# 第六阶段：交互式可视化展示系统

这是基于第五阶段输出构建的本地 Web 可视化系统。它包含一个轻量 Python 后端和一个原生 Web 前端。

## 运行方式

在 `mora` 环境中，从 `D:\The_Mora\vgae` 运行：

```powershell
conda activate mora
cd D:\The_Mora\vgae
python .\viz_system\server.py
```

默认地址：

```text
http://127.0.0.1:8066
```

如果端口被占用：

```powershell
python .\viz_system\server.py --port 8070
```

## 功能

- 版本社区图
  - 展示版本节点和版本相似边。
  - 节点颜色表示 Girvan-Newman 社区。
  - 边宽表示版本相似度。
  - 悬停高亮关联边，点击固定节点。

- 章节查询
  - 输入章节号或用下拉框选择章节。
  - 展示该章中 12 个版本的章节社区关系。
  - 右侧展示所有版本的句子文本。

- 亲缘矩阵
  - 展示 `book_affinity_matrix.csv` 的书籍级亲缘矩阵。

- 版本相似度排序
  - 左侧展示 `version_pair_similarity_ranking.csv` 前若干名。
  - 鼠标悬停排序项时，高亮对应版本关系。

## 后端接口

- `GET /api/overview`
- `GET /api/version-graph?limit=42`
- `GET /api/version-ranking?limit=66`
- `GET /api/chapters`
- `GET /api/chapter/{chapter_number}`
- `GET /api/book-affinity`

## 数据来源

系统读取以下已有文件：

- `mora_v4.1_0406.json`
- `gvnm/output/version_pair_similarity_ranking.csv`
- `gvnm/output/version_communities.json`
- `gvnm/output/chapter_affinity_edges.csv`
- `gvnm/output/chapter_community_membership.csv`
- `gvnm/output/book_affinity_matrix.csv`
- `gvnm/output/community_detection_summary.json`

## 设计说明

当前 `mora` 环境未安装 FastAPI / Flask。为了保证项目开箱即用，后端使用 Python 标准库 `http.server` 实现 REST 风格接口和静态资源服务。接口边界已经按后续迁移 FastAPI 的方式组织。
