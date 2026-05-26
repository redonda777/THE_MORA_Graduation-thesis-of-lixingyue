# VGAE-codex 代码归档记录

## 目录

1. VGAE 初始训练脚本归档
2. 2026-05-02 改进归档：修复向量坍缩并加入自动超参数迭代
3. 2026-05-02 二次修复归档：修复 NaN、KL 压缩和初始特征过强
4. 2026-05-02 三次修复归档：修正 KL 退火并拆分重构损失
5. 2026-05-02 四次修复归档：同时保存综合最佳和句子最佳 trial
6. 2026-05-02 五次修复归档：新增章节节点和版本节点聚合图
7. 2026-05-02 第五阶段前评估归档：社区发现可行性判断
8. 2026-05-02 第五阶段代码归档：Girvan-Newman 社区发现实现
9. 2026-05-02 第六阶段代码归档：交互式可视化展示系统
10. 2026-05-02 第六阶段补充归档：动态相似度排行
11. 2026-05-02 第六阶段补充归档：版本图保底连边与矩阵缺失值区分
12. 2026-05-02 第六阶段补充归档：无重叠章节版本对改用 VGAE 补全
13. 2026-05-02 第六阶段补充归档：确认无共同章节后恢复 0 值并加说明
14. 2026-05-02 第六阶段补充归档：对读区双重索引修正
15. 2026-05-02 第六阶段补充归档：对读区编号改为全章句位编号
16. 2026-05-02 第六阶段补充归档：高亮子图、层级颜色与非相关节点保留
17. 2026-05-02 第六阶段补充归档：章节视图加入全书句子搜索与自动跳章高亮
18. 2026-05-02 第六阶段补充归档：章节查询骨架节点从 version 修正为 chapter
19. 2026-05-02 第六阶段补充归档：六项交互与展示问题修正
20. 2026-05-02 第六阶段补充归档：空白点击清除与 site3 视觉细化
21. 2026-05-02 第六阶段补充归档：高亮改为点击触发并贴近 site3 节点样式
22. 2026-05-02 第六阶段补充归档：章节图星团力导向布局
23. 2026-05-02 第六阶段补充归档：章节图加入句子节点与 normalized_distance 边
24. 2026-05-02 第六阶段补充归档：点击聚焦联动高亮动画
25. 2026-05-02 第六阶段补充归档：版本图加入点击聚合动画
26. 2026-05-03 第六阶段补充归档：修复节点点击后的文本光标闪烁
27. 2026-05-03 第六阶段补充归档：矩阵视图隐藏右侧对读栏
28. 2026-05-03 第六阶段补充归档：矩阵视图彻底移除右侧 aside

归档时间：2026-05-01  
工作目录：`D:\The_Mora\vgae`  
交付脚本：`vgae_training.py`

## 任务目标

本次代码实现第四阶段“图节点向量学习（Graph Representation Learning）”，用于基于异构统一图对《道德经》多版本文本进行无监督节点嵌入学习。脚本融合两类数据源：

- `mora_v4.1_0406.json`：版本-章节-句子的层级树结构。
- `total_formal_all_sentence_adjusted_distance_aggressive_llm.json`：句子-句子的修正编辑距离相似性边。

最终通过 VGAE（变分图自编码器）学习版本节点、章节节点、句子节点的统一 128 维向量表示，并输出向量、版本相似度矩阵和 t-SNE 可视化。

## 新增文件

### `vgae_training.py`

单文件训练脚本，运行方式：

```powershell
python vgae_training.py
```

脚本内所有路径、训练参数和剪枝参数均集中在 `CONFIG` 字典中配置。

## 核心实现内容

### 1. 节点编码

脚本从 `mora_v4.1_0406.json` 中遍历根节点、版本节点、章节节点和句子节点，为三类节点分别创建全局唯一 ID 与索引：

- 版本节点：`version_{name}`，如 `version_hj`
- 章节节点：`chapter_{version}_{chapter_number}`，如 `chapter_hj_0`
- 句子节点：`sent_{version}_{chapter}_{sentence}`，如 `sent_hj_0_0`

同时保存节点元信息：

- `type`
- `version`
- `chapter`
- `sentence`
- `text`
- `sentence_count` 等辅助字段

### 2. 层级边构建

脚本保留全部层级边，不参与剪枝：

- 版本 -> 章节，权重 `1.0`
- 章节 -> 句子，权重 `1.0`

后续统一通过 PyG 的 `to_undirected(..., reduce="mean")` 转为无向传播结构。

### 3. 相似性边剪枝

脚本仅对句子-句子相似性边执行剪枝，层级边不剪枝。

相似性边使用 `normalized_distance` 分层处理：

- L0：`d == 0`，全部保留，权重 `10.0`
- L1：`0 < d <= 0.1`，全部保留，权重 `exp(-d / 0.05)`
- L2：`0.1 < d <= 0.2`，按每个句子节点保留 top-3 邻居，权重 `exp(-d / 0.05)`
- L3：`d > 0.2`，全部删除

兜底机制：

- 剪枝后仅检查句子-句子相似性边度数，层级边不计入。
- 若句子节点相似性边度数小于 3，补回该节点候选集中距离最小的边，权重 `0.01`。
- 若节点没有足够候选边，则添加自环，权重 `0.01`。

### 4. PyTorch Geometric 图构建

脚本将层级边和剪枝后的相似性边合并，构建 PyG `Data` 对象：

- `edge_index`
- `edge_weight`
- `x`
- `num_nodes`

节点特征采用统一随机初始化：

- 输入维度：256
- 类型：`torch.nn.Parameter`
- 训练过程中可学习

### 5. VGAE 模型

模型结构：

- 编码器：两层 `GCNConv`
- 第一层：`input_dim -> hidden_dim`
- 第二层分成两个分支：
  - `conv_mu`
  - `conv_logvar`
- 隐空间维度：128

训练配置：

- epoch：400
- learning rate：0.005
- KL 权重：1.0
- 优先 CUDA，缺失 CUDA 时自动使用 CPU

损失函数：

- 重构损失：基于正边和负采样边的 BCE 形式
- KL 散度：标准 VGAE KL 项
- 总损失：`recon_loss + kl_weight * kl_loss`

### 6. 分层评估

脚本实现三类评估：

- 句子节点：按章节编号计算 Silhouette Score，并生成 t-SNE 可视化。
- 章节节点：按同名章节编号计算 Silhouette Score。
- 版本节点：计算版本-版本余弦相似度矩阵。

### 7. 输出文件

训练完成后输出：

- `vgae_output.pt`
  - `model_state_dict`
  - `node_embeddings`
  - `node_id_map`
  - `node_meta`
  - `config`
  - `graph_stats`
  - `pruning_stats`
  - `eval_stats`
  - `training_history`
- `node_vectors.csv`
  - `node_id`
  - `type`
  - `version`
  - `chapter`
  - `sentence`
  - `dim_0 ... dim_127`
- `version_similarity_matrix.csv`
  - 版本名称 x 版本名称的余弦相似度矩阵
- `tsne_visualization.png`
  - 句子节点 t-SNE 图，按章节着色
- `missing_similarity_edges.json`
  - 若相似性边中的句子节点不在树图中，记录样例和统计

## 已完成验证

### 语法检查

执行：

```powershell
python -m py_compile .\vgae_training.py
```

结果：通过。

### 数据加载与剪枝轻量验证

在不依赖 PyTorch 的情况下，单独验证了树加载、相似性边映射和剪枝流程。

验证结果：

- 总节点数：9336
- 版本节点：12
- 章节节点：599
- 句子节点：8725
- 层级边：9324
- 原始相似性边：28550
- 成功映射的唯一句子对：28550
- 缺失节点边：0
- 剪枝后相似性边：23578
- 兜底补边：1968
- 兜底自环：150

剪枝前分层统计：

- L0：7603
- L1：11923
- L2：2113
- L3：6911

剪枝后关键统计：

- L0 保留：7603
- L1 保留：11923
- L2 top-3 union 保留：2084
- L3 删除：6911
- 最终相似性边：23578

## 当前环境阻塞

当前本机 Python 环境尚不能完整训练，原因如下：

- 未安装 `torch`
- 未安装 `torch_geometric`
- 当前 `numpy==2.4.4` 与已安装的 `sklearn`、`matplotlib`、`scipy` 二进制包 ABI 不兼容

实际运行 `python vgae_training.py` 时，脚本会在依赖检查阶段停止并输出明确错误提示。修复依赖后，可直接运行完整训练流程。

## 后续建议

建议为该实验单独创建 Python 3.10 环境，并安装版本兼容的依赖组合：

```powershell
pip install torch torch-geometric scikit-learn matplotlib numpy
```

若使用 CUDA，应按本机 CUDA 版本安装对应的 PyTorch 与 PyG wheel。依赖修复后再次运行：

```powershell
python vgae_training.py
```

即可生成 `vgae_output.pt`、`node_vectors.csv`、`version_similarity_matrix.csv` 和 `tsne_visualization.png`。

---

# 2026-05-02 改进归档：修复向量坍缩并加入自动超参数迭代

## 改进背景

上一版脚本虽然可以完成 VGAE 训练流程，但训练结果出现明显坍缩：

- 同章句子轮廓系数为负。
- 同名章节轮廓系数为负。
- 128 维节点向量全局标准差过小。
- 句子向量均值出现异常偏移。

诊断原因集中在三点：

- 层级边权重过强，亲缘相似性边信号被层级结构淹没。
- 节点初始特征过于同质化，版本、章节、句子节点没有足够区分。
- GCN 消息传递缺少残差保护，容易过度平滑。

## 已完成的五项核心修复

### 1. 层级边权重降级

新增配置：

```python
"hierarchy_weight": 0.2
```

`load_tree_graph` 中版本-章节、章节-句子层级边不再使用硬编码 `1.0`，改为读取配置权重。自动 trial 中还会根据 trial 序号继续衰减该权重，最低保留到 `0.05`。

### 2. 节点特征按类型区分初始化

新增 `init_node_features(...)`，替代原来的统一小随机初始化。

初始化策略：

- 版本节点：较大随机范围，并按版本索引加入偏移。
- 章节节点：中等随机范围，并按章节编号加入偏移。
- 句子节点：较小随机范围，让相似性边主导细粒度学习。

该改动用于打破节点特征同质化，降低所有节点被卷积平均到同一位置的风险。

### 3. 三层 GCN + 残差连接

编码器从原来的单个中间 GCN 层扩展为：

- `conv1`
- `conv2`
- `conv_mu`
- `conv_logvar`

新增 `conv2` 后，在第二层后加入残差连接：

```python
h2 = h2 + h1
```

同时新增配置：

```python
"dropout": 0.3
```

用于中间层 dropout，缓解过拟合与过度平滑。

### 4. KL 散度退火

新增 `get_kl_weight(...)`：

- 前 100 轮：KL 权重为 `0.0`
- 第 100 到 300 轮：线性增加到 `1.0`
- 第 300 轮后：保持 `1.0`

新增配置：

```python
"kl_anneal_epochs": 100,
"kl_full_epochs": 300
```

这样模型前期先拟合图结构，后期再逐步接受变分正则约束。

### 5. 相似性边剪枝收紧

新增配置：

```python
"layer3_threshold": 0.15,
"l2_top_k": 2
```

剪枝变化：

- L2 区间从 `0.1 < d <= 0.2` 收紧为 `0.1 < d <= 0.15`
- L3 改为 `d > 0.15`
- L2 top-k 从 3 改为 2
- L2 排序显式改为按 `distance` 升序，即优先保留距离小、权重大的边

## 新增自动超参数迭代机制

新增函数：

- `make_trial_config(...)`
- `train_single_trial(...)`
- `auto_train_loop(...)`
- `evaluate_metrics_only(...)`
- `save_best_artifacts(...)`

自动迭代逻辑：

- 最大 trial 次数：10
- 每次 trial 使用独立 seed
- trial 越靠后，层级边权重和学习率按 `0.7 ** trial` 衰减
- 每个 trial 都重新剪枝、重新构图、重新训练
- 任一 trial 报错不会中断总流程，会写入日志并继续下一次
- 每个 trial 结束主动执行 `gc.collect()`
- 若 CUDA 可用，执行 `torch.cuda.empty_cache()`
- 单 trial 训练超过 1800 秒会在 epoch 边界抛出 `TimeoutError`

收敛标准：

- 同章句子轮廓系数 > 0.15
- 同名章节轮廓系数 > 0.10
- 向量标准差 > 0.10

一旦达到标准，自动提前停止。

## 新增输出

除原有输出外，新增：

- `training_log.json`
  - 每个 trial 的配置、指标、剪枝统计、loss 历史和错误信息。
- `best_config.json`
  - 当前最佳 trial 的最终配置。

每当出现新的最佳 trial，会立即覆盖保存：

- `vgae_output.pt`
- `node_vectors.csv`
- `best_config.json`

主流程结束后，再基于最佳 trial 生成：

- `version_similarity_matrix.csv`
- `tsne_visualization.png`

## 轻量验证结果

执行了语法检查：

```powershell
python -m py_compile .\vgae_training.py
```

结果：通过。

在不依赖 PyTorch 的情况下，重新验证树加载、相似性边映射和剪枝流程。

验证结果：

- 总节点数：9336
- 层级边首条权重：0.2
- 原始相似性边：28550
- 成功映射的唯一句子对：28550
- 缺失节点边：0

剪枝前分层统计：

- L0：7603
- L1：11923
- L2：556
- L3：8468

剪枝后关键统计：

- L0 保留：7603
- L1 保留：11923
- L2 top-2 union 保留：544
- L3 删除：8468
- 兜底补边：2636
- 兜底自环：150
- 最终相似性边：22706

## 当前交付状态

`vgae_training.py` 已覆盖更新为自动调参版本。运行方式保持不变：

```powershell
python vgae_training.py
```

---

# 2026-05-02 二次修复归档：修复 NaN、KL 压缩和初始特征过强

## 问题复盘

用户在 `(mora)` 环境中完整运行上一版 `vgae_training.py` 后，10 个 trial 中有 5 个出现 `nan` / `inf`，其余 trial 虽然能完成，但最佳结果仍然很差：

- 最佳 trial：第 9 次。
- 同章句子 silhouette：`-0.2199`。
- 同名章节 silhouette：`-0.2707`。
- 向量标准差：`0.0047`。
- 多个 trial 第 1 轮 KL 已经达到极大值，部分为 `inf`。

这说明上一版的“节点类型差异化初始化 + KL 退火”没有真正解决坍缩，反而引入了新的数值不稳定。核心原因有四点：

- 版本和章节初始特征使用了沿所有维度叠加的大偏移，章节编号最高可把整行特征推到很大数值，GCN 后容易产生极端 `logvar`。
- L0 相似边权重 `10.0` 对 GCN 消息传播过强，容易放大节点表示。
- `kl_weight` 最终升到 `1.0`，对当前亲缘表示任务过强，会把后验重新压回标准正态，造成向量标准差过小。
- reconstruction loss 使用 `sigmoid + log`，没有 logits 形式稳定；训练中也缺少 `logvar` clamp、非有限 loss 检查和梯度裁剪。

## 本轮代码修改

### 1. 降低默认训练强度

更新 `CONFIG`：

```python
"learning_rate": 0.0015
"kl_weight": 0.001
"hierarchy_weight": 0.05
"dropout": 0.15
"kl_anneal_epochs": 40
"kl_full_epochs": 160
"embedding_std_target": 0.02
```

目标是让模型先稳定学结构，KL 只作为轻量正则，不再主导压缩隐空间。

### 2. 降低 L0 边和 encoder 边权重上限

新增：

```python
"l0_similarity_weight": 2.0
"max_encoder_edge_weight": 2.0
```

`similarity_weight(...)` 中完全相同句子的权重从固定 `10.0` 改为配置项 `2.0`。构建 PyG Data 后，对传入 GCN 的 `edge_weight` 执行 clamp，避免少数高权重边让卷积输出爆炸。

### 3. 重写节点初始特征

`init_node_features(...)` 不再用“章节编号乘以常数并加到所有维度”的方式。新策略为：

- 全体节点只有小随机噪声。
- 前几个维度编码节点类型 one-hot。
- 少数维度编码版本顺序、章节位置、句子位置的有界归一化坐标。
- 使用 `sin/cos` 周期特征表达位置，所有结构特征都限制在小范围内。

这样既保留类型和层级信号，又不会制造极端初值。

### 4. 稳定 VGAE 数值

模型部分新增：

- `logvar` clamp：默认限制到 `[-6.0, 2.0]`。
- 默认用 `mu` 做 reconstruction，避免训练早期采样噪声放大。
- reconstruction loss 改为 `binary_cross_entropy_with_logits`。
- 训练循环加入非有限 loss 检查，一旦出现 `nan/inf` 立即停止该 trial。
- 加入梯度裁剪：

```python
"gradient_clip_norm": 1.0
"weight_decay": 1e-4
```

### 5. 调整自动 trial 网格

`make_trial_config(...)` 不再用简单的 `0.7 ** trial` 同时衰减层级权重和学习率，而是使用显式网格：

- 层级权重：`0.02` 到 `0.10`
- 学习率：`0.0015` 到 `0.0005`
- KL 权重：`0.001` 到 `0.0001`

这样每个 trial 都是明确的稳定配置组合，不会在后半段重复跑同一个容易 NaN 的配置。

### 6. 调整 best trial 打分

`trial_combined_score(...)` 加入 embedding 标准差奖励和坍缩惩罚：

- `std < 0.01` 或 `l2_mean < 0.1` 会扣分。
- 标准差较健康的模型会得到小幅加分。

避免再把 silhouette “没那么负”但向量已经坍缩的 trial 选为最佳。

## 验证结果

### 语法检查

执行：

```powershell
python -m py_compile .\vgae_training.py
```

结果：通过。

### `(mora)` 环境 2 epoch 冒烟训练

使用解释器：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe
```

结果：

- 第 1 轮：`loss=1.4326`，`recon=1.4326`，`kl=0.8750`。
- 第 2 轮：`loss=1.2951`，`recon=1.2951`，`kl=0.5859`。
- 向量标准差：`0.0584`。
- `l2_mean`：`0.6487`。
- 未出现 `nan/inf`。

### `(mora)` 环境 20 epoch 轻量训练

结果：

- 第 20 轮：`loss=0.9912`，`recon=0.9912`，`kl=1.0668`。
- 同章句子 silhouette：`-0.1272`。
- 同名章节 silhouette：`0.1273`。
- 向量标准差：`0.1180`。
- `l2_mean`：`1.3142`。
- 未出现 `nan/inf`。

### `(mora)` 环境 80 epoch 退火验证

结果：

- 第 80 轮：`loss=1.1823`，`recon=0.9611`，`kl=0.6637`，`kl_w=0.3333`。
- 同章句子 silhouette：`0.0105`。
- 同名章节 silhouette：`0.3134`。
- 向量标准差：`0.0985`。
- `l2_mean`：`1.0936`。
- KL 开始退火后仍未出现向量坍缩。

## 当前交付状态

`vgae_training.py` 已完成二次修复。旧的 `node_vectors.csv`、`vgae_output.pt`、`best_config.json`、`training_log.json`、`version_similarity_matrix.csv` 和 `tsne_visualization.png` 仍然是上一轮完整训练生成的旧结果；本轮只进行了不覆盖主输出的轻量验证。下一次运行：

```powershell
python vgae_training.py
```

会用新的稳定配置重新生成完整输出并覆盖上述文件。

---

# 2026-05-02 三次修复归档：修正 KL 退火并拆分重构损失

## 问题复盘

用户用二次修复后的脚本重新完整训练，结果已经明显稳定：

- 8 个 trial 全部完成，没有 `nan/inf`。
- 最佳 trial 为第 7 次。
- 同章句子 silhouette：`-0.0190`。
- 同名章节 silhouette：`0.2829`。
- 向量标准差：`0.1522`。

该结果说明数值稳定性和章节级表示已经可用，但仍不是理想状态。继续检查训练日志后发现一个关键问题：KL 退火函数在第 40 到 160 轮之间返回的是 `0.0 -> 1.0` 的进度值，而不是 `0.0 -> kl_weight`。因此日志中出现：

```text
Epoch 0060 | kl_w=0.167
Epoch 0080 | kl_w=0.333
Epoch 0100 | kl_w=0.500
Epoch 0140 | kl_w=0.833
Epoch 0160 | kl_w=0.001
```

这会导致中段 KL 正则突然过强，随后又掉回很小的目标权重，训练轨迹被人为折断。

同时，上一版 reconstruction loss 仍然把所有传播边作为同一种正边重构，层级边和句子相似边没有区别。对于本任务，句子-句子的相似性边应当是主要学习信号，版本-章节-句子的层级边更适合作为辅助结构约束。

## 本轮代码修改

### 1. 修正 KL 退火函数

`get_kl_weight(...)` 改为先读取目标权重：

```python
target = float(config.get("kl_weight", 1.0))
```

然后在退火区间返回：

```python
target * progress
```

这样 `kl_weight=0.001` 时，退火过程只会从 `0.0` 平滑升到 `0.001`，不会再中途冲到 `0.833`。

### 2. 在 PyG Data 中保留边类型索引

`build_pyg_data_with_meta(...)` 新增：

- `data.hierarchy_edge_index`
- `data.similarity_edge_index`

GCN 编码器仍然使用合并后的 `data.edge_index` 和 `data.edge_weight` 做消息传播；但 reconstruction loss 不再只看合并后的边，而是按边类型分别计算。

### 3. 拆分 reconstruction loss

新增配置：

```python
"similarity_recon_weight": 1.0
"hierarchy_recon_weight": 0.25
```

训练循环中改为：

```python
similarity_recon = model.recon_loss(...)
hierarchy_recon = model.recon_loss(...)
recon = similarity_weight * similarity_recon + hierarchy_weight * hierarchy_recon
```

负采样仍使用完整图边作为排除集合，避免把已有的另一类边误采成负例。

### 4. 训练日志增强

每次记录 history 时新增：

- `similarity_recon_loss`
- `hierarchy_recon_loss`

终端日志也会显示：

```text
loss=... recon=... sim=... hier=... kl=... kl_w=...
```

`kl_w` 改为 6 位小数输出，便于看清 `0.0002`、`0.0005`、`0.0010` 这类轻量正则。

### 5. 自动 trial 网格调整

`make_trial_config(...)` 的 trial grid 增加 `hierarchy_recon_weight`，让不同 trial 同时搜索：

- GCN 传播层级权重 `hierarchy_weight`
- 学习率 `learning_rate`
- KL 目标权重 `kl_weight`
- 层级重构辅助权重 `hierarchy_recon_weight`

层级重构权重范围为 `0.10` 到 `0.25`，相似边重构保持主项权重 `1.0`。

## 验证结果

### 语法检查

执行：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\vgae_training.py
python -m py_compile .\vgae_training.py
```

结果：均通过。

### 80 epoch 验证

使用 `(mora)` 环境的 Python 执行单 trial，不覆盖正式输出。

关键日志：

```text
Epoch 0001 | loss=1.7725 recon=1.7725 sim=1.3855 hier=1.5480 kl=0.8750 kl_w=0.000000
Epoch 0080 | loss=1.0520 recon=1.0515 sim=0.8298 hier=0.8869 kl=1.5644 kl_w=0.000333
```

指标：

- 同章句子 silhouette：`-0.0613`
- 同名章节 silhouette：`0.2692`
- 向量标准差：`0.1528`
- `l2_mean`：`1.7115`

验证重点：第 80 轮 KL 权重已正确变为 `0.000333`，不再是上一版的 `0.333`。

### 400 epoch 单 trial 验证

使用第一组 trial 配置跑满 400 epoch，不覆盖正式输出。

关键日志：

```text
Epoch 0001 | loss=1.7725 recon=1.7725 sim=1.3855 hier=1.5480 kl=0.8750 kl_w=0.000000
Epoch 0100 | loss=1.0382 recon=1.0374 sim=0.8202 hier=0.8689 kl=1.6198 kl_w=0.000500
Epoch 0200 | loss=1.0094 recon=1.0076 sim=0.8002 hier=0.8297 kl=1.7604 kl_w=0.001000
Epoch 0400 | loss=0.9902 recon=0.9883 sim=0.7862 hier=0.8082 kl=1.8812 kl_w=0.001000
```

指标：

- 同章句子 silhouette：`-0.0106`
- 同名章节 silhouette：`0.1401`
- 向量标准差：`0.1688`
- `l2_mean`：`1.8913`

验证重点：

- KL 权重全程最高只到 `0.001`。
- 相似边重构损失从 `1.3855` 降到 `0.7862`。
- 未出现 `nan/inf`。

## 当前交付状态

`vgae_training.py` 已完成三次修复。旧的正式输出文件仍是用户上一轮完整训练的结果；本轮验证只通过脚本内函数执行训练，没有调用保存最佳模型流程，因此没有覆盖 `node_vectors.csv`、`vgae_output.pt`、`best_config.json`、`training_log.json`、`version_similarity_matrix.csv` 和 `tsne_visualization.png`。

下一次在 `(mora)` 环境、`D:\The_Mora\vgae` 目录下运行：

```powershell
python vgae_training.py
```

会使用修正后的 KL 退火与拆分 reconstruction loss 重新进行 8 个 trial，并覆盖生成新的正式输出。

---

# 2026-05-02 四次修复归档：同时保存综合最佳和句子最佳 trial

## 问题复盘

用户用三次修复后的脚本再次完整训练，整体结果继续改善：

- 最佳综合 trial：第 6 次。
- 同章句子 silhouette：`0.0247`。
- 同名章节 silhouette：`0.4516`。
- 向量标准差：`0.1601`。
- 版本相似矩阵分辨率明显改善，`gd3` 与主高相似组已被拉开。

但日志也显示：第 3 次 trial 的句子 silhouette 更高：

- trial 3 同章句子 silhouette：`0.0755`。
- trial 3 同名章节 silhouette：`0.3487`。

原脚本只按综合分保存单一最佳结果，因此会把句子表现更好的 trial 3 覆盖掉。对于后续分析，综合最佳和句子最佳都值得保留。

## 本轮代码修改

### 1. 新增 sentence-best 输出路径

在 `CONFIG` 中新增：

```python
"sentence_best_vgae_output_path": "sentence_best_vgae_output.pt"
"sentence_best_node_vectors_csv_path": "sentence_best_node_vectors.csv"
"sentence_best_version_similarity_csv_path": "sentence_best_version_similarity_matrix.csv"
"sentence_best_tsne_png_path": "sentence_best_tsne_visualization.png"
"sentence_best_config_path": "sentence_best_config.json"
```

原有输出文件仍表示综合最佳：

- `vgae_output.pt`
- `node_vectors.csv`
- `version_similarity_matrix.csv`
- `tsne_visualization.png`
- `best_config.json`

新增文件专门保存句子优先最佳 trial。

### 2. 新增 artifact 路径切换工具

新增函数：

```python
with_artifact_paths(config, "sentence_best")
```

它会复制当前 trial 配置，并把输出路径替换为 `sentence_best_*` 文件，避免复用主输出路径。

### 3. 新增句子优先评分

新增函数：

```python
trial_sentence_priority_score(...)
```

评分逻辑：

- 以同章句子 silhouette 为主。
- 同名章节 silhouette 只作为较小辅助项。
- 健康的 embedding 标准差提供少量加分。
- 若出现向量坍缩则扣分。

公式近似为：

```python
sent + 0.25 * chap + 0.25 * min(std, 0.2) - collapse_penalty
```

这样 trial 3 这类句子表现更好的模型不会再被综合最佳覆盖。

### 4. 自动训练循环同时追踪两套最佳

`auto_train_loop(...)` 现在同时维护：

- `best_result` / `best_config`：综合最佳。
- `best_sentence_result` / `best_sentence_config`：句子优先最佳。

每个 trial 日志新增：

```json
"sentence_priority_score": ...
```

终端日志也会显示：

```text
combined=..., sentence_priority=...
```

出现新的 sentence-best 时，会立即保存：

- `sentence_best_node_vectors.csv`
- `sentence_best_vgae_output.pt`
- `sentence_best_config.json`

### 5. 主流程结束后补全 sentence-best 衍生输出

主流程仍会对综合最佳生成原来的版本相似矩阵和 t-SNE。随后会对 sentence-best 额外生成：

- `sentence_best_version_similarity_matrix.csv`
- `sentence_best_tsne_visualization.png`
- `sentence_best_node_vectors.csv`
- `sentence_best_vgae_output.pt`
- `sentence_best_config.json`

## 验证结果

执行：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\vgae_training.py
python -m py_compile .\vgae_training.py
```

结果：均通过。

同时检查代码中已出现以下关键项：

- `sentence_best_*` 输出路径。
- `trial_sentence_priority_score(...)`。
- `sentence_priority_score` 写入训练日志。
- `best_sentence_result` / `best_sentence_config` 返回主流程。

## 当前交付状态

`vgae_training.py` 已完成四次修复。下一次完整运行后，会同时得到两套结果：

综合最佳：

- `vgae_output.pt`
- `node_vectors.csv`
- `version_similarity_matrix.csv`
- `tsne_visualization.png`
- `best_config.json`

句子最佳：

- `sentence_best_vgae_output.pt`
- `sentence_best_node_vectors.csv`
- `sentence_best_version_similarity_matrix.csv`
- `sentence_best_tsne_visualization.png`
- `sentence_best_config.json`

运行方式不变：

```powershell
python vgae_training.py
```

---

# 2026-05-02 五次修复归档：新增章节节点和版本节点聚合图

## 改进背景

上一版脚本已经同时保存综合最佳和句子最佳 trial，但可视化仍主要是句子节点 t-SNE：

- `tsne_visualization.png`
- `sentence_best_tsne_visualization.png`

这两张图都只展示 sentence nodes，无法直观看到章节节点和版本节点在同一嵌入空间中的聚合效果。用户要求额外生成章节节点和版本节点的聚合图。

## 本轮代码修改

### 1. 新增综合最佳可视化输出路径

在 `CONFIG` 中新增：

```python
"chapter_tsne_png_path": "chapter_tsne_visualization.png"
"version_tsne_png_path": "version_tsne_visualization.png"
```

### 2. 新增句子最佳可视化输出路径

在 `CONFIG` 中新增：

```python
"sentence_best_chapter_tsne_png_path": "sentence_best_chapter_tsne_visualization.png"
"sentence_best_version_tsne_png_path": "sentence_best_version_tsne_visualization.png"
```

`with_artifact_paths(..., "sentence_best")` 也同步替换这两个路径，确保 sentence-best 的章节/版本图不会覆盖综合最佳图。

### 3. 新增通用节点类型 t-SNE 绘图函数

新增函数：

```python
save_node_type_tsne_plot(...)
```

用途：

- 可绘制章节节点 t-SNE。
- 可绘制版本节点 t-SNE。
- 版本节点图会标注版本名。
- 小样本版本节点自动使用较小 perplexity。

### 4. 扩展 `evaluate_and_save(...)`

`evaluate_and_save(...)` 现在除了原有句子节点 t-SNE 外，还会额外生成：

- 章节节点 t-SNE：按章节编号着色。
- 版本节点 t-SNE：按版本索引着色，并标注版本名。

## 已生成的新文件

基于当前已训练好的 `vgae_output.pt` 和 `sentence_best_vgae_output.pt`，直接重新执行评估绘图流程，未重跑 8 个训练 trial。

综合最佳新增：

- `chapter_tsne_visualization.png`
- `version_tsne_visualization.png`

句子最佳新增：

- `sentence_best_chapter_tsne_visualization.png`
- `sentence_best_version_tsne_visualization.png`

同时重新生成了原有：

- `tsne_visualization.png`
- `sentence_best_tsne_visualization.png`
- `version_similarity_matrix.csv`
- `sentence_best_version_similarity_matrix.csv`

## 验证结果

执行：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\vgae_training.py
python -m py_compile .\vgae_training.py
```

结果：均通过。

直接从已有模型包生成图时的指标：

综合最佳：

- 同章句子 silhouette：`0.0247`
- 同名章节 silhouette：`0.4513`

句子最佳：

- 同章句子 silhouette：`0.0760`
- 同名章节 silhouette：`0.3493`

## 当前交付状态

后续完整运行：

```powershell
python vgae_training.py
```

会自动为综合最佳和句子最佳各生成三层可视化：

- 句子节点 t-SNE。
- 章节节点 t-SNE。
- 版本节点 t-SNE。

---

# 2026-05-02 第五阶段前评估归档：社区发现可行性判断

## 用户问题

用户提出下一步要进入第五阶段“社区发现与亲缘分析（Community Detection）”，计划使用 NetworkX 的 Girvan-Newman 算法，并询问当前 VGAE 聚合程度是否足以支撑后续社区发现。如果不可以，需要说明改进方向，并归档本次讨论。

## 当前结果复核

当前已经生成两套 VGAE 结果：

### 综合最佳

配置文件：

- `best_config.json`
- 对应 `vgae_output.pt`

关键指标：

- 同章句子 silhouette：`0.0247`
- 同名章节 silhouette：`0.4513`
- 向量标准差约：`0.1601`

综合最佳版本相似矩阵具有较清晰的分辨率：

- `hj / ba / bb / wb / hs / fy` 形成高相似组。
- `gd / gd1` 明显接近。
- `gd2` 与主高相似组距离较远。
- `gd3` 被明显拉开，与主高相似组出现低相似甚至负相似。
- `yz / xr` 也呈现区别于主高相似组的中间/边缘关系。

### 句子最佳

配置文件：

- `sentence_best_config.json`
- 对应 `sentence_best_vgae_output.pt`

关键指标：

- 同章句子 silhouette：`0.0760`
- 同名章节 silhouette：`0.3493`

该结果更适合做句子层观察，但版本相似矩阵更粘连，书籍级分辨率弱于综合最佳。

## 判断结论

当前结果可以支撑第五阶段，但应限定入口层级：

### 可以开始做

1. 版本级社区发现。
2. 章节级社区发现。
3. 章节级亲缘关系映射。
4. 书籍级亲缘关系汇总。

推荐优先使用综合最佳输出：

- `vgae_output.pt`
- `node_vectors.csv`
- `version_similarity_matrix.csv`
- `chapter_tsne_visualization.png`
- `version_tsne_visualization.png`

原因是综合最佳的章节指标和版本相似矩阵更适合做亲缘关系。

### 暂不建议直接做

不建议直接在全量句子节点图上跑原始 Girvan-Newman：

- 当前图约 9336 个节点，to-undirected 后约 63910 条边，Girvan-Newman 反复计算边介数，复杂度很高。
- 句子级 silhouette 最高只有 `0.0760`，说明句子级聚合有改善但还不够强。
- 全量句子图中句子、章节、版本三类节点混合，直接社区发现容易得到结构层级社区，而不是文本亲缘社区。

## 第五阶段建议路线

### 1. 先做版本图社区发现

从 `version_similarity_matrix.csv` 构建 12 节点加权图：

- 节点：版本。
- 边权：版本余弦相似度。
- 可先过滤低相似边，例如只保留 `similarity >= 0.6` 或每个版本 top-k。
- Girvan-Newman 中将距离/代价设为：

```python
distance = 1.0 / max(similarity, epsilon)
```

或：

```python
distance = 1.0 - similarity
```

NetworkX 的 shortest path / betweenness 使用 `weight` 时通常把它当作距离，因此不能直接把 similarity 当作 weight 传给带权最短路。

### 2. 再做章节压缩图社区发现

不要把句子节点直接喂给 Girvan-Newman。建议构建章节级压缩图：

- 节点：599 个章节节点，或按同名章节聚合后的章节-版本节点。
- 边：由句子相似边聚合到章节对。
- 边权：章节之间的平均相似度、最大相似度、top-k 均值或相似句子覆盖率。
- 可叠加 VGAE 章节向量余弦相似度作为第二信号。

这样图规模小很多，也更贴近“章节级亲缘关系映射”。

### 3. 句子层只做局部验证

句子最佳输出可以用于：

- 检查某一章内部的跨版本句子社区。
- 验证章节级结论是否有句子证据支持。
- 作为可解释案例抽样。

但不建议作为全书级主社区发现入口。

## 如果要继续改进 VGAE

若第五阶段希望更依赖句子级社区，建议在进入正式社区发现前增加两个评价：

1. 相似边邻居召回率：
   - 对每个句子，检查向量空间 top-k 邻居是否覆盖原始 L0/L1 相似边。

2. 同章同句号跨版本聚合指标：
   - 比“same-chapter silhouette”更贴近数据构造，因为原始相似边本来就是同章节、同句号的跨版本比较。

如果这两个指标较好，即便 same-chapter silhouette 不高，也可以认为句子向量对亲缘分析是有效的。

## 推荐第五阶段实现策略

建议新建独立脚本：

```text
community_detection.py
```

脚本输出：

- `version_communities.json`
- `version_community_matrix.csv`
- `chapter_communities.json`
- `chapter_affinity_edges.csv`
- `book_affinity_matrix.csv`

第一版不需要直接读取全量 sentence graph，而应读取：

- `vgae_output.pt`
- `node_vectors.csv`
- `version_similarity_matrix.csv`
- 原始树结构 `mora_v4.1_0406.json`
- 相似边 JSON `total_formal_all_sentence_adjusted_distance_aggressive_llm.json`

## 最终判断

当前 VGAE 结果已经足够支撑第五阶段的“版本级 + 章节级”社区发现和亲缘分析。

但当前结果还不足以支撑“全量句子图上的 Girvan-Newman 作为主算法”。句子层应作为局部验证和案例解释，而不是主干社区发现入口。

---

# 2026-05-02 第五阶段代码归档：Girvan-Newman 社区发现实现

## 任务目标

用户要求在 `D:\The_Mora\vgae` 下新建子目录 `gvnm`，并在该目录构建第五阶段“社区发现与亲缘分析”代码，同时说明运行方式并归档。

## 新增目录

```text
D:\The_Mora\vgae\gvnm
```

目录内容：

- `community_detection.py`
- `README.md`
- `output/`

## 核心脚本

### `gvnm/community_detection.py`

该脚本实现两类任务：

1. 版本两两相似度排序。
2. Girvan-Newman 社区发现。

重要设计：

- 版本排序和社区发现分开输出。
- 版本级社区发现读取 `version_similarity_matrix.csv`。
- 章节级社区发现不直接跑 599 节点大图，而是按每章构建一个 12 版本小图。
- 章节图边权由原始句子相似 JSON 聚合得到。
- 书籍级亲缘矩阵由章节/句子相似边汇总得到。

## 带权 Girvan-Newman 适配

NetworkX 中带权最短路通常把 `weight` 当作距离/代价，但当前版本矩阵中的值是相似度。因此脚本为每条边保存：

- `similarity`：原始相似度。
- `weight`：用于 modularity 的相似度权重。
- `distance = 1 - similarity`：用于带权 edge betweenness。

Girvan-Newman 删除边时使用自定义 `most_valuable_edge(...)`：

```python
nx.edge_betweenness_centrality(current_graph, weight="distance")
```

即按带权边介数选择要删除的边。

## 输出文件

默认输出目录：

```text
D:\The_Mora\vgae\gvnm\output
```

已生成文件：

- `version_pair_similarity_ranking.csv`
  - 66 个版本对的完整相似度排名。
- `version_communities.json`
  - 版本级社区发现结果。
- `version_community_membership.csv`
  - 每个版本所属社区。
- `chapter_affinity_edges.csv`
  - 每章内部版本对聚合相似边。
- `chapter_communities.json`
  - 每章内部版本社区。
- `chapter_community_membership.csv`
  - 每章每个版本所属社区。
- `book_affinity_matrix.csv`
  - 由原始句子相似边聚合得到的书籍级亲缘矩阵。
- `community_detection_summary.json`
  - 运行摘要和前 20 个版本相似对。

## 版本相似度排序结果示例

`version_pair_similarity_ranking.csv` 前几名：

```text
1. hs - wb    0.99839509
2. bb - wb    0.99837494
3. bb - hs    0.99822336
4. fy - hj    0.99816686
5. fy - hs    0.99776489
6. bb - hj    0.99724531
7. hj - wb    0.99713671
8. hj - hs    0.99686277
9. fy - wb    0.99659729
10. bb - fy   0.99628079
```

这部分直接回答“谁和谁第一相似、第二相似”。

## 版本社区发现结果

默认参数下，版本级 Girvan-Newman 得到 2 个社区：

```json
[
  ["ba", "bb", "fy", "hj", "hs", "wb", "xr"],
  ["gd", "gd1", "gd2", "gd3", "yz"]
]
```

版本图参数：

- `version_threshold = 0.6`
- `version_top_k = 3`
- `version_max_communities = 8`
- modularity：约 `0.0429`

## 章节级处理

章节级不直接在全量句子图上跑 Girvan-Newman，而是：

1. 从 `total_formal_all_sentence_adjusted_distance_aggressive_llm.json` 读取句子相似边。
2. 用 `1 - normalized_distance` 转成句子相似度。
3. 按章节和版本对聚合。
4. 每一章构建一个最多 12 个版本节点的小图。
5. 对每章的小图运行带权 Girvan-Newman。

默认章节参数：

- `chapter_aggregation = mean`
- `sentence_similarity_transform = linear`
- `chapter_threshold = 0.35`
- `chapter_top_k = 4`
- `chapter_max_communities = 6`

本轮实际生成了 77 章的章节社区结果。

## 聚合策略修正

初版章节/书籍聚合使用 `exp(-distance / tau)` 加 `top3_mean`，会因少量完全相同句子导致 `book_affinity_matrix.csv` 过度饱和，很多版本对变成 `1.0`。

随后修正为：

- 默认 `sentence_similarity_transform = linear`，即 `1 - normalized_distance`。
- 默认 `chapter_aggregation = mean`。

修正后书籍级亲缘矩阵分辨率更合理，不再一片 `1.0`。

## 运行方式

在 `mora` 环境中，从 `D:\The_Mora\vgae` 运行：

```powershell
conda activate mora
cd D:\The_Mora\vgae
python .\gvnm\community_detection.py
```

快速只跑版本排序和版本社区：

```powershell
python .\gvnm\community_detection.py --skip-chapter
```

若想用句子最佳版本矩阵测试：

```powershell
python .\gvnm\community_detection.py `
  --version-matrix .\sentence_best_version_similarity_matrix.csv `
  --output-dir .\gvnm\output_sentence_best
```

## 验证结果

执行语法检查：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\gvnm\community_detection.py
```

结果：通过。

执行快速版本流程：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe .\gvnm\community_detection.py --skip-chapter
```

结果：

- 生成 66 个版本对排名。
- 生成版本社区。
- 输出目录：`D:\The_Mora\vgae\gvnm\output`

执行完整流程：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe .\gvnm\community_detection.py
```

结果：

- 生成版本对排名。
- 生成版本社区。
- 生成 77 章的章节社区。
- 生成章节亲缘边和书籍级亲缘矩阵。

## 当前交付状态

第五阶段第一版代码已完成，可直接运行。该版本重点完成：

- 版本相似度排序。
- 版本级 Girvan-Newman 社区发现。
- 章节级按章 Girvan-Newman 社区发现。
- 章节亲缘边存储。
- 书籍级亲缘矩阵汇总。

后续可在该脚本基础上继续加入可视化、社区稳定性评估、以及亲缘方向性解释。

---

# 2026-05-02 第六阶段代码归档：交互式可视化展示系统

## 任务目标

用户要求进入第六阶段“交互式可视化展示（Visualization System）”，并提供 `tosee_ref/site3` 作为样式参考。要求在 `D:\The_Mora\vgae` 下新建一个前后端完备项目，完成可交互的亲缘关系探索界面，并归档。

## 新增目录

```text
D:\The_Mora\vgae\viz_system
```

目录内容：

- `server.py`
- `README.md`
- `static/index.html`
- `static/styles.css`
- `static/app.js`
- `cache/`

## 技术实现

当前 `mora` 环境中未安装 FastAPI / Flask：

```text
fastapi MISSING
uvicorn MISSING
flask MISSING
```

为了保证系统可立即运行，本阶段后端使用 Python 标准库实现：

- `http.server.ThreadingHTTPServer`
- REST 风格 API
- 静态资源服务

前端使用：

- 原生 HTML / CSS / JavaScript
- SVG 图渲染
- 自实现轻量力导向布局

这样无需安装额外依赖即可运行。

## 后端接口

`viz_system/server.py` 提供：

- `GET /api/overview`
  - 返回版本、章节数、句子数、社区摘要、前 10 个版本相似对。
- `GET /api/version-graph?limit=42`
  - 返回版本节点和版本相似边。
- `GET /api/version-ranking?limit=66`
  - 返回版本两两相似度排序。
- `GET /api/chapters`
  - 返回章节列表和每章是否有社区结果。
- `GET /api/chapter/{chapter_number}`
  - 返回某章内 12 个版本节点、章节社区、版本间章节亲缘边、各版本句子文本。
- `GET /api/book-affinity`
  - 返回书籍级亲缘矩阵。

## 前端功能

前端保留 `site3` 的深色图谱风格，并扩展为三种视图：

### 1. 版本社区图

- 展示版本节点。
- 边宽表示版本相似度。
- 节点颜色表示 Girvan-Newman 社区。
- 悬停高亮关联边。
- 点击固定节点并显示关联版本排序。

### 2. 章节查询

- 支持输入章节号。
- 支持下拉选择章节。
- 展示该章 12 个版本的社区关系图。
- 右侧展示该章所有版本的句子文本。
- 节点颜色表示该章内的社区编号。

### 3. 书籍级亲缘矩阵

- 展示 `book_affinity_matrix.csv`。
- 用颜色深浅表达版本间聚合亲缘强度。

### 4. 版本相似度排序

左侧面板展示 `version_pair_similarity_ranking.csv` 的前若干项，例如：

- `hs - wb`
- `bb - wb`
- `bb - hs`
- `fy - hj`

悬停排序项会在图中高亮相关节点。

## 数据来源

系统读取以下已有文件：

- `mora_v4.1_0406.json`
- `gvnm/output/version_pair_similarity_ranking.csv`
- `gvnm/output/version_communities.json`
- `gvnm/output/chapter_affinity_edges.csv`
- `gvnm/output/chapter_community_membership.csv`
- `gvnm/output/book_affinity_matrix.csv`
- `gvnm/output/community_detection_summary.json`

## 运行方式

从 `D:\The_Mora\vgae` 运行：

```powershell
conda activate mora
cd D:\The_Mora\vgae
python .\viz_system\server.py
```

默认访问地址：

```text
http://127.0.0.1:8066
```

如端口被占用：

```powershell
python .\viz_system\server.py --port 8070
```

## 验证结果

执行语法检查：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
node --check .\viz_system\static\app.js
```

结果：均通过。

临时启动服务后验证接口：

- `GET /api/overview`：成功。
- `GET /api/version-graph?limit=5`：成功。
- `GET /api/chapter/0`：成功。
- `GET /`：成功返回页面。

## 当前交付状态

第六阶段第一版交互式可视化系统已完成。它已经可以支持：

- 版本社区探索。
- 版本相似度排序查看。
- 章节号查询。
- 章节社区高亮。
- 版本间亲缘矩阵查看。
- 句子文本对照阅读。

后续可继续加入：

- 拖拽节点固定布局。
- 导出当前视图截图。
- 对某个版本对展示逐章亲缘曲线。
- 从章节图进一步下钻到句子级证据边。

# 2026-05-02 第六阶段补充归档：动态相似度排行

## 本轮用户需求

用户在浏览器查看第六阶段可视化系统后，希望右下角/侧栏中的相似度排行不再只是固定全局榜单，而是在点亮一个版本或章节视图中的节点之后，动态显示与该版本/该章节节点相关的相似度排名。

## 修改文件

- `viz_system/static/index.html`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `viz_system/server.py`

## 主要修改

### 1. 排行标题可动态更新

为排行标题增加 `id="rankingTitle"`，前端可以根据当前上下文修改标题：

- 默认版本图：`版本相似度排序`
- 选中版本节点：`{version} 的版本相似度排行`
- 章节图默认：`第 N 章：版本相似度排行`
- 章节图选中节点：`第 N 章：{version} 的章节相似度排行`

### 2. 版本图动态排行

在 `app.js` 中新增：

- `getVersionRankingForNode(nodeId)`
- `updateContextRanking()`

当用户悬停或点击版本节点时，左侧排行会过滤为该版本相关的版本对，并按相似度从高到低重新编号。

### 3. 章节图动态排行

在 `server.py` 的 `/api/chapter/{chapter_number}` 返回值中新增：

- `ranking`

该字段返回该章节内版本对的完整相似度排行。前端在章节图中悬停或点击某个版本节点时，会显示该章节内该版本与其他版本的相似度排名，并显示 supporting sentence pair 数量。

### 4. 排行行交互保留

排行行本身仍然支持悬停高亮图中的相关边；这次修改避免了排行 hover 时反复重绘排行列表导致交互抖动的问题。

### 5. 样式补充

为排行行中的 supporting sentence pair 文本增加小号弱化样式：

```css
.rank-row small {
  grid-column: 2 / -1;
  color: var(--muted);
}
```

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
```

结果：均通过。

重启本地服务：

```powershell
python .\viz_system\server.py --port 8070
```

接口验证：

- `GET /api/overview`：200
- `GET /api/chapter/0`：200，并包含 `ranking` 字段

当前运行地址：

```text
http://127.0.0.1:8070
```

---

# 2026-05-12 第六阶段补充归档：句子节点原文双栏对读

## 本轮用户需求

用户希望在章节句子图中点亮一个句子节点后，右上角出现“对读”按钮。点击按钮后，从 `viz_system/Mora.xlsx` 抽取原文，按左右两栏展示：

- 左栏为当前点亮句子节点所属版本。
- 右栏为与该句子亲缘最近的版本。
- 只展示当前章内容。
- 点击任意一栏的句子，另一栏同一章内句位同步高亮。
- 对读文本必须来自原始 Excel，而不是图节点中的清洗文本。
- `~`、`29` 等缺文/占位标记过滤掉，不作为正文展示。

## 实现判断

当前图节点来自 `mora_v4.1_0406.json`，节点文本是清洗后的文本；`Mora.xlsx` 的结构为：

```text
seg, ln, hj, gd, gd1, gd2, gd3, ba, bb, wb, hs, yz, xr, fy
```

其中 `seg` 是 0 基章节号，`ln` 是 0 基全章句位，各版本列保存原文。因此新增原文对读层时，使用“版本 + 章节 + 句位”回到 Excel 抽取原文，并用现有句子节点索引过滤，避免显示没有对应节点的占位行。

## 修改文件

- `viz_system/server.py`
- `viz_system/static/index.html`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 后端修改

### 1. 新增轻量 XLSX 读取

`DataStore` 新增 `original_text_by_chapter`，启动时读取 `viz_system/Mora.xlsx` 的 `Sheet4`。

实现使用 Python 标准库 `zipfile` + `xml.etree.ElementTree` 解析 `.xlsx`，不引入额外依赖。读取时：

- 解析 shared strings。
- 定位 `Sheet4`。
- 读取 `seg`、`ln` 和版本列。
- 过滤空值、`~`、纯数字占位值。
- 只保留能够映射到现有句子节点的原文行。

### 2. 新增对读接口

新增：

```text
GET /api/parallel-reading?chapter={chapter}&version={version}&sentence={sentence}
```

返回：

- 当前焦点句子。
- 最近对读版本。
- 左右两栏的原文行。

最近版本优先使用当前句子节点的 `sentence_similarity_edges`，按 `normalized_distance` 从小到大选取；若该句子没有相似边，则回退到本章版本相似边中与当前版本最相近的版本。

## 前端修改

### 1. 句子详情加入“对读”按钮

`showSentenceDetail()` 的标题行改为左右布局，右侧加入 `对读` 按钮。按钮只在句子节点详情中出现。

### 2. 新增双栏对读弹层

`index.html` 新增 `parallelReader` 弹层结构；`app.js` 新增：

- `openParallelReading(node)`
- `renderParallelReading(payload)`
- `setParallelHighlight(sentence)`
- `closeParallelReader()`

对读弹层包含：

- 标题：第 N 章对读。
- 元信息：版本对与 `normalized_distance` 或章节相似度。
- 左右两列原文。
- 点击任意句位后，两列相同 `sentence` 值的行同步高亮并滚动到中部。

切换版本图、章节图或矩阵图时会自动关闭对读弹层，避免保留旧章节内容。

### 3. 样式

新增 `.parallel-reader`、`.parallel-reader-panel`、`.parallel-column`、`.parallel-row` 等样式。对读弹层采用宽屏两栏，不强塞进右侧窄栏，以保证原文可读性。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
python -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/api/parallel-reading?chapter=14&version=gd&sentence=7'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 当前页面返回 HTTP 200。
- 对读接口返回 HTTP 200。
- 示例 `gd` 第 15 章第 8 句自动选中最近版本 `gd2`，两栏均从全章第 8 句起展示原文，`29` 占位行已过滤。

当前访问地址：

```text
http://127.0.0.1:8070
```

## 2026-05-12 补充修正：对读正文改为书籍式连续排版

用户反馈双栏对读的风格方向正确，但不希望每一句单独占一行，而是像书籍正文一样排满一行后自然换行。

本轮只修改前端渲染与样式：

- `renderParallelReading()` 不再把每句渲染为块级行。
- 每句改为内联可点击片段 `.parallel-segment`。
- 句位编号改为小上标，保留定位能力但降低列表感。
- 同步高亮逻辑从 `.parallel-row` 改为 `.parallel-segment`，仍按句位联动左右栏。
- `.parallel-text-flow` 使用较大的正文字号、宽行距和自然换行，使对读区域更接近书籍排版。

验证：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/app.js'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/styles.css'
```

结果：`app.js` 语法检查通过，静态资源均返回 HTTP 200。

## 2026-05-12 补充修正：句子内部可自然换行

用户截图显示：当当前行放不下下一整个句子时，下一句会整体挪到下一行，仍不像书本排版。

判断原因：

- 对读区每句使用 `<button>` 渲染。
- 即使 CSS 设为 `display: inline`，按钮在浏览器布局中仍更接近不可拆分的交互控件，句子内部不能像普通正文一样自然断行。

本轮修改：

- 将 `.parallel-segment` 从 `<button>` 改为真正的内联 `<span role="button" tabindex="0">`。
- 保留点击高亮逻辑。
- 补充 Enter/Space 键盘触发，维持基本可访问性。
- CSS 去掉按钮继承字体相关设置，让片段完全融入正文流。

验证：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/app.js'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/styles.css'
```

结果：`app.js` 语法检查通过，静态资源均返回 HTTP 200。

## 2026-05-12 补充修正：对读正文收紧行距并增加首行缩进

用户反馈对读正文中句子之间隔得太远，且缺少书籍式首行缩进。

本轮修改 `.parallel-text-flow`：

- `line-height` 从 `2.05` 收紧到 `1.72`。
- 增加 `text-indent: 2em`，让每栏正文首行缩进两个字符。

验证：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/styles.css'
```

结果：`app.js` 语法检查通过，样式资源返回 HTTP 200。

## 2026-05-12 补充修正：消除对读正文句间拉伸

用户截图显示对读正文仍然像被分散排开，句子之间空得过大，不像书籍正文。

判断原因：

- `.parallel-text-flow` 使用了 `text-align: justify`，浏览器会把内联句子片段之间的空白强行拉伸到整行宽度。
- 前端模板中的换行和缩进也会在内联按钮之间形成额外空白。

本轮修改：

- 对读正文从 `text-align: justify` 改为 `text-align: left`。
- 增加 `word-break: break-all`，让中文正文自然连续换行。
- `renderParallelReading()` 中每个 `.parallel-segment` 改为紧凑字符串拼接，不再在句子按钮之间输出模板换行空白。
- 行距进一步收紧为 `1.62`。

验证：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/app.js'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/styles.css'
```

结果：`app.js` 语法检查通过，静态资源均返回 HTTP 200。

## 2026-05-12 补充修正：去掉句前编号并改为纯文本高亮

用户进一步要求对读区更接近书本排版：

- 去掉每句前的小数字。
- 不要让句子显示成一个框被点亮。
- 只让对应文本本身亮起来。

本轮修改：

- `renderParallelReading()` 去掉句位上标输出。
- `.parallel-segment` 去掉边框、圆角、内边距和背景框。
- hover 与 active 状态改为文本颜色和 `text-shadow` 发光，不再出现块状高亮。

验证：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/app.js'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/styles.css'
```

结果：`app.js` 语法检查通过，静态资源均返回 HTTP 200。

---

# 2026-05-10 对照组归档：标准 edit_count 输入的 VGAE-GVNM 实验

## 本轮任务

论文实验缺少对照组。用户要求从句子连边输入文件中取标准编辑距离 `edit_count`，重新跑一次 VGAE-GVNM，并在新目录中保留结果图片和社区发现输出。

## 新增目录

```text
D:\The_Mora\vgae\control-group
```

目录中新增：

- `generate_edit_count_control_input.py`
- `run_vgae_control.py`
- `run_gvnm_control.py`
- `README.md`
- `CONTROL_GROUP_SUMMARY.md`

## 控制输入构造

原始句子连边文件：

```text
D:\The_Mora\vgae\total_formal_all_sentence_adjusted_distance_aggressive_llm.json
```

`edit_count` 位于 `relation_summary.edit_count`。控制组把原始的修正距离替换为标准编辑距离归一化距离：

```text
normalized_distance = relation_summary.edit_count / max(len(original_text), len(modified_text), 1)
```

生成文件：

```text
D:\The_Mora\vgae\control-group\standard_edit_count_sentence_edges.json
```

输入统计：

- 原始边：28,550
- 写入边：28,550
- 缺失 `edit_count`：0
- 无效记录：0
- VGAE 剪枝前分层：L0 = 7,603，L1 = 37，L2 = 616，L3 = 20,294

## VGAE 控制组输出

运行命令：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe .\control-group\run_vgae_control.py
```

最佳 trial 与句子最佳 trial 均为第 3 个 trial（配置中 `trial_index = 2`）：

- seed：44
- hierarchy_weight：0.08
- learning_rate：0.001
- kl_weight：0.0007
- hierarchy_recon_weight：0.2
- sentence same-chapter silhouette：0.0280154496
- chapter same-name silhouette：0.2522784770
- combined score：0.4437035322
- sentence priority score：0.1319374703

生成的 VGAE 结果图：

- `control-group\tsne_visualization.png`
- `control-group\chapter_tsne_visualization.png`
- `control-group\version_tsne_visualization.png`
- `control-group\sentence_best_tsne_visualization.png`
- `control-group\sentence_best_chapter_tsne_visualization.png`
- `control-group\sentence_best_version_tsne_visualization.png`

同时生成：

- `control-group\vgae_output.pt`
- `control-group\node_vectors.csv`
- `control-group\version_similarity_matrix.csv`
- `control-group\best_config.json`
- `control-group\training_log.json`
- `control-group\sentence_best_vgae_output.pt`
- `control-group\sentence_best_node_vectors.csv`
- `control-group\sentence_best_version_similarity_matrix.csv`
- `control-group\sentence_best_config.json`

## GVNM 控制组输出

运行命令：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe .\control-group\run_gvnm_control.py
```

输出目录：

```text
D:\The_Mora\vgae\control-group\gvnm_output
```

关键结果：

- 版本对数量：66
- 版本社区数：2
- 版本 modularity：-0.0007768735
- 章节社区结果：77 章
- 书籍级亲缘矩阵缺失版本对：3

前三个版本相似对：

1. `fy` - `hs`：0.99971116
2. `hs` - `wb`：0.99965185
3. `fy` - `wb`：0.99958026

生成文件：

- `gvnm_output\version_pair_similarity_ranking.csv`
- `gvnm_output\version_communities.json`
- `gvnm_output\version_community_membership.csv`
- `gvnm_output\chapter_affinity_edges.csv`
- `gvnm_output\chapter_communities.json`
- `gvnm_output\chapter_community_membership.csv`
- `gvnm_output\book_affinity_matrix.csv`
- `gvnm_output\book_affinity_sources.csv`
- `gvnm_output\community_detection_summary.json`

## 验证

执行：

```powershell
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\control-group\generate_edit_count_control_input.py .\control-group\run_vgae_control.py .\control-group\run_gvnm_control.py
```

结果：通过。

PNG 检查：

- 6 张 t-SNE 可视化图片均可由 PIL 打开。
- 尺寸为 2420x1980 或 2200x1760。
- RGB 通道 extrema 均包含 0 到 255，确认不是空白图。

## 与原修正距离实验的简要对照

根目录原实验 `training_log.json` 记录：

- 综合最佳：sentence silhouette = 0.0247064643，chapter silhouette = 0.4513472915，combined = 0.6361582391。
- 句子最佳：sentence silhouette = 0.0760090947，chapter silhouette = 0.3492985964，sentence priority = 0.2048311159。

标准 `edit_count` 对照组：

- 综合/句子最佳同为 trial 3。
- sentence silhouette = 0.0280154496。
- chapter silhouette = 0.2522784770。
- combined = 0.4437035322。
- sentence priority = 0.1319374703。

因此，对照组在综合最佳的句子 silhouette 上略高于原综合最佳，但弱于原句子最佳；章节聚合、综合分和句子优先分均低于修正距离实验。这说明论文中的修正编辑距离信号对章节级聚合和句子优先目标有更明显贡献。

---

# 2026-05-02 第六阶段补充归档：版本图保底连边与矩阵缺失值区分

## 本轮用户反馈

用户在完成可视化后发现版本社区图中有版本节点几乎孤立，同时书籍级亲缘矩阵中出现若干 `0.0000`。经排查：

- 版本社区图此前请求 `/api/version-graph?limit=42`，只取全局排名前 42 条版本相似边，导致 `gd2`、`gd3` 的弱边未进入可视化图。
- `book_affinity_matrix.csv` 来自原始句子相似 JSON 聚合；若某个版本对在原始句子相似数据中没有任何记录，旧代码会默认写为 `0.00000000`，容易被误解为“真实相似度为 0”。
- 原始句子相似 JSON 中缺失证据的版本对为：`gd1-gd2`、`gd2-gd3`、`yz-xr`。

## 主要修改

### 1. 版本图改为“阈值 + 每节点 top-k”保底连边

后端 `version_graph` 不再简单按全局前 N 条截断，而是使用 `threshold=0.6` 和 `top_k=3`，保留所有达到阈值的边，并为每个版本节点补入相似度最高的 3 条邻接边。前端请求改为：

```text
/api/version-graph?threshold=0.6&top_k=3
```

### 2. 书籍亲缘矩阵区分“缺失证据”和“真实低相似”

`community_detection.py` 写出 `book_affinity_matrix.csv` 时，有原始句子证据的版本对继续写数值；没有任何原始句子相似记录的版本对写为 `NA`。

`viz_system/server.py` 读取 `NA` 后在 `/api/book-affinity` 中返回 `null`；前端热力图将其显示为 `NA`，并使用独立样式表示“无原始句子相似证据”。

### 3. 汇总文件记录缺失证据版本对

`community_detection_summary.json` 新增 `book_affinity_missing_pair_count` 和 `book_affinity_missing_pairs`。当前记录为 3 对：`gd1-gd2`、`gd2-gd3`、`yz-xr`。

## 修改文件

- `gvnm/community_detection.py`
- `viz_system/server.py`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `gvnm/output/book_affinity_matrix.csv`
- `gvnm/output/community_detection_summary.json`
- `gvnm/output/version_pair_similarity_ranking.csv`
- `gvnm/output/version_communities.json`
- `gvnm/output/version_community_membership.csv`
- `gvnm/output/chapter_affinity_edges.csv`
- `gvnm/output/chapter_communities.json`
- `gvnm/output/chapter_community_membership.csv`

## 输出文件更新

重新执行：

```powershell
python .\gvnm\community_detection.py
```

已替换 `gvnm/output` 下由第五阶段脚本生成的相关输出文件。新的 `book_affinity_matrix.csv` 中，缺失证据的双向单元格已由 `0.00000000` 改为 `NA`。

## 验证结果

执行语法检查：

```powershell
python -m py_compile .\gvnm\community_detection.py .\viz_system\server.py
node --check .\viz_system\static\app.js
```

结果：均通过。

启动临时服务并验证接口：

```powershell
python .\viz_system\server.py --port 8099
```

接口验证结果：

- `/api/version-graph?threshold=0.6&top_k=3` 返回 12 个版本节点、49 条边。
- 版本图最小节点度数为 3。
- `gd2` 度数为 4。
- `gd3` 度数为 3。
- `/api/book-affinity` 返回 6 个 `null` 矩阵单元，对应 3 个无原始句子证据的无向版本对。
- `/api/overview` 中 `book_affinity_missing_pair_count = 3`。

## 当前状态

本轮已完成用户决策的 B+C 方案：

- 版本图不会再因全局截断造成版本节点孤立。
- 书籍亲缘矩阵不会再把“缺失证据”伪装成“相似度 0”。
- 后端 API、前端热力图和第五阶段输出文件已打通并通过验证。

---

# 2026-05-02 第六阶段补充归档：无重叠章节版本对改用 VGAE 补全

## 本轮用户反馈

用户指出上一轮将缺失格子显示为 `NA` 仍未解决问题，因为这些版本都属于同一本书的不同版本，不应在书籍级亲缘矩阵中没有数据。用户怀疑是否与 VGAE 阶段的相似边剪枝有关，并要求回顾 VGAE 和社区发现产出，找到联系消失的真正原因。

## 排查结论

问题不是 VGAE 剪枝造成的。

证据：

- `missing_similarity_edges.json` 显示原始相似边 `raw_similarity_edges = 28550`，`missing_edge_count = 0`，说明进入 VGAE 的候选边本身没有节点缺失。
- 原始句子相似 JSON 中，`gd1-gd2`、`gd2-gd3`、`yz-xr` 三个版本对在剪枝前就是 0 条记录。
- 树结构检查显示这三对版本没有任何重叠章节：
  - `gd1`：20 章；`gd2`：8 章；共同章节数 0。
  - `gd2`：8 章；`gd3`：4 章；共同章节数 0。
  - `yz`：0-43 章；`xr`：46-76 章；共同章节数 0。
- 当前社区发现的 `book_affinity_matrix.csv` 原本只聚合原始句子相似边。原始句子相似边依赖 `chapter_number + sentence_number + version pair`，没有重叠章节时自然无法生成直接句子证据。

因此，“联系消失”的实际原因是：

> 章节/书籍聚合矩阵只使用直接句子相似边；而这三对版本在原始树中没有重叠章节，所以直接句子证据为空。VGAE 本身已经通过版本-章节-句子层级结构以及其他版本桥接关系推断出了非零版本相似度，但社区发现的书籍级矩阵没有使用这部分 VGAE 输出作为补全。

## VGAE 产出对照

综合最佳 `version_similarity_matrix.csv` 中，这三对都有非零版本相似度：

- `gd1-gd2 = 0.51221961`
- `gd2-gd3 = 0.22164167`
- `yz-xr = 0.31553042`

句子最佳 `sentence_best_version_similarity_matrix.csv` 中同样非零：

- `gd1-gd2 = 0.50994968`
- `gd2-gd3 = 0.29499745`
- `yz-xr = 0.66948843`

第五阶段社区发现主流程使用综合最佳 `version_similarity_matrix.csv`，因此本轮选择综合最佳 VGAE 版本相似度作为补全来源。

## 修改文件

- `gvnm/community_detection.py`
- `viz_system/server.py`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `gvnm/output/book_affinity_matrix.csv`
- `gvnm/output/book_affinity_sources.csv`
- `gvnm/output/community_detection_summary.json`
- 以及重跑第五阶段后同步更新的社区发现输出文件。

## 主要修改

### 1. 书籍级亲缘矩阵补全策略

`community_detection.py` 现在写 `book_affinity_matrix.csv` 时：

- 若版本对存在直接句子相似证据，继续使用原始句子相似边聚合值。
- 若版本对没有直接句子证据，则使用 `version_similarity_matrix.csv` 中的 VGAE 版本余弦相似度补全。
- 补充值按亲缘矩阵显示需求 clamp 到 `[0, 1]`。

### 2. 新增来源矩阵

新增输出：

```text
gvnm/output/book_affinity_sources.csv
```

用于标记每个格子的来源：

- `sentence_aggregation`
- `vgae_fallback`
- `self`

### 3. 可视化显示补值来源

`/api/book-affinity` 新增 `sources` 字段。前端热力图中，`vgae_fallback` 的格子仍显示数值，并用细边框/斜体标记；鼠标悬停会提示该值来自 VGAE 版本相似度补全。

## 输出文件更新

重新执行：

```powershell
python .\gvnm\community_detection.py
```

新的 `book_affinity_matrix.csv` 中，原本缺失的三个版本对已补全为：

- `gd1-gd2 = 0.51221961`
- `gd2-gd3 = 0.22164167`
- `yz-xr = 0.31553042`

## 验证结果

执行：

```powershell
python -m py_compile .\gvnm\community_detection.py .\viz_system\server.py
node --check .\viz_system\static\app.js
```

结果：均通过。

重启可视化服务：

```powershell
python .\viz_system\server.py --port 8070
```

接口验证：

- `/api/book-affinity` 中 `nullCells = 0`。
- `gd1-gd2 = 0.51221961`，来源 `vgae_fallback`。
- `gd2-gd3 = 0.22164167`，来源 `vgae_fallback`。
- `yz-xr = 0.31553042`，来源 `vgae_fallback`。
- `/api/version-graph?threshold=0.6&top_k=3` 仍返回 49 条边，版本图保底连边保持有效。

## 当前状态

书籍级亲缘矩阵已恢复为全数值矩阵。无重叠章节导致的直接句子证据缺失不会再显示为 `0` 或 `NA`，而是由 VGAE 版本相似度补全，并在可视化中保留来源标记。

---

# 2026-05-02 第六阶段补充归档：确认无共同章节后恢复 0 值并加说明

## 本轮用户决策

用户进一步确认：既然 `gd1-gd2`、`gd2-gd3`、`yz-xr` 这三对版本没有共同章节，那么在“直接句子证据聚合”的书籍级亲缘矩阵里，它们的值可以为 `0`。关键不是强行补值，而是在可视化里说明这些 `0` 的含义：它们不是算法把亲缘关系算没了，而是原始数据没有可直接比较的共同章节。

## 结论回顾

本轮排查确认：

- `mora_v4.1_0406.json` 中存在这些版本及其章节、句子。
- `total_formal_all_sentence_adjusted_distance_aggressive_llm.json` 中没有这三对版本的句子相似边。
- 原因不是 VGAE 剪枝；剪枝之前候选边就不存在。
- 候选边不存在的原因是这些版本对在树结构中没有共同章节：
  - `gd1-gd2`：共同章节数 0。
  - `gd2-gd3`：共同章节数 0。
  - `yz-xr`：共同章节数 0。

因此，书籍级亲缘矩阵中的 `0.0000` 应解释为：

> 该版本对没有共同章节，因此没有直接句子级亲缘证据；在直接句子聚合矩阵中记为 0。

## 修改文件

- `gvnm/community_detection.py`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `gvnm/output/book_affinity_matrix.csv`
- `gvnm/output/book_affinity_sources.csv`
- `gvnm/output/community_detection_summary.json`
- 以及重跑第五阶段后同步更新的社区发现输出文件。

## 主要修改

### 1. 恢复书籍级亲缘矩阵中的 0

`community_detection.py` 写 `book_affinity_matrix.csv` 时：

- 有直接句子证据：继续写句子相似边聚合值。
- 无直接句子证据：写 `0.00000000`。

原先临时使用 VGAE 版本相似度补全的逻辑已撤回。

### 2. 保留来源矩阵作为解释层

保留并更新：

```text
gvnm/output/book_affinity_sources.csv
```

来源类型调整为：

- `sentence_aggregation`
- `no_shared_chapter`
- `self`

其中 `no_shared_chapter` 用于解释值为 0 的特殊格子。

### 3. 可视化说明

`/api/book-affinity` 继续返回 `sources` 字段。前端热力图中，`no_shared_chapter` 的格子仍显示 `0.0000`，并加独立样式；鼠标悬停说明：

```text
两个版本没有共同章节，因此没有直接句子级亲缘证据，书籍级直接聚合值记为 0
```

## 输出文件更新

重新执行：

```powershell
python .\gvnm\community_detection.py
```

新的 `book_affinity_matrix.csv` 中：

- `gd1-gd2 = 0.00000000`
- `gd2-gd3 = 0.00000000`
- `yz-xr = 0.00000000`

对应 `book_affinity_sources.csv` 中均标记为：

```text
no_shared_chapter
```

## 验证结果

执行：

```powershell
python -m py_compile .\gvnm\community_detection.py .\viz_system\server.py
node --check .\viz_system\static\app.js
```

结果：均通过。

重启可视化服务：

```powershell
python .\viz_system\server.py --port 8070
```

接口验证：

- `gd1-gd2 = 0.0`，来源 `no_shared_chapter`。
- `gd2-gd3 = 0.0`，来源 `no_shared_chapter`。
- `yz-xr = 0.0`，来源 `no_shared_chapter`。

## 当前状态

本轮最终方案为：书籍级亲缘矩阵保持直接句子聚合语义；无共同章节的版本对显示为 `0.0000`，并在可视化中明确解释其原因为“无共同章节/无直接句子级证据”。

---

# 2026-05-02 第六阶段补充归档：对读区双重索引修正

## 本轮用户反馈

用户指出章节对读区出现双重索引，例如：

```text
1. 0 天下有始
2. 1 可以爲天下母
```

以及某些版本从大于 0 的原始句号开始，例如：

```text
1. 7 閟亓門
2. 8 賽其𨓚
```

这说明部分版本在该章前面的原始句位缺失。用户要求确认哪个索引是正确的，并将对读区显示改为从 1 开始的阅读友好型索引。

## 判断

两个索引含义不同：

- `row.sentence` 是原始数据中的句子定位索引，用于匹配句子节点和句子相似边；它可能从 7 等大于 0 的位置开始，反映该版本该章前面存在缺失句位。
- `<ol>` 自动生成的 1、2、3... 是阅读友好型序号，更适合在对读区展示。

因此，对读区应只直接显示阅读友好型序号；原始句号作为辅助信息保留，不再和正文混排。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 主要修改

### 1. 移除正文前的原始句号

`renderSentences()` 中原先生成：

```html
<li><span>${row.sentence}</span> 文本</li>
```

现改为：

```html
<li title="原始句号：${row.sentence}">文本</li>
```

页面上只显示 `<ol>` 的 1 起阅读序号。

### 2. 对缺失前置句位的版本加标题提示

若某版本该章第一条 `row.sentence > 0`，版本标题旁显示：

```text
原始句号从 N 起
```

悬停说明：

```text
该版本本章前 N 个句位在原始数据中缺失
```

例如第 15 章中 `gd` 第一条原始句号为 7，则标题旁提示 `原始句号从 7 起`，正文列表仍从 1 开始显示。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
python -m py_compile .\viz_system\server.py
```

结果：均通过。

重启服务：

```powershell
python .\viz_system\server.py --port 8070
```

接口/静态文件验证：

- `GET /` 返回 HTTP 200。
- 线上 `app.js` 已包含 `原始句号从` 和 `原始句号：`。
- 旧的 `<li><span>${row.sentence}</span>` 混排逻辑已移除。

## 当前状态

章节对读区现在只显示从 1 开始的阅读友好型编号；原始句号保留在悬停提示和标题提示中，用于说明该版本该章是否存在前置缺失句位。

---

# 2026-05-02 第六阶段补充归档：对读区编号改为全章句位编号

## 本轮用户反馈

用户进一步指出，对读区不应让每个版本都从 `1` 开始编号；如果某个版本本章是从后面的句位开始的，那么显示编号也应从对应句位开始。

## 修正判断

原始 `sentence` 是 0 基定位索引；对读区应显示 1 基阅读编号，但保留全章句位位置：

- 原始 `sentence = 0` 显示为第 `1` 句。
- 原始 `sentence = 7` 显示为第 `8` 句。

这比每个版本都局部从 `1` 开始更准确，因为它能直接表现该版本本章前面句位缺失。

## 修改文件

- `viz_system/static/app.js`
- `log-of-code.md`

## 主要修改

`renderSentences()` 中，版本文本列表 `<ol>` 现在根据该版本本章第一条原始句号设置 `start`：

```js
const firstDisplaySentence = Number(firstRawSentence) + 1;
<ol start="${firstDisplaySentence}">
```

标题提示也从：

```text
原始句号从 N 起
```

改为：

```text
从第 N+1 句起
```

例如第 15 章中 `gd` 第一条原始句号为 `7`，则对读区显示列表从 `8.` 开始，并在标题旁提示 `从第 8 句起`。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
```

结果：通过。

重启服务：

```powershell
python .\viz_system\server.py --port 8070
```

验证：

- `GET /` 返回 HTTP 200。
- 线上 `app.js` 已包含 `<ol start=...>`。
- 线上 `app.js` 已包含 `从第 ${firstDisplaySentence} 句起`。

## 当前状态

对读区编号现在是全章位置感知的 1 基阅读编号：不再显示双重索引，也不再让缺失前置句位的版本从局部 `1` 开始。

---

# 2026-05-02 第六阶段补充归档：高亮子图、层级颜色与非相关节点保留

## 本轮用户反馈

用户希望可视化高亮效果更接近 `tosee_ref/site3`：

- 所有亮起来的节点之间，若存在连边，连边也应亮起来。
- 亮起节点和连边应根据相似度/距离层级使用不同颜色。
- 右侧关联/对读详情框里的文字也应使用节点高亮颜色区分。
- 选中节点为白色最亮，其他相关节点依次变暗。
- 高亮时不要让其他节点直接消失，应保留上下文。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 主要修改

### 1. 引入 site3 风格亮度色阶

在 `app.js` 中新增：

```js
const BRIGHTNESS_COLORS = [
  '#ffffff',
  '#fef08a',
  '#fde68a',
  '#fcd34d',
  '#fbbf24',
  '#f59e0b',
  '#d97706',
  '#b45309',
  '#92400e',
];
```

含义：

- 焦点节点：白色。
- 最近/最强相关节点：浅黄。
- 更远/更弱相关节点：逐步过渡到橙褐色。

### 2. 高亮逻辑从“直接边”扩展为“亮起节点子图”

`rankedFocus()` 原先只给选中节点的直接边设置高亮等级。现在改为：

1. 先按距离/相似度找出与选中节点直接相关的节点，并给这些节点分配 rank。
2. 再遍历所有边。
3. 如果一条边的两端都在亮起节点集合中，则这条边也获得 rank 并亮起。

这样可以实现：

> 所有亮起来的节点之间，只要原图里存在连边，就一起亮。

### 3. 右侧关联列表加入同色层级

`renderRanking()`、`showNodeDetail()`、`showSentenceDetail()` 中的关联行现在会带有：

- `rank-1` 到 `rank-8`
- `--rank-color`

CSS 会使用该颜色渲染编号、关联节点名、句子片段左边框等，让详情框与图上的节点层级一致。

### 4. 非相关节点不再接近消失

旧样式：

```css
.dimmed {
  opacity: 0.1;
}
```

新样式区分节点和边：

```css
.node.dimmed {
  opacity: 0.38;
}

.edge.dimmed {
  opacity: 0.24;
}
```

这样高亮时仍能看到其他节点和边的位置，不会破坏整体上下文。

### 5. 连边按层级着色

`edge.rank-1` 到 `edge.rank-8` 现在使用从黄到橙褐的不同颜色和透明度，而不再只有一组近似黄色。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
python -m py_compile .\viz_system\server.py
```

结果：均通过。

重启服务：

```powershell
python .\viz_system\server.py --port 8070
```

验证：

- `GET /` 返回 HTTP 200。
- 线上 `app.js` 已包含 `BRIGHTNESS_COLORS`。
- 线上 `app.js` 已包含亮起节点子图判断：`activeIds.has(link.source)`。
- 线上 `app.js` 已包含 `relation-row`。
- 线上 `styles.css` 已包含 `.node.dimmed` 和 `.edge.dimmed`。

## 当前状态

点击节点后：

- 焦点节点白色最亮。
- 相关节点按相似度/距离排名逐级变暗。
- 亮起节点之间存在的边会同步亮起。
- 右侧关联列表使用相同色阶标记相关项。
- 非相关节点和边保留可见，只轻度弱化。

---

# 2026-05-02 第六阶段补充归档：章节视图加入全书句子搜索与自动跳章高亮

## 本轮用户需求

用户希望在“章节查询”视图中增加句子搜索功能：

- 搜索框放在上侧标题旁。
- 输入句子文本后，全书搜索。
- 支持模糊搜索。
- 如果命中其他章节，需要自动跳转到对应章节。
- 找到句子节点后，触发与鼠标点击节点相同的高亮效果。
- 多个句子节点符合时，排序规则为：
  1. 匹配质量更高优先。
  2. 该版本该章句子数更多优先。
  3. 版本名字母顺序更靠前优先。
  4. 句子编号更靠前优先。

## 修改文件

- `viz_system/server.py`
- `viz_system/static/index.html`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 后端修改

新增 API：

```text
GET /api/search-sentences?q={query}&limit=12
```

新增方法：

- `_normalize_search_text(value)`
- `_subsequence_score(query, text)`
- `_search_score(query, text)`
- `search_sentences(query, limit)`

搜索范围：

- 全书所有章节。
- 每章所有版本。
- 每个版本内所有句子。

匹配逻辑：

- 先做规范化：去掉空白、`#`、`□`，并转小写。
- 若查询文本是句子文本的连续子串，记为 `contains`。
- 否则使用简单子序列匹配作为 `fuzzy`。
- 模糊分数低于阈值的结果不返回。

返回字段包括：

- `chapter`
- `chapterDisplay`
- `version`
- `sentence`
- `sentenceDisplay`
- `sentenceCount`
- `nodeId`
- `text`
- `score`
- `matchType`
- `rank`

## 前端修改

### 1. 标题栏搜索框

`index.html` 在 toolbar 中新增：

```html
<div class="global-search hidden" id="sentenceSearch">
  <input id="sentenceSearchInput" type="search" placeholder="搜索全书句子" />
  <button id="sentenceSearchBtn">搜索</button>
  <div class="search-status" id="sentenceSearchStatus"></div>
</div>
```

该搜索框只在章节查询模式显示：

- `loadChapterView()`：显示。
- `loadVersionView()` / `loadMatrixView()`：隐藏。

### 2. 搜索与自动定位

新增：

- `searchAndFocusSentence()`
- `focusNode(node, { center: true })`
- `centerGraphOnNode(node)`

流程：

1. 用户输入文本并按 Enter 或点击“搜索”。
2. 调用 `/api/search-sentences`。
3. 取排序后的最佳结果。
4. 若最佳结果不在当前章节，自动 `loadChapterView(best.chapter)`。
5. 在新章节图中找到 `best.nodeId` 对应句子节点。
6. 设置为固定焦点节点。
7. 居中该节点。
8. 调用现有详情面板和高亮逻辑。

因此搜索命中后的效果与鼠标点击句子节点一致。

### 3. 搜索状态提示

搜索框下方显示：

- `搜索中...`
- `未找到匹配句子`
- `第 N 章 · version · 第 M 句 · contains/fuzzy`

## 样式修改

新增 `.global-search` 和 `.search-status` 样式，使搜索框位于标题栏右侧，保持与现有深色工具栏风格一致。

## 验证

执行：

```powershell
python -m py_compile .\viz_system\server.py
node --check .\viz_system\static\app.js
```

结果：均通过。

重启服务：

```powershell
python .\viz_system\server.py --port 8070
```

接口验证：

```text
GET /api/search-sentences?q=上德&limit=3
```

返回示例：

```text
rank chapter version sentence sentenceCount matchType text
1    1       fy      1        25            contains  上德不德
2    1       hj      1        25            contains  上德不德
3    1       hs      1        25            contains  上德不德
```

静态文件验证：

- `GET /` 返回 HTTP 200。
- 页面包含 `sentenceSearchInput`。
- `app.js` 包含 `searchAndFocusSentence`。
- `app.js` 包含自动跳章逻辑 `await loadChapterView(Number(best.chapter))`。

## 当前状态

章节查询视图已经支持全书句子搜索。搜索命中后会自动跳转章节、定位句子节点、居中并触发节点高亮与右侧详情更新。

# 2026-05-02 第六阶段补充归档：章节查询骨架节点从 version 修正为 chapter

## 本轮用户反馈

用户指出章节查询视图中，被高亮的骨架节点应当是章节节点，但右侧详情显示为：

```text
类型 version
```

这说明前端图结构里，章节查询视图仍然把每个版本在该章中的代表节点建模为 `version`，语义不正确。用户希望该节点应显示为 `chapter`。

## 修改文件

- `viz_system/server.py`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 后端修改

### 1. 新增章节节点 ID

新增：

```python
def _chapter_node_id(self, version: str, chapter: int) -> str:
    return f"chap_{version}_{chapter}"
```

### 2. 章节接口节点类型修正

`GET /api/chapter/{chapter_number}` 中，每个版本对应的该章骨架节点从：

```json
{
  "id": "hj",
  "type": "version"
}
```

改为：

```json
{
  "id": "chap_hj_11",
  "label": "hj",
  "type": "chapter",
  "version": "hj",
  "chapter": 11
}
```

### 3. 包含边修正

版本/章节骨架到句子的包含边从：

```text
hj -> sent_hj_11_0
```

改为：

```text
chap_hj_11 -> sent_hj_11_0
```

### 4. 章节相似边类型修正

章节查询中的版本间章节相似边从：

```text
type = version_similarity
```

改为：

```text
type = chapter_similarity
```

同时保留：

- `sourceVersion`
- `targetVersion`

方便后续需要显示原版本名。

### 5. 章节排行同步改为章节节点 ID

`ranking` 中的 `source` / `target` 也改为章节节点 ID，使左侧排行点击能够正确高亮章节节点：

```text
chap_hj_11 - chap_ba_11
```

前端显示时仍通过 `sentenceLabel()` 展示为：

```text
hj - ba
```

## 前端修改

### 1. 新增 chapter 节点颜色

`nodeVisualColor()` 新增：

```js
if (node.type === 'chapter') return '#60a5fa';
```

与 `site3` 的章节节点蓝色一致。

### 2. 章节布局识别 chapter 节点

`prepareChapterLayout()` 中，骨架节点从：

```js
nodes.filter(node => node.type === 'version')
```

改为：

```js
nodes.filter(node => node.type === 'chapter')
```

句子星团仍按 `node.version` 分组，因此布局行为保持一致，但语义正确。

### 3. 章节节点详情

`showNodeDetail()` 中为 `chapter` 节点增加：

- 版本。
- 第几章。
- 类型显示为 `chapter`。

用户点击 `hj` 的第 12 章节点时，右侧应显示：

```text
类型 chapter
版本 hj
章节 第 12 章
```

### 4. 章节节点命中半径与样式

新增：

- `nodeHitRadius()` 对 chapter 返回 11。
- `.node.chapter circle`
- `.node.chapter.focus circle`
- `.node.chapter.rank-* circle`
- `.edge.chapter_similarity`

使章节节点的视觉和交互尺寸与 `site3` 的章节层级语义匹配。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
```

结果：均通过。

重启服务并验证章节接口：

```powershell
GET http://127.0.0.1:8070/api/chapter/11
```

验证结果：

- HTTP 状态：200
- `CHAPTER_NODES=12`
- `VERSION_NODES=0`
- `CHAPTER_EDGES=21`
- `FIRST_NODE_TYPE=chapter`

当前服务进程：

```text
PID=16900
```

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-02 第六阶段补充归档：六项交互与展示问题修正

## 本轮用户反馈

用户在浏览器中测试后提出 6 个问题：

1. 点亮一次节点之后点击空白处没有恢复。
2. 点击节点不是稳定触发高亮，而是有一定概率触发。
3. `site3` 里的高亮聚拢动画没有了，现在只是高亮，不会聚拢。
4. 版本亲缘矩阵消失了。
5. 左边显示的章节号和“第 x 章”需要把索引对齐，面向普通人时索引不应从 0 开始。
6. 页面标题旁边的“社区 i”图例需要去掉。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/index.html`
- `log-of-code.md`

## 逐项修正

### 问题 1：空白点击不恢复

原来的空白点击依赖 `click` 事件里的 `drag` 状态，但 `pointerup` 已经把 `drag` 清空，导致空白点击经常无法识别。

本轮改为在 `pointerup` 中统一判断：

- 如果拖动距离小于 `BLANK_CLICK_DRIFT = 4`，认为是点击。
- 若点击位置没有命中任何节点，则调用 `clearHighlight()`。
- 同时恢复右侧空状态文案。

### 问题 2：点击节点概率触发

章节图有动画重绘，DOM 节点可能在 `pointerdown` 和 `click` 之间被替换，导致原先依赖 DOM click 的节点点击不稳定。

本轮改为几何命中测试：

- 新增 `graphPointFromClient()`
- 新增 `nodeHitRadius()`
- 新增 `findNodeAtClientPoint()`
- 新增 `toggleNodeFocus()`

现在点击节点不再依赖当前 DOM 元素，而是根据鼠标坐标和当前图中节点坐标计算是否命中，因此动画重绘时也能稳定选择节点。

### 问题 3：高亮后不聚拢

在 `tickChapter()` 中新增焦点聚拢力：

- 根据 `rankedFocus(state.fixed, links)` 找到焦点节点和邻居。
- 焦点邻居按 rank 围绕焦点重新产生目标位置。
- 句子节点焦点使用更强聚拢力。
- 点击节点或清空高亮后都会重新启动章节动画。

修改后，点击节点不仅会亮起，还会把相关节点向焦点附近聚拢，接近 `site3` 的交互效果。

### 问题 4：版本亲缘矩阵消失

原因是章节图动画与 SVG 状态可能残留，切换矩阵页时没有强制停止动画和隐藏 SVG。

本轮在 `loadMatrixView()` 中：

- 调用 `stopLayoutAnimation()`。
- 清空图例。
- 强制 `svg.style.display = 'none'`。
- 强制 `heatmap.style.display = 'block'`。
- 保留 `hidden` class 的切换。

在 `renderGraph()` 中恢复：

- `svg.style.display = ''`
- `heatmap.style.display = ''`

确保矩阵页和图页切换不会互相污染。

### 问题 5：章节索引对齐为普通人习惯

前端内部仍然使用 0-based chapter index 读取接口，但用户输入和显示改为 1-based：

- 输入框最小值改为 `min="1"`。
- 默认值改为 `value="1"`。
- `loadChapterView()` 中显示 `state.chapter + 1`。
- 输入定位时将用户输入转换为内部索引：`Number(input) - 1`。
- 下拉框继续显示“第 N 章”，value 保持内部索引。
- 统计卡片从“章节编号”改为“章节数”，避免误解。

### 问题 6：去掉标题旁社区图例

`renderLegend()` 改为清空：

```js
function renderLegend(maxCommunity = 6) {
  legend.innerHTML = '';
}
```

矩阵页切换时也显式清空 `legend.innerHTML`。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
Select-String -Path .\viz_system\static\app.js,.\\viz_system\\static\\index.html -Pattern "mouseenter|mouseleave|社区 \\$|renderLegend|targetIsBlank|findNodeAtClientPoint|loadMatrixView|第几章"
```

结果：

- `app.js` 语法检查通过。
- 当前页面返回 HTTP 200。
- 前端无 `mouseenter` / `mouseleave` 高亮逻辑。
- 已存在几何命中点击逻辑。
- 已存在矩阵页强制显示/隐藏逻辑。
- 输入框文案已改为“第几章”。
- 标题旁社区图例已被清空。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-02 第六阶段补充归档：空白点击清除与 site3 视觉细化

## 本轮用户需求

用户提出两个明确需求：

1. 新增点击空白处取消高亮，否则选择一次后体验上像是无法切换/清除。
2. 节点样式和交互后的样式要尽量与 `vgae\tosee_ref\site3` 一致，因为 `site3` 是用户自己写的、认为最美观的版本。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 交互修改

### 1. 空白点击清除高亮

在 SVG 上新增空白点击逻辑：

```js
svg.addEventListener('click', event => {
  if (event.target !== svg || !drag?.targetIsBlank) return;
  const moved = Math.hypot(event.clientX - drag.x, event.clientY - drag.y);
  if (moved > BLANK_CLICK_DRIFT) return;
  clearHighlight();
  detailPanel.innerHTML = '<div class="empty-state">点击节点查看详情并高亮关联</div>';
});
```

要点：

- 只有点击真正空白处才清除。
- 点击节点不会冒泡触发清除。
- 拖动画布不会被误判为空白点击。
- `BLANK_CLICK_DRIFT = 4`，允许极小鼠标抖动。

### 2. 节点点击阻止冒泡

节点点击事件中增加：

```js
event.stopPropagation();
```

避免点击节点后又被 SVG 空白点击逻辑清空。

## site3 样式对齐

### 1. 普通节点尺寸

对齐 `site3` 的半径：

- 版本节点：`r = 6`
- 句子节点：`r = 2`

### 2. 普通节点颜色

对齐 `site3`：

- 版本节点：`#a78bfa`
- 句子节点：`#3b82f6`

### 3. 边颜色

对齐 `site3` 的暗色低透明边：

- 普通句子边：`rgba(59, 130, 246, 0.10)`
- 层级/包含边：`rgba(139, 92, 246, 0.15)`
- 高亮边：`rgba(234, 179, 8, 0.45)`

### 4. 点击后亮度等级

按 `site3` 的亮度色表调整：

```text
#ffffff
#fef08a
#fde68a
#fcd34d
#fbbf24
#f59e0b
#d97706
#b45309
#92400e
```

焦点节点白色，邻居按 rank 逐级变黄、金、橙。

### 5. 画布标签策略

为了更接近 `site3` 的 canvas 图面：

- 图中普通节点不显示文字标签。
- 焦点和邻居节点也不在图面显示标签。
- 节点信息保留在右侧详情面板与左侧排行中。

这样图面更接近用户提供的星团参考图，不会被标签干扰。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
Select-String -Path .\viz_system\static\app.js -Pattern "targetIsBlank|clearHighlight|mouseenter|mouseleave"
```

结果：

- `app.js` 语法检查通过。
- 当前页面返回 HTTP 200。
- 存在空白点击清除逻辑。
- 无 `mouseenter` / `mouseleave` 高亮逻辑。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-02 第六阶段补充归档：高亮改为点击触发并贴近 site3 节点样式

## 本轮用户需求

用户希望修改当前可视化交互：

- 不再鼠标移上去就高亮。
- 改为鼠标点击后才高亮。
- 节点样式改成和 `vgae\tosee_ref\site3` 更一致。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `viz_system/static/index.html`
- `log-of-code.md`

## 交互逻辑修改

### 1. 取消节点 hover 高亮

从 `app.js` 中移除了节点的：

- `mouseenter`
- `mouseleave`

现在鼠标经过节点不会再改变：

- 当前焦点节点。
- 图中高亮状态。
- 左侧动态排行。

### 2. 点击节点才高亮

节点点击逻辑保留并成为唯一高亮入口：

```js
state.fixed = state.fixed?.id === node.id ? null : node;
```

再次点击同一个节点会取消高亮。

### 3. 排行项也改为点击触发

左侧相似度排行从 hover 改成 click：

```js
row.addEventListener('click', () => highlightPair(...));
```

这样排行项不会因为鼠标扫过而频繁改变图面状态。

### 4. 高亮计算只看固定焦点

修改为：

```js
const activeNode = state.fixed;
```

因此图中的发光联动只由点击产生。

## 节点样式修改

参考 `site3`：

- 版本节点：紫色 `#a78bfa`。
- 句子节点：亮蓝色 `#3b82f6`。
- 普通边：暗蓝低透明度。
- 树/包含边：紫色弱线。
- 点击焦点：白色发光。
- 相邻节点：按 rank 分级显示黄/金/橙色。
- 非相关节点：弱化。

### 视觉策略

为了更接近用户给出的星团参考图：

- 普通节点默认更小。
- 普通标签默认隐藏。
- 只有焦点节点和最近邻节点显示标签。
- 节点 cursor 改为 pointer，暗示点击交互。

## 文案同步

将页面初始提示从：

```text
悬停高亮关联边，点击固定节点。
```

改为：

```text
点击节点高亮关联结构。
```

右侧空状态也改为：

```text
点击节点查看详情并高亮关联
```

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
Select-String -Path .\viz_system\static\app.js -Pattern "mouseenter|mouseleave"
```

结果：

- `app.js` 语法检查通过。
- 当前页面返回 HTTP 200。
- 前端脚本中已无 `mouseenter` / `mouseleave` 高亮逻辑。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-02 第六阶段补充归档：章节图星团力导向布局

## 本轮用户需求

用户提供参考图，希望章节图不再像表格一样把节点写死在固定列位置，而是更接近 `site3` 和参考图中的“星团”结构：

- 节点像星星一样分布。
- 版本/章节骨架在中心。
- 句子节点围绕形成多个自然团簇。
- 点击后仍保留联动高亮。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 核心修改

### 1. 章节图从列式布局改为星团布局

原来的章节视图是：

- 版本节点横向排列在顶部。
- 句子节点垂直排列在各自版本下方。

本轮改为：

- 版本节点围绕中心形成内圈。
- 每个版本的句子节点围绕该版本方向形成一个外部星团。
- 句子节点采用黄金角 `goldenAngle` 撒点，形成更自然的星云/星团分布。

### 2. 新增章节图专用力导向模拟

新增：

- `tickChapter(nodes, links)`
- `stabilizeChapter(nodes, links, count)`
- `startChapterAnimation()`
- `stopLayoutAnimation()`

章节图现在不是静态定位，而是：

1. 先根据版本方向生成星团初始位置。
2. 再运行若干轮物理稳定。
3. 页面打开后继续执行约 260 帧轻量动画，使节点自然散开并安定。

### 3. 力导向规则

章节图的力包括：

- 节点锚点力：句子节点倾向留在所属版本星团附近，版本节点倾向留在中心附近。
- 节点斥力：避免句子节点重叠，形成星点分布。
- `contains_sentence` 边：弱弹簧，保持版本-句子结构骨架。
- `sentence_similarity` 边：按 `normalized_distance` 调整目标距离，距离越小拉得越近。
- `version_similarity` 边：轻微拉近版本节点。

### 4. 点击联动高亮继续保留

上轮实现的：

- 焦点节点发光。
- 相邻节点按 rank 分级亮起。
- 相邻边同步高亮。
- 非相关节点弱化。

本轮继续保留，并作用在新的星团布局上。

### 5. 句子标签弱化

为了更像参考图中的星点，默认隐藏句子节点上的数字标签：

- 普通句子节点只显示小点。
- 焦点句子、最近邻句子才显示句号标签。

这样图面更接近“星空结构”，不会被大量句号文本挤满。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/api/chapter/0'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 第 1 章接口返回 HTTP 200。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-02 第六阶段补充归档：章节图加入句子节点与 normalized_distance 边

## 本轮用户需求

用户希望章节查询部分不只显示版本/章节级关系，还要在章节图中加入：

- 句子节点。
- 句子节点与句子节点之间的边。
- 点击句子节点后的交互逻辑。
- 句子边的参考值使用 `normalized_distance`。

用户指出 `normalized_distance` 的字段位置可参考 `VGAE_v2_heterogeneous.md`，实际来源是句子连边文件 `total_formal_all_sentence_adjusted_distance_aggressive_llm.json`。

## 数据依据

根据 `VGAE_v2_heterogeneous.md`：

```json
{
  "chapter_number": 0,
  "sentence_number": 0,
  "original_text_version": "ba",
  "modified_text_version": "bb",
  "original_text": "####",
  "modified_text": "上德不德",
  "normalized_distance": 0.05
}
```

节点匹配规则：

```text
sent_{version}_{chapter}_{sentence}
```

## 修改文件

- `viz_system/server.py`
- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 后端修改

### 1. 加载句子相似边

在 `DataStore` 中新增：

- `sentence_similarity_edges`
- `_load_sentence_similarity_edges()`
- `_sentence_node_id()`
- `_sentence_exists()`

后端启动时读取：

```text
total_formal_all_sentence_adjusted_distance_aggressive_llm.json
```

并从每条记录中提取：

- `chapter_number`
- `sentence_number`
- `original_text_version`
- `modified_text_version`
- `normalized_distance`
- `original_text`
- `modified_text`

转换为前端可使用的边：

```json
{
  "source": "sent_ba_0_0",
  "target": "sent_bb_0_0",
  "type": "sentence_similarity",
  "sentence": 0,
  "sourceVersion": "ba",
  "targetVersion": "bb",
  "distance": 0.05,
  "similarity": 0.95
}
```

其中：

```text
similarity = 1 - normalized_distance
```

### 2. 章节接口扩展

`GET /api/chapter/{chapter_number}` 现在返回三类图元素：

1. 版本节点：`type = version`
2. 句子节点：`type = sentence`
3. 边：
   - `contains_sentence`：版本节点到句子节点的包含边。
   - `version_similarity`：章节级版本相似边。
   - `sentence_similarity`：句子-句子 normalized_distance 边。

同时新增字段：

```json
"sentenceRanking": [...]
```

用于前端在点击句子节点后显示该句子的最近句子排行。

## 前端修改

### 1. 重写章节图布局

章节图现在采用列式布局：

- 每个版本节点位于顶部。
- 该版本下属句子节点垂直排列在版本节点下方。
- 句子-句子边横向连接不同版本中的同号/相关句子。

这样比纯力导向更适合文献学阅读：版本结构清晰，句子对齐关系也更容易观察。

### 2. 句子节点样式

句子节点使用更小的圆点和句号标签：

- 版本节点：大圆点。
- 句子节点：小圆点。
- 颜色继承章节社区颜色。

### 3. 句子边样式

新增三类边样式：

- `contains_sentence`：弱化虚线。
- `version_similarity`：版本/章节级关系边。
- `sentence_similarity`：蓝色细边，表示句子间 normalized distance 关系。

### 4. 点击句子节点逻辑

点击或悬停句子节点时，右侧详情面板显示：

- 版本。
- 章节。
- 句号。
- 当前句子文本。
- 与它最近的句子列表。
- 每条最近句子的 `normalized_distance`。

### 5. 动态排行扩展到句子

原来的动态排行继续保留：

- 版本图：选中版本后显示该版本的相似度排行。
- 章节图：选中版本后显示该章节内版本相似度排行。
- 章节图：选中句子后显示该句子的句子距离排行。

句子排行按 `normalized_distance` 从小到大排序。

## 验证

语法检查：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
```

结果：均通过。

重启服务并验证：

```text
http://127.0.0.1:8070
```

接口验证结果：

- `GET /`：200
- `GET /api/chapter/0`：200
- 第 1 章返回句子节点数：173
- 第 1 章返回句子相似边数：514
- `sentenceRanking` 中包含 `distance` 字段：是

当前服务进程：

```text
PID=28596
```

# 2026-05-02 第六阶段补充归档：点击聚焦联动高亮动画

## 本轮用户需求

用户希望当前可视化系统的点击交互更接近 `vgae\tosee_ref\site3`：点击节点后，相关节点和边一起亮起，形成更动态的聚焦效果。用户同时指出当前节点数量比 `site3` 少，因此这种交互应该会更流畅。

## 参考实现

查看了 `tosee_ref/site3/src/App.tsx`，其核心交互逻辑是：

- 点击节点后设置 focused node。
- 根据该节点的连接关系生成 brightness map。
- 焦点节点最高亮。
- 相邻节点按编辑距离/连接强度分级亮起。
- 相关边同步高亮。
- 非相关节点和边弱化。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 前端逻辑修改

### 1. 新增分级聚焦计算

在 `app.js` 中新增：

- `RANK_CLASS_COUNT`
- `rankedFocus(activeNode, links)`

该函数根据当前焦点节点计算：

- `activeIds`：需要亮起的节点集合。
- `ranks`：节点亮度等级。
- `edgeRanks`：边亮度等级。

对句子节点：

- 按 `normalized_distance` 从小到大排序。
- 距离越小，亮度等级越高。

对版本节点：

- 按相似度从高到低排序。
- 版本相似边优先高亮，版本-句子的包含边也会一起显示。

### 2. 点击固定优先级

修改焦点优先级：

```js
const activeNode = state.fixed || state.selected;
```

也就是说：

- 点击节点后，该节点成为固定焦点。
- 鼠标经过其他节点时，不会抢走已点击节点的主高亮。
- 未点击固定节点时，悬停仍然可临时触发高亮。

### 3. 节点和边的分级 class

前端渲染时会给节点和边附加：

- `focus`
- `rank-1`
- `rank-2`
- ...
- `rank-8`
- `dimmed`

这样 CSS 可以控制发光层次。

## 样式修改

### 1. 边高亮

新增不同 rank 的边颜色和透明度：

- `rank-1` 到 `rank-3`：亮黄色。
- `rank-4` 到 `rank-5`：金色。
- `rank-6` 到 `rank-8`：橙色弱光。

### 2. 节点高亮

新增：

- 焦点节点白色发光。
- 最近邻节点黄色发光。
- 更远邻居逐级转为金色/橙色。
- 非相关节点透明度降低。

### 3. 动画

为焦点节点添加轻量 pulse：

```css
@keyframes node-pulse {
  0%, 100% {
    stroke-width: 2px;
    filter: drop-shadow(0 0 7px rgba(254, 240, 138, 0.85));
  }
  50% {
    stroke-width: 3px;
    filter: drop-shadow(0 0 12px rgba(254, 240, 138, 1));
  }
}
```

同时给节点、文字、边增加 `transition`，使点击后的亮起不再是突然切换。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 当前页面返回 HTTP 200。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-02 第六阶段补充归档：版本图加入点击聚合动画

## 本轮用户需求

用户询问句子节点点击后的“节点之间聚合动画效果”是否也可以用在版本图中，并特别担心如果对整个版本图施加聚合力，是否会把图结构整体“撤飞出去”。用户希望在不会失控的前提下尝试实现，并同步归档。

## 判断

版本图节点数量较少，边也远少于章节句子图，因此可以加入同类聚合动画。为避免整个图被拖出画布，本轮实现同时加入三类约束：

- 每个版本节点保留原环形布局锚点 `anchorX / anchorY`。
- 聚合力使用较小 `pull`，只把相似版本轻轻拉到焦点周围。
- 每帧限制速度并把节点坐标 clamp 在 SVG 画布内。

因此点击聚合不会无限漂移，也不会把整张图甩出视野；清除焦点后，节点会在锚点力作用下回到稳定环形结构附近。

## 修改文件

- `viz_system/static/app.js`
- `log-of-code.md`

## 前端逻辑修改

### 1. 版本节点加入锚点

`prepareVersionLayout(...)` 不再只写入 `x / y`，同时记录：

```js
node.anchorX = width / 2 + Math.cos(angle) * radius;
node.anchorY = height / 2 + Math.sin(angle) * radius;
```

版本图后续动画会持续受到弱锚点力约束，避免焦点聚合后整体漂移。

### 2. 抽取通用 rank 聚合函数

新增：

```js
applyRankAggregation(nodes, links, activeNode, options = {})
```

该函数复用既有 `rankedFocus(...)` 结果：

- 焦点节点保持中心。
- 相邻节点按 rank 使用黄金角分布到焦点周围。
- rank 越靠前，目标环半径越近。
- 句子图和版本图可传入不同 `baseRing / rankRing / pull` 参数。

章节图原本内联的聚合逻辑也改为调用该函数，避免两套逻辑分叉。

### 3. 版本图 tick 加入防漂移机制

`tick(...)` 中新增：

- 中心弱吸引。
- 锚点弱吸引。
- 版本相似边弹簧。
- 点击焦点后的 rank 聚合力。
- 阻尼从 `0.82` 调整为 `0.76`。
- 每帧速度上限限制为 `5.2`。
- 节点位置限制在画布边界 `28px` 内。

### 4. 新增版本图动画循环

新增：

```js
startVersionAnimation()
```

版本图现在在以下场景会启动短时动画：

- 初次加载版本图。
- 点击版本节点聚焦。
- 点击排行项触发聚焦。
- 清除焦点。
- 搜索/程序式聚焦到节点时，如果当前处于版本图。

动画上限为约 `220` 帧，足够看到聚合过程，也不会长期占用渲染。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/api/version-graph?threshold=0.6&top_k=3'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 当前页面返回 HTTP 200。
- 版本图接口返回 HTTP 200。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-03 第六阶段补充归档：修复节点点击后的文本光标闪烁

## 本轮用户反馈

用户反馈点击图中节点后，某个节点位置会出现类似文本输入框的闪烁光标。根据现象判断，该问题更可能来自 SVG 节点标签文本被浏览器选中或进入文本选择状态，而不是实际输入框获得焦点。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 修复内容

### 1. 禁用图区域文本选择

在 `.canvas-wrap`、`#graphSvg` 和 `.node` 上加入：

```css
user-select: none;
-webkit-user-select: none;
```

同时在 `#graphSvg` 上加入：

```css
caret-color: transparent;
```

避免浏览器在 SVG 图层中绘制文本插入光标。

### 2. 节点标签不再接收鼠标事件

在 `.node text` 上加入：

```css
pointer-events: none;
user-select: none;
-webkit-user-select: none;
```

节点命中继续由 JS 的坐标检测逻辑处理，SVG 文本本身不再成为点击目标，也不会触发文本选择状态。

### 3. SVG 点击时阻止默认选择行为

在 `svg` 的 `pointerdown` 处理中新增：

```js
event.preventDefault();
if (document.activeElement && document.activeElement !== document.body) {
  document.activeElement.blur();
}
```

这样点击/拖动画布时不会启动浏览器默认文本选择；如果之前搜索框或章节输入框有焦点，也会在操作图时主动失焦。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 当前页面返回 HTTP 200。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-03 第六阶段补充归档：矩阵视图隐藏右侧对读栏

## 本轮用户反馈

用户反馈从版本图或章节查询切换到亲缘矩阵时，右侧对读栏仍然保留上一视图内容。该行为不符合视图语义：矩阵视图不需要章节对读内容，对读栏应隐藏；切回版本图或章节查询时再恢复。

## 修改文件

- `viz_system/static/app.js`
- `log-of-code.md`

## 修复内容

在三个视图加载函数中明确控制 `sentencesPanel` 的显示状态：

```js
// 版本图
sentencesPanel.classList.remove('hidden');

// 章节查询
sentencesPanel.classList.remove('hidden');

// 亲缘矩阵
sentencesPanel.classList.add('hidden');
```

这样矩阵视图不会继续显示上一章的对读内容；再次进入版本图或章节查询时，对读栏恢复显示。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 当前页面返回 HTTP 200。

当前访问地址：

```text
http://127.0.0.1:8070
```

# 2026-05-03 第六阶段补充归档：矩阵视图彻底移除右侧 aside

## 本轮用户反馈

用户指出上一轮只隐藏 `sentencesPanel` 不彻底，矩阵页面本身就不应该保留右侧 `aside` 标签。该反馈成立：矩阵视图是独立表格视图，不需要右侧详情/对读区域，也不应继续占据三列布局。

## 修改文件

- `viz_system/static/app.js`
- `viz_system/static/styles.css`
- `log-of-code.md`

## 修复内容

### 1. 矩阵模式从 DOM 中移除右侧 aside

新增：

```js
const appShell = document.querySelector('.app-shell');
const detailAside = document.querySelector('.detail');

function showDetailAside() {
  appShell.classList.remove('matrix-view');
  if (!detailAside.isConnected) appShell.appendChild(detailAside);
}

function removeDetailAside() {
  appShell.classList.add('matrix-view');
  if (detailAside.isConnected) detailAside.remove();
}
```

进入矩阵视图时调用 `removeDetailAside()`，右侧 `<aside class="detail">` 会从当前 DOM 中移除。切回版本图或章节查询时调用 `showDetailAside()`，再把该 aside 插回主容器。

### 2. 矩阵模式改为两列布局

新增：

```css
.app-shell.matrix-view {
  grid-template-columns: 320px minmax(520px, 1fr);
}
```

这样矩阵视图不再保留第三列空位，页面结构变为左侧控制栏 + 主矩阵区域。

### 3. 版本图和章节查询恢复 aside

`loadVersionView()` 和 `loadChapterView()` 中新增 `showDetailAside()`；`loadMatrixView()` 中改为 `removeDetailAside()`。

## 验证

执行：

```powershell
node --check .\viz_system\static\app.js
C:\Users\YUE20\anaconda3\envs\mora\python.exe -m py_compile .\viz_system\server.py
Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8070/'
```

结果：

- `app.js` 语法检查通过。
- `server.py` 语法检查通过。
- 当前页面返回 HTTP 200。

当前访问地址：

```text
http://127.0.0.1:8070
```
