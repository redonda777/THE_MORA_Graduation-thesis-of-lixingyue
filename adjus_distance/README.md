# adjus_distance 方案实现

该目录实现了一个独立算法：以 `operations_llm` 为主输入，判断特殊字符关系并赋予自定义代价，输出方案 A 的详细结构。

## 已实现特性

- 显式输出 `match(a, pos)`：通过回放操作得到最终文本后，抽取保持不变字符作为 `match_operations`
- 保留近义关系：关系标签包含 `synonym`
- 激进代价：默认读取 `config_aggressive.json`
- 兼容回退：若缺少 `operations_llm`，自动回退 `operations_dp`
- 支持大模型补判：规则先判，`default` 操作可用大模型重判（如 `弗 -> 不` 判为 `synonym`）

## 关系标签

- `interchangeable`：异体/通假
- `homophone`：同音/近音
- `synonym`：近义
- `function_word`：虚词
- `missing_char`：占位符（如 `#` / `□`）
- `private_char`：Unicode 私有区字符
- `default`：普通操作
- `exact`：仅用于 `match`，代价固定 0

## 输出结构（方案 A）

每条记录输出以下核心字段：

- 元信息：`chapter_number`、`sentence_number`、版本、原文、改文
- `operations_scored`：基于 `operations_llm` 的逐步打分结果
- `match_operations`：显式 `match` 列表
- `operations_all`：合并后的全操作视图（按位置排序）
- `relation_summary`：关系计数、match 数量、编辑步数
- `distance_breakdown`：代价拆解
- `adjusted_edit_distance` 与 `normalized_distance`
- `validation`：操作回放校验结果和告警

## 运行方式

在仓库根目录执行：

```bash
python adjus_distance/score_special_relations.py \
  --input connect_edit_distance/llm_edit_distance_0410/formal_all_sentence_edit_distance_llm_v2_merged.json \
  --output adjus_distance/formal_all_sentence_adjusted_distance_aggressive.json \
  --config adjus_distance/config_aggressive.json
```

启用大模型补判（仅对规则判为 default 的操作补判）：

```bash
python adjus_distance/score_special_relations.py \
  --input connect_edit_distance/llm_edit_distance_0410/formal_all_sentence_edit_distance_llm_v2_merged.json \
  --output adjus_distance/formal_all_sentence_adjusted_distance_aggressive_llm.json \
  --config adjus_distance/config_aggressive.json \
  --use-llm \
  --llm-config adjus_distance/llm_config.json
```

只跑局部数据（例如前 100 条）：

```bash
python adjus_distance/score_special_relations.py \
  --start 0 \
  --end 100
```

## 配置文件

`config_aggressive.json` 可直接改：

- `pair_lexicons`：关系词表（异体/同音/近义）
- `function_words`：虚词表
- `missing_chars`：占位符
- `costs`：各关系代价（当前是激进策略）

`llm_config.json`（新增）可直接改：

- `provider`：`dashscope` 或 `openai_compatible`
- `model`：模型名
- `api_key_env`：API Key 对应环境变量名（例如 `DASHSCOPE_API_KEY`）
- `base_url`：接口地址
- `timeout_seconds` / `max_retries` / `retry_sleep_seconds`
- `retry_backoff_multiplier`：指数退避倍数（例如 2.0）
- `retry_max_sleep_seconds`：单次重试等待上限
- `request_interval_seconds`：请求节流间隔（降低限流与抖动错误）
- `temperature` / `max_tokens`

### 稳定性策略（已实现）

- **多返回格式解析**：兼容 `output.text` 与 `output.choices[0].message.content`（含数组内容片段）
- **请求节流**：每次模型请求之间按 `request_interval_seconds` 间隔发送
- **指数退避重试**：失败后按指数退避等待，并受 `retry_max_sleep_seconds` 限制
- **失败类型统计**：脚本结束时输出 `LLM失败类型统计`，用于定位问题（如超时、429、解析失败）

## 环境变量

启用 `--use-llm` 前先设置 API Key（PowerShell）：

```powershell
$env:DASHSCOPE_API_KEY="你的key"
```
