#!/bin/bash

# 设置脚本错误时退出
set -e

# 定义默认值
DEFAULT_PKL_FILE="no_version_non_empty_graph.pkl"
DEFAULT_STEPS="1 2 3 4"  # 默认执行所有步骤

# 解析命令行参数
while getopts "p:s:" opt; do
  case $opt in
    p) PKL_FILE="$OPTARG"
    ;;
    s) STEPS="$OPTARG"
    ;;
    \?) echo "用法: $0 [-p pkl_file] [-s 步骤列表]"
        echo "  步骤列表格式: 1 2 3 4 (空格分隔，默认执行所有步骤)"
        exit 1
    ;;
  esac
done

# 如果未指定pkl_file，使用默认值
PKL_FILE=${PKL_FILE:-$DEFAULT_PKL_FILE}

# 如果未指定steps，使用默认值
STEPS=${STEPS:-$DEFAULT_STEPS}

# 检查步骤参数是否有效
for step in $STEPS; do
  if [[ $step -lt 1 || $step -gt 4 ]]; then
    echo "错误: 无效的步骤号 $step，必须是 1-4 之间的整数"
    echo "用法: $0 [-p pkl_file] [-s 步骤列表]"
    echo "  步骤列表格式: 1 2 3 4 (空格分隔，默认执行所有步骤)"
    exit 1
  fi
done

# 创建logs目录（如果不存在）
mkdir -p logs

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/text_variant_pipeline_${TIMESTAMP}.log"

# 定义日志函数
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# 记录开始时间
START_TIME=$(date +%s)

# 打印当前目录
pwd | tee -a "$LOG_FILE"

# 输出执行开始时间
log "开始执行文本变体处理管道"
log "使用的pkl文件: $PKL_FILE"
log "执行的步骤: $STEPS"

# 检查是否包含特定步骤的函数
contains_step() {
  [[ " $STEPS " == *" $1 "* ]]
}

# 1. 运行数据处理脚本
if contains_step 1; then
  log "=== 步骤1: 处理原始数据 ==="
  log "运行 read_data.py..."
  python read_data.py 2>&1 | tee -a "$LOG_FILE"
else
  log "=== 步骤1: 处理原始数据 (跳过) ==="
fi

# 2. 构建图结构
if contains_step 2; then
  log "\n=== 步骤2: 构建文本变体图 ==="
  log "运行 build_graph.py..."
  python build_graph.py --resume_from_cache --no_use_all_version --output_file "$PKL_FILE" 2>&1 | tee -a "$LOG_FILE"
else
  log "\n=== 步骤2: 构建文本变体图 (跳过) ==="
fi

# 3. 运行图神经网络模型
if contains_step 3; then
  log "\n=== 步骤3: 运行图神经网络模型 ==="
  log "运行 vgae.py..."
  PLOT_PREFIX=$(basename "$PKL_FILE" .pkl)
  PLOTS_DIR="${PLOT_PREFIX}_plots"
  python vgae.py --feature_type hidden --feature_dim 64 --pkl_file "$PKL_FILE" --result_dir "$PLOTS_DIR" 2>&1 | tee -a "$LOG_FILE"
else
  log "\n=== 步骤3: 运行图神经网络模型 (跳过) ==="
fi

# 4. 压缩相似性图文件夹
if contains_step 4; then
  # 提取PKL文件名前缀（去掉.pkl后缀）
  ZIP_PREFIX=$(basename "$PKL_FILE" .pkl)
  ZIP_FILE="${ZIP_PREFIX}_plots.zip"
  log "\n=== 步骤4: 压缩相似性图文件夹 ==="
  if [ -d "$PLOTS_DIR" ]; then
      log "压缩文件夹: $PLOTS_DIR -> $ZIP_FILE"
      # 使用zip命令压缩文件夹
      zip -r "$ZIP_FILE" "$PLOTS_DIR" 2>&1 | tee -a "$LOG_FILE"
      log "压缩完成: $ZIP_FILE"
  else
      log "警告: 文件夹 $PLOTS_DIR 不存在，跳过压缩"
  fi
else
  log "\n=== 步骤4: 压缩相似性图文件夹 (跳过) ==="
fi

# 计算执行时间
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# 输出执行完成时间
log "\n文本变体处理管道执行完成"
log "总执行时间: $((ELAPSED_TIME/3600))小时 $(((ELAPSED_TIME%3600)/60))分钟 $((ELAPSED_TIME%60))秒"
log "日志文件已保存至: $LOG_FILE"