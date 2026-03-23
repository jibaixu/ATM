#!/bin/bash

# 确保 log 目录存在
mkdir -p log

# ==========================================
# 定义任务列表
# 格式: "显卡编号:数据集编号:view:episode_min"
# ==========================================
TASKS=(
    # "0:4:high:800"
    # "0:4:left:800"
    # "0:4:right:800"
    
    # "2:4:high:1000"
    # "2:4:left:1000"
    # "2:4:right:1000"

    # "3:5:high:700"
    # "3:5:left:700"
    # "3:5:right:700"

    "0:4:high:700"
    "0:4:left:700"
    "2:4:right:700"

    "2:5:high:650"
    "3:5:left:650"
    "3:5:right:650"
)

echo "🚀 开始批量提交预处理任务..."
echo "------------------------------------------"

# 遍历数组中的每一个任务
for TASK in "${TASKS[@]}"; do
    # 使用 ':' 作为分隔符，将字符串拆分并赋值给对应的变量
    IFS=':' read -r GPU DATASET VIEW MIN_EP <<< "$TASK"
    
    # 动态生成日志文件名
    LOG_FILE="log/${DATASET}_${VIEW}_${MIN_EP}.log"
    
    echo "➡️ 启动任务: [显卡 $GPU] 数据集=$DATASET | View=$VIEW | Min_Ep=$MIN_EP"
    
    # 执行 Python 脚本并放入后台
    CUDA_VISIBLE_DEVICES=$GPU python scripts/preprocess_robocoin_2.py \
        --dataset_idx $DATASET \
        --view $VIEW \
        --min_episode $MIN_EP > "$LOG_FILE" 2>&1 &
done

echo "------------------------------------------"
echo "✅ 所有任务已成功提交至后台！"
echo "💡 提示："
echo "  - 使用 'jobs' 命令可查看当前终端的后台任务。"
echo "  - 使用 'tail -f log/2_high_700.log' 可实时查看某个任务的输出。"
