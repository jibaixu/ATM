#!/bin/bash

# 确保 log 目录存在
mkdir -p log

# ==========================================
# 定义任务列表
# 格式: "显卡编号:数据集编号列表:view:episode_min:episode_max"
# 数据集编号列表使用逗号分隔，例如 "0,1,2"
# ==========================================
TASKS=(
    # GPU 0: 处理数据集 0-12 (4个程序)
    # "0:0,1,2,3:cam_high_rgb:0:-1"
    # "0:4,5,6,7:cam_high_rgb:0:-1"
    # "0:8,9,10:cam_high_rgb:0:-1"
    # "0:11,12:cam_high_rgb:0:-1"
    # "0:0,1,2,3,4:cam_high_rgb:0:-1"
    # "0:5,6,7,8,9:cam_high_rgb:0:-1"
    # "0:10,11,12:cam_high_rgb:0:-1"
    # "0:4:cam_high_rgb:0:-1"
    # "0:8,9:cam_high_rgb:0:-1"
    # "0:12:cam_high_rgb:0:-1"

    # GPU 1: 处理数据集 13-25 (4个程序)
    # "1:13,14,15,16:cam_high_rgb:0:-1"
    # "1:17,18,19,20:cam_high_rgb:0:-1"
    # "1:21,22,23:cam_high_rgb:0:-1"
    # "1:24,25:cam_high_rgb:0:-1"
    # "1:13,14,15,16,17:cam_high_rgb:0:-1"
    # "1:18,19,20,21,22:cam_high_rgb:0:-1"
    # "1:23,24,25:cam_high_rgb:0:-1"
    # "1:17:cam_high_rgb:0:-1"
    # "1:21,22:cam_high_rgb:0:-1"
    # "1:25:cam_high_rgb:0:-1"

    # "1:2:cam_high_rgb:25:-1"
    # "1:2:cam_high_rgb:37:-1"
    # "1:3:cam_high_rgb:45:-1"
    # "2:44:cam_high_rgb:32:-1"
    # "2:44:cam_high_rgb:27:-1"

    # GPU 2: 处理数据集 26-38 (4个程序)
    # "2:26,27,28,29:cam_high_rgb:0:-1"
    # "2:30,31,32,33:cam_high_rgb:0:-1"
    # "2:34,35,36:cam_high_rgb:0:-1"
    # "2:37,38:cam_high_rgb:0:-1"
    # "2:26,27,28,29,30:cam_high_rgb:0:-1"
    # "2:31,32,33,34,35:cam_high_rgb:0:-1"
    # "2:36,37,38:cam_high_rgb:0:-1"
    # "2:29,30:cam_high_rgb:0:-1"
    # "2:34,35:cam_high_rgb:0:-1"
    # "2:38:cam_high_rgb:0:-1"

    # GPU 3: 处理数据集 39-49 (4个程序)
    # "3:39,40,41,42:cam_high_rgb:0:-1"
    # "3:43,44,45,46:cam_high_rgb:0:-1"
    # "3:47,48,49:cam_high_rgb:0:-1"
    # "3:39,40,41,42:cam_high_rgb:0:-1"
    # "3:43,44,45,46:cam_high_rgb:0:-1"
    # "3:47,48,49:cam_high_rgb:0:-1"
    # "3:42:cam_high_rgb:0:-1"
    # "3:46:cam_high_rgb:0:-1"
    # "3:49:cam_high_rgb:0:-1"

    # "0:45:cam_high_rgb:0:-1"
)

echo "🚀 开始批量提交 worldarena 预处理任务..."
echo "------------------------------------------"

for TASK in "${TASKS[@]}"; do
    IFS=':' read -r GPU DATASETS VIEW MIN_EP MAX_EP <<< "$TASK"

    IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
    DATASET_ARGS=()
    for DATASET_IDX in "${DATASET_ARRAY[@]}"; do
        if [[ -n "$DATASET_IDX" ]]; then
            DATASET_ARGS+=(--dataset_idx "$DATASET_IDX")
        fi
    done

    LOG_DATASETS="${DATASETS//,/+}"
    LOG_FILE="log/${LOG_DATASETS}_${VIEW}_${MIN_EP}_${MAX_EP}.log"

    echo "➡️ 启动任务: [显卡 $GPU] 数据集=$DATASETS | View=$VIEW | Min_Ep=$MIN_EP | Max_Ep=$MAX_EP"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/preprocess_worldarena.py \
        "${DATASET_ARGS[@]}" \
        --view "$VIEW" \
        --min_episode "$MIN_EP" \
        --max_episode "$MAX_EP" > "$LOG_FILE" 2>&1 &
done

echo "------------------------------------------"
echo "✅ 所有 worldarena 任务已成功提交至后台！"
echo "💡 提示："
echo "  - 使用 'jobs' 命令可查看当前终端的后台任务。"
echo "  - 使用 'tail -f log/0_cam_high_rgb_0_-1.log' 可实时查看某个任务的输出。"
