import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import decord

from atm.utils.cotracker_utils import Visualizer

# --- 基础配置 ---
# 根据 preprocess_robocoin_2.py 中的设置 
BASE_DIR = Path("/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320")
VAL_JSONL = BASE_DIR / "episodes_clipped_train.jsonl"
SAVE_DIR = "results/vis_dataset/resize_240_320"
INDEX = 3009

# --- 辅助函数：加载视频 ---
def load_video_to_tensor(video_path):
    """将视频加载为 (B, T, C, H, W) 格式的 Tensor"""
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy() # T, H, W, C
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() # T, C, H, W
    return frames.unsqueeze(0) # B, T, C, H, W

# --- 主程序 ---
def main():
    # 1. 随机选取数据
    with open(VAL_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        item = json.loads(lines[INDEX]) # 随机选一行 

    print(f"Visualizing episode: {item['episode_index']} from {item['dataset_name']}")

    start_frame, end_frame = item['start_frame'], item['end_frame']

    # 2. 加载视频和轨迹
    video_path = BASE_DIR / item['video']
    track_path = BASE_DIR / item['track']
    
    video_tensor = load_video_to_tensor(video_path) # (B, T, C, H, W)
    video_tensor = video_tensor[:, start_frame:end_frame+1, :, :, :] # 裁剪到指定帧范围
    _, T, _, H, W = video_tensor.shape

    # 加载轨迹并还原归一化坐标 
    data = np.load(track_path)
    tracks_np = data['tracks'][start_frame:end_frame+1, :, :] # (T, N, 2)
    vis_np = data['vis'][start_frame:end_frame+1, :] # (T, N)

    # 还原坐标：x * W, y * H 
    tracks_np[:, :, 0] *= W
    tracks_np[:, :, 1] *= H

    tracks_tensor = torch.from_numpy(tracks_np).unsqueeze(0).float() # (1, T, N, 2)
    visibility_tensor = torch.from_numpy(vis_np).unsqueeze(0).unsqueeze(-1).bool() # (1, T, N, 1)

    # 3. 执行可视化
    viz = Visualizer(save_dir=SAVE_DIR, fps=20, tracks_leave_trace=15)
    viz.visualize(
        video=video_tensor,
        tracks=tracks_tensor,
        visibility=visibility_tensor,
        filename=f"ep_{item['episode_index']}_index{INDEX}_{Path(item['video']).stem}"
    )

if __name__ == "__main__":
    main()
