import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import decord

import cv2
import imageio

from atm.utils.cotracker_utils import Visualizer

# --- 基础配置 ---
# 根据 preprocess_robocoin_2.py 中的设置 
BASE_DIR = Path("/data_jbx/Datasets/RoboTwin2.0_lerobot_v2")
VAL_JSONL = BASE_DIR / "episodes_val_worldarena.jsonl"
SAVE_DIR = "results/vis_dataset/worldarena"
INDEX = 0

def _load_video_frames_decord(video_path, frame_indices=None):
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    if frame_indices is None:
        frame_indices = range(len(vr))
    return vr.get_batch(frame_indices).asnumpy()


def _load_video_frames_imageio(video_path, frame_indices=None):
    if imageio is None:
        raise RuntimeError("imageio is not available")

    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        if frame_indices is None:
            for frame in reader:
                frames.append(frame)
        else:
            for idx in frame_indices:
                frames.append(reader.get_data(int(idx)))
    finally:
        reader.close()

    if not frames:
        raise RuntimeError("imageio decoded 0 frames")
    return np.stack(frames, axis=0)


def _load_video_frames_cv2(video_path, frame_indices=None):
    if cv2 is None:
        raise RuntimeError("opencv-python is not available")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("cv2.VideoCapture failed to open video")

    frames = []
    try:
        if frame_indices is None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"cv2 failed to read frame {idx}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("cv2 decoded 0 frames")
    return np.stack(frames, axis=0)


def _load_video_frames(video_path, frame_indices=None):
    errors = []
    loaders = [
        ("decord", _load_video_frames_decord),
        ("imageio", _load_video_frames_imageio),
        ("cv2", _load_video_frames_cv2),
    ]

    for backend_name, loader in loaders:
        try:
            return loader(video_path, frame_indices=frame_indices)
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")

    raise RuntimeError(
        f"Failed to decode video {video_path}. "
        f"Backends tried: {'; '.join(errors)}"
    )


# --- 辅助函数：加载视频 ---
def load_video_to_tensor(video_path):
    """将视频加载为 (B, T, C, H, W) 格式的 Tensor"""
    frames = _load_video_frames(video_path)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return frames.unsqueeze(0)

# --- 主程序 ---
def main():
    # 1. 随机选取数据
    with open(VAL_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        item = json.loads(lines[INDEX]) # 随机选一行 

    print(f"Visualizing episode: {item['episode_index']}")

    start_frame, end_frame = item['start_frame'], item['end_frame']

    # 2. 加载视频和轨迹
    video_path = BASE_DIR / item['video']
    episode_name = Path(item['video']).stem
    track_path = BASE_DIR / item['track']
    
    video_tensor = load_video_to_tensor(video_path) # (B, T, C, H, W)
    video_tensor = video_tensor[:, start_frame:end_frame+1, :, :, :] # 裁剪到指定帧范围
    # 双线性插值调整到 240*320 分辨率
    B, T, C, H, W = video_tensor.shape
    video_reshaped = video_tensor.view(B * T, C, H, W)
    video_tensor = F.interpolate(video_reshaped, size=(240, 320), mode='bilinear', align_corners=False)
    video_tensor = video_tensor.view(B, T, C, 240, 320)
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
        filename=f"ep_{item['episode_index']}_index{INDEX}_{Path(item['video']).stem}_1000"
    )

if __name__ == "__main__":
    main()
