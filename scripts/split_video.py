import cv2
import os
from pathlib import Path
import subprocess

def split_video_single(input_video_path, output_dir, segment_frames=81):
    """
    将单个视频按固定帧数切分为多个片段
    
    Args:
        input_video_path: 输入视频路径（字符串或Path）
        output_dir: 输出目录
        segment_frames: 每个片段的帧数（默认81）
    """
    input_path = Path(input_video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"处理视频: {input_path}")
    print(f"输出目录: {output_dir}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"总帧数: {total_frames}, FPS: {fps}, 分辨率: {width}x{height}")
    
    # 计算片段数
    num_segments = (total_frames + segment_frames - 1) // segment_frames
    print(f"切分为 {num_segments} 个片段 (每段 {segment_frames} 帧)")
    print("-" * 50)
    
    # 逐段切分（使用FFmpeg保证Decord兼容）
    for seg_idx in range(num_segments):
        start_frame = seg_idx * segment_frames
        end_frame = min(start_frame + segment_frames, total_frames)
        actual_frames = end_frame - start_frame
        
        output_name = f"{input_path.stem}_seg{seg_idx+1:04d}.mp4"
        output_path = output_dir / output_name
        
        # FFmpeg切分 - Decord兼容格式 (H.264 + yuv420p)
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', str(start_frame / fps),      # 起始时间
            '-i', str(input_path),
            '-c:v', 'libx264',                   # H.264编码
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',               # Decord必需
            '-movflags', '+faststart',
            '-frames:v', str(actual_frames),     # 只编码指定帧数
            '-r', str(fps),                      # 保持原FPS
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {output_name} (帧 {start_frame}-{end_frame-1}, {actual_frames}帧)")
        else:
            print(f"✗ {output_name} 失败: {result.stderr[:100]}")
    
    print("-" * 50)
    print(f"完成！共生成 {num_segments} 个片段")
    return num_segments

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 修改为你的路径
    VIDEO_PATH = "results/vis_dataset/tmp5/ep_8_episode_000008_pred_track.mp4"    # ← 修改：输入视频路径
    OUTPUT_DIR = "results/vis_dataset/tmp5"    # ← 修改：输出目录
    
    # 单次处理一个视频
    split_video_single(VIDEO_PATH, OUTPUT_DIR, segment_frames=81)
