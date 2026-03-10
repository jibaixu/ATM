"""
display_2d_traj.py

示例：将世界坐标系下的笛卡尔位置投影到第一摄像机视频的像素平面，
并把轨迹绘制到第一帧上保存为图片。

说明/假设：
- 搜索并使用 parquet 文件中的 `observation.state.cartesian_position` 作为 3D 轨迹（取前三维 x,y,z）。
- 优先使用 parquet 中的 `camera_extrinsics.<camera>`（格式：[x,y,z,roll,pitch,yaw]，位置(m) + 欧拉角(弧度)）；
  若不存在则尝试从 `episodes_stats.jsonl` 中读取均值；仍不存在则使用默认猜测值。
- 若缺少相机内参（intrinsics），脚本会使用默认内参（fx=fy=500，cx=width/2, cy=height/2），
  可通过命令行参数 `--focal` 调整焦距。

用法示例：
python display_2d_traj.py --episode 2348 --camera exterior_1_left --data_root . --out outputs --focal 800
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import mediapy as mp
from PIL import Image, ImageDraw


def euler_to_rot(roll, pitch, yaw):
	"""按照 roll(x), pitch(y), yaw(z) 顺序生成旋转矩阵：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
	cx, sx = np.cos(roll), np.sin(roll)
	cy, sy = np.cos(pitch), np.sin(pitch)
	cz, sz = np.cos(yaw), np.sin(yaw)

	Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
	Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
	Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
	R = Rz @ Ry @ Rx
	return R


def project_points(points_world, extrinsic, intrinsics):
	"""points_world: (N,3); extrinsic: [tx,ty,tz,roll,pitch,yaw]
	intrinsics: dict with fx,fy,cx,cy
	返回像素坐标 (N,2) 和 深度 (N,)
	"""
	t = np.asarray(extrinsic[:3], dtype=float)
	roll, pitch, yaw = map(float, extrinsic[3:6])
	R = euler_to_rot(roll, pitch, yaw)

	# 相机位姿给出的是相机在世界坐标系的位置与姿态（camera-to-world），
	# 所以将世界点变换到相机坐标：p_cam = R.T @ (p_world - t)
	p_cam = (R.T @ (points_world - t).T).T

	X = p_cam[:, 0]
	Y = p_cam[:, 1]
	Z = p_cam[:, 2]

	fx = intrinsics['fx']
	fy = intrinsics['fy']
	cx = intrinsics['cx']
	cy = intrinsics['cy']

	# 防止除 0
	eps = 1e-6
	u = fx * (X / (Z + eps)) + cx
	v = fy * (Y / (Z + eps)) + cy
	return np.stack([u, v], axis=1), Z


def draw_trajectory_on_frames(frames, points, depths) -> list:
	# 仅保留前景（深度>0）和在图像内的点
	H, W = frames[0].shape[0], frames[0].shape[1]
	ok = (depths > 0) & (points[:, 0] >= 0) & (points[:, 0] < W) & (points[:, 1] >= 0) & (points[:, 1] < H)
	points_ok = points[ok]

	frames_with_traj = []
	# 核心修改：遍历每一帧，将整条轨迹绘制上去
	for i in range(len(frames)):
		im = Image.fromarray(frames[i])
		draw = ImageDraw.Draw(im)
		
		if len(points_ok) >= 2:
			pts = [tuple(p) for p in points_ok.tolist()]
			# 绘制整条轨迹线
			draw.line(pts, fill=(255, 0, 0), width=2)
			
			# 突出显示当前帧所在的轨迹点
			if ok[i]:
				curr_p = points[i]
				r = 4
				draw.ellipse((curr_p[0]-r, curr_p[1]-r, curr_p[0]+r, curr_p[1]+r), fill=(0, 255, 0))
		
		frames_with_traj.append(np.array(im))
	return frames_with_traj


"""
K = get_camera_intrinsic_matrix(env.sim, "agentview", RESIZED_SIZE[1], RESIZED_SIZE[0])
R_world_to_cam = get_camera_transform_matrix(env.sim, "agentview",  RESIZED_SIZE[1], RESIZED_SIZE[0])
K: 
[[154.510  0.000    64.000]
 [0.000    154.510  64.000]
 [0.000    0.000    1.000]]
R_world_to_cam:
[[-54.321  154.510 -33.841  70.700]
 [ 27.379  0.000   -164.984 82.692]
 [-0.849  -0.000   -0.529   1.105]
 [ 0.000   0.000    0.000   1.000]]
"""
def main():
	# python test.py --episode 2348 --camera exterior_1_left --data_root . --out outputs --focal 800
	parser = argparse.ArgumentParser()
	parser.add_argument('--episode', type=int, default=8331)
	parser.add_argument('--camera', type=str, default='exterior_1_left',
						help='列名或后缀，如 exterior_1_left / exterior_2_left / wrist_left')
	parser.add_argument('--data_root', type=str, default='/data2/jibaixu/Datasets/droid_1.0.1', help='数据根路径（包含 data/ videos/ 等）')
	parser.add_argument('--out', type=str, default='outputs', help='输出目录')
	parser.add_argument('--focal', type=float, default=125.0, help='默认焦距 fx=fy（若无内参时使用）')
	args = parser.parse_args()

	episode = args.episode
	# 支持 camera 参数传 'exterior_1_left' 或完整列名
	cam_suffix = args.camera
	if cam_suffix.startswith('observation.images'):
		cam_name = cam_suffix.split('.')[-1]
	else:
		cam_name = cam_suffix

	# 文件路径
	chunk_id = int(episode / 1000)
	parquet_path = os.path.join(args.data_root, f'data/chunk-{chunk_id:03d}', f'episode_{episode:06d}.parquet')
	video_path = os.path.join(args.data_root, f'videos/chunk-{chunk_id:03d}', f'observation.images.{cam_name}', f'episode_{episode:06d}.mp4')

	if not os.path.exists(parquet_path):
		raise FileNotFoundError(f'找不到 parquet 文件: {parquet_path}')
	if not os.path.exists(video_path):
		raise FileNotFoundError(f'找不到视频文件: {video_path}')

	print('读取 parquet:', parquet_path)
	df = pd.read_parquet(parquet_path)

	# 读取 3D 轨迹（取前三维）
	if 'observation.state.cartesian_position' not in df.columns:
		raise KeyError('parquet 中缺少 observation.state.cartesian_position 字段')
	states = np.array([np.asarray(x, dtype=float)[:3] for x in df['observation.state.cartesian_position'].tolist()])
	print(f"episode {episode} 读取到 {states.shape[0]} 个 3D 轨迹点")
	print(f"轨迹点示例: {states[:3]} ... {states[-3:]}")

	# 读取视频（只需要第一帧用于绘制）
	print('读取视频:', video_path)
	video = mp.read_video(video_path)  # (T,H,W,3) uint8
	if video is None or len(video) == 0:
		raise RuntimeError('无法读取视频或视频为空')
	first_frame = video[0]
	H, W = first_frame.shape[0], first_frame.shape[1]

	# 尝试从 parquet 或 episodes_stats.jsonl 中获取相机外参
	cam_key = f'camera_extrinsics.{cam_name}'
	extr = load_camera_extrinsic_from_parquet(df, cam_key)
	if extr is None:
		print(f'parquet 中未找到 {cam_key}，尝试从 episodes_stats.jsonl 获取均值')
		extr = load_camera_extrinsic_stats(args.data_root, cam_key)
	if extr is None:
		print('未找到相机外参，使用默认近似值（位置与朝向为经验猜测）')
		extr = np.array([0.33, 0.7, 0.49, -2.05, -0.015, -2.71])

	# 相机内参
	intrinsics = {'fx': args.focal, 'fy': args.focal, 'cx': W / 2.0, 'cy': H / 2.0}

	# 将轨迹与视频帧对齐：按视频帧数均匀采样轨迹点
	assert len(video) == states.shape[0], "视频帧数与轨迹点数不匹配，无法对齐"

	os.makedirs(args.out, exist_ok=True)
	video_out_path = os.path.join(args.out, f'traj_vis_{episode}_{cam_name}.mp4')

	pixels, depths = project_points(states, extr, intrinsics)
	frames_with_traj = draw_trajectory_on_frames(video, pixels, depths)

	# 使用 mediapy 保存视频，保持原始 FPS
	fps = 15 # 也可以从 info.json 或视频元数据获取
	mp.write_video(video_out_path, frames_with_traj, fps=fps)
	print('保存视频结果到', video_out_path)


if __name__ == '__main__':
	main()
