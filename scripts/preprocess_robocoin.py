import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click
import numpy as np
import torch
import decord
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# 导入您环境中的 CoTracker Utils
from atm.utils.flow_utils import sample_from_mask, sample_double_grid

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# --- 切分与数据量常量设置 ---
WINDOW_SIZE = 81
STRIDE = 81
VAL_PER_TASK = 2
TRAIN_TARGET_PER_TASK = 5000
TRAIN_OVERFLOW_MAX = 500
SEED = 42

BASE_DIR = Path("/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320")
TRAIN_OUTPUT_PATH = BASE_DIR / "episodes_clipped_train.jsonl"
VAL_OUTPUT_PATH = BASE_DIR / "episodes_clipped_val.jsonl"

VIDEO_DIR_NAME = "videos_clipped"
ACTION_DIR_NAME = "data_clipped"
TRACK_DIR_NAME = "tracks_clipped"


def get_task_embs(descriptions: List[str]):
    tz = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")
    tokens = tz(
        text=descriptions,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()
    return task_embs


def track_and_remove(tracker, video, points, var_threshold=5.):
    B, T, C, H, W = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=True)

    var = torch.var(pred_tracks, dim=1)
    var = torch.sum(var, dim=-1)[0]

    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        assert len(idx) > 0, 'No points with low variance'

    new_points = points[:, idx].clone()
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, :points.shape[1]]
    
    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * H
    new_points[:, :, 1:] += noise

    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=True)
    return pred_tracks, pred_vis


def track_through_video(video, track_model, num_points=1000):
    T, C, H, W = video.shape
    video_tensor = torch.from_numpy(video).cuda().float()

    points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().cuda()
    points = torch.cat([torch.randint_like(points[:, :1], 0, T), points], dim=-1).cuda()

    grid_points = sample_double_grid(7, device="cuda")
    grid_points[:, 0] = grid_points[:, 0] * H
    grid_points[:, 1] = grid_points[:, 1] * W
    grid_points = torch.cat([torch.randint_like(grid_points[:, :1], 0, T), grid_points], dim=-1).cuda()

    pred_tracks, pred_vis = track_and_remove(track_model, video_tensor[None], points[None])
    pred_grid_tracks, pred_grid_vis = track_and_remove(track_model, video_tensor[None], grid_points[None], var_threshold=0.)

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


def track_and_remove_chunk(tracker, video, points, var_threshold=5.):
    B, T, C, H, W = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=False)

    var = torch.var(pred_tracks, dim=1)
    var = torch.sum(var, dim=-1)[0]

    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        return pred_tracks, pred_vis

    new_points = points[:, idx].clone()
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, :points.shape[1]]
    
    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * H
    new_points[:, :, 1:] += noise

    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=False)
    return pred_tracks, pred_vis


def track_through_video_sliding_window(video, track_model, num_points=1000, chunk_size=60, overlap=10):
    T_total, C, H, W = video.shape
    video_tensor = torch.from_numpy(video).cuda().float().unsqueeze(0) 
    
    if T_total <= chunk_size:
        rand_pts = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
        rand_pts = torch.from_numpy(rand_pts).float().cuda()
        rand_queries = torch.cat([torch.zeros_like(rand_pts[:, :1]), rand_pts], dim=-1).unsqueeze(0)

        grid_points = sample_double_grid(7, device="cuda")
        grid_points[:, 0] = grid_points[:, 0] * H
        grid_points[:, 1] = grid_points[:, 1] * W
        grid_queries = torch.cat([torch.zeros_like(grid_points[:, :1]), grid_points], dim=-1).unsqueeze(0)

        pred_tracks_r, pred_vis_r = track_and_remove_chunk(track_model, video_tensor, rand_queries, var_threshold=5.)
        pred_tracks_g, pred_vis_g = track_and_remove_chunk(track_model, video_tensor, grid_queries, var_threshold=0.)
        
        pred_tracks = torch.cat([pred_tracks_g, pred_tracks_r], dim=2)
        pred_vis = torch.cat([pred_vis_g, pred_vis_r], dim=2)
        return pred_tracks, pred_vis

    grid_points = sample_double_grid(7, device="cuda")
    num_grid = grid_points.shape[0]
    
    global_tracks_rand = torch.zeros(1, T_total, num_points, 2).cuda()
    global_vis_rand = torch.zeros(1, T_total, num_points).cuda()
    
    global_tracks_grid = torch.zeros(1, T_total, num_grid, 2).cuda()
    global_vis_grid = torch.zeros(1, T_total, num_grid).cuda()

    rand_pts = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
    rand_pts = torch.from_numpy(rand_pts).float().cuda()
    rand_queries = torch.cat([torch.zeros_like(rand_pts[:, :1]), rand_pts], dim=-1).unsqueeze(0)
    
    grid_pts = grid_points.clone()
    grid_pts[:, 0] = grid_pts[:, 0] * H
    grid_pts[:, 1] = grid_pts[:, 1] * W
    grid_queries = torch.cat([torch.zeros_like(grid_pts[:, :1]), grid_pts], dim=-1).unsqueeze(0)

    step = chunk_size - overlap
    
    for start_idx in range(0, T_total, step):
        end_idx = min(start_idx + chunk_size, T_total)
        current_chunk_size = end_idx - start_idx
        chunk_video = video_tensor[:, start_idx:end_idx]
        
        if start_idx == 0:
            pred_tracks_r, pred_vis_r = track_and_remove_chunk(track_model, chunk_video, rand_queries, var_threshold=5.)
            pred_tracks_g, pred_vis_g = track_and_remove_chunk(track_model, chunk_video, grid_queries, var_threshold=0.)
        else:
            pred_tracks_r, pred_vis_r = track_model(chunk_video, queries=rand_queries, backward_tracking=False)
            pred_tracks_g, pred_vis_g = track_model(chunk_video, queries=grid_queries, backward_tracking=False)
            
        global_tracks_rand[:, start_idx:end_idx] = pred_tracks_r
        global_vis_rand[:, start_idx:end_idx] = pred_vis_r
        
        global_tracks_grid[:, start_idx:end_idx] = pred_tracks_g
        global_vis_grid[:, start_idx:end_idx] = pred_vis_g
        
        if end_idx < T_total:
            t_next = step
            if t_next >= current_chunk_size:
                break
                
            x_r = pred_tracks_r[0, t_next, :, 0]
            y_r = pred_tracks_r[0, t_next, :, 1]
            rand_queries = torch.stack([torch.zeros_like(x_r), x_r, y_r], dim=-1).unsqueeze(0)
            
            x_g = pred_tracks_g[0, t_next, :, 0]
            y_g = pred_tracks_g[0, t_next, :, 1]
            grid_queries = torch.stack([torch.zeros_like(x_g), x_g, y_g], dim=-1).unsqueeze(0)

    pred_tracks = torch.cat([global_tracks_grid, global_tracks_rand], dim=2)
    pred_vis = torch.cat([global_vis_grid, global_vis_rand], dim=2)
    
    return pred_tracks, pred_vis


def track_by_independent_segments(video, track_model, num_points=1000, segment_size=81):
    T_total, C, H, W = video.shape
    video_tensor = torch.from_numpy(video).cuda().float().unsqueeze(0) 
    
    grid_points_template = sample_double_grid(7, device="cuda")
    num_grid = grid_points_template.shape[0]
    
    global_tracks = torch.zeros(1, T_total, num_grid + num_points, 2).cuda()
    global_vis = torch.zeros(1, T_total, num_grid + num_points).cuda()

    # 计算分段索引：确保后面的段是完整的 segment_size
    indices = []
    curr_end = T_total
    while curr_end > 0:
        start = max(0, curr_end - segment_size)
        indices.append((start, curr_end))
        curr_end -= segment_size
    
    # 按时间顺序处理分段：[(0, 19), (19, 100)] (假设总长100)
    indices.sort()

    for start_idx, end_idx in indices:
        current_chunk = video_tensor[:, start_idx:end_idx]
        curr_h, curr_w = current_chunk.shape[3], current_chunk.shape[4]
        
        # 这里的逻辑保持不变，但 current_chunk 的范围已经对齐了末尾
        rand_pts = sample_from_mask(np.ones((curr_h, curr_w, 1)) * 255, num_samples=num_points)
        rand_pts = torch.from_numpy(rand_pts).float().cuda()
        rand_queries = torch.cat([torch.zeros_like(rand_pts[:, :1]), rand_pts], dim=-1).unsqueeze(0)

        grid_pts = grid_points_template.clone()
        grid_pts[:, 0] *= curr_h
        grid_pts[:, 1] *= curr_w
        grid_queries = torch.cat([torch.zeros_like(grid_pts[:, :1]), grid_pts], dim=-1).unsqueeze(0)

        with torch.no_grad():
            pred_tracks_r, pred_vis_r = track_and_remove_chunk(track_model, current_chunk, rand_queries, var_threshold=5.)
            pred_tracks_g, pred_vis_g = track_and_remove_chunk(track_model, current_chunk, grid_queries, var_threshold=0.)
        
        chunk_tracks = torch.cat([pred_tracks_g, pred_tracks_r], dim=2)
        chunk_vis = torch.cat([pred_vis_g, pred_vis_r], dim=2)

        global_tracks[:, start_idx:end_idx] = chunk_tracks
        global_vis[:, start_idx:end_idx] = chunk_vis

    return global_tracks, global_vis


def load_video_frames(video_path: str):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy() 
    frames = rearrange(frames, "t h w c -> t c h w")
    return frames


# --- 统一的切分逻辑 ---
def build_segments(episodes: List[Dict], target_indices: Set[int]) -> List[Dict]:
    """
    修改后的切分函数：从视频末尾向前回溯，确保最后一段窗口是完整的 81 帧。
    """
    segments = []
    for ep in episodes:
        if ep["episode_index"] not in target_indices:
            continue
            
        raw_length = ep["raw_length"]
        
        # 从 (总长度 - 窗口大小) 开始倒序计算起始点
        # 例如 raw_length=100, WINDOW_SIZE=81: start 依次为 19, -62(停止)
        # 这样生成的窗口就是 [19, 99]，而舍弃开头的 [0, 18]
        current_starts = []
        for start in range(raw_length - WINDOW_SIZE, -1, -STRIDE):
            current_starts.append(start)
        
        # 为了保持 jsonl 文件中的时间顺序，我们反转回来
        current_starts.sort()

        for start in current_starts:
            end = start + WINDOW_SIZE - 1
            
            # 由于起始点计算逻辑，这里 end 必然 <= raw_length - 1
            ep_copy = ep.copy()
            ep_copy.update({
                "length": WINDOW_SIZE,
                "start_frame": start,
                "end_frame": end,
            })
            segments.append(ep_copy)
                
    return segments


def select_train_segments_by_episode(segments, rng, target_per_task, max_overflow):
    by_episode = {}
    for seg in segments:
        episode_index = int(seg["episode_index"])
        by_episode.setdefault(episode_index, []).append(seg)

    episode_indices = list(by_episode.keys())
    rng.shuffle(episode_indices)

    selected_episode_indices = []
    overflow_candidates = []
    skipped_too_long = 0
    kept_clip_count = 0

    for ep_idx in episode_indices:
        clip_count = len(by_episode[ep_idx])
        if clip_count > target_per_task:
            skipped_too_long += 1
            continue
        if kept_clip_count + clip_count <= target_per_task:
            selected_episode_indices.append(ep_idx)
            kept_clip_count += clip_count
        else:
            overflow_candidates.append(ep_idx)

    overflow_applied = 0
    best_overflow_episode, best_overflow_total = None, None
    overflow_limit = target_per_task + max_overflow
    
    for ep_idx in overflow_candidates:
        candidate_total = kept_clip_count + len(by_episode[ep_idx])
        if target_per_task < candidate_total <= overflow_limit:
            if best_overflow_total is None or candidate_total < best_overflow_total:
                best_overflow_total = candidate_total
                best_overflow_episode = ep_idx

    if best_overflow_episode is not None:
        selected_episode_indices.append(best_overflow_episode)
        kept_clip_count += len(by_episode[best_overflow_episode])
        overflow_applied = 1

    kept_segments = [seg for ep_idx in selected_episode_indices for seg in by_episode[ep_idx]]

    stats = {
        "train_generated_total": len(segments),
        "train_kept": len(kept_segments),
        "selected_train_episodes": len(selected_episode_indices),
        "skipped_too_long": skipped_too_long,
        "overflow_applied": overflow_applied,
    }
    return kept_segments, stats


def write_jsonl(path: Path, entries: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


@click.command()
@click.option("--skip_exist", type=bool, default=True, help="Skip tracking if .npz file already exists")
@click.option("--dataset_idx", type=int, default=-1, help="Index of the dataset to process (e.g., 2)")
@click.option("--view", type=click.Choice(['high', 'left', 'right', 'all']), default="all", help="Camera view to process")
@click.option("--min_episode", type=int, default=0, help="Minimum episode index to start processing")
def main(skip_exist, dataset_idx, view, min_episode):
    rng = random.Random(SEED)
    
    cotracker_path = os.path.join(os.path.expanduser("~"), "/home/jibaixu/Codes/co-tracker/")
    if os.path.exists(cotracker_path):
        cotracker = torch.hub.load(cotracker_path, "cotracker2", source="local")
    else:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    cotracker = cotracker.eval().cuda()

    datasets = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("Cobot_Magic_")])
    if dataset_idx >= 0:
        datasets = datasets[dataset_idx : dataset_idx + 1]
    print(f"##### Preprocess Datasets Robocoin include: {datasets}")
    print(f"##### Processing view: {view}, min_episode: {min_episode}")

    all_train = []
    all_val = []
    dataset_summaries = []

    print(f"Found {len(datasets)} datasets.")

    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        meta_dir = dataset_dir / "meta"
        tasks_file = meta_dir / "tasks.jsonl"
        episodes_file = meta_dir / "episodes_clipped.jsonl"
        
        assert all(p.exists() for p in [tasks_file, episodes_file])

        print(f"\nProcessing {dataset_name}...")

        task_embed_dir = dataset_dir / "task_embed" / "bert"
        task_embed_dir.mkdir(parents=True, exist_ok=True)
        
        tasks_map = {}
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                tasks_map[item['task']] = item['task_index']
                
        descriptions = list(tasks_map.keys())
        task_embs = get_task_embs(descriptions)
        
        for desc, emb in zip(descriptions, task_embs):
            idx = tasks_map[desc]
            pt_path = task_embed_dir / f"{idx}.pt"
            torch.save(emb, pt_path)

        # 将硬编码的 video_views 替换为字典映射
        view_map = {
            'high': 'observation.images.cam_high_rgb',
            'left': 'observation.images.cam_left_wrist_rgb',
            'right': 'observation.images.cam_right_wrist_rgb'
        }
        if view == 'all':
            views = ['high', 'left', 'right']
        else:
            views = [view]
        video_views = [view_map[v] for v in views]

        base_episodes = []
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting Tracks"):
                item = json.loads(line.strip())
                if item["episode_index"] < min_episode or item["episode_index"] > 700:
                    continue
                
                # --- 新增过滤逻辑：过滤掉初始长度不够 WINDOW_SIZE (81帧) 的片段 ---
                if item["length"] < WINDOW_SIZE:
                    continue
                
                ep_idx = int(item["episode_index"])
                prompt = item["prompt"]
                task_index = tasks_map[prompt]
                
                records = []

                video_parts = Path(item["video"]).parts
                chunk_dir = [p for p in video_parts if "chunk-" in p][0]
                ep_filename = Path(item["video"]).stem
                
                for view in video_views:
                    track_view_name = view.replace("images", "tracks")

                    npz_save_dir = dataset_dir / TRACK_DIR_NAME / chunk_dir / track_view_name
                    npz_save_dir.mkdir(parents=True, exist_ok=True)
                    npz_save_path = npz_save_dir / f"{ep_filename}.npz"
                    
                    actual_video_path = dataset_dir / VIDEO_DIR_NAME / chunk_dir / view / f"{ep_filename}.mp4"

                    video_path = f"{dataset_name}/{VIDEO_DIR_NAME}/{chunk_dir}/{view}/{ep_filename}.mp4"
                    action_path = f"{dataset_name}/{ACTION_DIR_NAME}/{chunk_dir}/{ep_filename}.parquet"
                    track_path = f"{dataset_name}/{TRACK_DIR_NAME}/{chunk_dir}/{track_view_name}/{ep_filename}.npz"

                    record = {
                        "episode_index": ep_idx,
                        "video": video_path,
                        "action": action_path,
                        "raw_action": action_path,
                        "track": track_path,
                        "prompt": prompt,
                        "dataset_name": dataset_name,
                        "prompt_embed_bert": f"{dataset_name}/task_embed/bert/{task_index}.pt"
                    }

                    record["raw_length"] = item['length']
                    records.append(record)
                    assert actual_video_path.exists(), f"Video not found: {actual_video_path}"
                    if skip_exist and npz_save_path.exists():
                        continue
                    frames = load_video_frames(str(actual_video_path))
                    raw_length, H, W = frames.shape[0], frames.shape[2], frames.shape[3]
                    assert raw_length == item['length'], f"Raw length mismatch for {actual_video_path}: expected {item['length']}, got {raw_length}"

                    with torch.no_grad():
                        pred_tracks, pred_vis = track_by_independent_segments(frames, cotracker, num_points=1000, segment_size=WINDOW_SIZE)
                    
                    pred_tracks[:, :, :, 0] /= W
                    pred_tracks[:, :, :, 1] /= H
                    tracks_np = pred_tracks[0].cpu().numpy()
                    vis_np = pred_vis[0].cpu().numpy()

                    np.savez_compressed(npz_save_path, tracks=tracks_np, vis=vis_np)
                
                base_episodes.extend(records)
                
        prompt_to_eps = {}
        for ep in base_episodes:
            p = ep["prompt"]
            if p not in prompt_to_eps:
                prompt_to_eps[p] = set()
            prompt_to_eps[p].add(ep["episode_index"])

        val_indices = set()
        for p, ep_ids in prompt_to_eps.items():
            sorted_ids = sorted(list(ep_ids))
            task_val_ids = sorted_ids[-VAL_PER_TASK:]
            val_indices.update(task_val_ids)

        # 所有的 episode ID
        all_ep_ids = set(ep["episode_index"] for ep in base_episodes)
        # 训练集的 episode ID
        train_indices = all_ep_ids - val_indices

        # --- 统一调用 build_segments 进行定长切分 ---
        val_entries = build_segments(base_episodes, val_indices)
        train_entries = build_segments(base_episodes, train_indices)
        
        if not train_entries:
            print(f"Warning: No train segments for {dataset_name} after validation split.")
            train_stats = {
                "selected_train_episodes": 0, "train_generated_total": 0, 
                "train_kept": 0, "skipped_too_long": 0, "overflow_applied": 0
            }
        else:
            train_entries, train_stats = select_train_segments_by_episode(
                segments=train_entries,
                rng=rng,
                target_per_task=TRAIN_TARGET_PER_TASK,
                max_overflow=TRAIN_OVERFLOW_MAX,
            )

        all_val.extend(val_entries)
        all_train.extend(train_entries)

        total_clips = len(train_entries) + len(val_entries)
        total_episodes = train_stats["selected_train_episodes"] + len(val_indices)
        dataset_summaries.append({
            "dataset": dataset_name,
            "episodes_total": len(base_episodes),
            "train_clips": len(train_entries),
            "train_episodes": train_stats["selected_train_episodes"],
            "val_clips": len(val_entries),
            "val_episodes": len(val_indices),
            "total_clips": total_clips,
            "total_episodes": total_episodes,
        })

    all_train.sort(key=lambda x: (x["dataset_name"], x["episode_index"], x["start_frame"]))
    all_val.sort(key=lambda x: (x["dataset_name"], x["episode_index"], x["start_frame"]))

    write_jsonl(TRAIN_OUTPUT_PATH, all_train)
    write_jsonl(VAL_OUTPUT_PATH, all_val)

    print(f"\nWrote train: {len(all_train)} -> {TRAIN_OUTPUT_PATH}")
    print(f"Wrote val:   {len(all_val)} -> {VAL_OUTPUT_PATH}")
    print("\n=== Per-Dataset Summary ===")
    for item in dataset_summaries:
        print(
            f"{item['dataset']}: "
            f"total_clips={item['total_clips']}, total_episodes={item['total_episodes']}, "
            f"train_clips={item['train_clips']}, val_clips={item['val_clips']}"
        )


if __name__ == "__main__":
    main()
