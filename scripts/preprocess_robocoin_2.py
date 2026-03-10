import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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

# --- 切分与数据量常量设置 (与 build_train_jsonl.py 保持一致) ---
WINDOW_SIZE = 81
STRIDE = 80
VAL_PER_TASK = 10
TRAIN_TARGET_PER_TASK = 4000
TRAIN_OVERFLOW_MAX = 200
SEED = 42

BASE_DIR = Path("/data1/jibaixu/Datasets/Cobot_Magic_all_extracted")
TRAIN_OUTPUT_PATH = BASE_DIR / "episodes_clipped_train.jsonl"
VAL_OUTPUT_PATH = BASE_DIR / "episodes_clipped_val.jsonl"


def get_task_embs(descriptions: List[str]):
    tz = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")
    tokens = tz(
        text=descriptions,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        # truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()
    return task_embs


def track_and_remove(tracker, video, points, var_threshold=10.):
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


def load_video_frames(video_path: str):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy() 
    frames = rearrange(frames, "t h w c -> t c h w")
    return frames


# --- 切分与采样逻辑提取 ---
def build_val_entries(episodes: List[Dict], val_indices: Set[int]) -> List[Dict]:
    entries = []
    for ep in episodes:
        if ep["episode_index"] not in val_indices:
            continue
        raw_length = ep["raw_length"]
        if raw_length <= 0:
            continue
        ep_copy = ep.copy()
        ep_copy.update({
            "length": raw_length,
            "start_frame": 0,
            "end_frame": raw_length - 1,
        })
        entries.append(ep_copy)
    return entries


def build_train_segments(episodes: List[Dict], excluded_indices: Set[int]) -> List[Dict]:
    segments = []
    for ep in episodes:
        if ep["episode_index"] in excluded_indices:
            continue
        raw_length = ep["raw_length"]
        if raw_length <= 0:
            continue

        for start in range(0, raw_length, STRIDE):
            end = min(start + WINDOW_SIZE - 1, raw_length - 1)
            seg_len = end - start + 1
            if end == raw_length - 1 and seg_len < 30:
                continue
            
            ep_copy = ep.copy()
            ep_copy.update({
                "length": seg_len,
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
def main(skip_exist):
    rng = random.Random(SEED)
    
    # 1. 挂载 CoTracker
    cotracker_path = os.path.join(os.path.expanduser("~"), "/data2/jibaixu/Codes/co-tracker/")
    if os.path.exists(cotracker_path):
        cotracker = torch.hub.load(cotracker_path, "cotracker2", source="local")
    else:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    cotracker = cotracker.eval().cuda()

    datasets = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("Cobot_Magic_")])

    all_train = []          # 存储到episode_xxx_rain.jsonl
    all_val = []            # 存储到episode_xxx_val.jsonl
    dataset_summaries = []  # 仅打印输出展示

    print(f"Found {len(datasets)} datasets.")

    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        meta_dir = dataset_dir / "meta"
        tasks_file = meta_dir / "tasks.jsonl"
        episodes_file = meta_dir / "episodes_clipped.jsonl"
        info_file = meta_dir / "info.json"
        
        assert all(p.exists() for p in [tasks_file, episodes_file, info_file])
            
        print(f"\nProcessing {dataset_name}...")

        # --- A. 处理并保存文本 Embedding ---
        task_embed_dir = dataset_dir / "task_embed"
        task_embed_dir.mkdir(exist_ok=True)
        
        tasks_map = {}  # prompt -> task_index mapping
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                tasks_map[item['task']] = item['task_index']
                
        # 计算整个数据集的所有 unique descriptions
        descriptions = list(tasks_map.keys())
        task_embs = get_task_embs(descriptions)
        
        # 将每个 task 单独存入 task_embed/xxx.pt
        for desc, emb in zip(descriptions, task_embs):
            idx = tasks_map[desc]
            pt_path = task_embed_dir / f"{idx}.pt"
            torch.save(emb, pt_path)

        # --- B. 读取 info.json 以确定多视角 ---
        with open(info_file, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        video_views = [k for k, v in info_data['features'].items() if v.get('dtype') == 'video']

        # --- C. 构建基础轨迹信息并运行 CoTracker ---
        base_episodes = []
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Extracting Tracks"):
                item = json.loads(line.strip())
                ep_idx = int(item["episode_index"])
                prompt = item["prompt"]
                raw_length = int(item.get("raw_length", item["length"]))
                task_index = tasks_map[prompt]
                
                # 初始化轨迹字典
                record = {
                    "episode_index": ep_idx,
                    "raw_length": raw_length,
                    "video": str(item["video"]),
                    "action": str(item["action"]),
                    "track": {},  # 按照视角记录 npz 路径
                    "prompt": prompt,
                    "task": dataset_name,
                    "task_index": task_index,
                    "task_embed": f"{dataset_name}/task_embed/{task_index}.pt"
                }

                # 从 video 路径中提取 chunk 和文件名，例如：
                # video: Cobot_Magic_.../videos_clipped/chunk-000/observation.images.cam_all_views_rgb/episode_000000.mp4
                video_parts = Path(item["video"]).parts
                chunk_dir = [p for p in video_parts if "chunk-" in p][0]
                ep_filename = Path(item["video"]).stem  # episode_000000
                
                # 针对每个视角进行 Tracking
                for view in video_views:
                    view_name = view.split('.')[-1]
                    track_view_name = view.replace("images", "tracks")
                    
                    # 构造原有视频路径
                    actual_video_path = dataset_dir / "videos_clipped" / chunk_dir / view / f"{ep_filename}.mp4"
                    # 构造新的 npz 保存路径
                    npz_save_dir = dataset_dir / "tracks_clipped" / chunk_dir / track_view_name
                    npz_save_dir.mkdir(parents=True, exist_ok=True)
                    npz_save_path = npz_save_dir / f"{ep_filename}.npz"
                    
                    # 记录到 record
                    rel_npz_path = f"{dataset_name}/tracks_clipped/{chunk_dir}/{track_view_name}/{ep_filename}.npz"
                    record["track"][view] = rel_npz_path

                    if not actual_video_path.exists():
                        continue
                        
                    if skip_exist and npz_save_path.exists():
                        continue

                    # 加载视频并追踪
                    frames = load_video_frames(str(actual_video_path))
                    H, W = frames.shape[2], frames.shape[3]
                    
                    with torch.no_grad():
                        pred_tracks, pred_vis = track_through_video(frames, cotracker, num_points=1000)
                    
                    # 归一化并转移到CPU
                    pred_tracks[:, :, :, 0] /= W
                    pred_tracks[:, :, :, 1] /= H
                    tracks_np = pred_tracks[0].cpu().numpy()
                    vis_np = pred_vis[0].cpu().numpy()

                    # 使用压缩模式保存为 npz
                    np.savez_compressed(npz_save_path, tracks=tracks_np, vis=vis_np)
                
                base_episodes.append(record)
                
        # --- D. 切分数据集 ---
        if len(base_episodes) < VAL_PER_TASK:
            print(f"Warning: {dataset_name} has only {len(base_episodes)} episodes, skipping split.")
            continue
            
        all_indices = [ep["episode_index"] for ep in base_episodes]
        val_indices = set(rng.sample(all_indices, VAL_PER_TASK))

        val_entries = build_val_entries(base_episodes, val_indices)
        train_entries = build_train_segments(base_episodes, val_indices)
        train_entries, train_stats = select_train_segments_by_episode(
            segments=train_entries,
            rng=rng,
            target_per_task=TRAIN_TARGET_PER_TASK,
            max_overflow=TRAIN_OVERFLOW_MAX,
        )

        all_val.extend(val_entries)
        all_train.extend(train_entries)

        total_clips = len(train_entries) + len(val_entries)
        total_episodes = train_stats["selected_train_episodes"] + len(val_entries)
        dataset_summaries.append({
            "dataset": dataset_name,
            "episodes_total": len(base_episodes),
            "train_clips": len(train_entries),
            "train_episodes": train_stats["selected_train_episodes"],
            "val_clips": len(val_entries),
            "val_episodes": len(val_entries),
            "total_clips": total_clips,
            "total_episodes": total_episodes,
        })

    # 排序并写入全局文件
    all_train.sort(key=lambda x: (x["task"], x["episode_index"], x["start_frame"]))
    all_val.sort(key=lambda x: (x["task"], x["episode_index"], x["start_frame"]))

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
