# 将robocoin数据集(lerobot格式)中加入track信息，parquet中添加track和task_embed，并更新stas和info
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from glob import glob
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel
import decord

from atm.utils.flow_utils import sample_from_mask, sample_double_grid

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
VIDEO_DIR_NAME = "videos_new_81"
DATA_DIR_NAME = "data_new_81"

def get_task_embs(descriptions):
    """
    Bert embeddings for task embeddings.
    """
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
        print(torch.max(var))
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

def load_video_frames(video_path):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy() # Shape: [T, H, W, C]
    # Transform to [T, C, H, W]
    frames = rearrange(frames, "t h w c -> t c h w")
    return frames

def calculate_stats(array):
    """Calculate min, max, mean, std for a numpy array to match episodes_stats.jsonl format"""
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [len(array)]
    }

@click.command()
@click.option("--root", type=str, default="/data1/jibaixu/Datasets/Cobot_Magic_all_extracted/", help="Root directory containing Cobot_Magic_* datasets")
def main(root):
    root_path = Path(root)
    
    # Setup CoTracker
    cotracker_path = os.path.join(os.path.expanduser("~"), "/data2/jibaixu/Codes/co-tracker/")
    if not os.path.exists(cotracker_path):
        # Fallback to download if local path is invalid
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    else:
        cotracker = torch.hub.load(cotracker_path, "cotracker2", source="local")
    cotracker = cotracker.eval().cuda()

    # Discover datasets
    datasets = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Cobot_Magic_")]
    print(f"Found {len(datasets)} datasets.")

    for dataset_dir in datasets:
        print(f"\nProcessing dataset: {dataset_dir.name}")
        meta_dir = dataset_dir / "meta"
        tasks_file = meta_dir / "tasks.jsonl"
        info_file = meta_dir / "info.json"
        stats_file = meta_dir / "episodes_stats.jsonl"
        
        if not all(p.exists() for p in [tasks_file, info_file, stats_file]):
            print(f"Missing meta files in {dataset_dir.name}, skipping.")
            continue

        # Load tasks and generate BERT embeddings
        tasks_map = {}
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                tasks_map[item['task_index']] = item['task']
        
        # Precompute BERT embeddings for all tasks in this dataset
        task_embs = get_task_embs(list(tasks_map.values())).cpu().numpy()
        task_emb_dict = {task_idx: emb for task_idx, emb in zip(tasks_map.keys(), task_embs)}

        # Load info.json
        with open(info_file, 'r', encoding='utf-8') as f:
            info_data = json.load(f)

        # Identify video features to track
        video_views = [key for key, val in info_data['features'].items() if val.get('dtype') == 'video']

        # Load stats mapping
        stats_data = []
        with open(stats_file, 'r', encoding='utf-8') as f:
            for line in f:
                stats_data.append(json.loads(line.strip()))
        stats_dict = {item['episode_index']: item for item in stats_data}

        # Process all parquet files in data_clipped
        parquet_files = list((dataset_dir / DATA_DIR_NAME).rglob("*.parquet"))
        
        for parquet_path in tqdm(parquet_files, desc="Processing Episodes"):
            # Load parquet
            df = pd.read_parquet(parquet_path)
            T = len(df)
            
            # 1. Add BERT embedding
            # Assuming task_index is consistent within an episode
            # episode_task_idx = df['task_index'].iloc[0]
            # emb = task_emb_dict[episode_task_idx]
            # df['task_emb_bert'] = [emb] * T

            episode_idx = df['episode_index'].iloc[0]
            
            # 2. Process videos
            chunk_dir_name = parquet_path.parent.name # e.g. chunk-000
            episode_filename = parquet_path.stem # e.g. episode_000000
            
            for view in video_views:
                # view format: observation.images.cam_high_rgb
                view_name = view.split('.')[-1]
                video_path = dataset_dir / VIDEO_DIR_NAME / chunk_dir_name / view / f"{episode_filename}.mp4"
                
                assert video_path.exists(), "Missing video!"
                
                frames = load_video_frames(str(video_path))
                H, W = frames.shape[2], frames.shape[3]
                
                with torch.no_grad():
                    pred_tracks, pred_vis = track_through_video(frames, cotracker, num_points=1000)
                
                # Normalize coordinates to [0, 1]
                pred_tracks[:, :, :, 0] /= W
                pred_tracks[:, :, :, 1] /= H
                
                tracks_np = pred_tracks[0].cpu().numpy() # [T, N, 2]
                vis_np = pred_vis[0].cpu().numpy()       # [T, N]
                N_points = tracks_np.shape[1]

                # Convert to list of arrays to store in Parquet cell
                track_col_name = view.replace("images", "tracks")
                vis_col_name = view.replace("images", "vis")
                
                df[track_col_name] = list(tracks_np)
                df[vis_col_name] = list(vis_np)

                # Update stats for this episode
                stats_dict[episode_idx]["stats"][track_col_name] = calculate_stats(tracks_np)
                stats_dict[episode_idx]["stats"][vis_col_name] = calculate_stats(vis_np)
                
                # Update info.json feature definitions
                if track_col_name not in info_data['features']:
                    info_data['features'][track_col_name] = {
                        "dtype": "float32",
                        "shape": [N_points, 2],
                        "names": ["point_x", "point_y"]
                    }
                    info_data['features'][vis_col_name] = {
                        "dtype": "float32",
                        "shape": [N_points],
                        "names": ["visibility"]
                    }

            # 3. Add task_emb_bert to info.json if not present
            if "task_emb_bert" not in info_data['features']:
                info_data['features']["task_emb_bert"] = {
                    "dtype": "float32",
                    "shape": [768],
                    "names": None
                }

            # Save updated Parquet
            df.to_parquet(parquet_path, engine="pyarrow")

        # Save updated info.json
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=4)

        # Save updated episodes_stats.jsonl
        with open(stats_file, 'w', encoding='utf-8') as f:
            for ep_idx in sorted(stats_dict.keys()):
                f.write(json.dumps(stats_dict[ep_idx]) + "\n")
                
    print("All datasets processed successfully.")

if __name__ == "__main__":
    main()
