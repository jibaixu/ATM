import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click

try:
    import imageio
except ImportError:
    imageio = None
try:
    import cv2
except ImportError:
    cv2 = None
import decord
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from atm.utils.flow_utils import sample_double_grid, sample_from_mask

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

BASE_DIR = Path("/data_jbx/Datasets/Realbot")
TRAIN_OUTPUT_PATH = BASE_DIR / "episodes_train_realbot.jsonl"
VAL_OUTPUT_PATH = BASE_DIR / "episodes_val_realbot.jsonl"

VIDEO_DIR_NAME = "videos"
TRACK_DIR_NAME = "tracks"
PROMPT_EMB_DIR_NAME = "prompt_emb"
NUM_TRACK_POINTS = 1000
WINDOW_SIZE = 81
STRIDE = 10
VAL_PER_TASK = 5

VIEW_MAP = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}


def get_task_embs(descriptions: List[str]) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")
    tokens = tokenizer(
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


def track_and_remove(tracker, video, points, var_threshold=5.0):
    _, _, _, height, _ = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=True)

    var = torch.var(pred_tracks, dim=1)
    var = torch.sum(var, dim=-1)[0]

    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        print(torch.max(var))
        assert len(idx) > 0, "No points with low variance"

    new_points = points[:, idx].clone()
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, : points.shape[1]]

    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * height
    new_points[:, :, 1:] += noise

    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=True)
    return pred_tracks, pred_vis


def track_through_video(video, track_model, num_points=NUM_TRACK_POINTS):
    num_frames, _, height, width = video.shape
    video_tensor = torch.from_numpy(video).cuda().float()

    points = sample_from_mask(np.ones((height, width, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().cuda()
    points = torch.cat([torch.randint_like(points[:, :1], 0, num_frames), points], dim=-1).cuda()

    grid_points = sample_double_grid(7, device="cuda")
    grid_points[:, 0] = grid_points[:, 0] * height
    grid_points[:, 1] = grid_points[:, 1] * width
    grid_points = torch.cat(
        [torch.randint_like(grid_points[:, :1], 0, num_frames), grid_points],
        dim=-1,
    ).cuda()

    pred_tracks, pred_vis = track_and_remove(track_model, video_tensor[None], points[None])
    pred_grid_tracks, pred_grid_vis = track_and_remove(
        track_model,
        video_tensor[None],
        grid_points[None],
        var_threshold=0.0,
    )

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


def _resize_video_frames(frames: np.ndarray, target_size=None) -> np.ndarray:
    if target_size is None:
        return frames

    frames_tensor = torch.from_numpy(frames).float()
    frames_tensor = F.interpolate(
        frames_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return frames_tensor.numpy()


def _load_video_frames_decord(video_path: str) -> np.ndarray:
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy()
    frames = rearrange(frames, "t h w c -> t c h w")
    if frames.shape[0] == 0:
        raise RuntimeError("decord decoded 0 frames")
    return frames


def _load_video_frames_cv2(video_path: str) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is not available")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("cv2.VideoCapture failed to open video")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("cv2 decoded 0 frames")

    frames = np.stack(frames, axis=0)
    frames = rearrange(frames, "t h w c -> t c h w")
    return frames


def _load_video_frames_imageio(video_path: str) -> np.ndarray:
    if imageio is None:
        raise RuntimeError("imageio is not available")

    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        raise RuntimeError("imageio decoded 0 frames")

    frames = np.stack(frames, axis=0)
    frames = rearrange(frames, "t h w c -> t c h w")
    return frames


def load_video_frames(
    video_path: str,
    target_size=None,
    dataset_name: str = "",
    episode_index: Optional[int] = None,
    video_view: str = "",
):
    errors = []
    loaders = [
        ("decord", _load_video_frames_decord),
        ("imageio", _load_video_frames_imageio),
        ("cv2", _load_video_frames_cv2),
    ]

    for backend_name, loader in loaders:
        try:
            frames = loader(video_path)
            return _resize_video_frames(frames, target_size=target_size)
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")

    context_parts = [f"path={video_path}"]
    if dataset_name:
        context_parts.append(f"dataset={dataset_name}")
    if episode_index is not None:
        context_parts.append(f"episode_index={episode_index}")
    if video_view:
        context_parts.append(f"view={video_view}")

    context = ", ".join(context_parts)
    error_summary = "; ".join(errors)
    raise RuntimeError(f"Failed to decode video ({context}). Backends tried: {error_summary}")


def write_jsonl(path: Path, entries: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def resolve_prompt(episode_item: Dict) -> str:
    prompt = episode_item.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt

    tasks = episode_item.get("tasks")
    assert isinstance(tasks, list) and len(tasks) > 0, f"Invalid tasks field: {tasks}"
    prompt = tasks[0]
    assert isinstance(prompt, str) and prompt, f"Invalid task prompt: {prompt}"
    return prompt


def get_selected_video_views(view: str) -> List[str]:
    if view == "all":
        return list(VIEW_MAP.values())
    return [VIEW_MAP[view]]


def get_video_relative_path(entry: Dict, video_view: str) -> str:
    video_field = entry["video"]
    if isinstance(video_field, str):
        candidates = [video_field]
    else:
        candidates = list(video_field)

    for rel_path in candidates:
        if video_view in rel_path:
            return rel_path

    raise KeyError(f"Could not find view {video_view} in video field: {video_field}")


def build_track_relative_path(video_relative_path: str) -> str:
    video_path = Path(video_relative_path)
    parts = video_path.parts
    assert len(parts) >= 4 and parts[0] == VIDEO_DIR_NAME, f"Unexpected video path: {video_relative_path}"

    chunk_name = parts[1]
    view_name = parts[2]
    track_view_name = view_name.replace("images", "tracks")
    return str(Path(TRACK_DIR_NAME) / chunk_name / track_view_name / f"{video_path.stem}.npz")


def prefix_dataset_path(dataset_name: str, relative_path: str) -> str:
    return f"{dataset_name}/{Path(relative_path).as_posix()}"


def load_tasks(tasks_file: Path) -> List[Dict]:
    task_items = read_jsonl(tasks_file)
    task_items.sort(key=lambda item: int(item["task_index"]))
    return task_items


def save_task_bert_embeddings(task_items: List[Dict], dataset_dir: Path) -> Dict[str, int]:
    prompt_embed_dir = dataset_dir / PROMPT_EMB_DIR_NAME / "bert"
    prompt_embed_dir.mkdir(parents=True, exist_ok=True)

    descriptions = [item["task"] for item in task_items]
    task_embs = get_task_embs(descriptions)

    task_to_index = {}
    for item, emb in zip(task_items, task_embs):
        task_index = int(item["task_index"])
        task_to_index[item["task"]] = task_index
        torch.save(emb, prompt_embed_dir / f"pos_{task_index}.pt")

    return task_to_index


def build_base_entries(
    episode_items: List[Dict],
    dataset_name: str,
    task_to_index: Dict[str, int],
    selected_video_views: List[str],
) -> List[Dict]:
    base_entries = []

    for item in episode_items:
        raw_length = int(item["raw_length"])
        if raw_length < WINDOW_SIZE:
            continue

        prompt = resolve_prompt(item)
        assert prompt in task_to_index, f"Task not found in tasks.jsonl: {prompt}"
        task_index = task_to_index[prompt]

        shared_entry = {
            "episode_index": int(item["episode_index"]),
            "action": prefix_dataset_path(dataset_name, item["action"]),
            "prompt": prompt,
            "dataset_name": dataset_name,
            "prompt_embed_bert": f"{dataset_name}/{PROMPT_EMB_DIR_NAME}/bert/pos_{task_index}.pt",
            "raw_length": raw_length,
        }
        if "prompt_emb" in item and isinstance(item["prompt_emb"], str):
            shared_entry["prompt_emb"] = prefix_dataset_path(dataset_name, item["prompt_emb"])

        for video_view in selected_video_views:
            video_relative_path = get_video_relative_path(item, video_view)
            track_relative_path = build_track_relative_path(video_relative_path)

            entry = shared_entry.copy()
            entry["video"] = prefix_dataset_path(dataset_name, video_relative_path)
            entry["track"] = prefix_dataset_path(dataset_name, track_relative_path)
            entry["video_relative_path"] = video_relative_path
            entry["track_relative_path"] = track_relative_path
            entry["video_view"] = video_view
            base_entries.append(entry)

    return base_entries


def collect_track_jobs(base_entries: List[Dict]) -> List[Dict]:
    jobs = {}
    for item in base_entries:
        track_relative_path = item["track_relative_path"]
        expected_raw_length = int(item["raw_length"])

        if track_relative_path in jobs:
            assert jobs[track_relative_path]["expected_raw_length"] == expected_raw_length, (
                f"Mismatched raw_length for {track_relative_path}: "
                f"{jobs[track_relative_path]['expected_raw_length']} vs {expected_raw_length}"
            )
            continue

        jobs[track_relative_path] = {
            "episode_index": int(item["episode_index"]),
            "video_view": item["video_view"],
            "video_relative_path": item["video_relative_path"],
            "track_relative_path": track_relative_path,
            "expected_raw_length": expected_raw_length,
        }

    return list(jobs.values())


def extract_tracks_for_dataset(
    dataset_dir: Path,
    dataset_name: str,
    track_jobs: List[Dict],
    track_model,
    target_video_size,
    skip_exist: bool,
):
    for job in tqdm(track_jobs, desc=f"Extracting full-video tracks for {dataset_name}"):
        video_relative_path = job["video_relative_path"]
        track_relative_path = job["track_relative_path"]
        episode_index = job["episode_index"]
        video_view = job["video_view"]
        expected_raw_length = int(job["expected_raw_length"])

        actual_video_path = dataset_dir / video_relative_path
        npz_save_path = dataset_dir / track_relative_path

        assert actual_video_path.exists(), f"Video not found: {actual_video_path}"
        npz_save_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_exist and npz_save_path.exists():
            with np.load(npz_save_path) as npz_data:
                cached_length = int(npz_data["tracks"].shape[0])
            assert cached_length == expected_raw_length, (
                f"Cached track length mismatch for {npz_save_path}: "
                f"expected {expected_raw_length}, got {cached_length}"
            )
            continue

        frames = load_video_frames(
            str(actual_video_path),
            target_size=target_video_size,
            dataset_name=dataset_name,
            episode_index=episode_index,
            video_view=video_view,
        )
        raw_length, height, width = frames.shape[0], frames.shape[2], frames.shape[3]
        assert raw_length == expected_raw_length, (
            f"Raw length mismatch for {actual_video_path}: "
            f"expected {expected_raw_length}, got {raw_length}"
        )

        with torch.no_grad():
            pred_tracks, pred_vis = track_through_video(frames, track_model, num_points=NUM_TRACK_POINTS)

        pred_tracks[:, :, :, 0] /= width
        pred_tracks[:, :, :, 1] /= height

        np.savez_compressed(
            npz_save_path,
            tracks=pred_tracks[0].cpu().numpy(),
            vis=pred_vis[0].cpu().numpy(),
        )


def split_episode_indices_by_prompt(base_entries: List[Dict]) -> Tuple[Set[int], Set[int]]:
    prompt_to_episode_indices: Dict[str, Set[int]] = {}
    for item in base_entries:
        prompt_to_episode_indices.setdefault(item["prompt"], set()).add(int(item["episode_index"]))

    val_indices: Set[int] = set()
    for episode_indices in prompt_to_episode_indices.values():
        sorted_indices = sorted(episode_indices)
        val_indices.update(sorted_indices[-VAL_PER_TASK:])

    all_episode_indices = {int(item["episode_index"]) for item in base_entries}
    train_indices = all_episode_indices - val_indices
    return train_indices, val_indices


def build_backward_window_starts(raw_length: int) -> List[int]:
    starts = list(range(raw_length - WINDOW_SIZE, -1, -STRIDE))
    starts.reverse()
    return starts


def build_segment_entries(base_entries: List[Dict], target_episode_indices: Set[int]) -> List[Dict]:
    output_entries = []

    for item in base_entries:
        if int(item["episode_index"]) not in target_episode_indices:
            continue

        raw_length = int(item["raw_length"])
        for start in build_backward_window_starts(raw_length):
            entry = {
                "episode_index": int(item["episode_index"]),
                "action": item["action"],
                "prompt": item["prompt"],
                "dataset_name": item["dataset_name"],
                "prompt_embed_bert": item["prompt_embed_bert"],
                "length": WINDOW_SIZE,
                "start_frame": start,
                "end_frame": start + WINDOW_SIZE - 1,
            }
            if "prompt_emb" in item:
                entry["prompt_emb"] = item["prompt_emb"]
            entry["video"] = item["video"]
            entry["track"] = item["track"]
            entry["raw_length"] = raw_length
            output_entries.append(entry)

    return output_entries


@click.command()
@click.option("--skip_exist", type=bool, default=True, help="Skip tracking if .npz file already exists")
@click.option("--dataset_idx", type=int, default=-1, help="Index of the dataset to process")
@click.option("--view", type=click.Choice(["image", "wrist_image", "all"]), default="all", help="Camera view to process")
@click.option("--min_episode", type=int, default=0, help="Minimum episode index to start processing")
@click.option("--max_episode", type=int, default=-1, help="Maximum episode index to process")
def main(skip_exist, dataset_idx, view, min_episode, max_episode):
    target_video_size = (240, 320)
    cotracker_path = "/data_jbx/Codes/co-tracker/"

    if os.path.exists(cotracker_path):
        cotracker = torch.hub.load(cotracker_path, "cotracker2", source="local")
    else:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    cotracker = cotracker.eval().cuda()

    datasets = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("4_4_four_tasks_wan")])
    if dataset_idx >= 0:
        datasets = datasets[dataset_idx : dataset_idx + 1]

    selected_video_views = get_selected_video_views(view)

    print(f"##### Preprocess Datasets Realbot include: {datasets}")
    print(f"##### Processing view: {view}, min_episode: {min_episode}")
    print(f"##### Resize video frames to: {target_video_size[0]}x{target_video_size[1]}")
    print(f"##### Sliding window config: window_size={WINDOW_SIZE}, stride={STRIDE}, val_per_task={VAL_PER_TASK}")
    print(f"Found {len(datasets)} datasets.")

    all_train = []
    all_val = []
    dataset_summaries = []

    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        meta_dir = dataset_dir / "meta"
        tasks_file = meta_dir / "tasks.jsonl"
        episodes_file = meta_dir / "episodes.jsonl"

        assert all(path.exists() for path in [tasks_file, episodes_file])
        print(f"\nProcessing {dataset_name}...")

        task_items = load_tasks(tasks_file)
        task_to_index = save_task_bert_embeddings(task_items, dataset_dir)

        episode_items = [
            item
            for item in read_jsonl(episodes_file)
            if int(item["episode_index"]) >= min_episode
            and (max_episode == -1 or int(item["episode_index"]) < max_episode)
        ]
        base_entries = build_base_entries(
            episode_items=episode_items,
            dataset_name=dataset_name,
            task_to_index=task_to_index,
            selected_video_views=selected_video_views,
        )

        track_jobs = collect_track_jobs(base_entries)
        extract_tracks_for_dataset(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            track_jobs=track_jobs,
            track_model=cotracker,
            target_video_size=target_video_size,
            skip_exist=skip_exist,
        )

        train_indices, val_indices = split_episode_indices_by_prompt(base_entries)
        train_entries = build_segment_entries(base_entries, train_indices)
        val_entries = build_segment_entries(base_entries, val_indices)

        all_train.extend(train_entries)
        all_val.extend(val_entries)

        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "tracked_videos": len(track_jobs),
                "eligible_episodes": len({int(item["episode_index"]) for item in base_entries}),
                "train_episodes": len(train_indices),
                "val_episodes": len(val_indices),
                "train_segments": len(train_entries),
                "val_segments": len(val_entries),
            }
        )

    all_train.sort(key=lambda x: (x["dataset_name"], x["episode_index"], x["start_frame"], x["video"]))
    all_val.sort(key=lambda x: (x["dataset_name"], x["episode_index"], x["start_frame"], x["video"]))

    write_jsonl(TRAIN_OUTPUT_PATH, all_train)
    write_jsonl(VAL_OUTPUT_PATH, all_val)

    print(f"\nWrote train: {len(all_train)} -> {TRAIN_OUTPUT_PATH}")
    print(f"Wrote val:   {len(all_val)} -> {VAL_OUTPUT_PATH}")
    print("\n=== Per-Dataset Summary ===")
    for item in dataset_summaries:
        print(
            f"{item['dataset']}: "
            f"tracked_videos={item['tracked_videos']}, "
            f"eligible_episodes={item['eligible_episodes']}, "
            f"train_episodes={item['train_episodes']}, "
            f"val_episodes={item['val_episodes']}, "
            f"train_segments={item['train_segments']}, "
            f"val_segments={item['val_segments']}"
        )


if __name__ == "__main__":
    main()
