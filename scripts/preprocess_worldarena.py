import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click

import imageio
import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from atm.utils.flow_utils import sample_double_grid, sample_from_mask

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

BASE_DIR = Path("/data_jbx/Datasets/RoboTwin2.0_lerobot_0423")
TRAIN_OUTPUT_PATH = BASE_DIR / "episodes_train_worldarena.jsonl"
VAL_OUTPUT_PATH = BASE_DIR / "episodes_val_worldarena.jsonl"

VIDEO_DIR_NAME = "videos"
TRACK_DIR_NAME = "tracks"
BERT_DIR_NAME = "bert"
NUM_TRACK_POINTS = 1000
GLOBAL_CHUNK_SIZE = 200
GLOBAL_CHUNK_OVERLAP = 20
WINDOW_SIZE = 20
STRIDE = 10
VAL_PER_TASK = 5
TRACKING_MODES = ("chunk", "global", "global_chunked")
LEGACY_TRACKING_MODE = "global_chunked"

VIEW_MAP = {
    "cam_high_rgb": "observation.images.cam_high_rgb",
    "cam_left_wrist_rgb": "observation.images.cam_left_wrist_rgb",
    "cam_right_wrist_rgb": "observation.images.cam_right_wrist_rgb",
    "cam_front_rgb": "observation.images.cam_front_rgb",
}

_TOKENIZER = None
_TEXT_MODEL = None


def get_task_embs(descriptions: List[str]) -> torch.Tensor:
    global _TOKENIZER, _TEXT_MODEL

    if _TOKENIZER is None or _TEXT_MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
        _TEXT_MODEL = AutoModel.from_pretrained("bert-base-cased").eval()

    tokens = _TOKENIZER(
        text=descriptions,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        task_embs = _TEXT_MODEL(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()
    return task_embs


def track_and_remove(tracker, video, points, var_threshold=5.0, backward_tracking=False):
    _, _, _, height, _ = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=backward_tracking)

    var = torch.var(pred_tracks, dim=1)
    var = torch.sum(var, dim=-1)[0]

    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        return pred_tracks, pred_vis

    new_points = points[:, idx].clone()
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, : points.shape[1]]

    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * height
    new_points[:, :, 1:] += noise

    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=backward_tracking)
    return pred_tracks, pred_vis


def build_overlapping_chunk_ranges(
    total_frames: int,
    chunk_size: int = GLOBAL_CHUNK_SIZE,
    overlap: int = GLOBAL_CHUNK_OVERLAP,
) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, {chunk_size}), got {overlap}")

    ranges = []
    chunk_start = 0
    while chunk_start < total_frames:
        chunk_end = min(chunk_start + chunk_size, total_frames)
        ranges.append((chunk_start, chunk_end))
        if chunk_end == total_frames:
            break
        chunk_start = chunk_end - overlap

    return ranges


def build_random_timed_queries(
    num_frames: int,
    height: int,
    width: int,
    num_points: int,
    grid_points_template: torch.Tensor,
):
    points = sample_from_mask(np.ones((height, width, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().cuda()
    point_start_frames = torch.randint(0, num_frames, (num_points, 1), device="cuda").float()
    point_queries = torch.cat([point_start_frames, points], dim=-1).unsqueeze(0)

    grid_points = grid_points_template.clone()
    grid_points[:, 0] *= height
    grid_points[:, 1] *= width
    grid_start_frames = torch.randint(0, num_frames, (grid_points.shape[0], 1), device="cuda").float()
    grid_queries = torch.cat([grid_start_frames, grid_points], dim=-1).unsqueeze(0)

    return point_queries, grid_queries


def track_by_independent_segments(
    video,
    track_model,
    num_points=NUM_TRACK_POINTS,
    segment_size=WINDOW_SIZE,
):
    total_frames, _, _, _ = video.shape
    video_tensor = torch.from_numpy(video).cuda().float().unsqueeze(0)

    grid_points_template = sample_double_grid(7, device="cuda")
    num_grid = grid_points_template.shape[0]

    global_tracks = torch.zeros(1, total_frames, num_grid + num_points, 2, device="cuda")
    global_vis = torch.zeros(1, total_frames, num_grid + num_points, device="cuda")

    # Split from the tail so the last chunk is always a full 81-frame segment when possible.
    chunk_ranges = []
    chunk_end = total_frames
    while chunk_end > 0:
        chunk_start = max(0, chunk_end - segment_size)
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_end -= segment_size

    chunk_ranges.sort()

    for chunk_start, chunk_end in chunk_ranges:
        chunk_video = video_tensor[:, chunk_start:chunk_end]
        _, _, _, height, width = chunk_video.shape

        points = sample_from_mask(np.ones((height, width, 1)) * 255, num_samples=num_points)
        points = torch.from_numpy(points).float().cuda()
        point_queries = torch.cat([torch.zeros_like(points[:, :1]), points], dim=-1).unsqueeze(0)

        grid_points = grid_points_template.clone()
        grid_points[:, 0] *= height
        grid_points[:, 1] *= width
        grid_queries = torch.cat([torch.zeros_like(grid_points[:, :1]), grid_points], dim=-1).unsqueeze(0)

        pred_tracks, pred_vis = track_and_remove(
            track_model,
            chunk_video,
            point_queries,
            var_threshold=5.0,
            backward_tracking=False,
        )
        pred_grid_tracks, pred_grid_vis = track_and_remove(
            track_model,
            chunk_video,
            grid_queries,
            var_threshold=0.0,
            backward_tracking=False,
        )

        chunk_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
        chunk_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)

        global_tracks[:, chunk_start:chunk_end] = chunk_tracks
        global_vis[:, chunk_start:chunk_end] = chunk_vis

    return global_tracks, global_vis


def track_through_video(video, track_model, num_points=NUM_TRACK_POINTS):
    num_frames, _, height, width = video.shape
    video_tensor = torch.from_numpy(video).cuda().float()
    grid_points_template = sample_double_grid(7, device="cuda")

    point_queries, grid_queries = build_random_timed_queries(
        num_frames=num_frames,
        height=height,
        width=width,
        num_points=num_points,
        grid_points_template=grid_points_template,
    )

    pred_tracks, pred_vis = track_and_remove(
        track_model,
        video_tensor[None],
        point_queries,
        var_threshold=5.0,
        backward_tracking=True,
    )
    pred_grid_tracks, pred_grid_vis = track_and_remove(
        track_model,
        video_tensor[None],
        grid_queries,
        var_threshold=0.0,
        backward_tracking=True,
    )

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


def track_through_video_chunked(
    video,
    track_model,
    num_points=NUM_TRACK_POINTS,
    chunk_size=GLOBAL_CHUNK_SIZE,
    overlap=GLOBAL_CHUNK_OVERLAP,
):
    total_frames, _, _, _ = video.shape
    video_tensor = torch.from_numpy(video).cuda().float().unsqueeze(0)

    grid_points_template = sample_double_grid(7, device="cuda")
    num_grid = grid_points_template.shape[0]
    global_tracks = torch.zeros(1, total_frames, num_grid + num_points, 2, device="cuda")
    global_vis = torch.zeros(1, total_frames, num_grid + num_points, device="cuda")

    for chunk_start, chunk_end in build_overlapping_chunk_ranges(
        total_frames=total_frames,
        chunk_size=chunk_size,
        overlap=overlap,
    ):
        chunk_video = video_tensor[:, chunk_start:chunk_end]
        chunk_num_frames = chunk_end - chunk_start
        _, _, _, height, width = chunk_video.shape

        point_queries, grid_queries = build_random_timed_queries(
            num_frames=chunk_num_frames,
            height=height,
            width=width,
            num_points=num_points,
            grid_points_template=grid_points_template,
        )

        pred_tracks, pred_vis = track_and_remove(
            track_model,
            chunk_video,
            point_queries,
            var_threshold=5.0,
            backward_tracking=True,
        )
        pred_grid_tracks, pred_grid_vis = track_and_remove(
            track_model,
            chunk_video,
            grid_queries,
            var_threshold=0.0,
            backward_tracking=True,
        )

        chunk_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
        chunk_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)

        global_tracks[:, chunk_start:chunk_end] = chunk_tracks
        global_vis[:, chunk_start:chunk_end] = chunk_vis

    return global_tracks, global_vis


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


def get_episode_raw_length(episode_item: Dict) -> int:
    if "raw_length" in episode_item:
        return int(episode_item["raw_length"])

    if "length" in episode_item:
        return int(episode_item["length"])

    raise KeyError(f"Missing raw length field in episode item: {episode_item.keys()}")


def resolve_prompt_indices(episode_item: Dict) -> List[int]:
    prompt = episode_item.get("prompt")
    if isinstance(prompt, int):
        return [int(prompt)]

    if isinstance(prompt, list) and len(prompt) > 0:
        return [int(task_index) for task_index in prompt]

    raise ValueError(f"Invalid prompt index list: {prompt}")


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


def save_task_bert_embeddings(task_items: List[Dict], dataset_dir: Path) -> Dict[int, Dict[str, str]]:
    bert_dir = dataset_dir / BERT_DIR_NAME
    bert_dir.mkdir(parents=True, exist_ok=True)

    descriptions = [item["task"] for item in task_items]
    task_embs = get_task_embs(descriptions)

    task_index_to_assets: Dict[int, Dict[str, str]] = {}
    for item, emb in zip(task_items, task_embs):
        task_index = int(item["task_index"])
        prompt_emb_relative_path = item.get("prompt_emb")
        assert isinstance(prompt_emb_relative_path, str) and prompt_emb_relative_path, (
            f"Invalid prompt_emb for task_index={task_index}: {prompt_emb_relative_path}"
        )

        bert_filename = Path(prompt_emb_relative_path).name
        bert_relative_path = str((Path(BERT_DIR_NAME) / bert_filename).as_posix())
        torch.save(emb, dataset_dir / bert_relative_path)

        task_index_to_assets[task_index] = {
            "task_text": item["task"],
            "prompt_emb": prompt_emb_relative_path,
            "prompt_embed_bert": bert_relative_path,
        }

    return task_index_to_assets


def build_base_entries(
    episode_items: List[Dict],
    dataset_name: str,
    task_index_to_assets: Dict[int, Dict[str, str]],
    selected_video_views: List[str],
) -> List[Dict]:
    base_entries = []

    for item in episode_items:
        raw_length = get_episode_raw_length(item)

        prompt_indices = resolve_prompt_indices(item)
        task_group = item.get("task", dataset_name)
        assert isinstance(task_group, str) and task_group, f"Invalid task name: {task_group}"

        shared_entry = {
            "episode_index": int(item["episode_index"]),
            "action": prefix_dataset_path(dataset_name, item["action"]),
            "dataset_name": dataset_name,
            "task_group": task_group,
            "raw_length": raw_length,
        }

        for video_view in selected_video_views:
            video_relative_path = get_video_relative_path(item, video_view)
            track_relative_path = build_track_relative_path(video_relative_path)

            for task_index in prompt_indices:
                assert task_index in task_index_to_assets, (
                    f"Prompt task_index={task_index} missing from tasks.jsonl for dataset {dataset_name}"
                )
                task_assets = task_index_to_assets[task_index]

                entry = shared_entry.copy()
                entry["prompt"] = task_index
                entry["task"] = task_assets["task_text"]
                entry["prompt_emb"] = prefix_dataset_path(dataset_name, task_assets["prompt_emb"])
                entry["prompt_embed_bert"] = prefix_dataset_path(
                    dataset_name, task_assets["prompt_embed_bert"]
                )
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


def get_cached_npz_scalar(npz_data, field_name: str):
    if field_name not in npz_data.files:
        return None

    value = npz_data[field_name]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def get_cached_tracking_mode(npz_data) -> Optional[str]:
    tracking_mode = get_cached_npz_scalar(npz_data, "tracking_mode")
    if tracking_mode is None:
        return None
    return str(tracking_mode)


def get_cached_num_track_points(npz_data) -> Optional[int]:
    num_track_points = get_cached_npz_scalar(npz_data, "num_track_points")
    if num_track_points is None:
        return None
    return int(num_track_points)


def get_cached_global_chunk_size(npz_data) -> Optional[int]:
    global_chunk_size = get_cached_npz_scalar(npz_data, "global_chunk_size")
    if global_chunk_size is None:
        return None
    return int(global_chunk_size)


def get_cached_global_chunk_overlap(npz_data) -> Optional[int]:
    global_chunk_overlap = get_cached_npz_scalar(npz_data, "global_chunk_overlap")
    if global_chunk_overlap is None:
        return None
    return int(global_chunk_overlap)


def cached_track_matches_mode(
    npz_save_path: Path,
    expected_raw_length: int,
    tracking_mode: str,
    num_track_points: int,
    global_chunk_size: int,
    global_chunk_overlap: int,
) -> Tuple[bool, Optional[str], Optional[int], Optional[int], Optional[int]]:
    with np.load(npz_save_path) as npz_data:
        cached_length = int(npz_data["tracks"].shape[0])
        cached_tracking_mode = get_cached_tracking_mode(npz_data)
        cached_num_track_points = get_cached_num_track_points(npz_data)
        cached_global_chunk_size = get_cached_global_chunk_size(npz_data)
        cached_global_chunk_overlap = get_cached_global_chunk_overlap(npz_data)

    assert cached_length == expected_raw_length, (
        f"Cached track length mismatch for {npz_save_path}: "
        f"expected {expected_raw_length}, got {cached_length}"
    )

    if cached_tracking_mode is None:
        return (
            tracking_mode == LEGACY_TRACKING_MODE,
            None,
            cached_num_track_points,
            cached_global_chunk_size,
            cached_global_chunk_overlap,
        )
    if cached_tracking_mode != tracking_mode:
        return (
            False,
            cached_tracking_mode,
            cached_num_track_points,
            cached_global_chunk_size,
            cached_global_chunk_overlap,
        )
    if cached_num_track_points is None:
        return (
            tracking_mode == LEGACY_TRACKING_MODE,
            cached_tracking_mode,
            None,
            cached_global_chunk_size,
            cached_global_chunk_overlap,
        )

    if cached_num_track_points != num_track_points:
        return (
            False,
            cached_tracking_mode,
            cached_num_track_points,
            cached_global_chunk_size,
            cached_global_chunk_overlap,
        )

    if tracking_mode != "global_chunked":
        return (
            True,
            cached_tracking_mode,
            cached_num_track_points,
            cached_global_chunk_size,
            cached_global_chunk_overlap,
        )

    return (
        cached_global_chunk_size == global_chunk_size
        and cached_global_chunk_overlap == global_chunk_overlap,
        cached_tracking_mode,
        cached_num_track_points,
        cached_global_chunk_size,
        cached_global_chunk_overlap,
    )


def extract_tracks_for_dataset(
    dataset_dir: Path,
    dataset_name: str,
    track_jobs: List[Dict],
    track_model,
    target_video_size,
    skip_exist: bool,
    tracking_mode: str,
):
    for job in tqdm(track_jobs, desc=f"Extracting full-video tracks for {dataset_name}"):
        video_relative_path = job["video_relative_path"]
        track_relative_path = job["track_relative_path"]
        episode_index = job["episode_index"]
        video_view = job["video_view"]
        expected_raw_length = int(job["expected_raw_length"])

        actual_video_path = dataset_dir / video_relative_path
        npz_save_path = dataset_dir / track_relative_path
        num_track_points = NUM_TRACK_POINTS
        global_chunk_size = GLOBAL_CHUNK_SIZE if tracking_mode == "global_chunked" else 0
        global_chunk_overlap = GLOBAL_CHUNK_OVERLAP if tracking_mode == "global_chunked" else 0

        assert actual_video_path.exists(), f"Video not found: {actual_video_path}"
        npz_save_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_exist and npz_save_path.exists():
            (
                can_reuse_cache,
                cached_tracking_mode,
                cached_num_track_points,
                cached_global_chunk_size,
                cached_global_chunk_overlap,
            ) = cached_track_matches_mode(
                npz_save_path=npz_save_path,
                expected_raw_length=expected_raw_length,
                tracking_mode=tracking_mode,
                num_track_points=num_track_points,
                global_chunk_size=global_chunk_size,
                global_chunk_overlap=global_chunk_overlap,
            )
            if can_reuse_cache:
                continue

            cached_mode_desc = cached_tracking_mode or f"legacy_{LEGACY_TRACKING_MODE}"
            cached_num_points_desc = (
                "missing"
                if cached_num_track_points is None
                else str(cached_num_track_points)
            )
            cached_chunk_size_desc = (
                "missing"
                if cached_global_chunk_size is None
                else str(cached_global_chunk_size)
            )
            cached_chunk_overlap_desc = (
                "missing"
                if cached_global_chunk_overlap is None
                else str(cached_global_chunk_overlap)
            )
            print(
                f"Recomputing {npz_save_path} because cached tracking_mode={cached_mode_desc} "
                f"or cached num_track_points={cached_num_points_desc} does not match "
                f"requested tracking_mode={tracking_mode}, num_track_points={num_track_points}, "
                f"global_chunk_size={global_chunk_size}, global_chunk_overlap={global_chunk_overlap}; "
                f"cached global_chunk_size={cached_chunk_size_desc}, "
                f"cached global_chunk_overlap={cached_chunk_overlap_desc}."
            )

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
            if tracking_mode == "chunk":
                pred_tracks, pred_vis = track_by_independent_segments(
                    frames,
                    track_model,
                    num_points=num_track_points,
                    segment_size=WINDOW_SIZE,
                )
            elif tracking_mode == "global":
                print(
                    f"Global tracking {actual_video_path}: raw_length={raw_length}, "
                    f"num_track_points={num_track_points}"
                )
                pred_tracks, pred_vis = track_through_video(
                    frames,
                    track_model,
                    num_points=num_track_points,
                )
            elif tracking_mode == "global_chunked":
                print(
                    f"Global chunked tracking {actual_video_path}: raw_length={raw_length}, "
                    f"num_track_points={num_track_points}, chunk_size={GLOBAL_CHUNK_SIZE}, "
                    f"overlap={GLOBAL_CHUNK_OVERLAP}"
                )
                pred_tracks, pred_vis = track_through_video_chunked(
                    frames,
                    track_model,
                    num_points=num_track_points,
                    chunk_size=GLOBAL_CHUNK_SIZE,
                    overlap=GLOBAL_CHUNK_OVERLAP,
                )
            else:
                raise ValueError(f"Unsupported tracking_mode: {tracking_mode}")

        pred_tracks[:, :, :, 0] /= width
        pred_tracks[:, :, :, 1] /= height

        np.savez_compressed(
            npz_save_path,
            tracks=pred_tracks[0].cpu().numpy(),
            vis=pred_vis[0].cpu().numpy(),
            tracking_mode=np.array(tracking_mode),
            num_track_points=np.array(num_track_points),
            global_chunk_size=np.array(global_chunk_size),
            global_chunk_overlap=np.array(global_chunk_overlap),
        )


def split_episode_indices_by_task(base_entries: List[Dict]) -> Tuple[Set[int], Set[int]]:
    task_to_episode_indices: Dict[Tuple[str, str], Set[int]] = {}
    for item in base_entries:
        # Preserve the original action-level split even though each sample now carries one prompt.
        # Short clips still participate in the episode split even if they cannot yield WINDOW_SIZE-frame windows.
        task_key = (item["dataset_name"], item["task_group"])
        task_to_episode_indices.setdefault(task_key, set()).add(int(item["episode_index"]))

    val_indices: Set[int] = set()
    for episode_indices in task_to_episode_indices.values():
        sorted_indices = sorted(episode_indices)
        val_indices.update(sorted_indices[-VAL_PER_TASK:])

    all_episode_indices = {int(item["episode_index"]) for item in base_entries}
    train_indices = all_episode_indices - val_indices
    return train_indices, val_indices


def is_segment_eligible(raw_length: int) -> bool:
    return raw_length >= WINDOW_SIZE


def build_backward_window_starts(raw_length: int) -> List[int]:
    if not is_segment_eligible(raw_length):
        return []

    starts = list(range(raw_length - WINDOW_SIZE, -1, -STRIDE))
    starts.reverse()
    return starts


def build_stitched_chunk_intervals(
    raw_length: int,
    chunk_size: int = GLOBAL_CHUNK_SIZE,
    overlap: int = GLOBAL_CHUNK_OVERLAP,
) -> List[Tuple[int, int]]:
    if raw_length <= 0:
        return []

    chunk_ranges = build_overlapping_chunk_ranges(
        total_frames=raw_length,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if len(chunk_ranges) == 1:
        return [(0, raw_length)]

    intervals = []
    for chunk_index, (chunk_start, _) in enumerate(chunk_ranges):
        if chunk_index + 1 < len(chunk_ranges):
            interval_end = chunk_ranges[chunk_index + 1][0]
        else:
            interval_end = raw_length

        if interval_end > chunk_start:
            intervals.append((chunk_start, interval_end))

    return intervals


def build_valid_window_starts(raw_length: int, tracking_mode: str) -> Tuple[List[int], int]:
    starts = build_backward_window_starts(raw_length)
    if tracking_mode != "global_chunked" or not starts:
        return starts, 0

    stitched_intervals = build_stitched_chunk_intervals(raw_length)
    valid_starts = []
    dropped_count = 0

    for start in starts:
        end_exclusive = start + WINDOW_SIZE
        keep_window = any(
            interval_start <= start and end_exclusive <= interval_end
            for interval_start, interval_end in stitched_intervals
        )
        if keep_window:
            valid_starts.append(start)
        else:
            dropped_count += 1

    return valid_starts, dropped_count


def collect_episode_indices(base_entries: List[Dict], segment_eligible_only: bool = False) -> Set[int]:
    episode_indices = set()
    for item in base_entries:
        raw_length = int(item["raw_length"])
        if segment_eligible_only and not is_segment_eligible(raw_length):
            continue
        episode_indices.add(int(item["episode_index"]))
    return episode_indices


def build_segment_entries(
    base_entries: List[Dict],
    target_episode_indices: Set[int],
    tracking_mode: str,
) -> Tuple[List[Dict], int]:
    output_entries = []
    dropped_by_boundary = 0
    starts_cache: Dict[int, Tuple[List[int], int]] = {}

    for item in base_entries:
        if int(item["episode_index"]) not in target_episode_indices:
            continue

        raw_length = int(item["raw_length"])
        if raw_length not in starts_cache:
            starts_cache[raw_length] = build_valid_window_starts(raw_length, tracking_mode)

        valid_starts, boundary_drop_count = starts_cache[raw_length]
        dropped_by_boundary += boundary_drop_count

        for start in valid_starts:
            entry = {
                "episode_index": int(item["episode_index"]),
                "action": item["action"],
                "prompt": item["prompt"],
                "task": item["task"],
                "dataset_name": item["dataset_name"],
                "prompt_embed_bert": item["prompt_embed_bert"],
                "prompt_emb": item["prompt_emb"],
                "length": WINDOW_SIZE,
                "start_frame": start,
                "end_frame": start + WINDOW_SIZE - 1,
            }
            entry["video"] = item["video"]
            entry["track"] = item["track"]
            entry["raw_length"] = raw_length
            output_entries.append(entry)

    return output_entries, dropped_by_boundary


def normalize_dataset_indices(dataset_indices: Tuple[int, ...], total_datasets: int) -> List[int]:
    if not dataset_indices:
        return list(range(total_datasets))

    selected = []
    seen = set()
    for idx in dataset_indices:
        if idx < 0 or idx >= total_datasets:
            raise click.BadParameter(f"dataset_idx {idx} is out of range for {total_datasets} datasets")
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(idx)
    return selected


@click.command()
@click.option("--skip_exist", type=bool, default=True, help="Skip tracking if .npz file already exists")
@click.option(
    "--dataset_idx",
    type=int,
    multiple=True,
    default=(),
    help="Dataset index to process. Repeat the flag to select multiple datasets; omit it to process all datasets.",
)
@click.option(
    "--view",
    type=click.Choice(["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb", "cam_front_rgb", "all"]),
    default="cam_high_rgb",
    help="Camera view to process",
)
@click.option("--min_episode", type=int, default=0, help="Minimum episode index to start processing")
@click.option("--max_episode", type=int, default=-1, help="Maximum episode index to process")
@click.option(
    "--tracking_mode",
    type=click.Choice(TRACKING_MODES),
    default="global_chunked",
    help=(
        f"Tracking mode: process the whole video globally, by overlapping {GLOBAL_CHUNK_SIZE}-frame "
        f"global chunks, or track each {WINDOW_SIZE}-frame chunk independently"
    ),
)
def main(skip_exist, dataset_idx, view, min_episode, max_episode, tracking_mode):
    target_video_size = (240, 320)
    cotracker_path = "/data_jbx/Codes/co-tracker/"

    if os.path.exists(cotracker_path):
        cotracker = torch.hub.load(cotracker_path, "cotracker2", source="local")
    else:
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    cotracker = cotracker.eval().cuda()

    datasets = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    selected_dataset_indices = normalize_dataset_indices(tuple(dataset_idx), len(datasets))
    datasets = [datasets[i] for i in selected_dataset_indices]

    selected_video_views = get_selected_video_views(view)

    print(f"##### Preprocess Datasets WorldArena include: {datasets}")
    print(f"##### Processing view: {view}, min_episode: {min_episode}")
    print(f"##### Resize video frames to: {target_video_size[0]}x{target_video_size[1]}")
    print(f"##### Sliding window config: window_size={WINDOW_SIZE}, stride={STRIDE}, val_per_task={VAL_PER_TASK}")
    print(f"##### Tracking mode: {tracking_mode}")
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
        task_index_to_assets = save_task_bert_embeddings(task_items, dataset_dir)
        continue

        episode_items = [
            item
            for item in read_jsonl(episodes_file)
            if int(item["episode_index"]) >= min_episode
            and (max_episode == -1 or int(item["episode_index"]) < max_episode)
        ]
        base_entries = build_base_entries(
            episode_items=episode_items,
            dataset_name=dataset_name,
            task_index_to_assets=task_index_to_assets,
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
            tracking_mode=tracking_mode,
        )

        train_indices, val_indices = split_episode_indices_by_task(base_entries)
        train_entries, train_segments_dropped_by_boundary = build_segment_entries(
            base_entries,
            train_indices,
            tracking_mode=tracking_mode,
        )
        val_entries, val_segments_dropped_by_boundary = build_segment_entries(
            base_entries,
            val_indices,
            tracking_mode=tracking_mode,
        )

        all_train.extend(train_entries)
        all_val.extend(val_entries)

        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "tracked_videos": len(track_jobs),
                "tracked_episodes": len(collect_episode_indices(base_entries)),
                "segment_eligible_episodes": len(
                    collect_episode_indices(base_entries, segment_eligible_only=True)
                ),
                "tracked_short_episodes": len(collect_episode_indices(base_entries))
                - len(collect_episode_indices(base_entries, segment_eligible_only=True)),
                "train_episodes": len(train_indices),
                "val_episodes": len(val_indices),
                "train_segments": len(train_entries),
                "val_segments": len(val_entries),
                "train_segments_dropped_by_boundary": train_segments_dropped_by_boundary,
                "val_segments_dropped_by_boundary": val_segments_dropped_by_boundary,
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
            f"tracked_episodes={item['tracked_episodes']}, "
            f"segment_eligible_episodes={item['segment_eligible_episodes']}, "
            f"tracked_short_episodes={item['tracked_short_episodes']}, "
            f"train_episodes={item['train_episodes']}, "
            f"val_episodes={item['val_episodes']}, "
            f"train_segments={item['train_segments']}, "
            f"val_segments={item['val_segments']}, "
            f"train_segments_dropped_by_boundary={item['train_segments_dropped_by_boundary']}, "
            f"val_segments_dropped_by_boundary={item['val_segments_dropped_by_boundary']}"
        )


if __name__ == "__main__":
    main()
