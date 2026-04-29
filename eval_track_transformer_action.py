import json
import os
import sys
from contextlib import nullcontext

import cv2
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def debug_on():
    if len(sys.argv) > 1:
        return

    # 指定使用的 GPU ID
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    num_track_ids = 256
    eval_batch_size = 16
    vis_every_n_batches = 10
    pixel_eval_img_size = "[480,640]"
    ckpt_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "track_transformer",
        "0422_worldarena_track_transformer_001B_action_bs_64_grad_acc_4_numtrack_256_ep501_0243",
        "model_105.ckpt",
    )

    sys.argv = [
        "eval_track_transformer_action.py",
        "--config-name=eval_robocoin_track_transformer_action",
        f"num_track_ids={num_track_ids}",
        f"batch_size={eval_batch_size}",
        f"eval_batch_size={eval_batch_size}",
        f"vis_every_n_batches={vis_every_n_batches}",
        f"pixel_eval_img_size={pixel_eval_img_size}",
        f'ckpt_path="{ckpt_path}"',
        'val_jsonl="/data_jbx/Datasets/RoboTwin2.0_lerobot_v2/episodes_val_worldarena.jsonl"',
        'val_dataset_dir="/data_jbx/Datasets/RoboTwin2.0_lerobot_v2"',
        'stat_path="/data_jbx/Datasets/RoboTwin2.0_lerobot_v2/stat.json"',
    ]


debug_on()

from atm.dataloader.robocoin_action_dataloader import RoboCoinATMActionDataset
from atm.dataloader.utils import get_dataloader
from atm.model import *


def save_image(img_array, path):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def _resolve_optional_path(path):
    if path is None:
        return None

    resolved_path = str(path).strip()
    if not resolved_path:
        return None

    resolved_path = os.path.expanduser(resolved_path)
    return hydra.utils.to_absolute_path(resolved_path)


def _resolve_ckpt_path(cfg: DictConfig) -> str:
    ckpt_path = _resolve_optional_path(cfg.get("ckpt_path"))
    if ckpt_path is None:
        raise ValueError("Missing required `ckpt_path`. Pass it via config or CLI.")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _default_output_dir(ckpt_path: str) -> str:
    ckpt_dir_name = os.path.basename(os.path.dirname(ckpt_path))
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join(PROJECT_ROOT, "results", "eval_track_result", ckpt_dir_name, ckpt_name)


def _resolve_output_dir(cfg: DictConfig, ckpt_path: str) -> str:
    output_dir = _resolve_optional_path(cfg.get("output_dir"))
    if output_dir is not None:
        return output_dir
    return _default_output_dir(ckpt_path)


def _load_runtime_cfg(default_cfg: DictConfig, ckpt_path: str) -> DictConfig:
    saved_cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if os.path.exists(saved_cfg_path):
        print(f"==> Loading config from checkpoint directory: {saved_cfg_path}")
        return OmegaConf.load(saved_cfg_path)

    print("==> config.yaml not found next to checkpoint, falling back to Hydra config.")
    return default_cfg


def _iter_cli_override_keys():
    try:
        raw_overrides = HydraConfig.get().overrides.task
    except Exception:
        return []

    override_keys = []
    for override in raw_overrides:
        if "=" not in override:
            continue
        key = override.split("=", 1)[0].lstrip("+")
        if not key or key.startswith("~") or key.startswith("hydra."):
            continue
        override_keys.append(key)
    return override_keys


def _merge_eval_config(default_cfg: DictConfig, runtime_cfg: DictConfig) -> DictConfig:
    merged_cfg = OmegaConf.create(OmegaConf.to_container(runtime_cfg, resolve=False))
    eval_keys = (
        "ckpt_path",
        "output_dir",
        "vis_every_n_batches",
        "eval_batch_size",
        "pixel_eval_img_size",
    )

    with open_dict(merged_cfg):
        for key in eval_keys:
            if key in default_cfg:
                merged_cfg[key] = default_cfg.get(key)

        for key in _iter_cli_override_keys():
            try:
                value = OmegaConf.select(default_cfg, key, throw_on_missing=True)
            except Exception:
                continue
            OmegaConf.update(merged_cfg, key, value, merge=True)

    return merged_cfg


def _load_checkpoint_state_dict(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


############### 输入数据类型适配函数 ###############
def _get_model_input_dtype(model):
    img_proj = getattr(model, "img_proj_encoder", None)
    if img_proj is not None and hasattr(img_proj, "proj") and hasattr(img_proj.proj, "weight"):
        return img_proj.proj.weight.dtype

    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype

    return torch.float32


def _move_tensor_to_model_device(tensor, device, dtype):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor.to(device=device, non_blocking=True)


def _cast_batch_for_model(model, device, vid, track, vis, task_emb, action):
    target_dtype = _get_model_input_dtype(model)
    return (
        _move_tensor_to_model_device(vid, device, target_dtype),
        _move_tensor_to_model_device(track, device, target_dtype),
        _move_tensor_to_model_device(vis, device, target_dtype),
        _move_tensor_to_model_device(task_emb, device, target_dtype),
        _move_tensor_to_model_device(action, device, target_dtype),
    )


def _get_autocast_context(cfg: DictConfig, device: torch.device):
    if device.type == "cuda" and cfg.get("mix_precision", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _build_model(cfg: DictConfig, device: torch.device):
    model_cls = eval(cfg.model_name)
    with open_dict(cfg.model_cfg):
        cfg.model_cfg.load_path = None

    model = model_cls(**cfg.model_cfg).to(device=device)
    model.load_state_dict(_load_checkpoint_state_dict(cfg.ckpt_path), strict=True)
    model.eval()
    return model


def _update_metric_sums(metric_sums, ret_dict, batch_size: int):
    metric_sums["track_loss"] += float(ret_dict["track_loss"]) * batch_size
    metric_sums["img_loss"] += float(ret_dict["img_loss"]) * batch_size
    metric_sums["loss"] += float(ret_dict["loss"]) * batch_size


def _normalize_img_size(img_size, field_name: str):
    if img_size is None:
        raise ValueError(f"Missing required `{field_name}`.")

    if isinstance(img_size, int):
        normalized_size = (img_size, img_size)
    else:
        try:
            normalized_size = tuple(int(v) for v in img_size)
        except TypeError as exc:
            raise TypeError(
                f"`{field_name}` must be an int or a length-2 iterable of ints, got {type(img_size)!r}."
            ) from exc

    if len(normalized_size) != 2:
        raise ValueError(f"`{field_name}` must contain exactly 2 elements, got {normalized_size}.")
    if normalized_size[0] <= 0 or normalized_size[1] <= 0:
        raise ValueError(f"`{field_name}` values must be positive, got {normalized_size}.")
    return normalized_size


def _resolve_pixel_eval_img_size(cfg: DictConfig):
    pixel_eval_img_size = cfg.get("pixel_eval_img_size")
    if pixel_eval_img_size is None:
        pixel_eval_img_size = cfg.get("img_size")
    return _normalize_img_size(pixel_eval_img_size, field_name="pixel_eval_img_size")


def _tracks_to_pixel_coords(tracks: torch.Tensor, img_size):
    height, width = img_size
    scale = tracks.new_tensor([float(width), float(height)], dtype=torch.float32)
    return tracks.to(dtype=torch.float32) * scale


def _accumulate_pixel_error(metric_sums, sum_key, count_key, pixel_errors, mask=None):
    if mask is None:
        metric_sums[sum_key] += float(pixel_errors.sum().item())
        metric_sums[count_key] += int(pixel_errors.numel())
        return

    masked_count = int(mask.sum().item())
    if masked_count == 0:
        return

    metric_sums[sum_key] += float(pixel_errors[mask].sum().item())
    metric_sums[count_key] += masked_count


def _update_pixel_error_sums(metric_sums, rec_track, track, vis, pixel_eval_img_size):
    pixel_rec_track = _tracks_to_pixel_coords(rec_track, pixel_eval_img_size)
    pixel_track = _tracks_to_pixel_coords(track, pixel_eval_img_size)
    pixel_errors = torch.sqrt(torch.sum((pixel_rec_track - pixel_track) ** 2, dim=-1))
    _accumulate_pixel_error(
        metric_sums,
        "pixel_l2_error_visible_sum",
        "pixel_l2_error_visible_count",
        pixel_errors,
        mask=(vis > 0),
    )
    _accumulate_pixel_error(
        metric_sums,
        "pixel_l2_error_all_sum",
        "pixel_l2_error_all_count",
        pixel_errors,
    )


@hydra.main(
    config_path="./conf/train_track_transformer",
    config_name="eval_robocoin_track_transformer_action",
    version_base="1.3",
)
def main(cfg: DictConfig):
    ckpt_path = _resolve_ckpt_path(cfg)
    runtime_cfg = _load_runtime_cfg(cfg, ckpt_path)
    cfg = _merge_eval_config(cfg, runtime_cfg)

    with open_dict(cfg):
        cfg.ckpt_path = ckpt_path
        cfg.output_dir = _resolve_output_dir(cfg, ckpt_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    vis_every_n_batches = int(cfg.get("vis_every_n_batches", 10))
    eval_batch_size = int(cfg.get("eval_batch_size", 1))
    pixel_eval_img_size = _resolve_pixel_eval_img_size(cfg)
    vis_dir = os.path.join(cfg.output_dir, "visualizations")
    if vis_every_n_batches > 0:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"==> Loading model from: {cfg.ckpt_path}")
    model = _build_model(cfg, device)

    print("==> Preparing validation dataset...")
    val_dataset = RoboCoinATMActionDataset(
        jsonl_path=cfg.val_jsonl,
        dataset_dir=cfg.val_dataset_dir,
        **cfg.dataset_cfg,
        aug_prob=0.0,
    )
    val_dataloader = get_dataloader(
        val_dataset,
        mode="val",
        num_workers=cfg.num_workers,
        batch_size=eval_batch_size,
    )

    metric_sums = {
        "track_loss": 0.0,
        "img_loss": 0.0,
        "loss": 0.0,
        "pixel_l2_error_visible_sum": 0.0,
        "pixel_l2_error_visible_count": 0,
        "pixel_l2_error_all_sum": 0.0,
        "pixel_l2_error_all_count": 0,
    }
    num_samples = 0
    vis_disabled_for_batch_size = False

    print(f"==> Starting evaluation on {len(val_dataset)} samples...")

    with torch.inference_mode():
        for i, (vid, track, vis, task_emb, action) in enumerate(tqdm(val_dataloader)):
            batch_size = int(vid.shape[0])
            vid, track, vis, task_emb, action = _cast_batch_for_model(
                model,
                device,
                vid,
                track,
                vis,
                task_emb,
                action,
            )

            vis_for_loss = vis.clone()
            with _get_autocast_context(cfg, device):
                _, ret_dict, (rec_track, _) = model.forward_loss(
                    vid,
                    track,
                    task_emb,
                    lbd_track=cfg.lbd_track,
                    lbd_img=cfg.lbd_img,
                    p_img=cfg.p_img,
                    vis=vis_for_loss,
                    action=action,
                    return_outs=True,
                )

            _update_metric_sums(metric_sums, ret_dict, batch_size=batch_size)
            _update_pixel_error_sums(
                metric_sums,
                rec_track=rec_track,
                track=track,
                vis=vis,
                pixel_eval_img_size=pixel_eval_img_size,
            )
            num_samples += batch_size

            if vis_every_n_batches <= 0 or i % vis_every_n_batches != 0:
                continue

            if batch_size != 1:
                if not vis_disabled_for_batch_size:
                    print(
                        "==> Skipping visualizations because `forward_vis` requires batch size 1. "
                        f"Current eval_batch_size={eval_batch_size}."
                    )
                    vis_disabled_for_batch_size = True
                continue

            with _get_autocast_context(cfg, device):
                _, vis_ret = model.forward_vis(
                    vid,
                    track,
                    task_emb,
                    p_img=0,
                    action=action,
                )

            save_image(
                vis_ret["combined_image"],
                os.path.join(vis_dir, f"batch_{i}_reconstruction.png"),
            )
            save_image(
                vis_ret["combined_track_vid"],
                os.path.join(vis_dir, f"batch_{i}_track.png"),
            )

    avg_metrics = {
        key: (value / num_samples) if num_samples > 0 else float("nan")
        for key, value in metric_sums.items()
        if key
        not in {
            "pixel_l2_error_visible_sum",
            "pixel_l2_error_visible_count",
            "pixel_l2_error_all_sum",
            "pixel_l2_error_all_count",
        }
    }
    visible_point_count = metric_sums["pixel_l2_error_visible_count"]
    all_point_count = metric_sums["pixel_l2_error_all_count"]
    avg_metrics["pixel_l2_error_visible"] = (
        metric_sums["pixel_l2_error_visible_sum"] / visible_point_count
        if visible_point_count > 0
        else float("nan")
    )
    avg_metrics["pixel_l2_error_all"] = (
        metric_sums["pixel_l2_error_all_sum"] / all_point_count
        if all_point_count > 0
        else float("nan")
    )
    avg_metrics["pixel_eval_img_size"] = list(pixel_eval_img_size)
    avg_metrics["pixel_l2_error_visible_count"] = visible_point_count
    avg_metrics["pixel_l2_error_all_count"] = all_point_count
    avg_metrics["num_samples"] = num_samples
    avg_metrics["ckpt_path"] = cfg.ckpt_path
    avg_metrics["output_dir"] = cfg.output_dir

    print("\n==> Evaluation results:")
    for key, value in avg_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=4)

    print(f"\nResults saved to '{cfg.output_dir}'.")


if __name__ == "__main__":
    main()
