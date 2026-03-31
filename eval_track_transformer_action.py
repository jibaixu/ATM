import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXP_NAME = "0326_robocoin_track_transformer_001B_action_bs_8_grad_acc_4_numtrack_256_robocoin-object_ep1001_2109"
CKPT_NAME = "model_best"
CKPT_PATH = os.path.join("results", "track_transformer", EXP_NAME, f"{CKPT_NAME}.ckpt")
OUTPUT_DIR = os.path.join("results", "eval_track_result", EXP_NAME, CKPT_NAME)
VIS_EVERY_N_BATCHES = 10


def debug_on():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    batch_size = 8
    num_track_ids = 256
    gradient_accumulation_steps = 4
    sys.argv = [
        "eval_track_transformer_action.py",
        "--config-name=robocoin_track_transformer_action",
        f"num_track_ids={num_track_ids}",
        f"batch_size={batch_size}",
        f"gradient_accumulation_steps={gradient_accumulation_steps}",
        f"experiment=robocoin_track_transformer_001B_action_bs_{batch_size}_grad_acc_{gradient_accumulation_steps}_numtrack_{num_track_ids}_robocoin-object_ep1001",
        'val_jsonl="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320/episodes_clipped_val_4w.jsonl"',
        'val_dataset_dir="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320"',
        'stat_path="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320/stat.json"',
    ]
debug_on()


import json
from contextlib import nullcontext

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from atm.dataloader import RoboCoinATMActionDataset, get_dataloader
from atm.model import TrackTransformerAction


def save_image(img_array, path):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def _load_runtime_cfg(default_cfg: DictConfig, ckpt_path: str) -> DictConfig:
    saved_cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if os.path.exists(saved_cfg_path):
        print(f"==> Loading config from checkpoint directory: {saved_cfg_path}")
        return OmegaConf.load(saved_cfg_path)

    print("==> config.yaml not found next to checkpoint, falling back to Hydra config.")
    return default_cfg


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


@hydra.main(config_path="./conf/train_track_transformer", version_base="1.3")
def main(cfg: DictConfig):
    assert os.path.exists(CKPT_PATH)

    cfg = _load_runtime_cfg(cfg, CKPT_PATH)

    device = torch.device("cuda:0")
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"==> Loading model from: {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    with open_dict(cfg.model_cfg):
        cfg.model_cfg.action_dim = 14

    model = TrackTransformerAction(**cfg.model_cfg).to(device=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("==> Preparing validation dataset...")
    val_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.val_jsonl, dataset_dir=cfg.val_dataset_dir, **cfg.dataset_cfg, aug_prob=0.)
    val_dataloader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=1)

    metrics = {
        "track_loss": [],
        "img_loss": [],
        "loss": [],
    }

    print(f"==> Starting evaluation on {len(val_dataset)} samples...")

    with torch.inference_mode():
        for i, (vid, track, vis, task_emb, action) in enumerate(tqdm(val_dataloader)):
            vid, track, vis, task_emb, action = _cast_batch_for_model(
                model,
                device,
                vid,
                track,
                vis,
                task_emb,
                action,
            )

            with _get_autocast_context(cfg, device):
                _, ret_dict = model.forward_loss(
                    vid,
                    track,
                    task_emb,
                    lbd_track=cfg.lbd_track,
                    lbd_img=cfg.lbd_img,
                    p_img=cfg.p_img,
                    vis=vis,
                    action=action,
                )

            metrics["track_loss"].append(ret_dict["track_loss"])
            metrics["img_loss"].append(ret_dict["img_loss"])
            metrics["loss"].append(ret_dict["loss"])

            if VIS_EVERY_N_BATCHES > 0 and i % VIS_EVERY_N_BATCHES == 0:
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
        key: float(np.mean(values)) if values else float("nan")
        for key, values in metrics.items()
    }
    avg_metrics["num_samples"] = len(metrics["loss"])
    avg_metrics["ckpt_path"] = CKPT_PATH

    print("\n==> Evaluation results:")
    for key, value in avg_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=4)

    print(f"\nResults saved to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
