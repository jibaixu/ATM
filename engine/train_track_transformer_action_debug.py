import json
import math
import os
import sys
import traceback
from pathlib import Path


def debug_on():
    # 指定使用的 GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    batch_size = 4
    num_track_ids = 256
    sys.argv = [
        "train_track_transformer_action.py",
        "--config-name=robocoin_track_transformer_action",
        f"num_track_ids={num_track_ids}",
        f"batch_size={batch_size}",
        "train_gpus=[0]",
        f"experiment=robocoin_track_transformer_01B_action_bs_{batch_size}_numtrack_{num_track_ids}_robocoin-object_ep1001",
        "epochs=1001",
        'train_jsonl="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320/episodes_clipped_train.jsonl"',
        'val_jsonl="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320/episodes_clipped_val.jsonl"',
        'train_dataset_dir="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320"',
        'val_dataset_dir="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320"',
        'stat_path="/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320/stat.json"',
    ]


debug_on()

import hydra
import lightning
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from atm.dataloader import RoboCoinATMActionDataset, get_dataloader
from atm.model import *
from atm.utils.log_utils import BestAvgLoss, MetricLogger
from atm.utils.train_utils import init_wandb, setup_lr_scheduler, setup_optimizer


class NumericalDebugError(RuntimeError):
    pass


def _build_debug_cfg(cfg: DictConfig):
    defaults = {
        "enable": True,
        "check_inputs_every_n_steps": 1,
        "log_stats_every_n_steps": 50,
        "check_params_every_n_steps": 1,
        "fail_on_nonfinite": True,
        "fail_on_large_loss": True,
        "loss_threshold": 1e4,
        "fail_on_large_grad_norm": True,
        "grad_norm_threshold": 1e4,
        "fail_on_large_param_absmax": False,
        "param_absmax_threshold": 1e4,
        "detect_anomaly": False,
        "save_snapshot_on_error": True,
        "preview_items": 16,
        "topk_items": 8,
        "input_thresholds": {
            "vid_abs_max": 512.0,
            "track_abs_max": None,
            "action_abs_max": 5.0,
            "task_emb_abs_max": 1e3,
            "vis_min": -1e-3,
            "vis_max": 1.001,
        },
    }

    user_cfg = {}
    if cfg.get("debug") is not None:
        user_cfg = OmegaConf.to_container(cfg.debug, resolve=True)

    merged = {**defaults, **{k: v for k, v in user_cfg.items() if k != "input_thresholds"}}
    input_thresholds = {**defaults["input_thresholds"], **user_cfg.get("input_thresholds", {})}
    if input_thresholds["track_abs_max"] is None:
        input_thresholds["track_abs_max"] = float(max(cfg.img_size) * 4)
    merged["input_thresholds"] = input_thresholds
    return merged


def _should_run(step, every_n_steps):
    return every_n_steps <= 1 or step % every_n_steps == 0


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().cpu().item())
    return float(value)


def _format_float(value, precision=6):
    if value is None:
        return "None"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


def _tensor_stats(tensor, preview_items=16):
    tensor = tensor.detach()
    flat_float = tensor.reshape(-1).float()
    finite_mask = torch.isfinite(flat_float)
    finite_vals = flat_float[finite_mask]
    nan_count = int(torch.isnan(flat_float).sum().item())
    inf_count = int(torch.isinf(flat_float).sum().item())

    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(flat_float.numel()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_count": nan_count + inf_count,
        "preview": flat_float[:preview_items].cpu().tolist(),
    }

    if finite_vals.numel() == 0:
        stats.update({
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_max": None,
        })
        return stats

    stats.update({
        "min": float(finite_vals.min().item()),
        "max": float(finite_vals.max().item()),
        "mean": float(finite_vals.mean().item()),
        "std": float(finite_vals.std(unbiased=False).item()) if finite_vals.numel() > 1 else 0.0,
        "abs_max": float(finite_vals.abs().max().item()),
    })
    return stats


def _collect_batch_stats(batch_tensors, preview_items):
    return {name: _tensor_stats(tensor, preview_items=preview_items) for name, tensor in batch_tensors.items()}


def _collect_batch_preview(batch_tensors, preview_items):
    preview = {}
    for name, tensor in batch_tensors.items():
        flat = tensor.detach().reshape(-1).float()
        preview[name] = {
            "shape": list(tensor.shape),
            "preview": flat[:preview_items].cpu().tolist(),
        }
    return preview


def _validate_input_stats(batch_stats, debug_cfg):
    issues = []
    thresholds = debug_cfg["input_thresholds"]

    for name, stats in batch_stats.items():
        if debug_cfg["fail_on_nonfinite"] and stats["nonfinite_count"] > 0:
            issues.append(f"{name} contains {stats['nonfinite_count']} non-finite values.")

    vid_abs_max = batch_stats["vid"]["abs_max"]
    if vid_abs_max is not None and vid_abs_max > thresholds["vid_abs_max"]:
        issues.append(f"vid abs max {vid_abs_max:.4f} exceeds {thresholds['vid_abs_max']:.4f}.")

    track_abs_max = batch_stats["track"]["abs_max"]
    if track_abs_max is not None and track_abs_max > thresholds["track_abs_max"]:
        issues.append(f"track abs max {track_abs_max:.4f} exceeds {thresholds['track_abs_max']:.4f}.")

    action_abs_max = batch_stats["action"]["abs_max"]
    if action_abs_max is not None and action_abs_max > thresholds["action_abs_max"]:
        issues.append(f"action abs max {action_abs_max:.4f} exceeds {thresholds['action_abs_max']:.4f}.")

    task_emb_abs_max = batch_stats["task_emb"]["abs_max"]
    if task_emb_abs_max is not None and task_emb_abs_max > thresholds["task_emb_abs_max"]:
        issues.append(f"task_emb abs max {task_emb_abs_max:.4f} exceeds {thresholds['task_emb_abs_max']:.4f}.")

    vis_min = batch_stats["vis"]["min"]
    vis_max = batch_stats["vis"]["max"]
    if vis_min is not None and vis_min < thresholds["vis_min"]:
        issues.append(f"vis min {vis_min:.4f} is below {thresholds['vis_min']:.4f}.")
    if vis_max is not None and vis_max > thresholds["vis_max"]:
        issues.append(f"vis max {vis_max:.4f} exceeds {thresholds['vis_max']:.4f}.")

    return issues


def _collect_loss_info(loss, ret_dict):
    ret_dict = ret_dict or {}
    return {
        "loss": _to_float(loss),
        "track_loss": _to_float(ret_dict.get("track_loss")),
        "img_loss": _to_float(ret_dict.get("img_loss")),
        "ret_dict_loss": _to_float(ret_dict.get("loss")),
    }


def _validate_loss_info(loss_info, debug_cfg):
    issues = []
    for name, value in loss_info.items():
        if value is None:
            continue
        if debug_cfg["fail_on_nonfinite"] and not math.isfinite(value):
            issues.append(f"{name} is non-finite: {value}.")
        if debug_cfg["fail_on_large_loss"] and math.isfinite(value) and abs(value) > debug_cfg["loss_threshold"]:
            issues.append(f"{name}={value:.4f} exceeds {debug_cfg['loss_threshold']:.4f}.")
    return issues


def _collect_grad_stats(model, topk=8):
    largest_grad_norms = []
    nonfinite_grads = []
    missing_grads = []
    total_sq_norm = 0.0
    max_grad_abs = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing_grads.append(name)
            continue

        grad = param.grad.detach().float()
        is_all_finite = bool(torch.isfinite(grad).all().item())

        if is_all_finite:
            grad_norm = float(torch.linalg.vector_norm(grad).item())
            abs_max = float(grad.abs().max().item()) if grad.numel() > 0 else 0.0
            total_sq_norm += grad_norm ** 2
            max_grad_abs = max(max_grad_abs, abs_max)
            largest_grad_norms.append({"name": name, "norm": grad_norm, "abs_max": abs_max})
            continue

        finite_mask = torch.isfinite(grad)
        finite_vals = grad[finite_mask]
        nan_count = int(torch.isnan(grad).sum().item())
        inf_count = int(torch.isinf(grad).sum().item())
        nonfinite_grads.append({
            "name": name,
            "numel": int(grad.numel()),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "norm": float(torch.linalg.vector_norm(finite_vals).item()) if finite_vals.numel() > 0 else None,
            "abs_max": float(finite_vals.abs().max().item()) if finite_vals.numel() > 0 else None,
        })

    largest_grad_norms = sorted(largest_grad_norms, key=lambda item: item["norm"], reverse=True)[:topk]
    return {
        "global_grad_norm": math.sqrt(total_sq_norm) if total_sq_norm > 0 else 0.0,
        "max_grad_abs": max_grad_abs,
        "largest_grad_norms": largest_grad_norms,
        "nonfinite_grads": nonfinite_grads[:topk],
        "nonfinite_grad_count": len(nonfinite_grads),
        "missing_grad_count": len(missing_grads),
        "missing_grad_samples": missing_grads[:topk],
    }


def _validate_grad_stats(grad_stats, debug_cfg):
    issues = []
    grad_norm = grad_stats["global_grad_norm"]
    if debug_cfg["fail_on_nonfinite"] and not math.isfinite(grad_norm):
        issues.append(f"global grad norm is non-finite: {grad_norm}.")
    if grad_stats["nonfinite_grad_count"] > 0:
        bad_names = ", ".join(item["name"] for item in grad_stats["nonfinite_grads"])
        issues.append(f"found non-finite gradients in parameters: {bad_names}.")
    if debug_cfg["fail_on_large_grad_norm"] and math.isfinite(grad_norm) and grad_norm > debug_cfg["grad_norm_threshold"]:
        issues.append(f"global grad norm {grad_norm:.4f} exceeds {debug_cfg['grad_norm_threshold']:.4f}.")
    return issues


def _collect_param_stats(model, topk=8):
    largest_params = []
    nonfinite_params = []
    max_param_abs = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        data = param.detach().float()
        is_all_finite = bool(torch.isfinite(data).all().item())

        if is_all_finite:
            abs_max = float(data.abs().max().item()) if data.numel() > 0 else 0.0
            max_param_abs = max(max_param_abs, abs_max)
            largest_params.append({"name": name, "abs_max": abs_max})
            continue

        finite_mask = torch.isfinite(data)
        finite_vals = data[finite_mask]
        nonfinite_params.append({
            "name": name,
            "numel": int(data.numel()),
            "nan_count": int(torch.isnan(data).sum().item()),
            "inf_count": int(torch.isinf(data).sum().item()),
            "abs_max": float(finite_vals.abs().max().item()) if finite_vals.numel() > 0 else None,
        })

    largest_params = sorted(largest_params, key=lambda item: item["abs_max"], reverse=True)[:topk]
    return {
        "max_param_abs": max_param_abs,
        "largest_params": largest_params,
        "nonfinite_params": nonfinite_params[:topk],
        "nonfinite_param_count": len(nonfinite_params),
    }


def _validate_param_stats(param_stats, debug_cfg):
    issues = []
    if param_stats["nonfinite_param_count"] > 0:
        bad_names = ", ".join(item["name"] for item in param_stats["nonfinite_params"])
        issues.append(f"found non-finite parameters after optimizer.step(): {bad_names}.")
    max_param_abs = param_stats["max_param_abs"]
    if debug_cfg["fail_on_large_param_absmax"] and math.isfinite(max_param_abs) and max_param_abs > debug_cfg["param_absmax_threshold"]:
        issues.append(f"max parameter abs value {max_param_abs:.4f} exceeds {debug_cfg['param_absmax_threshold']:.4f}.")
    return issues


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


def _write_debug_snapshot(work_dir, fabric, epoch, step, stage, payload):
    debug_dir = Path(work_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rank{fabric.global_rank}_epoch{epoch:04d}_step{step:06d}_{stage}"
    pt_path = debug_dir / f"{stem}.pt"
    json_path = debug_dir / f"{stem}.json"

    torch.save(payload, pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)

    return {"pt": str(pt_path), "json": str(json_path)}


def _raise_debug_error(fabric,
                       work_dir,
                       epoch,
                       step,
                       stage,
                       message,
                       batch_tensors,
                       debug_cfg,
                       issues=None,
                       batch_stats=None,
                       loss_info=None,
                       grad_stats=None,
                       param_stats=None,
                       extra=None):
    payload = {
        "meta": {
            "epoch": epoch,
            "step": step,
            "stage": stage,
            "global_rank": fabric.global_rank,
            "message": message,
        },
        "issues": issues or [],
        "batch_stats": batch_stats or _collect_batch_stats(batch_tensors, debug_cfg["preview_items"]),
        "batch_preview": _collect_batch_preview(batch_tensors, debug_cfg["preview_items"]),
        "loss_info": loss_info,
        "grad_stats": grad_stats,
        "param_stats": param_stats,
        "extra": extra or {},
    }

    snapshot_paths = None
    if debug_cfg["save_snapshot_on_error"]:
        snapshot_paths = _write_debug_snapshot(work_dir, fabric, epoch, step, stage, payload)
        print(
            f"[Debug][Rank {fabric.global_rank}] {message}. Snapshot: {snapshot_paths['json']}",
            flush=True,
        )

    raise NumericalDebugError(
        f"{message}. Debug snapshot: {snapshot_paths['json']}" if snapshot_paths is not None else message
    )


def _log_health(fabric, epoch, step, loss_info, grad_stats, batch_stats):
    fabric.print(
        f"[Debug][Epoch {epoch:04d} Step {step:06d}] "
        f"loss={_format_float(loss_info['loss'])}, "
        f"track_loss={_format_float(loss_info['track_loss'])}, "
        f"img_loss={_format_float(loss_info['img_loss'])}, "
        f"grad_norm={_format_float(grad_stats['global_grad_norm'])}, "
        f"vid_abs_max={_format_float(batch_stats['vid']['abs_max'], precision=4)}, "
        f"track_abs_max={_format_float(batch_stats['track']['abs_max'], precision=4)}, "
        f"action_abs_max={_format_float(batch_stats['action']['abs_max'], precision=4)}"
    )


@hydra.main(config_path="../conf/train_track_transformer", version_base="1.3")
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))
    debug_cfg = _build_debug_cfg(cfg)

    fabric = Fabric(
        accelerator="cuda",
        devices=list(cfg.train_gpus),
        precision="bf16-mixed" if cfg.mix_precision else None,
        strategy="deepspeed",
    )
    fabric.launch()

    if debug_cfg["enable"]:
        torch.autograd.set_detect_anomaly(debug_cfg["detect_anomaly"])
        fabric.print(
            f"[Debug] Numerical checks enabled. "
            f"input_every={debug_cfg['check_inputs_every_n_steps']}, "
            f"log_every={debug_cfg['log_stats_every_n_steps']}, "
            f"param_every={debug_cfg['check_params_every_n_steps']}, "
            f"loss_threshold={debug_cfg['loss_threshold']}, "
            f"grad_norm_threshold={debug_cfg['grad_norm_threshold']}, "
            f"detect_anomaly={debug_cfg['detect_anomaly']}"
        )

    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)

    train_dataset = RoboCoinATMActionDataset(
        jsonl_path=cfg.train_jsonl,
        dataset_dir=cfg.train_dataset_dir,
        **cfg.dataset_cfg,
        aug_prob=cfg.aug_prob,
    )
    train_loader = get_dataloader(
        train_dataset,
        mode="train",
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
    )

    # train_vis_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.train_jsonl, dataset_dir=cfg.train_dataset_dir, vis=True, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    # train_vis_dataloader = get_dataloader(train_vis_dataset, mode="train", num_workers=1, batch_size=1)

    val_dataset = RoboCoinATMActionDataset(
        jsonl_path=cfg.val_jsonl,
        dataset_dir=cfg.val_dataset_dir,
        **cfg.dataset_cfg,
        aug_prob=0.,
    )
    val_loader = get_dataloader(
        val_dataset,
        mode="val",
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size * 2,
    )

    # val_vis_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.val_jsonl, dataset_dir=cfg.val_dataset_dir, vis=True, **cfg.dataset_cfg, aug_prob=0.)
    # val_vis_dataloader = get_dataloader(val_vis_dataset, mode="val", num_workers=1, batch_size=1)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    print(f"###### Model Structure ######")
    print(model)

    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method("forward_loss")
    model.mark_forward_method("forward_vis")
    train_loader = fabric.setup_dataloaders(train_loader)

    lbd_track = cfg.lbd_track
    lbd_img = cfg.lbd_img
    p_img = cfg.p_img

    # Pick ckpt based on the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            work_dir=work_dir,
            debug_cfg=debug_cfg,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            scheduler=scheduler,
            mix_precision=cfg.mix_precision,
            clip_grad=cfg.clip_grad,
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    lbd_track=lbd_track,
                    lbd_img=lbd_img,
                    p_img=p_img,
                    mix_precision=cfg.mix_precision,
                    tag="val",
                )

                # Save best checkpoint
                metric_logger.update(**val_metrics)

                val_metrics = {**val_metrics}
                loss_metric = val_metrics["val/loss"]
                is_best = best_loss_logger.update_best(loss_metric, epoch)

                if is_best:
                    model.save(f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "w") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.4f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

            if epoch % cfg.save_freq == 0:
                model.save(f"{work_dir}/model_{epoch}.ckpt")

                def vis_and_log(model, vis_dataloader, mode="train"):
                    vis_dict = visualize(model, vis_dataloader, mix_precision=cfg.mix_precision)

                    caption = f"reconstruction (right) @ epoch {epoch}; \n Track MSE: {vis_dict['track_loss']:.4f}"
                    wandb_vis_track = wandb.Video(
                        vis_dict["combined_track_vid"],
                        fps=10,
                        format="mp4",
                        caption=caption,
                    )
                    None if cfg.dry else wandb.log({f"{mode}/reconstruct_track": wandb_vis_track}, step=epoch)

                # vis_and_log(model, train_vis_dataloader, mode="train")
                # vis_and_log(model, val_vis_dataloader, mode="val")

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric,
                  model,
                  dataloader,
                  optimizer,
                  epoch,
                  work_dir,
                  debug_cfg,
                  lbd_track,
                  lbd_img,
                  p_img,
                  mix_precision=False,
                  scheduler=None,
                  clip_grad=1.0):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    track_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0
    grad_norm_sum, grad_norm_steps, grad_norm_max = 0.0, 0, 0.0

    model.train()
    for step, (vid, track, vis, task_emb, action) in enumerate(
        tqdm(dataloader, disable=not fabric.is_global_zero),
        start=1,
    ):
        batch_tensors = {
            "vid": vid,
            "track": track,
            "vis": vis,
            "task_emb": task_emb,
            "action": action,
        }
        batch_stats = None
        ret_dict = None
        loss_info = None
        grad_stats = None
        param_stats = None
        clipped_grad_norm = None

        try:
            if debug_cfg["enable"] and _should_run(step, debug_cfg["check_inputs_every_n_steps"]):
                batch_stats = _collect_batch_stats(batch_tensors, debug_cfg["preview_items"])
                input_issues = _validate_input_stats(batch_stats, debug_cfg)
                if input_issues:
                    _raise_debug_error(
                        fabric=fabric,
                        work_dir=work_dir,
                        epoch=epoch,
                        step=step,
                        stage="input_check",
                        message="Input batch failed numerical checks",
                        batch_tensors=batch_tensors,
                        debug_cfg=debug_cfg,
                        issues=input_issues,
                        batch_stats=batch_stats,
                    )

            b, t, c, h, w = vid.shape
            b, tl, n, _ = track.shape
            b, tl, n = vis.shape
            loss, ret_dict = model.forward_loss(
                vid,
                track,
                task_emb,
                lbd_track=lbd_track,
                lbd_img=lbd_img,
                p_img=p_img,
                action=action,
            )  # do not use vis

            loss_info = _collect_loss_info(loss, ret_dict)
            if debug_cfg["enable"]:
                loss_issues = _validate_loss_info(loss_info, debug_cfg)
                if loss_issues:
                    _raise_debug_error(
                        fabric=fabric,
                        work_dir=work_dir,
                        epoch=epoch,
                        step=step,
                        stage="forward_loss",
                        message="Forward loss failed numerical checks",
                        batch_tensors=batch_tensors,
                        debug_cfg=debug_cfg,
                        issues=loss_issues,
                        batch_stats=batch_stats,
                        loss_info=loss_info,
                        extra={"ret_dict": ret_dict},
                    )

            optimizer.zero_grad()
            fabric.backward(loss)

            grad_stats = _collect_grad_stats(model, topk=debug_cfg["topk_items"])
            grad_norm = grad_stats["global_grad_norm"]
            if math.isfinite(grad_norm):
                grad_norm_sum += grad_norm
                grad_norm_steps += 1
                grad_norm_max = max(grad_norm_max, grad_norm)

            if debug_cfg["enable"]:
                grad_issues = _validate_grad_stats(grad_stats, debug_cfg)
                if grad_issues:
                    _raise_debug_error(
                        fabric=fabric,
                        work_dir=work_dir,
                        epoch=epoch,
                        step=step,
                        stage="backward_grad",
                        message="Gradient check failed after backward",
                        batch_tensors=batch_tensors,
                        debug_cfg=debug_cfg,
                        issues=grad_issues,
                        batch_stats=batch_stats,
                        loss_info=loss_info,
                        grad_stats=grad_stats,
                        extra={"ret_dict": ret_dict},
                    )

            clipped_grad_norm = _to_float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad))
            if debug_cfg["enable"]:
                clip_issues = []
                if debug_cfg["fail_on_nonfinite"] and clipped_grad_norm is not None and not math.isfinite(clipped_grad_norm):
                    clip_issues.append(f"clip_grad_norm returned non-finite value: {clipped_grad_norm}.")
                if (
                    debug_cfg["fail_on_large_grad_norm"]
                    and clipped_grad_norm is not None
                    and math.isfinite(clipped_grad_norm)
                    and clipped_grad_norm > debug_cfg["grad_norm_threshold"]
                ):
                    clip_issues.append(
                        f"clip_grad_norm returned {clipped_grad_norm:.4f}, exceeding {debug_cfg['grad_norm_threshold']:.4f}."
                    )
                if clip_issues:
                    _raise_debug_error(
                        fabric=fabric,
                        work_dir=work_dir,
                        epoch=epoch,
                        step=step,
                        stage="clip_grad",
                        message="Gradient clipping check failed",
                        batch_tensors=batch_tensors,
                        debug_cfg=debug_cfg,
                        issues=clip_issues,
                        batch_stats=batch_stats,
                        loss_info=loss_info,
                        grad_stats=grad_stats,
                        extra={"clip_grad_norm": clipped_grad_norm, "ret_dict": ret_dict},
                    )

            optimizer.step()

            if debug_cfg["enable"] and _should_run(step, debug_cfg["check_params_every_n_steps"]):
                param_stats = _collect_param_stats(model, topk=debug_cfg["topk_items"])
                param_issues = _validate_param_stats(param_stats, debug_cfg)
                if param_issues:
                    _raise_debug_error(
                        fabric=fabric,
                        work_dir=work_dir,
                        epoch=epoch,
                        step=step,
                        stage="optimizer_step",
                        message="Parameter check failed after optimizer.step()",
                        batch_tensors=batch_tensors,
                        debug_cfg=debug_cfg,
                        issues=param_issues,
                        batch_stats=batch_stats,
                        loss_info=loss_info,
                        grad_stats=grad_stats,
                        param_stats=param_stats,
                        extra={"clip_grad_norm": clipped_grad_norm, "ret_dict": ret_dict},
                    )

            if debug_cfg["enable"] and _should_run(step, debug_cfg["log_stats_every_n_steps"]):
                if batch_stats is None:
                    batch_stats = _collect_batch_stats(batch_tensors, debug_cfg["preview_items"])
                _log_health(fabric, epoch, step, loss_info, grad_stats, batch_stats)

            track_loss += ret_dict["track_loss"]
            vid_loss += ret_dict["img_loss"]
            tot_loss += ret_dict["loss"]
            tot_items += b

        except Exception as exc:
            if debug_cfg["enable"] and not isinstance(exc, NumericalDebugError):
                snapshot_paths = _write_debug_snapshot(
                    work_dir=work_dir,
                    fabric=fabric,
                    epoch=epoch,
                    step=step,
                    stage="exception",
                    payload={
                        "meta": {
                            "epoch": epoch,
                            "step": step,
                            "stage": "exception",
                            "global_rank": fabric.global_rank,
                            "message": "Unhandled exception during training step",
                        },
                        "batch_stats": batch_stats or _collect_batch_stats(batch_tensors, debug_cfg["preview_items"]),
                        "batch_preview": _collect_batch_preview(batch_tensors, debug_cfg["preview_items"]),
                        "loss_info": loss_info,
                        "grad_stats": grad_stats,
                        "param_stats": param_stats,
                        "extra": {
                            "clip_grad_norm": clipped_grad_norm,
                            "ret_dict": ret_dict,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    },
                )
                print(
                    f"[Debug][Rank {fabric.global_rank}] Unhandled exception at epoch={epoch}, step={step}. "
                    f"Snapshot: {snapshot_paths['json']}",
                    flush=True,
                )
            raise

    out_dict = {
        "train/track_loss": track_loss / tot_items,
        "train/vid_loss": vid_loss / tot_items,
        "train/loss": tot_loss / tot_items,
        "train/grad_norm_avg": grad_norm_sum / grad_norm_steps if grad_norm_steps > 0 else 0.0,
        "train/grad_norm_max": grad_norm_max,
    }

    if scheduler is not None:
        scheduler.step()

    return out_dict


@torch.no_grad()
def evaluate(model, dataloader, lbd_track, lbd_img, p_img, mix_precision=False, tag="val"):
    track_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0
    model.eval()

    i = 0
    for vid, track, vis, task_emb, action in tqdm(dataloader):
        vid, track, vis, task_emb, action = vid.cuda(), track.cuda(), vis.cuda(), task_emb.cuda(), action.cuda()
        # if mix_precision:
        #     vid, track, vis, task_emb, action = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16(), action.bfloat16()
        b, t, c, h, w = vid.shape
        b, tl, n, _ = track.shape

        _, ret_dict = model.forward_loss(
            vid,
            track,
            task_emb,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            vis=vis,
            action=action,
        )

        track_loss += ret_dict["track_loss"]
        vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        tot_items += b

        i += 1

    out_dict = {
        f"{tag}/track_loss": track_loss / tot_items,
        f"{tag}/vid_loss": vid_loss / tot_items,
        f"{tag}/loss": tot_loss / tot_items,
    }

    return out_dict


@torch.no_grad()
def visualize(model, dataloader, mix_precision=False):
    model.eval()
    keep_eval_dict = None

    for i, (vid, track, vis, task_emb, action) in enumerate(dataloader):
        vid, track, task_emb, action = vid.cuda(), track.cuda(), task_emb.cuda(), action.cuda()
        # if mix_precision:
        #     vid, track, task_emb, action = vid.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
        _, eval_dict = model.forward_vis(vid, track, task_emb, p_img=0, action=action)
        if keep_eval_dict is None or torch.rand(1) < 0.1:
            keep_eval_dict = eval_dict

        if i == 10:
            break
    return keep_eval_dict


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()
