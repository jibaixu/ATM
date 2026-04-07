import os
import sys
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from atm.model.track_transformer_action import TrackTransformerAction
from atm.utils.flow_utils import draw_tracks_on_single_image


def _build_default_cfg():
    return OmegaConf.create(
        {
            "mix_precision": True,
            "model_cfg": {
                "transformer_cfg": {
                    "dim": 384,
                    "dim_head": None,
                    "heads": 8,
                    "depth": 8,
                    "attn_dropout": 0.2,
                    "ff_dropout": 0.2,
                },
                "track_cfg": {
                    "num_track_ts": 81,
                    "num_track_ids": 256,
                    "patch_size": 9,
                },
                "vid_cfg": {
                    "img_size": [240, 320],
                    "frame_stack": 1,
                    "patch_size": 16,
                },
                "language_encoder_cfg": {
                    "network_name": "MLPEncoder",
                    "input_size": 768,
                    "hidden_size": 128,
                    "num_layers": 1,
                },
            },
        }
    )


def _load_runtime_cfg(default_cfg, ckpt_path):
    saved_cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    if os.path.exists(saved_cfg_path):
        return OmegaConf.load(saved_cfg_path)
    return default_cfg


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


def _get_autocast_context(cfg, device):
    if device.type == "cuda" and cfg.get("mix_precision", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _build_query_points(num_track_ids, device, dtype, margin=0.02):
    side = int(np.sqrt(num_track_ids))
    if side * side != num_track_ids:
        raise ValueError(
            f"Square uniform grid requires num_track_ids to be a perfect square, got {num_track_ids}."
        )

    y = torch.linspace(margin, 1.0 - margin, side, device=device, dtype=dtype)
    x = torch.linspace(margin, 1.0 - margin, side, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


class ATMInference:
    def __init__(self, checkpoint_path, device="cuda"):
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")

        self.checkpoint_path = checkpoint_path
        self.cfg = _load_runtime_cfg(_build_default_cfg(), checkpoint_path)
        with open_dict(self.cfg.model_cfg):
            self.cfg.model_cfg.action_dim = 14

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model = TrackTransformerAction(**self.cfg.model_cfg).to(device=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.mix_precision = bool(self.cfg.get("mix_precision", False))
        self.num_track_ids = int(self.cfg.model_cfg.track_cfg.num_track_ids)
        self.num_track_ts = int(self.cfg.model_cfg.track_cfg.num_track_ts)
        self.frame_stack = int(self.cfg.model_cfg.vid_cfg.frame_stack)
        self.action_dim = int(self.cfg.model_cfg.action_dim)
        self.task_emb_dim = int(self.cfg.model_cfg.language_encoder_cfg.input_size)
        img_size_cfg = self.cfg.model_cfg.vid_cfg.img_size
        self.img_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))

    def _validate_inputs(self, video, task_emb, action):
        if not all(isinstance(tensor, torch.Tensor) for tensor in (video, task_emb, action)):
            raise TypeError("video, task_emb, and action must all be torch.Tensor instances.")
        if video.ndim != 5:
            raise ValueError(f"video must have shape (B, T, C, H, W), got {tuple(video.shape)}.")
        if task_emb.ndim != 2:
            raise ValueError(f"task_emb must have shape (B, E), got {tuple(task_emb.shape)}.")
        if action.ndim != 3:
            raise ValueError(f"action must have shape (B, T, A), got {tuple(action.shape)}.")
        if video.shape[0] != task_emb.shape[0] or video.shape[0] != action.shape[0]:
            raise ValueError(
                "Batch size mismatch among video, task_emb, and action: "
                f"{video.shape[0]}, {task_emb.shape[0]}, {action.shape[0]}."
            )
        if video.shape[1] < self.frame_stack:
            raise ValueError(
                f"video provides {video.shape[1]} frames, but model requires at least {self.frame_stack}."
            )
        if tuple(video.shape[-2:]) != self.img_size:
            raise ValueError(
                f"video spatial size must match checkpoint config {self.img_size}, "
                f"got {tuple(video.shape[-2:])}."
            )
        if task_emb.shape[1] != self.task_emb_dim:
            raise ValueError(
                f"task_emb dim must match checkpoint config {self.task_emb_dim}, "
                f"got {task_emb.shape[1]}."
            )
        if action.shape[1] != self.num_track_ts or action.shape[2] != self.action_dim:
            raise ValueError(
                f"action must have shape (B, {self.num_track_ts}, {self.action_dim}), "
                f"got {tuple(action.shape)}."
            )
        if not task_emb.is_floating_point():
            raise TypeError("task_emb must be a floating-point tensor.")
        if not action.is_floating_point():
            raise TypeError("action must be a floating-point tensor.")

    def _generate_query_tracks(self, batch_size):
        dtype = _get_model_input_dtype(self.model)
        points = _build_query_points(self.num_track_ids, self.device, dtype)
        return repeat(points, "n c -> b t n c", b=batch_size, t=self.num_track_ts)

    @torch.no_grad()
    def infer(self, video, task_emb, action):
        """
        Args:
            video: (B, T, C, H, W), raw image values in the same format used by eval.
            task_emb: (B, E), already prepared language embedding tensor.
            action: (B, num_track_ts, action_dim), already reordered and normalized.
        Returns:
            rec_track: (B, num_track_ts, num_track_ids, 2), normalized coordinates in [0, 1].
        """
        self._validate_inputs(video, task_emb, action)

        if not video.is_floating_point():
            video = video.float()

        track_grid = self._generate_query_tracks(video.shape[0])
        video, track_grid, _, task_emb, action = _cast_batch_for_model(
            self.model,
            self.device,
            video,
            track_grid,
            None,
            task_emb,
            action,
        )

        with torch.inference_mode():
            with _get_autocast_context(self.cfg, self.device):
                rec_track, _ = self.model.reconstruct(
                    vid=video,
                    track=track_grid,
                    task_emb=task_emb,
                    p_img=0.0,
                    action=action,
                )
        return rec_track

    def draw_tracks_on_image(self, image, predictions):
        """
        Draw a subset of predicted trajectories on a single RGB image.

        Args:
            image: RGB image in HWC or CHW format.
            predictions: (T, N, 2) or (1, T, N, 2), normalized coordinates.
            track_idx_to_show: optional indices of tracks to render.
        """
        if isinstance(predictions, torch.Tensor):
            pred_tensor = predictions.detach().cpu()
        else:
            pred_tensor = torch.as_tensor(predictions)

        if pred_tensor.ndim == 3:
            pred_tensor = pred_tensor.unsqueeze(0)
        if pred_tensor.ndim != 4 or pred_tensor.shape[0] != 1:
            raise ValueError(
                "predictions must have shape (T, N, 2) or (1, T, N, 2)."
            )

        return draw_tracks_on_single_image(
            pred_tensor,
            image,
            img_size=self.img_size,
            tracks_leave_trace=min(15, pred_tensor.shape[1] - 1),
        )

    def save_predictions_video(self, background_img, predictions, output_path="output_tracks.mp4", track_idx_to_show=None):
        """
        Save a simple trajectory video for qualitative inspection.

        Args:
            background_img: RGB image in HWC format.
            predictions: (T, N, 2) or (1, T, N, 2), normalized coordinates.
        """
        if isinstance(predictions, torch.Tensor):
            pred_tensor = predictions.detach().cpu()
        else:
            pred_tensor = torch.as_tensor(predictions)

        if pred_tensor.ndim == 3:
            pred_tensor = pred_tensor.unsqueeze(0)
        if pred_tensor.ndim != 4 or pred_tensor.shape[0] != 1:
            raise ValueError(
                "predictions must have shape (T, N, 2) or (1, T, N, 2)."
            )

        num_tracks = pred_tensor.shape[2]
        if track_idx_to_show is None:
            num_show = min(32, num_tracks)
            track_idx_to_show = np.linspace(0, num_tracks - 1, num_show, dtype=int)

        pred_tensor = pred_tensor[:, :, track_idx_to_show]
        height, width = self.img_size
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (width, height),
        )

        for t in range(pred_tensor.shape[1]):
            frame = draw_tracks_on_single_image(
                pred_tensor[:, : t + 1],
                background_img,
                img_size=self.img_size,
                tracks_leave_trace=min(15, t),
            )
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video_writer.release()
        print(f"Video saved to {output_path}")


if __name__ == "__main__":
    # 1. 路径配置（参考 eval_track_transformer_action.py 中的设置）
    #
    checkpoint_path = "/home/jibaixu/Codes/ATM/results/track_transformer/0326_robocoin_track_transformer_001B_action_bs_8_grad_acc_4_numtrack_256_robocoin-object_ep1001_2109/model_best.ckpt"

    dataset_root = "/home/jibaixu/Datasets/Cobot_Magic_all_extracted/resize_240_320"
    jsonl_path = os.path.join(dataset_root, "episodes_clipped_train_test.jsonl")
    stat_path = os.path.join(dataset_root, "stat.json")

    # 2. 初始化推理引擎
    #
    infer_engine = ATMInference(checkpoint_path)

    # 3. 初始化真实数据集
    # 按照 eval 脚本逻辑，关闭数据增强 (aug_prob=0) 并使用对应的统计文件
    from atm.dataloader.robocoin_action_dataloader import RoboCoinATMActionDataset
    
    dataset = RoboCoinATMActionDataset(
        jsonl_path=jsonl_path,
        dataset_dir=dataset_root,
        img_size=infer_engine.img_size,
        num_track_ts=infer_engine.num_track_ts,
        num_track_ids=infer_engine.num_track_ids,
        frame_stack=infer_engine.frame_stack,
        stat_path=stat_path,
        aug_prob=0.0,  # 推理时不需要增强
        cache_all=False
    )

    # 4. 取出一个真实数据样本
    # dataset[i] 返回: frames(T,C,H,W), tracks(T,N,2), vis(T,N), task_emb(E), actions(T,A)
    sample_idx = 0
    real_video, real_tracks, _, real_task_emb, real_action = dataset[sample_idx]

    # 5. 增加 Batch 维度 (B=1) 以适配推理接口
    #
    input_vid = real_video.unsqueeze(0)        # (1, T, C, H, W)
    input_task_emb = real_task_emb.unsqueeze(0) # (1, E)
    input_action = real_action.unsqueeze(0)    # (1, T_track, A)

    print(f"正在处理样本 {sample_idx}，任务维度: {input_task_emb.shape}")

    # 6. 执行推理
    #
    preds = infer_engine.infer(input_vid, input_task_emb, input_action)

    # 7. 保存可视化结果
    # 取最后一帧作为背景图
    img_np = real_video[-1].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # 保存静态轨迹图
    static_vis = infer_engine.draw_tracks_on_image(img_np, preds[0])
    cv2.imwrite("real_sample_tracks.jpg", cv2.cvtColor(static_vis, cv2.COLOR_RGB2BGR))

    # 保存轨迹动画视频
    infer_engine.save_predictions_video(
        img_np, 
        preds[0], 
        output_path="real_sample_inference.mp4"
    )
