import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


"""
#! 使用Accelerate进行多卡训练时, torch 2.7.1+cu128对应需要修改nvidia-nccl-cu12==2.26.5, 否则会导致NCCL错误
accelerate launch --num_processes 4 --main_process_port 29500 --gpu_ids 0,1,2,3 -m engine.train_track_transformer_action_acc
"""
def debug_on():
    # 指定使用的 GPU ID
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    batch_size = 16
    num_track_ids = 256
    gradient_accumulation_steps = 4
    sys.argv = [
        "train_track_transformer_action_acc.py",
        "--config-name=robocoin_track_transformer_action",
        f"num_track_ids={num_track_ids}",
        f"batch_size={batch_size}",
        f"gradient_accumulation_steps={gradient_accumulation_steps}",
        f"experiment=realbot_track_transformer_001B_action_bs_{batch_size}_grad_acc_{gradient_accumulation_steps}_numtrack_{num_track_ids}_ep1001",
        "epochs=1001",
        'train_jsonl="/data_jbx/Datasets/Realbot/episodes_train_realbot.jsonl"',
        'val_jsonl="/data_jbx/Datasets/Realbot/episodes_val_realbot.jsonl"',
        'train_dataset_dir="/data_jbx/Datasets/Realbot"',
        'val_dataset_dir="/data_jbx/Datasets/Realbot"',
        'stat_path="/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/stat.json"',
        'model_load_path="/data_jbx/Codes/ATM/results/track_transformer/RoboCoin_Pretrain_Track_Action_Transformer/model_best.ckpt"',
    ]
debug_on()

import hydra
import torch
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from atm.dataloader import RoboCoinATMActionDataset, get_dataloader
from atm.model import *
from atm.utils.log_utils import BestAvgLoss, MetricLogger, setup_for_distributed
from atm.utils.train_utils import init_wandb, setup_lr_scheduler, setup_optimizer


############### 1. 配置辅助函数 ###############
def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")
    set_seed(cfg.seed)


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    return cfg.get(key, default)


def _get_runtime_config(cfg: DictConfig):
    distributed_cfg = cfg.get("distributed", {})
    return {
        "mixed_precision": "bf16" if cfg.mix_precision else "no",
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "backend": _cfg_get(distributed_cfg, "backend", "deepspeed"),
        "zero_stage": int(_cfg_get(distributed_cfg, "zero_stage", 2)),
        "offload_optimizer_device": _cfg_get(distributed_cfg, "offload_optimizer_device", "none"),
        "offload_param_device": _cfg_get(distributed_cfg, "offload_param_device", "none"),
    }


def _resolve_optional_path(path):
    if path is None:
        return None

    resolved_path = str(path).strip()
    if not resolved_path:
        return None

    resolved_path = os.path.expanduser(resolved_path)
    return hydra.utils.to_absolute_path(resolved_path)


def _prepare_model_load_path(cfg: DictConfig):
    load_path = _cfg_get(cfg.get("model_cfg"), "load_path", None)
    resolved_path = _resolve_optional_path(load_path)
    if resolved_path is None:
        return None

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {resolved_path}")

    with open_dict(cfg.model_cfg):
        cfg.model_cfg.load_path = resolved_path

    return resolved_path


############### 2. Accelerator 多卡训练辅助函数 ###############
def _make_accelerator(cfg: DictConfig) -> Accelerator:
    runtime_cfg = _get_runtime_config(cfg)
    accelerator_kwargs = {
        "mixed_precision": runtime_cfg["mixed_precision"],
        "gradient_accumulation_steps": runtime_cfg["gradient_accumulation_steps"],
    }

    if runtime_cfg["backend"] == "ddp":
        return Accelerator(**accelerator_kwargs)

    if runtime_cfg["backend"] != "deepspeed":
        raise ValueError(
            f"Unsupported distributed backend: {runtime_cfg['backend']}. "
            "Expected one of ['ddp', 'deepspeed']."
        )

    if runtime_cfg["zero_stage"] not in (1, 2, 3):
        raise ValueError(
            f"Unsupported DeepSpeed ZeRO stage: {runtime_cfg['zero_stage']}. "
            "Expected one of [1, 2, 3]."
        )

    accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(
        zero_stage=runtime_cfg["zero_stage"],
        gradient_accumulation_steps=runtime_cfg["gradient_accumulation_steps"],
        gradient_clipping=cfg.clip_grad,
        offload_optimizer_device=runtime_cfg["offload_optimizer_device"],
        offload_param_device=runtime_cfg["offload_param_device"],
    )
    return Accelerator(**accelerator_kwargs)


def _accumulate_batch_metrics(accelerator: Accelerator, ret_dict, batch_size: int):
    batch_metrics = torch.tensor(
        [
            ret_dict["track_loss"],
            ret_dict["img_loss"],
            ret_dict["loss"],
        ],
        dtype=torch.float32,
        device=accelerator.device,
    )
    batch_metrics = batch_metrics.unsqueeze(0).repeat(batch_size, 1)
    gathered_metrics = accelerator.gather_for_metrics(batch_metrics)
    return gathered_metrics.sum(dim=0), gathered_metrics.shape[0]

############### 3. 模型包装类 ###############
#! 不修改 Model.forward_loss 接口，适配多卡包装类对 forward 接口的依赖
class LossAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        vid,
        track,
        task_emb,
        lbd_track,
        lbd_img,
        p_img,
        action,
        return_outs=False,
        vis=None,
    ):
        return self.model.forward_loss(
            vid,
            track,
            task_emb,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            action=action,
            return_outs=return_outs,
            vis=vis,
        )

    def forward_loss(self, *args, **kwargs):
        return self.model.forward_loss(*args, **kwargs)

    def forward_vis(self, *args, **kwargs):
        return self.model.forward_vis(*args, **kwargs)

    def save(self, path):
        self.model.save(path)


def _get_base_model(accelerator: Accelerator, model):
    raw_model = accelerator.unwrap_model(model)
    while isinstance(raw_model, LossAdapter):
        raw_model = raw_model.model
    return raw_model


def _save_model(accelerator: Accelerator, model, save_path: str):
    _get_base_model(accelerator, model).save(save_path)


############### 4. 输入数据类型适配函数 ###############
def _get_model_input_dtype(accelerator: Accelerator, model):
    base_model = _get_base_model(accelerator, model)
    img_proj = getattr(base_model, "img_proj_encoder", None)
    if img_proj is not None and hasattr(img_proj, "proj") and hasattr(img_proj.proj, "weight"):
        return img_proj.proj.weight.dtype

    for param in base_model.parameters():
        if param.is_floating_point():
            return param.dtype

    return torch.float32


def _move_tensor_to_model_device(tensor, device, dtype):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor.to(device=device, non_blocking=True)


def _cast_batch_for_model(accelerator: Accelerator, model, vid, track, vis, task_emb, action):
    target_dtype = _get_model_input_dtype(accelerator, model)
    device = accelerator.device
    return (
        _move_tensor_to_model_device(vid, device, target_dtype),
        _move_tensor_to_model_device(track, device, target_dtype),
        _move_tensor_to_model_device(vis, device, target_dtype),
        _move_tensor_to_model_device(task_emb, device, target_dtype),
        _move_tensor_to_model_device(action, device, target_dtype),
    )


############### 5. 主训练循环 ###############
@hydra.main(config_path="../conf/train_track_transformer", version_base="1.3")
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    runtime_cfg = _get_runtime_config(cfg)
    model_load_path = _prepare_model_load_path(cfg)

    accelerator = _make_accelerator(cfg)
    setup_for_distributed(accelerator.is_main_process)

    if accelerator.is_main_process:
        OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))
        print(
            "###### Distributed Setup ######\n"
            f"backend={runtime_cfg['backend']}, "
            f"mixed_precision={runtime_cfg['mixed_precision']}, "
            f"gradient_accumulation_steps={runtime_cfg['gradient_accumulation_steps']}"
        )
        if model_load_path is not None:
            print(f"Loading pretrained weights from: {model_load_path}")
        if runtime_cfg["backend"] == "deepspeed":
            print(
                "DeepSpeed settings: "
                f"zero_stage={runtime_cfg['zero_stage']}, "
                f"offload_optimizer_device={runtime_cfg['offload_optimizer_device']}, "
                f"offload_param_device={runtime_cfg['offload_param_device']}"
            )

    None if (cfg.dry or not accelerator.is_main_process) else init_wandb(cfg)

    train_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.train_jsonl, dataset_dir=cfg.train_dataset_dir, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    # train_vis_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.train_jsonl, dataset_dir=cfg.train_dataset_dir, vis=True, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    # train_vis_dataloader = get_dataloader(train_vis_dataset, mode="train", num_workers=1, batch_size=1)

    val_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.val_jsonl, dataset_dir=cfg.val_dataset_dir, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size * 2)

    # val_vis_dataset = RoboCoinATMActionDataset(jsonl_path=cfg.val_jsonl, dataset_dir=cfg.val_dataset_dir, vis=True, **cfg.dataset_cfg, aug_prob=0.)
    # val_vis_dataloader = get_dataloader(val_vis_dataset, mode="val", num_workers=1, batch_size=1)

    model_cls = eval(cfg.model_name)
    base_model = model_cls(**cfg.model_cfg)
    model = LossAdapter(base_model)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    if accelerator.is_main_process:
        print("###### Model Structure ######")
        print(base_model)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    lbd_track = cfg.lbd_track
    lbd_img = cfg.lbd_img
    p_img = cfg.p_img

    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)
    epoch_iter = (
        metric_logger.log_every(range(cfg.epochs), 1, "")
        if accelerator.is_main_process
        else range(cfg.epochs)
    )

    for epoch in epoch_iter:
        train_metrics = run_one_epoch(
            accelerator,
            model,
            train_loader,
            optimizer,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            scheduler=scheduler,
            clip_grad=cfg.clip_grad,
        )
        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]

        if accelerator.is_main_process:
            metric_logger.update(**train_metrics)
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

        if epoch % cfg.val_freq == 0:
            val_metrics = evaluate(
                accelerator,
                model,
                val_loader,
                lbd_track=lbd_track,
                lbd_img=lbd_img,
                p_img=p_img,
                tag="val",
            )
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                metric_logger.update(**val_metrics)
                loss_metric = val_metrics["val/loss"]
                is_best = best_loss_logger.update_best(loss_metric, epoch)

                if is_best:
                    _save_model(accelerator, model, f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "w", encoding="utf-8") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.4f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

        if epoch % cfg.save_freq == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                _save_model(accelerator, model, f"{work_dir}/model_{epoch}.ckpt")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_model(accelerator, model, f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(
    accelerator: Accelerator,
    model,
    dataloader,
    optimizer,
    lbd_track,
    lbd_img,
    p_img,
    scheduler=None,
    clip_grad=1.0,
):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    metric_sums = torch.zeros(3, dtype=torch.float32, device=accelerator.device)
    total_items = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for vid, track, vis, task_emb, action in tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
    ):
        batch_size = vid.shape[0]
        vid, track, vis, task_emb, action = _cast_batch_for_model(
            accelerator,
            model,
            vid,
            track,
            vis,
            task_emb,
            action,
        )
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss, ret_dict = model(
                    vid,
                    track,
                    task_emb,
                    lbd_track=lbd_track,
                    lbd_img=lbd_img,
                    p_img=p_img,
                    action=action,
                )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_metric_sums, batch_items = _accumulate_batch_metrics(
            accelerator,
            ret_dict,
            batch_size=batch_size,
        )
        metric_sums += batch_metric_sums
        total_items += batch_items

    out_dict = {
        "train/track_loss": (metric_sums[0] / total_items).item(),
        "train/vid_loss": (metric_sums[1] / total_items).item(),
        "train/loss": (metric_sums[2] / total_items).item(),
    }

    if scheduler is not None:
        scheduler.step()

    return out_dict


@torch.no_grad()
def evaluate(accelerator: Accelerator, model, dataloader, lbd_track, lbd_img, p_img, tag="val"):
    metric_sums = torch.zeros(3, dtype=torch.float32, device=accelerator.device)
    total_items = 0
    model.eval()

    for vid, track, vis, task_emb, action in tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
    ):
        batch_size = vid.shape[0]
        vid, track, vis, task_emb, action = _cast_batch_for_model(
            accelerator,
            model,
            vid,
            track,
            vis,
            task_emb,
            action,
        )

        with accelerator.autocast():
            _, ret_dict = model(
                vid,
                track,
                task_emb,
                lbd_track=lbd_track,
                lbd_img=lbd_img,
                p_img=p_img,
                vis=vis,
                action=action,
            )

        batch_metric_sums, batch_items = _accumulate_batch_metrics(
            accelerator,
            ret_dict,
            batch_size=batch_size,
        )
        metric_sums += batch_metric_sums
        total_items += batch_items

    return {
        f"{tag}/track_loss": (metric_sums[0] / total_items).item(),
        f"{tag}/vid_loss": (metric_sums[1] / total_items).item(),
        f"{tag}/loss": (metric_sums[2] / total_items).item(),
    }


if __name__ == "__main__":
    main()
