import os
import sys


CKPT_PATH = "/data2/jibaixu/Codes/ATM/results/track_transformer/0303_libero_track_transformer_action_bs_128_numtrack_128_libero-object_ep1001_1325/model_best.ckpt"
EXP_NAME = CKPT_PATH.split('/')[-2]
MODEL_NAME = CKPT_PATH.split('/')[0].split('.')[0]
OUTPUT_DIR = f"results/eval_track_result/{EXP_NAME}/{MODEL_NAME}"


def debug_on():
    # 指定使用的 GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    batch_size = int(EXP_NAME.split('_')[6])
    num_track_ids = int(EXP_NAME.split('_')[8])
    sys.argv = [
        "eval_track_transformer.py",
        "--config-name=libero_track_transformer_action",
        f"num_track_ids={num_track_ids}",
        f"batch_size={batch_size}",
        "train_gpus=[0]",
        f"experiment=eval_{EXP_NAME}",
        "epochs=1001",
        'train_dataset=["/data1/jibaixu/Datasets/ATM/atm_libero/libero_object/*/train/"]',
        'val_dataset=["/data1/jibaixu/Datasets/ATM/atm_libero/libero_object/*/val/"]',
    ]
debug_on()

import torch
import numpy as np
import json
import cv2
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from atm.model import TrackTransformerAction
from atm.dataloader import ATMActionPretrainDataset, get_dataloader


def save_image(img_array, path):
    """将 numpy 数组 (H, W, 3) 保存为本地图片"""
    # 模型输出通常是 RGB，OpenCV 需要 BGR
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

@hydra.main(version_base=None, config_path="./conf/train_track_transformer", config_name="libero_track_transformer_action")
def main(cfg: DictConfig):
    # 1. 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"==> Loading model from: {CKPT_PATH}")
    
    # 2. 初始化模型并加载权重
    # 注意：确保这里传入了 action_dim
    model = TrackTransformerAction(**cfg.model_cfg).to(device=device, dtype=torch.bfloat16)

    # 加载权重 (如果是从 checkpoint 加载)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    # 如果 checkpoint 包含 'model' 键则取其值，否则加载整个 state_dict
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 准备验证集
    print("==> Preparing Validation Dataset...")
    val_dataset = ATMActionPretrainDataset(dataset_dir=cfg.val_dataset, **cfg.dataset_cfg, aug_prob=0., uniform_sample=True)
    val_dataloader = get_dataloader(val_dataset, mode="val", num_workers=1, batch_size=1)

    # 4. 开始评估
    metrics = {
        "track_loss": [],
        "img_loss": [],
        "total_loss": []
    }

    print(f"==> Starting Evaluation on {len(val_dataset)} samples...")
    
    with torch.no_grad():
        for i, (vid, track, vis, task_emb, action) in enumerate(tqdm(val_dataloader)):
            vid, track, vis, task_emb, action = vid.cuda(), track.cuda(), vis.cuda(), task_emb.cuda(), action.cuda()
            # 混合精度处理
            if cfg.get("mix_precision", False):
                vid, track, vis, task_emb, action = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), task_emb.bfloat16(), action.bfloat16()

            # --- 计算指标 ---
            loss, ret_dict = model.forward_loss(
                vid,
                track,
                task_emb,
                lbd_track=cfg.lbd_track,
                lbd_img=cfg.lbd_img,
                p_img=cfg.p_img,
                vis=vis,
                action=action
            )
            
            metrics["track_loss"].append(ret_dict["track_loss"])
            metrics["img_loss"].append(ret_dict["img_loss"])
            metrics["total_loss"].append(ret_dict["loss"])

            # --- 生成并保存可视化 (每隔几个 batch 保存一次，防止磁盘爆满) ---
            if i % 10 == 0:
                _, vis_ret = model.forward_vis(
                    vid, track, task_emb, p_img=0, action=action
                )
                
                # 保存重建图像对比 (原图 vs 重建)
                save_image(
                    vis_ret["combined_image"], 
                    os.path.join(vis_dir, f"batch_{i}_reconstruction.png")
                )
                # 保存轨迹预测视频帧 (真实轨迹 vs 预测轨迹)
                save_image(
                    vis_ret["combined_track_vid"], 
                    os.path.join(vis_dir, f"batch_{i}_track.png")
                )

    # 5. 汇总指标并保存
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print("\n==> Evaluation Results:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.6f}")

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)
    
    print(f"\nResults saved to '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
