import torch
import numpy as np
import cv2
from einops import repeat
from atm.model.track_transformer_action import TrackTransformerAction

class ATMInference:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        # 模型配置需与训练保持一致
        model_cfg = {
            "transformer_cfg": {"dim": 384, "dim_head": None, "heads": 8, "depth": 8, "attn_dropout": 0.2, "ff_dropout": 0.2},
            "track_cfg": {"num_track_ts": 81, "num_track_ids": 256, "patch_size": 9},
            "vid_cfg": {"img_size": [240, 320], "frame_stack": 1, "patch_size": 16},
            "language_encoder_cfg": {"network_name": "MLPEncoder", "input_size": 768, "hidden_size": 128, "num_layers": 1}
        }
        
        self.model = TrackTransformerAction(**model_cfg).to(device)
        self.model.load(checkpoint_path) #
        self.model.eval()
        
        self.num_track_ids = model_cfg["track_cfg"]["num_track_ids"]
        self.num_track_ts = model_cfg["track_cfg"]["num_track_ts"]
        self.img_size = model_cfg["vid_cfg"]["img_size"] # [H, W]

    def _generate_grid_tracks(self):
        """生成覆盖全图的均匀网格查询点"""
        side = int(np.sqrt(self.num_track_ids))
        y = torch.linspace(0, 1, side)
        x = torch.linspace(0, 1, side)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        tracks = repeat(points, 'n c -> 1 t n c', t=self.num_track_ts)
        return tracks.to(self.device)

    @torch.no_grad()
    def infer(self, video, task_emb, action):
        """
        video: (B, T, C, H, W) 0-255
        """
        track_grid = self._generate_grid_tracks()
        
        video = video.to(self.device).bfloat16()
        track_grid = track_grid.bfloat16()
        task_emb = task_emb.to(self.device).bfloat16()
        action = action.to(self.device).bfloat16()

        # p_img=0 表示不进行遮挡推理
        rec_track, _ = self.model.reconstruct(
            vid=video, 
            track=track_grid, 
            task_emb=task_emb, 
            p_img=0.0, 
            action=action
        )
        return rec_track # (B, T, N, 2) 归一化坐标

    def draw_tracks_on_image(self, image, predictions, track_idx_to_show=None):
        """
        在单张图上绘制轨迹线
        image: numpy array (H, W, 3), RGB
        predictions: tensor (T, N, 2) 归一化坐标
        """
        h, w = image.shape[:2]
        vis_img = image.copy()
        pred_np = predictions.cpu().float().numpy() # (T, N, 2)
        
        # 如果不指定，随机选30个点显示，避免画面太乱
        if track_idx_to_show is None:
            track_idx_to_show = np.random.choice(self.num_track_ids, 30, replace=False)

        for n in track_idx_to_show:
            # 绘制轨迹线 (T个时间步)
            points = pred_np[:, n, :] # (T, 2)
            points[:, 0] *= w
            points[:, 1] *= h
            
            # 绘制折线
            for t in range(len(points) - 1):
                pt1 = tuple(points[t].astype(int))
                pt2 = tuple(points[t+1].astype(int))
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            
            # 绘制当前起始点
            cv2.circle(vis_img, tuple(points[0].astype(int)), 2, (0, 0, 255), -1)
            
        return vis_img

    def save_predictions_video(self, background_img, predictions, output_path="output_tracks.mp4"):
        """
        保存轨迹运动视频
        background_img: numpy array (H, W, 3), RGB
        predictions: tensor (T, N, 2)
        """
        h, w = background_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 10, (w, h))
        
        pred_np = predictions.cpu().float().numpy()
        T = pred_np.shape[0]

        for t in range(T):
            frame = background_img.copy()
            # 转换颜色空间给 CV2
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            for n in range(self.num_track_ids):
                x = int(pred_np[t, n, 0] * w)
                y = int(pred_np[t, n, 1] * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to {output_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 初始化
    infer_engine = ATMInference("model_best.ckpt")
    
    # 假设你有一帧图像 (1, 1, 3, 240, 320)
    img_np = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    input_vid = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    
    # 执行推理
    task_emb = torch.randn(1, 768)
    action = torch.randn(1, 81, 14)
    preds = infer_engine.infer(input_vid, task_emb, action) # (1, 81, 256, 2)
    
    # 1. 绘制静态轨迹图
    static_vis = infer_engine.draw_tracks_on_image(img_np, preds[0])
    cv2.imwrite("tracks_static.jpg", cv2.cvtColor(static_vis, cv2.COLOR_RGB2BGR))
    
    # 2. 保存动态视频
    infer_engine.save_predictions_video(img_np, preds[0], "atm_movement.mp4")
