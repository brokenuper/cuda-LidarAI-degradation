import numpy as np
import cv2
import torch
from PIL import Image

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RoadTestDegradationSimulator:
    """
    车端多传感器融合长尾退化合成器 (Badcase 制造机)
    支持在 Data Pipeline 中实时注入：
    1. 视觉单模态退化（掉线全黑、镜头泥水遮挡、隧道口强光过曝、雨雾模糊）
    2. 跨模态外参标定漂移（Pitch/Yaw/Roll 旋转偏移与平移偏移）
    3. 跨模态时间戳延迟（基于自车高速运动的几何位姿滞后等效）
    """
    def __init__(self,
                 # 1. 视觉退化配置
                 enable_cam_degrad=False,
                 target_cam_indices=[0],          # 默认 0 为 CAM_FRONT (正前视主相机)
                 cam_degrad_mode='blackout',      # 可选: 'blackout', 'overexposure', 'occlusion', 'blur'
                 occlusion_ratio=0.4,             # 遮挡面积比例
                 
                 # 2. 外参漂移配置
                 enable_extrinsic_drift=False,
                 drift_cam_indices=[0],           # 发生外参漂移的相机索引
                 pitch_drift_deg=1.0,             # 俯仰角偏移 (度)
                 yaw_drift_deg=0.5,               # 偏航角偏移 (度)
                 roll_drift_deg=0.0,              # 翻滚角偏移 (度)
                 trans_drift_m=[0.0, 0.0, 0.0],   # 平移偏移 [dx, dy, dz] (米)
                 is_random_jitter=False,          # False: 评测固定偏移; True: 训练随机扰动增强
                 
                 # 3. 时间戳异步配置
                 enable_temporal_desync=False,
                 desync_dt_sec=0.1,               # 模拟相机滞后雷达的时间差 (如 100ms)
                 simulated_ego_vel=[15.0, 0.0, 0.0] # 模拟自车速度 [vx, vy, vz] (m/s)
                 ):
        self.enable_cam_degrad = enable_cam_degrad
        self.target_cam_indices = target_cam_indices
        self.cam_degrad_mode = cam_degrad_mode
        self.occlusion_ratio = occlusion_ratio

        self.enable_extrinsic_drift = enable_extrinsic_drift
        self.drift_cam_indices = drift_cam_indices
        self.pitch_drift_deg = pitch_drift_deg
        self.yaw_drift_deg = yaw_drift_deg
        self.roll_drift_deg = roll_drift_deg
        self.trans_drift_m = np.array(trans_drift_m, dtype=np.float32)
        self.is_random_jitter = is_random_jitter

        self.enable_temporal_desync = enable_temporal_desync
        self.desync_dt_sec = desync_dt_sec
        self.simulated_ego_vel = np.array(simulated_ego_vel, dtype=np.float32)

    def _get_rotation_matrix(self, pitch_deg, yaw_deg, roll_deg):
        """根据欧拉角生成 3x3 旋转扰动矩阵"""
        rx = np.deg2rad(pitch_deg)
        ry = np.deg2rad(yaw_deg)
        rz = np.deg2rad(roll_deg)

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]], dtype=np.float32)
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0, 0, 1]], dtype=np.float32)
        return Rz @ Ry @ Rx

    def _degrade_single_numpy_img(self, img_np):
        """对单张 numpy [H, W, C] 图像执行退化"""
        H, W = img_np.shape[:2]
        if self.cam_degrad_mode == 'blackout':
            img_np[...] = 0
        elif self.cam_degrad_mode == 'overexposure':
            # 区分 0-255 uint8 与归一化后的 float32
            if img_np.dtype == np.uint8 or img_np.max() > 2.0:
                img_np = np.clip(img_np.astype(np.float32) * 2.5 + 100.0, 0, 255).astype(img_np.dtype)
            else:
                max_val = float(img_np.max()) if img_np.max() > 0 else 1.0
                img_np = np.clip(img_np * 2.5 + 0.5 * max_val, None, max_val)
        elif self.cam_degrad_mode == 'occlusion':
            h_occ, w_occ = int(H * self.occlusion_ratio), int(W * self.occlusion_ratio)
            y0, x0 = (H - h_occ) // 2, (W - w_occ) // 2
            img_np[y0:y0+h_occ, x0:x0+w_occ, ...] = 0
        elif self.cam_degrad_mode == 'blur':
            img_np = cv2.GaussianBlur(img_np, (21, 21), sigmaX=10.0)
        return img_np

    def _inject_camera_degradation(self, imgs):
        """
        对环视图像注入单模态退化
        完美兼容：
        1. List[PIL.Image] (MIT bevfusion 默认加载格式)
        2. List[np.ndarray] (MMDet3D 默认加载格式)
        3. torch.Tensor [N, C, H, W] (打包后的 Tensor 格式)
        """
        is_tensor = isinstance(imgs, torch.Tensor)
        for idx in self.target_cam_indices:
            if idx >= len(imgs):
                continue

            if is_tensor:
                # 处理 Tensor [C, H, W]
                _, H, W = imgs[idx].shape
                if self.cam_degrad_mode == 'blackout':
                    imgs[idx].zero_()
                elif self.cam_degrad_mode == 'overexposure':
                    max_val = imgs[idx].max()
                    imgs[idx] = torch.clamp(imgs[idx] * 2.5 + 0.5 * max_val, max=max_val)
                elif self.cam_degrad_mode == 'occlusion':
                    h_occ, w_occ = int(H * self.occlusion_ratio), int(W * self.occlusion_ratio)
                    y0, x0 = (H - h_occ) // 2, (W - w_occ) // 2
                    imgs[idx][:, y0:y0+h_occ, x0:x0+w_occ] = 0.0
                elif self.cam_degrad_mode == 'blur':
                    # Tensor 转 numpy 高斯模糊后再转回 Tensor
                    img_np = imgs[idx].permute(1, 2, 0).cpu().numpy()
                    img_blur = cv2.GaussianBlur(img_np, (21, 21), sigmaX=10.0)
                    imgs[idx] = torch.from_numpy(img_blur).permute(2, 0, 1).to(imgs.device, dtype=imgs.dtype)
            elif isinstance(imgs[idx], Image.Image):
                # 处理 PIL.Image (MIT bevfusion 在 ImageNormalize 前的格式)
                img_np = np.array(imgs[idx])
                img_np = self._degrade_single_numpy_img(img_np)
                imgs[idx] = Image.fromarray(img_np)
            else:
                # 处理 numpy.ndarray [H, W, C]
                imgs[idx] = self._degrade_single_numpy_img(imgs[idx])

        return imgs

    def _inject_extrinsic_drift(self, results):
        """
        注入跨模态外参标定漂移
        同步修改 lidar2camera, camera2lidar, lidar2image
        """
        for idx in self.drift_cam_indices:
            if self.is_random_jitter:
                p = np.random.uniform(-self.pitch_drift_deg, self.pitch_drift_deg)
                y = np.random.uniform(-self.yaw_drift_deg, self.yaw_drift_deg)
                r = np.random.uniform(-self.roll_drift_deg, self.roll_drift_deg)
                t = np.random.uniform(-np.abs(self.trans_drift_m), np.abs(self.trans_drift_m))
            else:
                p, y, r = self.pitch_drift_deg, self.yaw_drift_deg, self.roll_drift_deg
                t = self.trans_drift_m

            delta_T = np.eye(4, dtype=np.float32)
            delta_T[:3, :3] = self._get_rotation_matrix(p, y, r)
            delta_T[:3, 3] = t

            # 1. 更新 lidar2camera (4x4)
            if 'lidar2camera' in results and idx < len(results['lidar2camera']):
                results['lidar2camera'][idx] = delta_T @ results['lidar2camera'][idx]

            # 2. 同步更新 camera2lidar (4x4, BEVPool 直接依赖此矩阵)
            if 'camera2lidar' in results and idx < len(results['camera2lidar']):
                inv_delta_T = np.linalg.inv(delta_T)
                results['camera2lidar'][idx] = results['camera2lidar'][idx] @ inv_delta_T

            # 3. 同步更新 lidar2image (4x4)
            if 'lidar2image' in results and 'camera_intrinsics' in results:
                if idx < len(results['lidar2image']):
                    K = results['camera_intrinsics'][idx]
                    results['lidar2image'][idx] = K @ results['lidar2camera'][idx]

        return results

    def _inject_temporal_desync(self, results):
        """
        注入跨模态时间不同步 (Temporal Desynchronization)：
        几何运动滞后等效：模拟相机触发比雷达晚 dt 秒，导致外参产生 v * dt 的等效位移差
        """
        lag_T = np.eye(4, dtype=np.float32)
        lag_T[:3, 3] = self.simulated_ego_vel * self.desync_dt_sec
        inv_lag_T = np.linalg.inv(lag_T)

        if 'camera2lidar' in results:
            for i in range(len(results['camera2lidar'])):
                results['camera2lidar'][i] = lag_T @ results['camera2lidar'][i]

        if 'lidar2camera' in results:
            for i in range(len(results['lidar2camera'])):
                results['lidar2camera'][i] = results['lidar2camera'][i] @ inv_lag_T

        if 'lidar2image' in results and 'camera_intrinsics' in results:
            for i in range(len(results['lidar2image'])):
                K = results['camera_intrinsics'][i]
                results['lidar2image'][i] = K @ results['lidar2camera'][i]

        return results

    def __call__(self, results):
        # 1. 注入视觉单模态退化
        if self.enable_cam_degrad and 'img' in results:
            results['img'] = self._inject_camera_degradation(results['img'])

        # 2. 注入外参标定漂移
        if self.enable_extrinsic_drift:
            results = self._inject_extrinsic_drift(results)

        # 3. 注入时间戳异步
        if self.enable_temporal_desync:
            results = self._inject_temporal_desync(results)

        return results