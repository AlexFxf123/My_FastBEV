# -*- coding: utf-8 -*-
"""
毫米波雷达点云BEV特征编码器

将雷达点云编码为BEV特征图，用于与视觉分支融合。

架构:
  1. 点云体素化 (基于voxel_size和point_cloud_range)
  2. 使用简化的pillar特征编码 (类似PointPillars)
  3. 通过2D卷积网络生成BEV特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import NECKS


@NECKS.register_module()
class RadarBEVEncoder(nn.Module):
    """毫米波雷达点云BEV编码器

    将雷达点云(稀疏)转换为密集的BEV特征图。
    使用类似PointPillars的pillar特征编码方式。

    Args:
        in_channels (int): 输入点云特征维度 (默认5: x,y,z,rcs,doppler)
        feat_channels (list[int]): 编码器特征通道数
        grid_size (list[int]): BEV网格尺寸 [x, y, z]
        voxel_size (list[float]): 体素大小
        point_cloud_range (list[float]): 点云范围
        max_voxels (int): 最大体素数
        max_points_per_voxel (int): 每个体素最大点数
        use_doppler (bool): 是否使用多普勒速度特征
        use_rcs (bool): 是否使用RCS特征
    """

    def __init__(self,
                 in_channels=5,
                 feat_channels=[32, 64, 128],
                 grid_size=[400, 400, 1],
                 voxel_size=[0.25, 0.25, 0.4],
                 point_cloud_range=[-50, -50, -3, 50, 50, 3],
                 max_voxels=60000,
                 max_points_per_voxel=10,
                 use_doppler=True,
                 use_rcs=True):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_voxels = max_voxels
        self.max_points_per_voxel = max_points_per_voxel
        self.use_doppler = use_doppler
        self.use_rcs = use_rcs

        # Pillar特征网络: 每个pillar内点特征的MLP
        pillar_in_channels = 5 if use_rcs and use_doppler else 3
        if use_rcs:
            pillar_in_channels = max(pillar_in_channels, 4)
        if use_doppler:
            pillar_in_channels = max(pillar_in_channels, 5)

        # 加上pillar中心偏移 (x, y, z) 和点到中心的偏移
        pillar_encoder_in = pillar_in_channels + 6  # + x_center, y_center, z_center, x_off, y_off, z_off

        self.pillar_encoder = nn.Sequential(
            nn.Linear(pillar_encoder_in, feat_channels[0]),
            nn.BatchNorm1d(feat_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels[0], feat_channels[1]),
            nn.BatchNorm1d(feat_channels[1]),
            nn.ReLU(inplace=True),
        )

        # BEV特征提取网络 (2D卷积)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(feat_channels[1], feat_channels[1], 3, 1, 1),
            nn.BatchNorm2d(feat_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels[1], feat_channels[1], 3, 1, 1),
            nn.BatchNorm2d(feat_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels[1], feat_channels[2], 3, 2, 1),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels[2], feat_channels[2], 3, 1, 1),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True),
        )

        # 坐标偏移量
        self.register_buffer('pc_min', torch.tensor(point_cloud_range[:3]))
        self.register_buffer('pc_max', torch.tensor(point_cloud_range[3:]))
        self.register_buffer('voxel_size_t', torch.tensor(voxel_size))

        self.fp16_enabled = False

    def _voxelize(self, points):
        """将点云体素化

        Args:
            points: (N, C) 点云

        Returns:
            voxel_features: (M, max_pts, feat_dim) 体素特征
            voxel_coords: (M, 3) 体素坐标
            num_points_per_voxel: (M,) 每个体素的点数
        """
        N = points.shape[0]
        device = points.device

        # 计算每个点的体素坐标
        coords = torch.floor(
            (points[:, :3] - self.pc_min) / self.voxel_size_t
        ).long()

        # 过滤超出范围的点
        mask = (coords >= 0).all(dim=1) & (coords < torch.tensor(self.grid_size[:3], device=device)).all(dim=1)
        coords = coords[mask]
        points = points[mask]

        # 将3D坐标转换为唯一的key
        # grid_size[0]*grid_size[1] 个 pillar
        coord_key = coords[:, 0] * self.grid_size[1] * self.grid_size[2] + \
                    coords[:, 1] * self.grid_size[2] + \
                    coords[:, 2]

        # 对每个体素选择前max_points_per_voxel个点
        unique_keys, inverse_indices = torch.unique(coord_key, return_inverse=True)
        M = unique_keys.shape[0]

        voxel_features = torch.zeros(M, self.max_points_per_voxel, points.shape[1], device=device)
        num_points_per_voxel = torch.zeros(M, dtype=torch.long, device=device)

        # 为每个点分配体素内的索引
        voxel_point_counts = torch.zeros(M, dtype=torch.long, device=device)

        for i in range(min(N, points.shape[0])):
            voxel_idx = inverse_indices[i]
            count = voxel_point_counts[voxel_idx]
            if count < self.max_points_per_voxel:
                voxel_features[voxel_idx, count] = points[i]
                voxel_point_counts[voxel_idx] = count + 1

        num_points_per_voxel = voxel_point_counts

        # 体素坐标 (x, y, z)
        voxel_coords_x = unique_keys // (self.grid_size[1] * self.grid_size[2])
        remaining = unique_keys % (self.grid_size[1] * self.grid_size[2])
        voxel_coords_y = remaining // self.grid_size[2]
        voxel_coords_z = remaining % self.grid_size[2]
        voxel_coords = torch.stack([voxel_coords_z, voxel_coords_y, voxel_coords_x], dim=1)  # (z, y, x)

        return voxel_features, voxel_coords, num_points_per_voxel

    def _pillar_encoding(self, voxel_features, num_points_per_voxel):
        """Pillar特征编码

        Args:
            voxel_features: (M, max_pts, C) 体素特征
            num_points_per_voxel: (M,) 点数

        Returns:
            pillar_features: (M, C_out) 编码后的pillar特征
        """
        M, max_pts, C = voxel_features.shape
        device = voxel_features.device

        # 计算每个pillar的中心
        # voxel坐标中心 (x, y, z) 在LIDAR坐标系下
        # 这里简化处理：使用体素中心
        x_size, y_size, z_size = self.voxel_size
        x_min, y_min, z_min = self.point_cloud_range[:3]

        # 计算每个pillar的几何中心坐标
        # 这部分需要在forward中完成

        # 对每个pillar做max pooling (沿第1维)
        # 先mask掉无效点
        point_mask = torch.arange(max_pts, device=device).unsqueeze(0) < num_points_per_voxel.unsqueeze(1)
        point_mask = point_mask.unsqueeze(-1).float()  # (M, max_pts, 1)

        # 对有效点的特征做max pooling
        masked_features = voxel_features * point_mask
        pillar_feat = masked_features.max(dim=1)[0]  # (M, C)

        # 补充pillar中心偏移特征
        # (可以直接在forward中处理)

        return pillar_feat

    @auto_fp16()
    def forward(self, points):
        """雷达点云编码为BEV特征

        Args:
            points: (B, N, C) 或 list[(N, C)] 雷达点云

        Returns:
            bev_feat: (B, C_out, H, W) BEV特征图
        """
        # 处理输入
        if isinstance(points, list):
            batch_size = len(points)
        else:
            batch_size = points.shape[0]

        # 构建BEV特征图
        x_size, y_size, z_size = self.grid_size
        device = points[0].device if isinstance(points, list) else points.device

        bev_features = []
        for b in range(batch_size):
            pts = points[b] if isinstance(points, list) else points[b]

            if pts.shape[0] == 0:
                # 空点云
                bev_feat_map = torch.zeros(
                    self.feat_channels[1], y_size, x_size, device=device)
                bev_features.append(bev_feat_map)
                continue

            # 体素化
            voxel_features, voxel_coords, num_points = self._voxelize(pts)

            if voxel_features.shape[0] == 0:
                bev_feat_map = torch.zeros(
                    self.feat_channels[1], y_size, x_size, device=device)
                bev_features.append(bev_feat_map)
                continue

            # Pillar特征编码
            M = voxel_features.shape[0]

            # 加入pillar中心坐标 (体素中心在LIDAR坐标系)
            voxel_center_x = voxel_coords[:, 2].float() * self.voxel_size[0] + self.point_cloud_range[0] + self.voxel_size[0] / 2
            voxel_center_y = voxel_coords[:, 1].float() * self.voxel_size[1] + self.point_cloud_range[1] + self.voxel_size[1] / 2
            voxel_center_z = voxel_coords[:, 0].float() * self.voxel_size[2] + self.point_cloud_range[2] + self.voxel_size[2] / 2
            voxel_centers = torch.stack([voxel_center_x, voxel_center_y, voxel_center_z], dim=1)  # (M, 3)

            # 点到pillar中心的偏移
            points_center = voxel_features[:, :, :3]  # (M, max_pts, 3)
            center_expanded = voxel_centers.unsqueeze(1).expand(-1, self.max_points_per_voxel, -1)
            points_offset = points_center - center_expanded  # (M, max_pts, 3)

            # 拼接特征: [x,y,z,rcs,doppler, center_x,center_y,center_z, offset_x,offset_y,offset_z]
            # 实际使用有效的特征
            feat_dim = voxel_features.shape[-1]
            if feat_dim >= 3:
                # 基础特征 + 中心偏移
                center_repeat = voxel_centers.unsqueeze(1).expand(-1, self.max_points_per_voxel, -1)
                offset = points_center
                enhanced_features = torch.cat([
                    voxel_features,
                    center_repeat,
                    offset
                ], dim=-1)  # (M, max_pts, feat_dim + 6)
            else:
                enhanced_features = voxel_features

            # 通过pillar编码器
            # (M, max_pts, feat_dim+6) -> (M, max_pts, C1) -> (M, C1) (max pooling across points)
            pillar_feat = self.pillar_encoder(enhanced_features)  # (M, max_pts, C1)

            # 对每个pillar做max pooling
            point_mask = torch.arange(self.max_points_per_voxel, device=device).unsqueeze(0) < num_points.unsqueeze(1)
            point_mask = point_mask.unsqueeze(-1).float()  # (M, max_pts, 1)
            pillar_feat = (pillar_feat * point_mask).max(dim=1)[0]  # (M, C1)

            # 散射到BEV网格
            bev_feat_map = torch.zeros(
                self.feat_channels[1], y_size, x_size, device=device)

            x_coords = voxel_coords[:, 2].long()  # x index
            y_coords = voxel_coords[:, 1].long()  # y index

            # 使用scatter_填充
            for k in range(M):
                xi = x_coords[k].item()
                yi = y_coords[k].item()
                if 0 <= xi < x_size and 0 <= yi < y_size:
                    bev_feat_map[:, yi, xi] = pillar_feat[k]

            bev_features.append(bev_feat_map)

        bev_batch = torch.stack(bev_features, dim=0)  # (B, C, H, W)

        # 2D卷积编码
        bev_out = self.bev_encoder(bev_batch)  # (B, C_out, H/2, W/2)

        return bev_out