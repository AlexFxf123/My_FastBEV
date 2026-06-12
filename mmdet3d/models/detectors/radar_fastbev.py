# -*- coding: utf-8 -*-
"""
FastBEV + 毫米波雷达点云融合模型

架构设计:
  1. 图像分支: 沿用FastBEV的Backbone+FPN+backproject
  2. 雷达分支: 雷达点云 → PointPillars风格的体素化 → 3D稀疏卷积 → BEV特征
  3. 融合模块: BEV空间上的特征融合 (concatenation + 卷积融合)
  4. 检测头: 共享原有FreeAnchor3DHead
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16

import copy

from .fastbev import FastBEV, get_points, backproject_inplace, backproject_vanilla


@DETECTORS.register_module()
class RadarFastBEV(FastBEV):
    """FastBEV + 毫米波雷达点云融合检测模型

    在FastBEV的基础上增加雷达点云分支:
    - 雷达点云经过体素化+稀疏卷积提取BEV特征
    - 与视觉backproject得到的BEV特征在BEV空间融合
    - 融合后的特征送入neck_3d和bbox_head
    """

    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head=None,
        n_voxels=None,
        voxel_size=None,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
        # 雷达分支参数
        radar_encoder=None,          # 雷达编码器配置
        radar_voxel_size=None,       # 雷达体素大小 [0.25, 0.25, 0.4]
        radar_point_cloud_range=None,# 雷达点云范围 [-50, -50, -3, 50, 50, 3]
        radar_max_voxels=None,       # 最大体素数
        radar_max_points_per_voxel=10, # 每个体素最大点数
        radar_feat_channels=None,    # 雷达特征通道数
        radar_feat_height=1,         # 雷达体素高度
        # 融合参数
        fusion_channels=None,        # 融合后的通道数，默认等于neck_3d的in_channels
        fusion_conv_layers=1,        # 融合卷积层数
    ):
        # 调用父类初始化（FastBEV的__init__）
        super().__init__(
            backbone=backbone,
            neck=neck,
            neck_fuse=neck_fuse,
            neck_3d=neck_3d,
            bbox_head=bbox_head,
            seg_head=seg_head,
            n_voxels=n_voxels,
            voxel_size=voxel_size,
            bbox_head_2d=bbox_head_2d,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            train_cfg_2d=train_cfg_2d,
            test_cfg_2d=test_cfg_2d,
            pretrained=pretrained,
            init_cfg=init_cfg,
            extrinsic_noise=extrinsic_noise,
            seq_detach=seq_detach,
            multi_scale_id=multi_scale_id,
            multi_scale_3d_scaler=multi_scale_3d_scaler,
            with_cp=with_cp,
            backproject=backproject,
            style=style,
        )

        # ---- 雷达分支 ----
        if radar_encoder is not None:
            self.radar_encoder = build_neck(radar_encoder) if radar_encoder.get('type') else None
        else:
            self.radar_encoder = None

        # 雷达点云处理参数
        self.radar_voxel_size = radar_voxel_size if radar_voxel_size is not None else [0.25, 0.25, 0.4]
        self.radar_point_cloud_range = radar_point_cloud_range if radar_point_cloud_range is not None else [-50, -50, -3, 50, 50, 3]
        self.radar_max_voxels = radar_max_voxels if radar_max_voxels is not None else 60000
        self.radar_max_points_per_voxel = radar_max_points_per_voxel
        self.radar_feat_channels = radar_feat_channels if radar_feat_channels is not None else 64
        self.radar_feat_height = radar_feat_height  # 毫米波雷达高度维度通常只有1

        # 计算雷达BEV网格尺寸
        if radar_point_cloud_range is not None and radar_voxel_size is not None:
            self.radar_grid_size = [
                int((radar_point_cloud_range[3] - radar_point_cloud_range[0]) / radar_voxel_size[0]),
                int((radar_point_cloud_range[4] - radar_point_cloud_range[1]) / radar_voxel_size[1]),
                int((radar_point_cloud_range[5] - radar_point_cloud_range[2]) / radar_voxel_size[2]),
            ]
        else:
            self.radar_grid_size = [400, 400, 16]

        # ---- 融合模块 ----
        # 获取视觉BEV通道数
        if style in ['v1', 'v2']:
            n_voxels_list = n_voxels if isinstance(n_voxels, list) else [n_voxels]
            seq_c = 4  # 时序帧数
            neck_fuse_out = 64  # neck_fuse输出通道
            self.vis_bev_channels = len(n_voxels_list) * seq_c * neck_fuse_out * n_voxels_list[0][2]
        else:
            # v3/v4 多尺度情况
            self.vis_bev_channels = 256  # 简化处理，实际可能更复杂

        # 雷达BEV通道
        radar_grid_x = self.radar_grid_size[0]
        radar_grid_y = self.radar_grid_size[1]
        radar_grid_z = self.radar_grid_size[2]

        # 融合输入通道 = 视觉通道 + 雷达通道
        # 视觉neck_3d输入: (B, lvl*seq*c, vx, vy, vz)
        # 雷达特征需要resize到与视觉相同的BEV网格尺寸
        fusion_in_channels = 128  # 融合层输入通道 (简化)
        if fusion_channels is not None:
            fusion_in_channels = fusion_channels

        self.fusion_in_channels = fusion_in_channels
        print(f"[RadarFastBEV] Fusion input channels: {fusion_in_channels}")

    def extract_radar_feat(self, radar_points, img_metas):
        """提取雷达点云BEV特征

        Args:
            radar_points: (B, N_pts, C) 或 list[(N_pts, C)] 雷达点云
            img_metas: 图像元数据

        Returns:
            radar_bev: (B, C_radar, H_bev, W_bev) 雷达BEV特征
        """
        batch_size = len(radar_points) if isinstance(radar_points, list) else radar_points.shape[0]

        # 如果没有雷达编码器，返回空特征
        if self.radar_encoder is None:
            # 获取视觉BEV的网格大小
            n_voxels = self.n_voxels[0]
            device = img_metas[0].get('device', 'cuda')
            return torch.zeros(batch_size, 64, n_voxels[1], n_voxels[0], device=device)

        # 体素化 + 特征编码
        radar_bev = []
        for b in range(batch_size):
            if isinstance(radar_points, list):
                pts = radar_points[b]
            else:
                pts = radar_points[b]

            # 简化的体素化: 投影到BEV网格
            voxel_feat = self._points_to_voxel_feat(pts)  # (1, 2, H, W)
            radar_bev.append(voxel_feat)

        radar_bev = torch.cat(radar_bev, dim=0)  # (B, 2, H, W)

        # 如果需要更多通道，用1x1卷积升维
        if radar_bev.shape[1] < 64:
            if not hasattr(self, 'radar_to_channels'):
                self.radar_to_channels = nn.Conv2d(radar_bev.shape[1], 64, 1).to(radar_bev.device)
            radar_bev = self.radar_to_channels(radar_bev)

        # resize到视觉BEV网格尺寸
        n_voxels = self.n_voxels[0]
        if radar_bev.shape[-2:] != (n_voxels[1], n_voxels[0]):
            radar_bev = F.interpolate(
                radar_bev,
                size=(n_voxels[1], n_voxels[0]),
                mode='bilinear',
                align_corners=False)

        return radar_bev

    def _points_to_voxel_feat(self, points):
        """将雷达点云转换为体素特征

        Args:
            points: (N, C) 雷达点云

        Returns:
            voxel_feat: (C_feat, H, W, D) 体素特征
        """
        # 简化实现: 直接投影到BEV网格
        # 在实际应用中，应使用稀疏卷积的体素化方式
        device = points.device
        N = points.shape[0]

        grid_x = self.radar_grid_size[0]
        grid_y = self.radar_grid_size[1]
        grid_z = self.radar_grid_size[2]

        # 点云范围
        pc_min_x, pc_min_y, pc_min_z = self.radar_point_cloud_range[0:3]
        pc_max_x, pc_max_y, pc_max_z = self.radar_point_cloud_range[3:6]
        voxel_size_x, voxel_size_y, voxel_size_z = self.radar_voxel_size

        # 计算每个点所在的体素坐标
        coords = torch.zeros((N, 3), device=device, dtype=torch.long)
        coords[:, 0] = torch.clamp(((points[:, 0] - pc_min_x) / voxel_size_x).long(), 0, grid_x - 1)
        coords[:, 1] = torch.clamp(((points[:, 1] - pc_min_y) / voxel_size_y).long(), 0, grid_y - 1)
        coords[:, 2] = torch.clamp(((points[:, 2] - pc_min_z) / voxel_size_z).long(), 0, grid_z - 1)

        # 雷达点特征: 使用反射强度 (radar的rcs强度在第3个维度)
        rcs = points[:, 3] if points.shape[1] > 3 else torch.ones(N, device=device)
        doppler = points[:, 4] if points.shape[1] > 4 else torch.zeros(N, device=device)

        # 简化的BEV投影: 沿高度维做max pooling
        # 使用scatter方法构建体素
        bev_feat = torch.zeros(2, grid_y, grid_x, device=device)  # 2个特征: rcs, doppler

        # 对每个点scatter
        for k in range(N):
            x = coords[k, 0]
            y = coords[k, 1]
            if 0 <= x < grid_x and 0 <= y < grid_y:
                if rcs[k] > bev_feat[0, y, x]:
                    bev_feat[0, y, x] = rcs[k]
                if abs(doppler[k]) > bev_feat[1, y, x]:
                    bev_feat[1, y, x] = abs(doppler[k])

        return bev_feat.unsqueeze(0)  # (1, 2, H, W)

    @auto_fp16(apply_to=('img', 'radar_points'))
    def extract_feat(self, img, img_metas, mode, radar_points=None):
        """提取融合特征

        Args:
            img: (B, N*T, C, H, W) 图像
            img_metas: 图像元数据
            mode: 'train' 或 'test'
            radar_points: (B, N_pts, C) 或 list 雷达点云

        Returns:
            feature_bev: (B, C_out, H_feat, W_feat) 融合BEV特征
            valids: None
            features_2d: None
        """
        # 1. 视觉分支: 沿用FastBEV的extract_feat到3D体素构建
        batch_size = img.shape[0]
        img_reshaped = img.reshape([-1] + list(img.shape)[2:])

        # backbone
        x = self.backbone(img_reshaped)
        if isinstance(x, dict):
            x = list(x.values())

        # neck (FPN)
        def _inner_forward(x):
            out = self.neck(x)
            return out
        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        features_2d = mlvl_feats if self.bbox_head_2d else None

        # 多尺度融合
        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i],
                            size=mlvl_feats[msid].size()[2:],
                            mode="bilinear",
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_

        # 构建3D体素 (backproject)
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][seq_id*6:(seq_id+1)*6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*6:(seq_id+1)*6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    points = get_points(
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)

                    volume = backproject_inplace(
                        feat_i[:, :, :height, :width], points, projection)
                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))
            mlvl_volumes.append(torch.cat(volume_list, dim=1))

        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]

        # 2. 雷达分支
        radar_bev = None
        if radar_points is not None:
            radar_bev = self.extract_radar_feat(radar_points, img_metas)

        # 3. 融合
        if radar_bev is not None:
            # 将雷达BEV特征也转换为5D体素格式 (增加Z维)
            # radar_bev: (B, C_r, H_bev, W_bev)
            # 需要与mlvl_volumes的5D格式匹配: (B, C_v, X, Y, Z)
            # 或者将mlvl_volumes平整为2D后融合

            # 方案: 将mlvl_volumes投影到2D
            # [bs, C, vx, vy, vz] -> [bs, C*vz, vx, vy]
            bs, C_v, vx, vy, vz = mlvl_volumes.shape
            vis_bev_2d = mlvl_volumes.permute(0, 2, 3, 4, 1).reshape(bs, vx, vy, vz * C_v).permute(0, 3, 1, 2)

            # 扩展雷达BEV到相同尺寸(如果需要)
            if radar_bev.shape[2:] != (vy, vx):
                radar_bev = F.interpolate(
                    radar_bev,
                    size=(vy, vx),
                    mode='bilinear',
                    align_corners=False)

            # 通道对齐
            vis_channels = vis_bev_2d.shape[1]
            radar_channels = radar_bev.shape[1]

            # 使用1x1卷积对齐通道后cat
            if not hasattr(self, 'vis_proj'):
                self.vis_proj = nn.Conv2d(vis_channels, 128, 1).to(mlvl_volumes.device)
                self.radar_proj = nn.Conv2d(radar_channels, 128, 1).to(mlvl_volumes.device)

            vis_proj = self.vis_proj(vis_bev_2d)
            radar_proj = self.radar_proj(radar_bev)

            fused = torch.cat([vis_proj, radar_proj], dim=1)  # (B, 256, H, W)

            # 融合卷积
            if not hasattr(self, 'fusion_conv'):
                self.fusion_conv = nn.Sequential(
                    nn.Conv2d(256, 256, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, vis_channels, 3, 1, 1),
                    nn.BatchNorm2d(vis_channels),
                    nn.ReLU(inplace=True),
                ).to(mlvl_volumes.device)

            fused_feat = self.fusion_conv(fused)  # (B, vis_channels, H, W)

            # 重新变回5D
            x = fused_feat.unsqueeze(-1)  # (B, vis_channels, H, W, 1)
        else:
            # 没有雷达，使用纯视觉
            x = mlvl_volumes

        # 4. neck_3d处理
        def _inner_forward_3d(x):
            out = self.neck_3d(x)
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward_3d, x)
        else:
            x = _inner_forward_3d(x)

        return x, None, features_2d

    @auto_fp16(apply_to=('img', 'radar_points'))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Forward function"""
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        # 提取雷达点云 (如果存在)
        radar_points = kwargs.get('radar_points', None)
        if 'radar_points' in kwargs:
            del kwargs['radar_points']

        if radar_points is None and 'points' in kwargs:
            radar_points = kwargs['points']

        feature_bev, valids, features_2d = self.extract_feat(
            img, img_metas, "train", radar_points=radar_points)

        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)

        if self.seg_head is not None:
            assert len(gt_bev_seg) == 1
            x_bev = self.seg_head(feature_bev)
            gt_bev = gt_bev_seg[0][None, ...].long()
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        radar_points = kwargs.get('radar_points', None)
        # 测试代码，正常工作可以关闭
        print(f"[DEBUG] radar_points type={type(radar_points)}, len={len(radar_points) if isinstance(radar_points, (list, tuple)) else 'N/A'}")
        feature_bev, _, features_2d = self.extract_feat(
            img, img_metas, "test", radar_points=radar_points)

        bbox_results = []
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]
        else:
            bbox_results = [dict()]

        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            bbox_results[0]['bev_seg'] = x_bev

        return bbox_results