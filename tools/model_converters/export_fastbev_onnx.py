# -*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.
"""
导出FastBEV模型为ONNX格式（支持分阶段导出）

FastBEV模型结构复杂，包含2D backbone + 3D体素构建 + 3D neck + detection head。
由于backproject操作包含循环和索引赋值，无法直接端到端导出ONNX。
本脚本支持分阶段导出：

阶段1 - 2D backbone + neck: 图像 → 2D特征图
阶段2 - 3D neck + bbox_head: BEV体素 → 3D检测框

用法:TOOL_NAME: single_find_and_replace
BEGIN_ARG: filepath
"tools/model_converters/export_fastbev_onnx.py"
END_ARG
BEGIN_ARG: old_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            cls_scores: (B, num_anchors*num_classes, fy, fx)\n            bbox_preds: (B, num_anchors*code_size, fy, fx)\n            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        head_out = self.bbox_head(neck_out)\n        return head_out"
END_ARG
BEGIN_ARG: new_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        为了兼容ONNX导出，将 (cls_scores, bbox_preds, dir_cls_preds) 的tuple\n        扁平化为单个list: [cls_l0, cls_l1, ..., bbox_l0, bbox_l1, ..., dir_l0, dir_l1, ...]\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            list[torch.Tensor]: 扁平化后的预测结果\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        cls_scores, bbox_preds, dir_cls_preds = self.bbox_head(neck_out)\n        # 扁平化为list以便ONNX导出\n        outputs = list(cls_scores) + list(bbox_preds) + list(dir_cls_preds)\n        return outputs"
END_ARG
BEGIN_ARG: replace_all
false
END_ARGTOOL_NAME: single_find_and_replace
BEGIN_ARG: filepath
"tools/model_converters/export_fastbev_onnx.py"
END_ARG
BEGIN_ARG: old_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            cls_scores: (B, num_anchors*num_classes, fy, fx)\n            bbox_preds: (B, num_anchors*code_size, fy, fx)\n            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        head_out = self.bbox_head(neck_out)\n        return head_out"
END_ARG
BEGIN_ARG: new_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        为了兼容ONNX导出，将 (cls_scores, bbox_preds, dir_cls_preds) 的tuple\n        扁平化为单个list: [cls_l0, cls_l1, ..., bbox_l0, bbox_l1, ..., dir_l0, dir_l1, ...]\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            list[torch.Tensor]: 扁平化后的预测结果\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        cls_scores, bbox_preds, dir_cls_preds = self.bbox_head(neck_out)\n        # 扁平化为list以便ONNX导出\n        outputs = list(cls_scores) + list(bbox_preds) + list(dir_cls_preds)\n        return outputs"
END_ARG
BEGIN_ARG: replace_all
false
END_ARGTOOL_NAME: single_find_and_replace
BEGIN_ARG: filepath
"tools/model_converters/export_fastbev_onnx.py"
END_ARG
BEGIN_ARG: old_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            cls_scores: (B, num_anchors*num_classes, fy, fx)\n            bbox_preds: (B, num_anchors*code_size, fy, fx)\n            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        head_out = self.bbox_head(neck_out)\n        return head_out"
END_ARG
BEGIN_ARG: new_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        为了兼容ONNX导出，将 (cls_scores, bbox_preds, dir_cls_preds) 的tuple\n        扁平化为单个list: [cls_l0, cls_l1, ..., bbox_l0, bbox_l1, ..., dir_l0, dir_l1, ...]\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            list[torch.Tensor]: 扁平化后的预测结果\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        cls_scores, bbox_preds, dir_cls_preds = self.bbox_head(neck_out)\n        # 扁平化为list以便ONNX导出\n        outputs = list(cls_scores) + list(bbox_preds) + list(dir_cls_preds)\n        return outputs"
END_ARG
BEGIN_ARG: replace_all
false
END_ARGTOOL_NAME: single_find_and_replace
BEGIN_ARG: filepath
"tools/model_converters/export_fastbev_onnx.py"
END_ARG
BEGIN_ARG: old_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            cls_scores: (B, num_anchors*num_classes, fy, fx)\n            bbox_preds: (B, num_anchors*code_size, fy, fx)\n            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        head_out = self.bbox_head(neck_out)\n        return head_out"
END_ARG
BEGIN_ARG: new_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        为了兼容ONNX导出，将 (cls_scores, bbox_preds, dir_cls_preds) 的tuple\n        扁平化为单个list: [cls_l0, cls_l1, ..., bbox_l0, bbox_l1, ..., dir_l0, dir_l1, ...]\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            list[torch.Tensor]: 扁平化后的预测结果\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        cls_scores, bbox_preds, dir_cls_preds = self.bbox_head(neck_out)\n        # 扁平化为list以便ONNX导出\n        outputs = list(cls_scores) + list(bbox_preds) + list(dir_cls_preds)\n        return outputs"
END_ARG
BEGIN_ARG: replace_all
false
END_ARGTOOL_NAME: single_find_and_replace
BEGIN_ARG: filepath
"tools/model_converters/export_fastbev_onnx.py"
END_ARG
BEGIN_ARG: old_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            cls_scores: (B, num_anchors*num_classes, fy, fx)\n            bbox_preds: (B, num_anchors*code_size, fy, fx)\n            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        head_out = self.bbox_head(neck_out)\n        return head_out"
END_ARG
BEGIN_ARG: new_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        为了兼容ONNX导出，将 (cls_scores, bbox_preds, dir_cls_preds) 的tuple\n        扁平化为单个list: [cls_l0, cls_l1, ..., bbox_l0, bbox_l1, ..., dir_l0, dir_l1, ...]\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            list[torch.Tensor]: 扁平化后的预测结果\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        cls_scores, bbox_preds, dir_cls_preds = self.bbox_head(neck_out)\n        # 扁平化为list以便ONNX导出\n        outputs = list(cls_scores) + list(bbox_preds) + list(dir_cls_preds)\n        return outputs"
END_ARG
BEGIN_ARG: replace_all
false
END_ARGTOOL_NAME: single_find_and_replace
BEGIN_ARG: filepath
"tools/model_converters/export_fastbev_onnx.py"
END_ARG
BEGIN_ARG: old_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            cls_scores: (B, num_anchors*num_classes, fy, fx)\n            bbox_preds: (B, num_anchors*code_size, fy, fx)\n            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        head_out = self.bbox_head(neck_out)\n        return head_out"
END_ARG
BEGIN_ARG: new_string
"    def forward(self, x):\n        \"\"\"输入BEV体素，输出检测头预测\n\n        为了兼容ONNX导出，将 (cls_scores, bbox_preds, dir_cls_preds) 的tuple\n        扁平化为单个list: [cls_l0, cls_l1, ..., bbox_l0, bbox_l1, ..., dir_l0, dir_l1, ...]\n\n        Args:\n            x: (B, C_in, X, Y, Z) — 体素特征\n        Returns:\n            list[torch.Tensor]: 扁平化后的预测结果\n        \"\"\"\n        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)\n        cls_scores, bbox_preds, dir_cls_preds = self.bbox_head(neck_out)\n        # 扁平化为list以便ONNX导出\n        outputs = list(cls_scores) + list(bbox_preds) + list(dir_cls_preds)\n        return outputs"
END_ARG
BEGIN_ARG: replace_all
false
END_ARG
    # 导出2D backbone部分
    python tools/model_converters/export_fastbev_onnx.py \
        configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
        work_dir/epoch_5.pth \
        --stage 2d \
        --out fastbev_2d.onnx

    # 导出3D检测部分 (style v1)
    python tools/model_converters/export_fastbev_onnx.py \
        configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
        work_dir/epoch_5.pth \
        --stage 3d \
        --out fastbev_3d.onnx

    # 依次导出两个阶段，用于部署
    # 部署时：图像→2D.onnx→体素构建（用cuda/numpy）→3D.onnx→检测结果
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

# 设置mmdet3d根目录
mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and os.path.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet3d.models import build_model
from mmdet.models import build_backbone, build_neck, build_head
from mmseg.ops import resize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export FastBEV model to ONNX')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--stage', choices=['2d', '3d'], required=True,
        help='Export stage: 2d (backbone+neck) or 3d (neck_3d+head)')
    parser.add_argument('--out', default='fastbev.onnx', help='Output ONNX file')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--verbose', action='store_true', help='Print verbose info')
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size (number of multi-view sets)')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config settings')
    return parser.parse_args()


def create_fastbev_model(cfg, checkpoint_path, device='cpu'):
    """构建FastBEV模型并加载权重"""
    print("Building FastBEV model...")

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    print("Checkpoint loaded!")

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
        print(f"Model classes ({len(model.CLASSES)}): {model.CLASSES}")

    model = model.to(device)
    model.eval()
    return model


class FastBEV2DONNX(nn.Module):
    """FastBEV 2D部分: 图像 → 2D特征图

    输入: (B, 6, 3, H, W) — 多视图图像 (nuScenes 6相机)
    输出: (B*6, 64, H_feat, W_feat) — 2D特征图 (单尺度输出，已融合)
    """

    def __init__(self, backbone, neck, neck_fuse, multi_scale_id=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.multi_scale_id = multi_scale_id

        # neck_fuse可能是:
        # 1. nn.ModuleDict (多个 neck_fuse_i 模块)
        # 2. nn.Conv2d (单个模块)
        self.neck_fuse = neck_fuse

        # 如果有命名如 neck_fuse_0, neck_fuse_1 的子模块，需要单独注册
        # 这部分由外部调用者负责 setattr

    def forward(self, img):
        """输入多视图图像，输出2D特征图

        模仿 FastBEV.extract_feat 中2D部分的前向逻辑:
          1. backbone → 多尺度特征
          2. neck (FPN) → 多尺度输出
          3. 如果 multi_scale_id 有值: 多尺度融合
          4. 返回第一个尺度的特征

        Args:
            img: (B, N, 3, H, W) 多视图图像, N通常为6
        Returns:
            feat: (B*N, C_out, H_feat, W_feat) 特征图
        """
        # 展平batch和view维度: (B, N, C, H, W) -> (B*N, C, H, W)
        if img.dim() == 5:
            B, N, C, H, W = img.shape
            img = img.reshape(B * N, C, H, W)

        # backbone
        x = self.backbone(img)
        if isinstance(x, dict):
            x = list(x.values())

        # FPN neck
        mlvl_feats = list(self.neck(x))

        # 多尺度融合 (当 multi_scale_id 不为None时)
        if self.multi_scale_id is not None:
            ms_feats = []
            for msid in self.multi_scale_id:
                neck_fuse_name = f'neck_fuse_{msid}'
                if hasattr(self, neck_fuse_name):
                    neck_fuse_module = getattr(self, neck_fuse_name)
                    # 融合: 将当前以及更粗糙层级上采样融合
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized = resize(
                            mlvl_feats[i],
                            size=mlvl_feats[msid].size()[2:],
                            mode="bilinear",
                            align_corners=False)
                        fuse_feats.append(resized)
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = neck_fuse_module(fuse_feats)
                    ms_feats.append(fuse_feats)
                else:
                    ms_feats.append(mlvl_feats[msid])
            mlvl_feats = ms_feats
        elif isinstance(self.neck_fuse, nn.Conv2d):
            # 单尺度: 直接使用 neck_fuse 处理最高分辨率特征
            c1, c2, c3, c4 = mlvl_feats[0], mlvl_feats[1], mlvl_feats[2], mlvl_feats[3]
            c2 = resize(c2, size=c1.size()[2:], mode="bilinear", align_corners=False)
            c3 = resize(c3, size=c1.size()[2:], mode="bilinear", align_corners=False)
            c4 = resize(c4, size=c1.size()[2:], mode="bilinear", align_corners=False)
            x = torch.cat([c1, c2, c3, c4], dim=1)
            out = self.neck_fuse(x)
            return out

        # 多尺度输出的统一
        if len(mlvl_feats) > 1:
            target_h, target_w = mlvl_feats[0].shape[2:]
            resized_feats = [mlvl_feats[0]]
            for feat in mlvl_feats[1:]:
                resized = resize(
                    feat, size=(target_h, target_w),
                    mode="bilinear", align_corners=False)
                resized_feats.append(resized)
            out = torch.cat(resized_feats, dim=1)
        else:
            out = mlvl_feats[0]

        return out


class FastBEV3DONNX(nn.Module):
    """FastBEV 3D部分: BEV体素特征 → 检测结果

    输入: (B, C_in, X, Y, Z) — BEV体素特征 (v1风格，已collapse Z维度)
    输出: tuple( cls_scores, bbox_preds, dir_cls_preds )
    """

    def __init__(self, neck_3d, bbox_head):
        super().__init__()
        self.neck_3d = neck_3d
        self.bbox_head = bbox_head

    def forward(self, x):
        """输入BEV体素，输出检测头预测

        Args:
            x: (B, C_in, X, Y, Z) — 体素特征
        Returns:
            cls_scores: (B, num_anchors*num_classes, fy, fx)
            bbox_preds: (B, num_anchors*code_size, fy, fx)
            dir_cls_preds: (B, num_anchors*2, fy, fx) 或 None
        """
        neck_out = self.neck_3d(x)  # list of (B, C_out, fy, fx)
        head_out = self.bbox_head(neck_out)
        return head_out


def export_stage_2d(cfg, checkpoint_path, args):
    """导出2D backbone部分"""
    print("=" * 60)
    print("导出 FastBEV 2D Backbone 部分")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 构建完整模型并加载权重
    full_model = create_fastbev_model(cfg, checkpoint_path, device)

    # 提取子模块
    backbone = full_model.backbone
    neck = full_model.neck
    multi_scale_id = full_model.multi_scale_id

    # FastBEV中neck_fuse可能有两种形式：
    # 1. 单个 nn.Conv2d (self.neck_fuse = Conv2d(...))
    # 2. 多个 nn.Conv2d (self.add_module('neck_fuse_0', Conv2d(...)), ...)
    # 从配置的 neck_fuse.in_channels 是否为list来判断
    neck_fuse_config = cfg.model.neck_fuse
    if isinstance(neck_fuse_config['in_channels'], list):
        # 多个neck_fuse子模块 (self.neck_fuse_0, self.neck_fuse_1, ...)
        neck_fuse_modules = nn.ModuleDict()
        for i in range(len(neck_fuse_config['in_channels'])):
            module_name = f'neck_fuse_{i}'
            if hasattr(full_model, module_name):
                neck_fuse_modules[f'neck_fuse_{i}'] = getattr(full_model, module_name)
        neck_fuse = neck_fuse_modules  # 传入ModuleDict
    else:
        # 单个neck_fuse (self.neck_fuse = Conv2d(...))
        neck_fuse = full_model.neck_fuse

    # 构建导出模型
    export_model = FastBEV2DONNX(backbone, neck, neck_fuse, multi_scale_id)
    # 如果有多个命名方式为 self.neck_fuse_{i} 的子模块，也复制到export_model
    if isinstance(neck_fuse_config['in_channels'], list):
        for i in range(len(neck_fuse_config['in_channels'])):
            module_name = f'neck_fuse_{i}'
            if hasattr(full_model, module_name):
                setattr(export_model, module_name, getattr(full_model, module_name))

    export_model = export_model.to(device)
    export_model.eval()

    # 输入尺寸
    input_size = cfg.data_config.get('input_size', (256, 704))
    H, W = input_size
    B = args.batch_size
    N = 6  # nuScenes 6个相机
    dummy_input = torch.randn(B, N, 3, H, W, device=device)

    # 测试前向
    print(f"\n输入形状: {dummy_input.shape}")
    with torch.no_grad():
        output = export_model(dummy_input)
    print(f"输出形状: {output.shape}")

    # 导出ONNX
    input_names = ['input_images']
    output_names = ['feat_2d']
    dynamic_axes = {
        'input_images': {0: 'batch_size'},
        'feat_2d': {0: 'total_views'},
    }

    print(f"\n导出 ONNX: {args.out}")
    torch.onnx.export(
        export_model,
        dummy_input,
        args.out,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=args.verbose,
        export_params=True,
        keep_initializers_as_inputs=False,
    )
    print("2D Backbone ONNX 导出成功!")

    return args.out


def export_stage_3d(cfg, checkpoint_path, args):
    """导出3D检测部分"""
    print("=" * 60)
    print("导出 FastBEV 3D 检测部分")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 构建完整模型并加载权重
    full_model = create_fastbev_model(cfg, checkpoint_path, device)

    # 提取子模块
    neck_3d = full_model.neck_3d
    bbox_head = full_model.bbox_head

    # 构建导出模型
    export_model = FastBEV3DONNX(neck_3d, bbox_head)
    export_model = export_model.to(device)
    export_model.eval()

    # 获取体素和特征维度
    n_voxels = cfg.model.n_voxels[0]  # [200, 200, 4]
    backbone_out_channels = cfg.model.neck.out_channels  # 64
    seq_times = 4  # 时序帧数

    # 对于style='v1'且multi_scale_id=[0], FastBEV的extract_feat中:
    # 1. backbone -> neck 输出4层FPN特征
    # 2. 取第0层(multi_scale_id=[0]) -> 用neck_fuse_0融合cat
    # 3. neck_fuse输出channels = neck_fuse.out_channels[0]
    neck_fuse_cfg = cfg.model.neck_fuse
    if isinstance(neck_fuse_cfg['in_channels'], list):
        neck_fuse_out_ch = neck_fuse_cfg['out_channels'][0]  # 64
    else:
        neck_fuse_out_ch = neck_fuse_cfg['out_channels']

    # 经过backproject后volume: (C, X, Y, Z), 其中C=neck_fuse_out_ch
    # 时序拼接后: (T*C, X, Y, Z), 其中T=seq_times
    # M2BevNeck会reshape: (N, T*C, X, Y, Z) -> (N, X, Y, Z, T*C) -> (N, X, Y, Z*T*C) -> (N, Z*T*C, X, Y)
    # 所以neck_3d期望的输入形状为 (N, T*C, X, Y, Z)
    neck_in_channels = neck_fuse_out_ch * seq_times  # 64 * 4 = 256

    B = args.batch_size
    X, Y, Z = n_voxels  # 200, 200, 4

    # 创建虚拟输入: (B, C, X, Y, Z)
    dummy_input = torch.randn(B, neck_in_channels, X, Y, Z, device=device)

    # 测试前向
    print(f"\n输入形状: {dummy_input.shape}")
    with torch.no_grad():
        output = export_model(dummy_input)
    # output is (cls_scores, bbox_preds, dir_cls_preds)
    # 每个是list of tensors (多层级)
    cls_scores, bbox_preds, dir_cls_preds = output
    print(f"输出: cls_scores ({len(cls_scores)} levels), bbox_preds ({len(bbox_preds)} levels), dir_cls_preds ({len(dir_cls_preds)} levels)")
    for i in range(len(cls_scores)):
        print(f"  Level {i}: cls {cls_scores[i].shape}, bbox {bbox_preds[i].shape}, dir {dir_cls_preds[i].shape}")

    # 导出ONNX
    input_names = ['bev_volume']
    output_names = [f'pred_{i}' for i in range(len(output))]

    dynamic_axes = {'bev_volume': {0: 'batch_size'}}
    for i in range(len(output)):
        dynamic_axes[f'pred_{i}'] = {0: 'batch_size'}

    print(f"\n导出 ONNX: {args.out}")
    torch.onnx.export(
        export_model,
        dummy_input,
        args.out,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=args.verbose,
        export_params=True,
        keep_initializers_as_inputs=False,
    )
    print("3D Head ONNX 导出成功!")

    return args.out


def validate_onnx(onnx_path):
    """验证ONNX模型"""
    print("\n验证ONNX模型...")
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✓ ONNX模型验证通过!")
        print(f"  IR version: {model.ir_version}")
        print(f"  Opset: {model.opset_import[0].version}")
        print(f"  Nodes: {len(model.graph.node)}")
        print(f"  Params: {len(model.graph.initializer)}")

        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  Input: {inp.name}, shape={shape}")

        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  Output: {out.name}, shape={shape}")
        return True
    except ImportError:
        print("  ! 跳过验证 (需要安装 onnx: pip install onnx)")
        return False
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        return False


def simplify_onnx(onnx_path):
    """简化ONNX模型"""
    print("\n简化ONNX模型...")
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, onnx_path)
            print(f"  ✓ 简化成功!")
        else:
            print(f"  ! 简化检查未通过")
    except ImportError:
        print("  ! 跳过简化 (需要安装 onnx-simplifier: pip install onnx-simplifier)")
    except Exception as e:
        print(f"  ! 简化失败: {e}")


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 确保模型设置为eval模式
    if 'train_cfg' in cfg.model:
        cfg.model.train_cfg = None

    # 根据stage导出
    if args.stage == '2d':
        onnx_path = export_stage_2d(cfg, args.checkpoint, args)
    elif args.stage == '3d':
        onnx_path = export_stage_3d(cfg, args.checkpoint, args)

    # 验证
    validate_onnx(onnx_path)

    # 简化
    if args.simplify:
        simplify_onnx(onnx_path)

    print(f"\n✓ ONNX模型已保存到: {onnx_path}")

    # 打印部署指引
    print("\n" + "=" * 60)
    print("部署指引")
    print("=" * 60)
    input_size = cfg.data_config.get('test_input_size', cfg.data_config.get('input_size', '(256, 704)'))
    if args.stage == '2d':
        print(f"""
FastBEV 2D 部分导出完成!

部署流程:
  1. 图像预处理 (归一化、resize到 {input_size})
  2. 运行 2D ONNX 模型 → 获取2D特征图
  3. 使用 backproject 将2D特征投影到3D体素空间 (需要相机内外参)
  4. 运行 3D ONNX 模型 → 获取检测结果
  5. 后处理 (解码bbox、NMS等)

提示: 先导出2D部分，再导出3D部分。两个部分组合使用。
主配置文件: {args.config}
""")
    elif args.stage == '3d':
        print(f"""
FastBEV 3D 部分导出完成!

3D部分的输入是BEV体素特征，形状为:
  (batch, C, X, Y, Z) = (batch, {cfg.model.neck.out_channels * 4 * cfg.model.n_voxels[0][2]}, {cfg.model.n_voxels[0][0]}, {cfg.model.n_voxels[0][1]}, {cfg.model.n_voxels[0][2]})
  (其中 C = backbone_out_channels * FPN_levels * seq_times * Z)

输出为检测头预测值 (list)，包含:
  - cls_scores: 分类分数
  - bbox_preds: 回归预测
  - dir_cls_preds: 方向分类 (可选)

部署后处理需要:
  1. 解码bbox (使用 DeltaXYZWLHRBBoxCoder)
  2. sigmoid 分类分数
  3. 3D旋转NMS

建议: 使用ONNX Runtime推理
```python
import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('{args.out}')
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {{input_name: bev_volume_np}})
cls_scores, bbox_preds, dir_cls_preds = outputs  # 根据实际输出数量调整
```
""")


if __name__ == '__main__':
    main()