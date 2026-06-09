# Fast-BEV

[Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline](https://arxiv.org/abs/2301.12511)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/fast-bev++.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark_setting.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark.png)

## Better Inference Implementation
Thanks to the repository [CUDA-FastBEV](https://github.com/Mandylove1993/CUDA-FastBEV) inference using CUDA & TensorRT. And provide PTQ and QAT int8 quantization code.
You can refer to it to get faster speed.

## Usage

[usage](https://github.com/Sense-GVT/Fast-BEV/blob/dev/tools/fastbev_run.sh)

### Installation

* CUDA>=9.2
* GCC>=5.4
* Python>=3.6
* Pytorch>=1.8.1
* Torchvision>=0.9.1
* MMCV-full==1.4.0
* MMDetection==2.14.0
* MMSegmentation==0.14.1

### Dataset preparation

```
  .
  ├── data
  │   └── nuscenes
  │       ├── maps
  │       ├── maps_bev_seg_gt_2class
  │       ├── nuscenes_infos_test_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_train_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_val_4d_interval3_max60.pkl
  │       ├── v1.0-test
  │       └── v1.0-trainval
```

[download](https://drive.google.com/drive/folders/10KyLm0xW3QiLhAefxBbXR-Hw_7nel_tm?usp=sharing)

### Pretraining

```
  .
  ├── pretrained_models
  │   ├── cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth
  │   ├── cascade_mask_rcnn_r34_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5190_segm_mAP_0.4140.pth
  │   └── cascade_mask_rcnn_r50_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5400_segm_mAP_0.4300.pth
```

[download](https://drive.google.com/drive/folders/19BD4totDHtwnHtOqTdn0xYJh7stwYd9l?usp=sharing)

### Training

```
  .
  ├── work_dirs
    └── fastbev
      └── exp
          └── paper
              └── fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.02062323.txt
              │   └── log.test.fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.02062309.txt
              ├── fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4.02080000.txt
              │   └── log.test.fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4.02072346.txt
              ├── fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.02080021.txt
              │   └── log.test.fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.02080005.txt
              ├── fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.02080021.txt
              │   └── log.test.fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.02080005.txt
              └── fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4
                  ├── epoch_20.pth
                  ├── latest.pth -> epoch_20.pth
                  ├── log.eval.fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.02080021.txt
                  └── log.test.fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.02080001.txt
```

[download](https://drive.google.com/drive/folders/1Ja9mqOE0iGPysVxmLSrZyUoCEBYu5fMH?usp=sharing)

### Deployment
TODO

## View Transformation Latency on device
[2D-to-3D on CUDA & CPU](https://github.com/Sense-GVT/Fast-BEV/tree/dev/script/view_tranform_cuda)

## Citation
```
@article{li2023fast,
  title={Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline},
  author={Li, Yangguang and Huang, Bin and Chen, Zeren and Cui, Yufeng and Liang, Feng and Shen, Mingzhu and Liu, Fenggang and Xie, Enze and Sheng, Lu and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2301.12511},
  year={2023}
}
```
# 模型训练
python tools/train.py configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py --work-dir work_dir --gpu-ids 0

# 参考博文
https://blog.csdn.net/weixin_41691854/article/details/154491942?spm=1001.2014.3001.5502

# onnx导出
为什么需要分阶段导出？

FastBEV的推理流程包含：

2D图像编码（Backbone + FPN Neck）→ 可导出ONNX ✅
Backproject（将2D特征投影到3D体素空间）→ 包含循环和索引操作 ❌ ONNX不支持
3D Neck + Detection Head → 可导出ONNX ✅
所以需要将模型分为 2D阶段 和 3D阶段 分别导出，中间用自定义的 backproject（CUDA或Python实现）连接。

# 1️⃣ 导出2D Backbone部分
python tools/model_converters/export_fastbev_onnx.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dir/epoch_5.pth \
    --stage 2d \
    --out nnx_models/fastbev_2d.onnx

    输入: (B, 6, 3, 256, 704) — 6个视角的图像
    输出: (B*6, 256, H_feat, W_feat) — 融合后的2D特征图

# 2️⃣ 导出3D检测部分   
python tools/model_converters/export_fastbev_onnx.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dir/epoch_5.pth \
    --stage 3d \
    --out onnx_models/fastbev_3d.onnx
    输入: (B, C_in, 200, 200, 4) — 体素特征
    输出: [cls_scores, bbox_preds, dir_cls_preds] — 检测头预测

# 3️⃣ 部署完整流程
图像 → [2D ONNX] → 2D特征图 → [backproject (CUDA/Python)] → 3D体素 → [3D ONNX] → 检测结果