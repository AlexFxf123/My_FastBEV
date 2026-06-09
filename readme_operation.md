# 模型训练
python tools/train.py configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py --work-dir work_dir --gpu-ids 0

# 基础评估，输出mAP、NDS等指标
python tools/test.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dir/latest.pth \
    --eval mAP

# 保存结果pkl + 可视化BEV和图像
python tools/test.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dir/latest.pth \
    --eval mAP \
    --out results_latest.pkl \
    --show-dir vis_results/

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
    work_dir/latest.pth \
    --stage 2d \
    --out nnx_models/fastbev_2d.onnx

    输入: (B, 6, 3, 256, 704) — 6个视角的图像
    输出: (B*6, 256, H_feat, W_feat) — 融合后的2D特征图

# 2️⃣ 导出3D检测部分   
python tools/model_converters/export_fastbev_onnx.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dir/latest.pth \
    --stage 3d \
    --out onnx_models/fastbev_3d.onnx
    输入: (B, C_in, 200, 200, 4) — 体素特征
    输出: [cls_scores, bbox_preds, dir_cls_preds] — 检测头预测

# 3️⃣ 部署完整流程
图像 → [2D ONNX] → 2D特征图 → [backproject (CUDA/Python)] → 3D体素 → [3D ONNX] → 检测结果