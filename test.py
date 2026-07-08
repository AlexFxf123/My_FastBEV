
import pickle, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('work_dir_fusion/results_latest.pkl', 'rb') as f:
    out = pickle.load(f)

# 取第一个样本的预测框
pred_boxes = out[0]['boxes_3d'].tensor  # (N, 9)
pred_scores = out[0]['scores_3d']
pred_labels = out[0]['labels_3d']

# 只画分数 > 0.1 的框
mask = pred_scores > 0.1
pred_boxes = pred_boxes[mask]
pred_scores = pred_scores[mask]
pred_labels = pred_labels[mask]

print('预测框数量（score>0.1）:', len(pred_boxes))
if len(pred_boxes) > 0:
    print('位置范围: x=[%.2f, %.2f] y=[%.2f, %.2f]' % (
        pred_boxes[:,0].min(), pred_boxes[:,0].max(),
        pred_boxes[:,1].min(), pred_boxes[:,1].max()))
else:
    print('无高分预测框')
    
# 画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 预测框
for i in range(min(100, len(pred_boxes))):
    bx = pred_boxes[i]
    cx, cy, cz, w, l, h, yaw = bx[:7].tolist()
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]])
    corners = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners[:, 0] += cx
    corners[:, 1] += cy
    ax1.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=0.5, alpha=0.5)

ax1.set_xlim(-50, 50)
ax1.set_ylim(-50, 50)
ax1.set_aspect('equal')
ax1.set_title('Predicted Boxes (score>0.3)')
ax1.grid(True)

# 加载GT框
import pickle as pkl2
with open('/home/radardepth/data/nuscenes/nuscenes_infos_val_4d_interval3_max60_wradar.pkl', 'rb') as f:
    val_data = pkl2.load(f)
info = val_data['infos'][0]
gt_boxes = info['gt_boxes']  # (N, 7)

for i in range(min(200, len(gt_boxes))):
    gx, gy, gz, gw, gl, gh, gyaw = gt_boxes[i, :7].tolist()
    cos_a, sin_a = np.cos(gyaw), np.sin(gyaw)
    corners = np.array([[-gl/2, -gw/2], [gl/2, -gw/2], [gl/2, gw/2], [-gl/2, gw/2], [-gl/2, -gw/2]])
    corners = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners[:, 0] += gx
    corners[:, 1] += gy
    ax2.plot(corners[:, 0], corners[:, 1], 'g-', linewidth=0.5, alpha=0.7)

ax2.set_xlim(-50, 50)
ax2.set_ylim(-50, 50)
ax2.set_aspect('equal')
ax2.set_title('Ground Truth Boxes')
ax2.grid(True)

plt.tight_layout()
plt.savefig('test_vis/pred_vs_gt_sample0.png', dpi=150)
print('已保存到 test_vis/pred_vs_gt_sample0.png')
