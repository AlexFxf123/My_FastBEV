# -*- coding: utf-8 -*-
"""
模拟训练 pipeline，加载一组数据并可视化验证：
  1. 6 相机图片 + GT 3D 框投影
  2. BEV 俯视图：雷达点云 + GT 框
注意：此脚本直接调用数据集和 pipeline，不启动训练
"""
import os, sys, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont
import cv2

# ---------- 配置 ----------
PKL = '/home/radardepth/data/nuscenes/nuscenes_infos_val_4d_interval3_max60_wradar.pkl'
DATAROOT = '/home/radardepth/data/nuscenes/'
SAMPLE_IDX = 5  # 想看哪帧改这里

# ---------- 工具函数 ----------
def qR(w,x,y,z):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def read_radar_pcd(filepath):
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = f.read()
    for tag in (b'DATA binary\r\n', b'DATA binary\n'):
        i = data.find(tag)
        if i != -1:
            pos = i + len(b'DATA binary')
            while pos < len(data) and data[pos] in (0x0A, 0x0D, 32):
                pos += 1
            data = data[pos:]
            break
    nb = len(data)
    if nb % 43 != 0:
        data = data[: (nb // 43) * 43]
    if len(data) == 0:
        return None
    n = len(data) // 43
    pts = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        b = data[i*43:(i+1)*43]
        pts[i,0] = np.frombuffer(b, np.float32, 1, 0)[0]
        pts[i,1] = np.frombuffer(b, np.float32, 1, 4)[0]
        pts[i,2] = np.frombuffer(b, np.float32, 1, 8)[0]
        pts[i,3] = np.frombuffer(b, np.float32, 1, 27)[0]
        pts[i,4] = np.frombuffer(b, np.float32, 1, 31)[0]
        pts[i,5] = np.frombuffer(b, np.float32, 1, 15)[0]
    return pts

def project_3d_to_2d(points_3d, K, R, t):
    """3D点 (N,3) lidar→相机→像素"""
    pts_cam = (R @ points_3d.T + t).T  # (N,3)
    z = pts_cam[:, 2]
    valid = z > 0.25
    u = K[0,0] * pts_cam[:,0] / z + K[0,2]
    v = K[1,1] * pts_cam[:,1] / z + K[1,2]
    return u, v, valid

def draw_3d_box(draw, corners_3d, K, R, t, img_w, img_h, color='red', width=2):
    u, v, valid = project_3d_to_2d(corners_3d.T, K, R, t)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        if valid[a] and valid[b]:
            x1 = int(np.clip(round(u[a]), 0, img_w-1))
            y1 = int(np.clip(round(v[a]), 0, img_h-1))
            x2 = int(np.clip(round(u[b]), 0, img_w-1))
            y2 = int(np.clip(round(v[b]), 0, img_h-1))
            draw.line([(x1,y1),(x2,y2)], fill=color, width=width)

# ---------- 加载 pkl ----------
print(f'加载 pkl: {PKL}')
with open(PKL, 'rb') as f:
    data = pickle.load(f)
infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
info = infos[SAMPLE_IDX]
print(f'样本 {SAMPLE_IDX}: token={info.get("token","?")[:16]}...')
print(f'  gt_boxes: {len(info["gt_boxes"])}')
print(f'  radars: {list(info.get("radars",{}).keys())}')
print(f'  cams: {list(info.get("cams",{}).keys())}')

# ---------- 外参 ----------
R_le = qR(*info['lidar2ego_rotation'])          # lidar→ego
t_le = np.array(info['lidar2ego_translation'], dtype=np.float64).reshape(3,1)

# ---------- GT 框 8 个角点 (在 lidar 系) ----------
gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float64)
gt_names = np.asarray(info['gt_names'])
box_corners_lidar = []
for bi in range(gt_boxes.shape[0]):
    cx,cy,cz,w,l,h,yaw = gt_boxes[bi]
    c,s = np.cos(yaw), np.sin(yaw)
    dlocal = np.array([
        [ w/2,  l/2, -h/2], [ w/2, -l/2, -h/2],
        [-w/2, -l/2, -h/2], [-w/2,  l/2, -h/2],
        [ w/2,  l/2,  h/2], [ w/2, -l/2,  h/2],
        [-w/2, -l/2,  h/2], [-w/2,  l/2,  h/2]], dtype=np.float64).T
    Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    p_l = Rz @ dlocal + np.array([[cx],[cy],[cz]])
    box_corners_lidar.append(p_l)

# ===========================
# 图1: 6 相机 + GT 投影
# ===========================
print('\n=== 图1: 6 相机 + GT 框 ===')
CAM_ORDER = [
    ('CAM_FRONT_LEFT', 0, 0), ('CAM_FRONT', 0, 1), ('CAM_FRONT_RIGHT', 0, 2),
    ('CAM_BACK_LEFT',  1, 0), ('CAM_BACK',   1, 1), ('CAM_BACK_RIGHT',  1, 2),
]

cell_imgs = {}
for cam_key, gr, gc in CAM_ORDER:
    cam = info.get('cams', {}).get(cam_key)
    if not cam:
        print(f'  {cam_key}: ❌ 无数据')
        continue
    
    img_path = os.path.join(DATAROOT, cam['data_path'])
    if not os.path.isfile(img_path):
        print(f'  {cam_key}: ❌ 找不到 {img_path}')
        continue
    
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 相机外参
    R_ce = qR(*cam['sensor2ego_rotation'])
    t_ce = np.array(cam['sensor2ego_translation'], dtype=np.float64).reshape(3,1)
    K = np.array(cam['cam_intrinsic'], dtype=np.float64)
    
    # lidar→camera 旋转和平移
    R_lidar2cam = R_ce.T @ R_le
    t_lidar2cam = R_ce.T @ (t_le - t_ce)
    
    # 画框
    for bi, corners_l in enumerate(box_corners_lidar):
        u, v, valid = project_3d_to_2d(corners_l.T, K, R_lidar2cam, t_lidar2cam)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a,b in edges:
            if valid[a] and valid[b]:
                x1 = int(np.clip(round(u[a]), 0, img.width-1))
                y1 = int(np.clip(round(v[a]), 0, img.height-1))
                x2 = int(np.clip(round(u[b]), 0, img.width-1))
                y2 = int(np.clip(round(v[b]), 0, img.height-1))
                draw.line([(x1,y1),(x2,y2)], fill='red', width=2)
        # 标签
        if valid[0]:
            try:
                lbl = str(gt_names[bi]).split('.')[-1][:6]
            except:
                lbl = '?'
            uu = int(np.clip(round(u[0]), 0, img.width-1))
            vv = int(np.clip(round(v[0])-10, 0, img.height-1))
            draw.text((uu, vv), lbl, fill='yellow')
    
    # 相机名
    try:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except:
        fnt = ImageFont.load_default()
    draw.text((6, 6), cam_key.replace('CAM_', ''), fill=(0,255,0), font=fnt)
    
    cell_imgs[(gr, gc)] = img
    print(f'  {cam_key}: ✅ {img.size}')

# 拼图
if cell_imgs:
    Ws = [im.width for im in cell_imgs.values()]
    Hs = [im.height for im in cell_imgs.values()]
    CW, CH = max(Ws), max(Hs)
    canvas = Image.new('RGB', (CW*3, CH*2))
    for (r,c), im in cell_imgs.items():
        if im.size != (CW, CH):
            im = im.resize((CW, CH), Image.BILINEAR)
        canvas.paste(im, (c*CW, r*CH))
else:
    canvas = None

# ===========================
# 图2: BEV 俯视图 (雷达点云 + GT 框)
# ===========================
print('\n=== 图2: BEV 俯视图 ===')
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.grid(True, ls='--', alpha=0.3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_xlim(-55, 55)
ax.set_ylim(-55, 55)

# ---- 用 LoadRadarPointsFromFile 相同的方式加载雷达点 ----
RCHS = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
COLORS = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6']

total_pts = 0
for ch, col in zip(RCHS, COLORS):
    r = info.get('radars', {}).get(ch)
    if not r:
        continue
    
    fpath = os.path.join(DATAROOT, r['data_path'])
    pts6 = read_radar_pcd(fpath)
    if pts6 is None or pts6.shape[0] == 0:
        print(f'  {ch}: ❌ 无点')
        continue
    
    # sensor→lidar 外参（单位是 m，已正确）
    Rsl = np.asarray(r['sensor2lidar_rotation'], np.float32)
    tsl = np.asarray(r['sensor2lidar_translation'], np.float32).reshape(1, 3)
    xyz = pts6[:, :3] @ Rsl.T + tsl
    
    ax.scatter(xyz[:, 0], xyz[:, 1], s=2, alpha=0.6, c=col, label=ch.replace('RADAR_',''))
    total_pts += xyz.shape[0]
    print(f'  {ch}: {xyz.shape[0]} 点  x=[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] y=[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}]')

print(f'  雷达点云总计: {total_pts} 点')

# ---- GT 框 ----
for i in range(gt_boxes.shape[0]):
    cx,cy,_,w,l,h,yaw = gt_boxes[i]
    c,s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[ w/2, l/2],[ w/2,-l/2],[-w/2,-l/2],[-w/2, l/2]], dtype=np.float64).T
    pol = (np.array([[c,-s],[s,c]]) @ loc + [[cx],[cy]]).T
    ax.add_patch(Polygon(pol, closed=True, fill=False, edgecolor='lime', lw=1.5))
    try:
        lbl = str(gt_names[i]).split('.')[-1][:6]
    except:
        lbl = '?'
    ax.text(cx, cy, lbl, fontsize=6, color='lime', ha='center', va='center')

ax.legend(fontsize=7, markerscale=5)
ax.set_title(f'验证样本 {SAMPLE_IDX}: Radar BEV + GT boxes (训练pipeline加载方式)')

# ===========================
# 保存
# ===========================
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_output')
os.makedirs(OUT, exist_ok=True)

if canvas:
    p1 = os.path.join(OUT, f'verify_cam_gt_sample{SAMPLE_IDX}.png')
    canvas.save(p1)
    print(f'\n✅ 图1(cam+GT): {p1}')

p2 = os.path.join(OUT, f'verify_bev_sample{SAMPLE_IDX}.png')
fig.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ 图2(BEV): {p2}')
