# -*- coding: utf-8 -*-
"""
tools/verify.py
============================================================
定位方式：glob 搜 pkl → glob 搜 samples/ 下的图片/雷达文件
图1：6相机 2×3 拼图 + GT框投影到像素（红框）
图2：雷达 BEV 俯视（蓝点）+ GT框（绿框）
输出：脚本旁边 vis_output/  （打开就是工程目录，不写 /tmp）
用法:  python tools/verify.py
============================================================
"""
import os
import sys
import glob
import pickle
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1) 只靠 glob 找 pkl（不手写家目录）
# ============================================================
def find_pkl():
    seeds = [
        "/home/radardepth/data/nuscenes",
    ]
    for s in seeds:
        hits = sorted(glob.glob(os.path.join(s, "**", "nuscenes_infos_val_4d_interval3_max60_wradar.pkl"), recursive=True))
        if hits:
            return hits[0]
    # 全盘兜底
    hits = sorted(glob.glob(os.path.join("/", "home", "**", "*wradar*.pkl"), recursive=True))
    if hits: return hits[0]
    return None

PKL = find_pkl()
if not PKL:
    print("❌ 找不到 *_wradar*.pkl，在终端先执行：ls /home/radardepth/data/nuscenes/*wradar*")
    sys.exit(1)

print("PKL =", PKL)

with open(PKL, "rb") as f:
    obj = pickle.load(f)
infos = obj["infos"] if isinstance(obj, dict) and "infos" in obj else obj
info  = infos[0]
token = info.get("token", "?")
print("token =", str(token)[:20])
print("radars keys =", list(info.get("radars", {}).keys()))
print("cams   keys =", list(info.get("cams", {}).keys()))
print("gt_boxes shape =", np.asarray(info["gt_boxes"]).shape)

# ============================================================
# 2) glob 搜真实文件路径（图片 / 雷达）—— 零次 os.path.join(DATAROOT,...)
# ============================================================
def glob_find(data_root_dirs, subdir, basename):
    """
    data_root_dirs: 候选根目录列表（由 pkl 位置反推）
    subdir: 如 'CAM_FRONT' / 'RADAR_FRONT'
    basename: 文件名
    返回绝对路径或 None
    """
    for root in data_root_dirs:
        for sub in ("samples", "sweeps"):
            pat = os.path.join(root, sub, subdir, basename)
            hs = glob.glob(pat)
            if hs and os.path.isfile(hs[0]):
                return os.path.abspath(hs[0])
    return None

def make_search_roots(pkl_path):
    """从 pkl 所在位置反推所有可能包含 samples/ 的根目录"""
    roots = set()
    d = os.path.dirname(os.path.abspath(pkl_path))
    for _ in range(8):
        for prefix in ("",):
            for sub in ("samples", "sweeps"):
                roots.add(os.path.join(d, sub))
            roots.add(d)
        nd = os.path.dirname(d)
        if nd == d: break
        d = nd
    return list(roots)

SR = make_search_roots(PKL)

# ============================================================
# 3) 按你给的写法：读 43-byte radar pcd
# ============================================================
def read_radar_43(filepath):
    if not filepath_check(filepath): return None
    with open(filepath, "rb") as f:
        data = f.read()
    # 跳 header 到 DATA binary 后
    for tag in (b"DATA binary\r\n", b"DATA binary\n"):
        idx = data.find(tag)
        if idx != -1:
            pos = idx + len(b"DATA binary")
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
        b = data[i*43 : (i+1)*43]
        pts[i,0] = np.frombuffer(b, np.float32, 1, 0)[0]   # x
        pts[i,1] = np.frombuffer(b, np.float32, 1, 4)[0]   # y
        pts[i,2] = np.frombuffer(b, np.float32, 1, 8)[0]   # z
        pts[i,3] = np.frombuffer(b, np.float32, 1, 27)[0]  # vx_comp
        pts[i,4] = np.frombuffer(b, np.float32, 1, 31)[0]  # vy_comp
        pts[i,5] = np.frombuffer(b, np.float32, 1, 15)[0]  # rcs
    return pts

def filepath_check(p):
    return isinstance(p, str) and os.path.isfile(p)


# 按 test_pcd.py 的方式解析 43-byte 雷达 PCD（struct.unpack + 正确偏移）
def read_radar(filepath):
    """
    解析 43-byte PCD 雷达点云，完全匹配 test_pcd.py 的解析方式：

    字节偏移:
      0:4   ->  float x
      4:8   ->  float y
      8:12  ->  float z
      13:15 ->  uint16 id
      15:19 ->  float rcs
      19:23 ->  float vx
      23:27 ->  float vy
    """
    if not isinstance(filepath, str) or not os.path.isfile(filepath):
        return None
    with open(filepath, "rb") as f:
        data = f.read()
    # 跳过 PCD header，定位到 DATA binary 后的实际点云二进制数据
    for tag in (b"DATA binary\r\n", b"DATA binary\n"):
        idx = data.find(tag)
        if idx != -1:
            pos = idx + len(b"DATA binary")
            while pos < len(data) and data[pos] in (0x0A, 0x0D, 32):
                pos += 1
            data = data[pos:]
            break
    nb = len(data)


    point_size = 43
    if nb % point_size != 0:
        data = data[: (nb // point_size) * point_size]
    if len(data) == 0:
        return None


    n = len(data) // point_size
    # 输出 7 列: [x, y, z, id, rcs, vx, vy]
    pts = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):







        b = data[i * point_size : (i + 1) * point_size]
        pts[i, 0] = struct.unpack('f', b[0:4])[0]       # x
        pts[i, 1] = struct.unpack('f', b[4:8])[0]       # y
        pts[i, 2] = struct.unpack('f', b[8:12])[0]      # z
        pts[i, 3] = float(struct.unpack('H', b[13:15])[0])  # id (uint16)
        pts[i, 4] = struct.unpack('f', b[15:19])[0]     # rcs
        pts[i, 5] = struct.unpack('f', b[19:23])[0]     # vx
        pts[i, 6] = struct.unpack('f', b[23:27])[0]     # vy
    return pts

# ============================================================
# 4) 四元数(w,x,y,z) → R (3x3)  用 numpy 不依赖库
# ============================================================
def qR(w, x, y, z):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ], dtype=np.float64)

# ============================================================
# 5) 外参链：lidar 系 → camera 系
#    lidar→ego:  p_ego = R_le @ p_l + t_le
#    camera 在 ego 里:  R_ce, t_ce
#    p_cam = R_ce^T @ (p_ego - t_ce)
# ============================================================
R_le = qR(*info["lidar2ego_rotation"])          # lidar→ego
t_le = np.array(info["lidar2ego_translation"], dtype=np.float64).reshape(3,1)

# ============================================================
# 图1：6 相机拼图 + GT框
# ============================================================
print("\n=== 图1：6 cameras + GT boxes ===")

CAM_ORDER = [                                    # (name, grid_row, grid_col)
    ("CAM_FRONT",       0, 1),
    ("CAM_FRONT_LEFT",  0, 0),
    ("CAM_FRONT_RIGHT", 0, 2),
    ("CAM_BACK",        1, 1),
    ("CAM_BACK_LEFT",   1, 0),
    ("CAM_BACK_RIGHT",  1, 2),
]

GRID_R, GRID_C = 2, 3
CELLS = {}

for cam_key, gr, gc in CAM_ORDER:
    cam = info.get("cams", {}).get(cam_key)
    if not cam:
        print(f"  {cam_key}: ❌ 不在 info['cams']"); continue

    bn  = os.path.basename(cam["data_path"])
    img_path = glob_find(SR, cam_key, bn)
    print(f"  {cam_key}: basename={bn}")
    print(f"       -> {img_path}  exists={os.path.exists(img_path) if img_path else '?'}")
    if not img_path or not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ---- 相机外参（pkl 里直接给了） ----
    R_ce = qR(*cam["sensor2ego_rotation"])  # camera→ego
    t_ce = np.array(cam["sensor2ego_translation"], dtype=np.float64).reshape(3,1)
    K = np.array(cam["cam_intrinsic"], dtype=np.float64)

    # 预计算 R_ce^T
    RceT = R_ce.T

    # ---- GT boxes: lidar→camera→pixel ----
    gt = np.asarray(info["gt_boxes"], dtype=np.float64)  # (M,7)
    gn = np.asarray(info["gt_names"])
    M  = gt.shape[0]

    for bi in range(M):
        cx, cy, cz, w, l, h, yaw = gt[bi]
        c, s = np.cos(yaw), np.sin(yaw)

        # 8 corners in lidar frame
        # nuScenes: box centered at (cx,cy,cz), l along x(forward), w along y(left), h along z(up)
        dlocal = np.array([
            [ l/2,  w/2, -h/2],
            [ l/2, -w/2, -h/2],
            [-l/2, -w/2, -h/2],
            [-l/2,  w/2, -h/2],
            [ l/2,  w/2,  h/2],
            [ l/2, -w/2,  h/2],
            [-l/2, -w/2,  h/2],
            [-l/2,  w/2,  h/2],
        ], dtype=np.float64).T  # 3×8

        # rotate around z
        Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
        p_l = Rz @ dlocal + np.array([[cx],[cy],[cz]])  # 3×8

        # lidar→camera
        p_ego = R_le @ p_l + t_le
        p_cam = RceT @ (p_ego - t_ce)  # 3×8

        Z = p_cam[2]
        # 要求所有 8 个角点都在相机前方（不然后方物体穿到前方画面）
        if (Z <= 0.25).any():
            # 保守：跳过有角点在后方的 box（避免穿帮）
            # 你也可以改成只画 z>0 的边
            pass

        u = K[0,0] * p_cam[0] / Z + K[0,2]
        v = K[1,1] * p_cam[1] / Z + K[1,2]

        # 12 edges of a cuboid
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]

        for a, b_ in edges:
            if Z[a] <= 0.25 or Z[b_] <= 0.25:
                continue
            x1 = int(np.clip(round(u[a]), 0, img.width-1))
            y1 = int(np.clip(round(v[a]), 0, img.height-1))
            x2 = int(np.clip(round(u[b_]), 0, img.width-1))
            y2 = int(np.clip(round(v[b_]), 0, img.height-1))
            draw.line([(x1,y1),(x2,y2)], fill="red", width=2)

        # 标签
        if Z[0] > 0.3:
            try:
                lbl = str(gn[bi]).split(".")[-1][:6]
            except Exception:
                lbl = "?"
            uu = int(np.clip(round(u[0]), 0, img.width-1))
            vv = int(np.clip(round(v[0])-10, 0, img.height-1))
            draw.text((uu, vv), lbl, fill="yellow")

    # 相机名
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        fnt = ImageFont.load_default()
    draw.text((6, 6), cam_key.replace("CAM_", ""), fill=(0,255,0), font=fnt)

    CELLS[(gr, gc)] = img

# ---- 拼 2×3 grid（统一尺寸防止错位）----
if CELLS:
    Ws = [im.width for im in CELLS.values()]
    Hs = [im.height for im in CELLS.values()]
    CW, CH = max(Ws), max(Hs)
    canvas = Image.new("RGB", (CW*GRID_C, CH*GRID_R))
    for (r,c), im in CELLS.items():
        if im.size != (CW, CH):
            im = im.resize((CW, CH), Image.BILINEAR)
        canvas.paste(im, (c*CW, r*CH))
else:
    canvas = None
    print("⚠️  图1：没有成功加载任何相机图像")

# ============================================================
# 图2：雷达 BEV + GT 框
# ============================================================
print("\n=== 图2：雷达点云 + GT ===")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 只保留右图：lidar 坐标系雷达点云（缩小点尺寸）+ GT框
fig, ax = plt.subplots(figsize=(11, 11))
ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.5)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.axhline(0, color="k", lw=0.8, alpha=0.35)
ax.axvline(0, color="k", lw=0.8, alpha=0.35)

RCHS = ["RADAR_FRONT","RADAR_FRONT_LEFT","RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT","RADAR_BACK_RIGHT"]
COLORS = ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"]

got_radar = False
for ch, col in zip(RCHS, COLORS):
    r = info.get("radars", {}).get(ch)
    if not r:
        print(f"  {ch}: ❌ 不在 info['radars']"); continue

    bn = os.path.basename(r["data_path"])
    path = glob_find(SR, ch, bn)
    print(f"  {ch}: basename={bn}")
    print(f"       -> {path}  exists={os.path.exists(path) if path else '?'}")
    if not path or not os.path.exists(path):
        for root in SR:
            for sub in ("samples", "sweeps"):
                d2 = os.path.join(root, sub, ch)
                if os.path.isdir(d2):
                    print(f"       🔎 dir={d2}  files={sorted(os.listdir(d2))[:6]}")
        continue

    pts6 = read_radar(path)
    if pts6 is None:
        print(f"       ❌ read_radar returned None"); continue
    print(f"       pts = {pts6.shape[0]}")
    print(f"       前5个点 (x, y, z, rcs):")
    for pi in range(min(5, pts6.shape[0])):
        print(f"         [{pi}]: x={pts6[pi,0]:.4f}  y={pts6[pi,1]:.4f}  z={pts6[pi,2]:.4f}  rcs={pts6[pi,4]:.4f}")

    # sensor→lidar 外参变换，外参单位mm -> m
    Rsl = np.asarray(r["sensor2lidar_rotation"], np.float32)
    tsl = np.asarray(r["sensor2lidar_translation"], np.float32).reshape(1, 3)
    tsl_m = tsl / 1000.0
    xyz_lidar = pts6[:, :3] @ Rsl.T + tsl_m
    print(f"       变换后 xy 范围: x=[{xyz_lidar[:,0].min():.2f}, {xyz_lidar[:,0].max():.2f}]  y=[{xyz_lidar[:,1].min():.2f}, {xyz_lidar[:,1].max():.2f}]")

    # 点云缩小到 s=2
    ax.scatter(xyz_lidar[:,0], xyz_lidar[:,1], s=2, alpha=0.7, c=col,
               label=ch.replace("RADAR_",""), rasterized=False)
    got_radar = True

# GT boxes (也画在右图上)
gt = np.asarray(info["gt_boxes"])
gn = np.asarray(info["gt_names"])
for i in range(gt.shape[0]):
    cx,cy,_,w,l,h,yaw = gt[i]
    c,s=np.cos(yaw),np.sin(yaw)
    loc=np.array([[ l/2,w/2],[ l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]],dtype=np.float64).T
    pol=(np.array([[c,-s],[s,c]])@loc+[[cx],[cy]]).T
    ax.add_patch(Polygon(pol, closed=True, fill=False, edgecolor="lime", lw=1.4))

ax.set_title("Radar BEV (lidar frame) + GT boxes")
if got_radar:
    ax.legend(fontsize=7, markerscale=5)
else:
    print("⚠️  没有雷达点加载成功")

# ============================================================
# 保存
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "vis_output")
os.makedirs(OUT, exist_ok=True)

if canvas:
    p1 = os.path.join(OUT, "camera_gt.png")
    canvas.save(p1)
    print(f"\n✅ 图1 saved: {os.path.abspath(p1)}  ({os.path.getsize(p1)//1024} KB)")

p2 = os.path.join(OUT, "radar_bev.png")
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ 图2 saved: {os.path.abspath(p2)}  ({os.path.getsize(p2)//1024} KB)")
print("\nOpen folder:", os.path.abspath(OUT))