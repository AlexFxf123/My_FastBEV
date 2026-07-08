from mmdet.datasets.builder import PIPELINES
import os
import numpy as np


def _find_binary_payload(fpath: str) -> bytes:
    """定位 PCD 文件中 DATA binary 之后的裸二进制载荷。"""
    with open(fpath, "rb") as f:
        data = f.read()

    # 正常 nuScenes 雷达 PCD：找到 "DATA binary" 行，取其末尾到文件尾
    idx = data.find(b"DATA binary")
    if idx == -1:
        # fallback：也许是无头裸 bin（罕见）
        return data

    # 跳过 "DATA binary" 这几个字符 + 紧随的 \n / \r
    pos = idx + len(b"DATA binary")
    while pos < len(data) and data[pos] in (0x0A, 0x0D):
        pos += 1
    return data[pos:]


def load_pcd_file(data_path: str,
                  expected_point_stride: int = 43,
                  *,
                  extract_fields: str = "xy_vx_vy_rcs") -> np.ndarray:
    """
    加载 nuScenes 雷达 .pcd（DATA binary）为 (N, D) float32。

    默认 extract_fields="xy_vx_vy_rcs" 输出 (N, 6)：
        [x, y, z, vx, vy, rcs]

    如需全部 43 字节按结构拆开再挑字段，也可以改成返回 (N, 18) 之类——
    但 pipeline 里一般只需要空间位置 + 径向速度 + 强度。

    出错时返回 (0, D) 空数组而不是 raise（DataLoader worker 里更稳）。
    """
    if not os.path.isfile(data_path):
        # nuScenes 偶尔某个 sweep 缺文件；给空点云让训练继续
        if extract_fields == "xy_vx_vy_rcs":
            return np.zeros((0, 6), dtype=np.float32)
        return np.zeros((0, 0), dtype=np.float32)

    try:
        payload = _find_binary_payload(data_path)
    except Exception:
        return np.zeros((0, 6), dtype=np.float32)

    nbytes = len(payload)
    if nbytes == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # ---- 对齐检查 -----------------------------------------------------------
    if nbytes % expected_point_stride != 0:
        # 静默截断，不打印 warning
        n_keep = (nbytes // expected_point_stride) * expected_point_stride
        payload = payload[:n_keep]
        if n_keep == 0:
            return np.zeros((0, 6), dtype=np.float32)

    # 视作 uint8 矩阵做字节级切片（这是 nuScenes PCD binary 的正确读法）
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, expected_point_stride)

    # ---- 按你验证过的偏移，提取成 float32 ----------------------------------
    # x: bytes 0..3, y: 4..7, z: 8..11
    # rcs: bytes 15..18
    # vx: bytes 27..30
    # vy: bytes 31..34
    #
    # 【注意】对二维 slice 不能直接 .view(np.float32)（不是 C-contiguous），
    # 需要用 .copy() 或 cast 到 bytes 再 frombuffer。
    # 这里直接 struct 方式从原始 payload 一次提取最稳：

    n = raw.shape[0]
    pts = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        b = payload[i*43 : (i+1)*43]
        # x, y, z (float32 at 0,4,8)
        pts[i, 0] = np.frombuffer(b, np.float32, 1, 0)[0]
        pts[i, 1] = np.frombuffer(b, np.float32, 1, 4)[0]
        pts[i, 2] = np.frombuffer(b, np.float32, 1, 8)[0]
        # rcs (float32 at 15)
        pts[i, 5] = np.frombuffer(b, np.float32, 1, 15)[0]
        # vx_comp (float32 at 27)
        pts[i, 3] = np.frombuffer(b, np.float32, 1, 27)[0]
        # vy_comp (float32 at 31)
        pts[i, 4] = np.frombuffer(b, np.float32, 1, 31)[0]

    return pts

RADAR_CHANNELS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]


@PIPELINES.register_module()
class LoadRadarPointsFromFile(object):
    """从 info['radars'] 加载多个雷达通道的点云，并用 sensor2lidar 外参变换到 lidar 坐标系。"""

    def __init__(self, use_dim=None, max_points=30000, data_root=None):
        if use_dim is not None:
            self.use_dim = list(use_dim)
        else:
            self.use_dim = [0, 1, 2, 3, 4, 5]
        self.max_points = max_points
        self.data_root = data_root

    def _load_single_channel(self, data_path, R, t):
        pts6 = load_pcd_file(data_path)
        if pts6.shape[0] == 0:
            return pts6
        Rsl = np.asarray(R, dtype=np.float32)
        tsl = np.asarray(t, dtype=np.float32).reshape(1, 3)
        xyz_lidar = pts6[:, :3] @ Rsl.T + tsl
        pts6[:, :3] = xyz_lidar
        return pts6

    def __call__(self, results):
        radars_info = results.get('radars', {})
        data_root = self.data_root or results.get('data_root', '')
        all_pts = []
        for ch in RADAR_CHANNELS:
            r = radars_info.get(ch)
            if r is None:
                continue
            data_path = r.get('data_path', '')
            if not data_path:
                continue
            full_path = os.path.join(data_root, data_path)
            if not os.path.isfile(full_path):
                continue
            R = r.get('sensor2lidar_rotation', None)
            t = r.get('sensor2lidar_translation', None)
            if R is None or t is None:
                continue
            pts = self._load_single_channel(full_path, R, t)
            if pts.shape[0] > 0:
                all_pts.append(pts)
        if len(all_pts) == 0:
            results['radar_points'] = np.zeros((0, len(self.use_dim)), dtype=np.float32)
            return results
        radar_points = np.concatenate(all_pts, axis=0)
        if self.use_dim is not None and len(self.use_dim) < radar_points.shape[1]:
            radar_points = radar_points[:, self.use_dim]
        if self.max_points > 0 and radar_points.shape[0] > self.max_points:
            indices = np.random.permutation(radar_points.shape[0])[:self.max_points]
            radar_points = radar_points[indices]
        results['radar_points'] = radar_points.astype(np.float32)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(use_dim={self.use_dim}, max_points={self.max_points})'


@PIPELINES.register_module()
class CollectRadarPoints(object):
    def __call__(self, results):
        if 'radar_points' in results:
            pts = results['radar_points']
            if isinstance(pts, np.ndarray) and pts.dtype != np.float32:
                results['radar_points'] = pts.astype(np.float32)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


import pickle
from PIL import Image, ImageDraw, ImageFont


def _qR(w,x,y,z):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def _read_radar_pcd_simple(filepath):
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


def save_visual_debug(info, data_root, out_dir, tag=''):
    """
    保存一组可视化结果（6相机+GT投影，BEV+雷达点云+GT框）
    与 verify_pipeline_data.py 中逻辑完全相同
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    os.makedirs(out_dir, exist_ok=True)

    R_le = _qR(*info['lidar2ego_rotation'])
    t_le = np.array(info['lidar2ego_translation'], dtype=np.float64).reshape(3,1)

    gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float64)
    gt_names = np.asarray(info['gt_names'])

    # ---------- GT 8 角点 ----------
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

    # ---------- 图1: 6相机 ----------
    CAM_ORDER = [
        ('CAM_FRONT_LEFT', 0, 0), ('CAM_FRONT', 0, 1), ('CAM_FRONT_RIGHT', 0, 2),
        ('CAM_BACK_LEFT',  1, 0), ('CAM_BACK',   1, 1), ('CAM_BACK_RIGHT',  1, 2),
    ]
    cell_imgs = {}
    for cam_key, gr, gc in CAM_ORDER:
        cam = info.get('cams', {}).get(cam_key)
        if not cam:
            continue
        img_path = os.path.join(data_root, cam['data_path'])
        if not os.path.isfile(img_path):
            continue
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        R_ce = _qR(*cam['sensor2ego_rotation'])
        t_ce = np.array(cam['sensor2ego_translation'], dtype=np.float64).reshape(3,1)
        K = np.array(cam['cam_intrinsic'], dtype=np.float64)
        R_lidar2cam = R_ce.T @ R_le
        t_lidar2cam = R_ce.T @ (t_le - t_ce)

        for bi, corners_l in enumerate(box_corners_lidar):
            pts_cam = (R_lidar2cam @ corners_l + t_lidar2cam).T
            z = pts_cam[:, 2]
            valid = z > 0.25
            u = K[0,0] * pts_cam[:,0] / z + K[0,2]
            v = K[1,1] * pts_cam[:,1] / z + K[1,2]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a,b in edges:
                if valid[a] and valid[b]:
                    x1 = int(np.clip(round(u[a]), 0, img.width-1))
                    y1 = int(np.clip(round(v[a]), 0, img.height-1))
                    x2 = int(np.clip(round(u[b]), 0, img.width-1))
                    y2 = int(np.clip(round(v[b]), 0, img.height-1))
                    draw.line([(x1,y1),(x2,y2)], fill='red', width=2)
            if valid[0]:
                try: lbl = str(gt_names[bi]).split('.')[-1][:6]
                except: lbl = '?'
                uu = int(np.clip(round(u[0]), 0, img.width-1))
                vv = int(np.clip(round(v[0])-10, 0, img.height-1))
                draw.text((uu, vv), lbl, fill='yellow')
        try:
            fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
        except:
            fnt = ImageFont.load_default()
        draw.text((6,6), cam_key.replace('CAM_',''), fill=(0,255,0), font=fnt)
        cell_imgs[(gr, gc)] = img

    if cell_imgs:
        Ws = [im.width for im in cell_imgs.values()]
        Hs = [im.height for im in cell_imgs.values()]
        CW, CH = max(Ws), max(Hs)
        canvas = Image.new('RGB', (CW*3, CH*2))
        for (r,c), im in cell_imgs.items():
            if im.size != (CW, CH):
                im = im.resize((CW, CH), Image.BILINEAR)
            canvas.paste(im, (c*CW, r*CH))
        p1 = os.path.join(out_dir, f'cam_gt{tag}.png')
        canvas.save(p1)

    # ---------- 图2: BEV ----------
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_aspect('equal')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(-55, 55); ax.set_ylim(-55, 55)

    RCHS = ['RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
    COLORS = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6']

    for ch, col in zip(RCHS, COLORS):
        r = info.get('radars', {}).get(ch)
        if not r:
            continue
        fpath = os.path.join(data_root, r['data_path'])
        pts6 = _read_radar_pcd_simple(fpath)
        if pts6 is None or pts6.shape[0] == 0:
            continue
        Rsl = np.asarray(r['sensor2lidar_rotation'], np.float32)
        tsl = np.asarray(r['sensor2lidar_translation'], np.float32).reshape(1,3)
        xyz = pts6[:, :3] @ Rsl.T + tsl
        ax.scatter(xyz[:,0], xyz[:,1], s=2, alpha=0.6, c=col, label=ch.replace('RADAR_',''))

    for i in range(gt_boxes.shape[0]):
        cx,cy,_,w,l,h,yaw = gt_boxes[i]
        c,s = np.cos(yaw), np.sin(yaw)
        loc = np.array([[ w/2, l/2],[ w/2,-l/2],[-w/2,-l/2],[-w/2, l/2]], dtype=np.float64).T
        pol = (np.array([[c,-s],[s,c]]) @ loc + [[cx],[cy]]).T
        ax.add_patch(Polygon(pol, closed=True, fill=False, edgecolor='lime', lw=1.5))
        try: lbl = str(gt_names[i]).split('.')[-1][:6]
        except: lbl = '?'
        ax.text(cx, cy, lbl, fontsize=6, color='lime', ha='center', va='center')
    ax.legend(fontsize=7, markerscale=5)
    ax.set_title(f'BEV{tag}: radar + GT')
    p2 = os.path.join(out_dir, f'bev{tag}.png')
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
