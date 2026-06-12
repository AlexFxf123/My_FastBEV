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
