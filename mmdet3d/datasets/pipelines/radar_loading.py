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
        # 不硬炸，先 warn 再尽量 salvage
        import warnings
        warnings.warn(
            f"[radar] '{os.path.basename(data_path)}' "
            f"payload={nbytes}B not aligned to {expected_point_stride}B "
            f"(mod={nbytes % expected_point_stride}); truncating"
        )
        n_keep = (nbytes // expected_point_stride) * expected_point_stride
        payload = payload[:n_keep]
        if n_keep == 0:
            return np.zeros((0, 6), dtype=np.float32)

    # 视作 uint8 矩阵做字节级切片（这是 nuScenes PCD binary 的正确读法）
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, expected_point_stride)

    # ---- 按你验证过的偏移，提取成 float32 ----------------------------------
    # x: bytes 0..3, y: 4..7, z: 8..11
    # rcs: bytes 15..18
    # vx: bytes 27..30  (有的版本你标的是 19..22 对应 vx_comp——看你怎么定义)
    # vy: bytes 31..34
    #
    # 【重要】你脚本里实际可用的偏移抄下来是这样（与你打印一致）：
    #   x  = raw[:, 0:4].view(np.float32)   ← 但不能对二维 slice 直接 view
    # 正确做法是用 np.frombuffer 从每行 bytes 重解释：
    x = raw[:, 0:4].view(np.float32).reshape(-1, 1)
    y = raw[:, 4:8].view(np.float32).reshape(-1, 1)
    z = raw[:, 8:12].view(np.float32).reshape(-1, 1)
    # rcs
    rcs = raw[:, 15:19].view(np.float32).reshape(-1, 1)
    # vx_comp (官方名 vx_comp, 有些 pipeline 把它当 vx)
    vx = raw[:, 27:31].view(np.float32).reshape(-1, 1)
    # vy_comp
    vy = raw[:, 31:35].view(np.float32).reshape(-1, 1)

    pts = np.concatenate([x, y, z, vx, vy, rcs], axis=1).astype(np.float32)
    # 输出 shape: (N, 6) = [x, y, z, vx, vy, rcs]
    return pts