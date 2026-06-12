# -*- coding: utf-8 -*-
"""
verify_radar_2d.py  —— 终极版：用 glob 搜索雷达文件 + 直接画图
用法:
  python tools/verify_radar_2d.py
"""
from __future__ import annotations   # ← 放第一行（shebang/coding之后），让注解惰性化
                                    # 同时下面我们仍用 typing 写法保底

import os
import pickle
import glob
import numpy as np

# ============== 配置区（只改这里） ==============
DATAROOT = '/home/radardepth/data/nuscenes/'
PKL_NAME  = 'nuscenes_infos_val_4d_interval3_max60_wradar.pkl'
PLOT_FRAME = 0
PLOT_CH    = 'RADAR_FRONT'
# ================================================

PKL_PATH = os.path.join(DATAROOT, PKL_NAME)


# -------- 1. 找雷达文件（glob 搜索，不靠手工拼） --------
def find_radar_file_via_glob(dataroot: str, ch: str, basename: str):
    # type hint 在 3.8 下不能用 `str | None`，用纯 return 或 Optional
    # 但因为有 __future__ annotations 惰性化，上面签名也不会崩溃了
    # 我们仍写一个兼容注释：returns Optional[str]
    for sub in ('samples', 'sweeps'):
        pattern = os.path.join(dataroot, sub, ch, basename)
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    for sub in ('samples', 'sweeps'):
        pattern = os.path.join(dataroot, sub, '**', basename)
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    return None


# -------- 2. 读 43-byte/pt radar pcd binary --------
def load_radar_xy(filepath: str):
    """返回 xs, ys (N,) float32 ；失败返回 None"""
    if not os.path.isfile(filepath):
        print('  [load] not file:', filepath)
        return None

    with open(filepath, 'rb') as f:
        data = f.read()

    # 跳 PCD header：找 "DATA binary" 行尾
    for tag in (b'DATA binary\r\n', b'DATA binary\n'):
        idx = data.find(tag)
        if idx != -1:
            pos = idx + len(tag)
            while pos < len(data) and data[pos] in (0x0D, 0x0A, 0x20):
                pos += 1
            data = data[pos:]
            break

    nb = len(data)
    if nb % 43 != 0:
        # 截掉尾部残字节（安全：nuScenes 的 pcd binary 必须 43 对齐）
        data = data[: (nb // 43) * 43]
        nb = len(data)
    if nb == 0:
        print('  [load] payload empty after header')
        return None

    # --- 用 numpy frombuffer 直读，不碰 Python 逐点循环 ---
    pts = np.frombuffer(data, dtype=np.float32).reshape(nb // 4, 43 // 4)
    # 每点 43 bytes = 10 floats + 1 uint8(?) ，但 nuScenes RADAR pcd 布局公认是：
    # [0:4]=x  [4:8]=y  [8:12]=z  [12:16]=dyn  [16:20]=rcs  [20:24]=vx  [24:28]=vy  [28:32]=vr  [32:36]=amplitude?  [36:40]=unused?
    # 更简单：直接按 43-byte stride reinterpret：
    # 但最稳验证方式——按 float32 每点 11 字段？不对，43不是4的倍数？ 等等：43字节/point:
    # 实际上 nu 官方 pcd 的 "DATA binary" 格式：
    #   x y z dyn rcs vx vy vcomp pad(3bytes) ?  -> 布局各家解析一样：float x,float y,float z,uint8_t dyn,float rcs,sensorType,...
    # 但常见实践：直接读所有 float32 然后 stride slice：

    # === 稳妥直读（等同你原意，但连续内存、不copy-per-point）===
    n = nb // 43
    # reinterpret 43-byte rows as (float32 x, float32 y, ...)
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n, 43)
    xs = raw[:, 0:4].view(np.float32).ravel()
    ys = raw[:, 4:8].view(np.float32).ravel()
    return xs, ys


# -------- 3. 主流程 --------
def main():
    display = os.environ.get('DISPLAY', '')
    if not display:
        print('\n⚠️  DISPLAY 环境变量为空 → 无法弹窗')
        print('   如果你在 SSH，请用:  ssh -X user@host')
    else:
        print('(DISPLAY=%s  →  尝试弹窗)' % display)

    import matplotlib
    print('matplotlib default backend:', matplotlib.get_backend())

    import matplotlib.pyplot as plt

    print('PKL:', PKL_PATH)
    assert os.path.exists(PKL_PATH), 'pkl 不存在: ' + PKL_PATH

    with open(PKL_PATH, 'rb') as f:
        obj = pickle.load(f)
    infos = obj['infos'] if isinstance(obj, dict) and 'infos' in obj else obj
    info = infos[PLOT_FRAME]

    token = info.get('token', '?')
    radars = info.get('radars', {})
    print('\nFrame %d  token=%s' % (PLOT_FRAME, str(token)[:20]))
    print('radars keys:', list(radars.keys()))

    if not radars or PLOT_CH not in radars:
        print('\n❌  radars 字段缺少 [%s] → 你的 add_radar_info 没写进去' % PLOT_CH)
        print('     info keys:', list(info.keys()))
        return

    r = radars[PLOT_CH]
    raw_path = r.get('data_path', '')
    basename = os.path.basename(raw_path)

    print('pkl data_path =', raw_path)
    print('basename     =', basename)

    found = find_radar_file_via_glob(DATAROOT, PLOT_CH, basename)
    if not found:
        print('\n❌  glob 搜索不到文件')
        for sub in ('samples', 'sweeps'):
            d = os.path.join(DATAROOT, sub, PLOT_CH)
            if os.path.isdir(d):
                print('   ', d, '->', sorted(os.listdir(d))[:5], '...')
            else:
                print('   ', d, '不存在')
        return

    print('✅  找到雷达文件:', found)

    res = load_radar_xy(found)
    if res is None:
        print('❌ 读点失败')
        return
    xs, ys = res
    if xs.size == 0:
        print('❌ 读到 0 个点')
        return

    print('pts=%d  x=[%.1f, %.1f]  y=[%.1f, %.1f]' % (
        xs.size, xs.min(), xs.max(), ys.min(), ys.max()))

    # sensor→lidar 变换
    R = np.asarray(r['sensor2lidar_rotation'], dtype=np.float32)
    t = np.asarray(r['sensor2lidar_translation'], dtype=np.float32)
    xl = R[0, 0] * xs + R[0, 1] * ys + t[0]
    yl = R[1, 0] * xs + R[1, 1] * ys + t[1]

    # -------- 4. 画图 --------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Frame %d  %s' % (PLOT_FRAME, PLOT_CH), fontsize=12)

    a1.scatter(xs, ys, s=3, alpha=0.85, color='#e74c3c')
    a1.set_title('Sensor XY (raw)')
    a1.set_xlabel('X (m)'); a1.set_ylabel('Y (m)')
    a1.axis('equal'); a1.grid(ls='--', alpha=0.5)
    a1.set_xlim(-110, 110); a1.set_ylim(-65, 65)

    a2.scatter(xl, yl, s=3, alpha=0.85, color='#2980b9')
    a2.set_title('Lidar XY (after sensor2lidar_R,t)')
    a2.set_xlabel('X (m)'); a2.set_ylabel('Y (m)')
    a2.axis('equal'); a2.grid(ls='--', alpha=0.5)
    a2.set_xlim(-70, 85); a2.set_ylim(-70, 70)

    for ax in (a1, a2):
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.axvline(0, color='k', lw=0.8, alpha=0.4)

    plt.tight_layout()
    print('→ 弹窗中（关闭窗口即结束）')
    plt.show()


if __name__ == '__main__':
    main()