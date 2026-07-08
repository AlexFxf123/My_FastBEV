# -*- coding: utf-8 -*-
"""
单独调试：加载一个雷达 PCD 文件并绘制点云
"""
import struct
import numpy as np
import matplotlib.pyplot as plt

def read_radar(filepath):
    if not isinstance(filepath, str) or not os.path.isfile(filepath):
        return None
    with open(filepath, "rb") as f:
        data = f.read()
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
    pts = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        b = data[i * point_size : (i + 1) * point_size]
        pts[i, 0] = struct.unpack('f', b[0:4])[0]    # x
        pts[i, 1] = struct.unpack('f', b[4:8])[0]    # y
        pts[i, 2] = struct.unpack('f', b[8:12])[0]   # z
        pts[i, 3] = float(struct.unpack('H', b[13:15])[0])  # id
        pts[i, 4] = struct.unpack('f', b[15:19])[0]  # rcs
        pts[i, 5] = struct.unpack('f', b[19:23])[0]  # vx
        pts[i, 6] = struct.unpack('f', b[23:27])[0]  # vy
    return pts

import os
# 找一个雷达文件
pcd_path = "/home/radardepth/data/nuscenes/samples/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402927664178.pcd"

print(f"文件存在: {os.path.exists(pcd_path)}")

pts = read_radar(pcd_path)
print(f"点云形状: {pts.shape}")
print(f"点数: {pts.shape[0]}")

if pts is not None and pts.shape[0] > 0:
    print(f"\n原始坐标范围:")
    print(f"  x: [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
    print(f"  y: [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]")
    print(f"  z: [{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]")
    print(f"  rcs: [{pts[:,4].min():.4f}, {pts[:,4].max():.4f}]")
    
    print("\n前10个点原始数据:")
    for i in range(min(10, pts.shape[0])):
        print(f"  [{i}] x={pts[i,0]:.4f} y={pts[i,1]:.4f} z={pts[i,2]:.4f} id={pts[i,3]:.0f} rcs={pts[i,4]:.4f} vx={pts[i,5]:.4f} vy={pts[i,6]:.4f}")
    
    # 画图 - 黑色点
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：原始坐标（雷达自身坐标系）
    ax = axes[0]
    ax.scatter(pts[:,0], pts[:,1], c='black', s=20, alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"原始雷达坐标系 ({pts.shape[0]} points)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.scatter(0, 0, c='red', marker='^', s=100, label='sensor origin')
    ax.legend()
    
    # 右图：用坐标范围自动缩放
    ax = axes[1]
    ax.scatter(pts[:,0], pts[:,1], c='black', s=5, alpha=0.6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("自动范围 zoom in")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.scatter(0, 0, c='red', marker='^', s=100, label='sensor origin')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("没有点云数据!")