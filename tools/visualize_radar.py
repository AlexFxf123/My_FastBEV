# -*- coding: utf-8 -*-
"""
可视化nuScenes雷达点云与图像的对应关系

显示同一帧的:
  1. 6个相机图像
  2. 所有5-6个雷达点云投影到各相机图像上
  3. 颜色表示雷达点的RCS强度
"""
import argparse
import pickle
import numpy as np
import cv2
import os

from nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion


def project_lidar_to_image(xyz_lidar, cam_info):
    """将LIDAR坐标系下的3D点投影到相机图像上

    Args:
        xyz_lidar: (3, N) LIDAR坐标下的点
        cam_info: 相机info dict，需包含:
            - sensor2lidar_rotation, sensor2lidar_translation
            - cam_intrinsic
            (使用nuscenes原有的lidar2camera变换方式)

    Returns:
        uv: (2, N) 像素坐标
        depth: (N,) 深度值
        mask: (N,) 有效点标志
    """
    # 从cam_info中提取变换: lidar -> camera
    # cam_info中存储的是 sensor2lidar 的逆变换
    # 需要 lidar2camera 变换
    
    # 实际这里用cams中已有的lidar2img变换
    if 'lidar2img' in cam_info:
        # (4, 4) 从LIDAR到图像平面的投影矩阵
        lidar2img = np.array(cam_info['lidar2img'])
        
        N = xyz_lidar.shape[1]
        ones = np.ones((1, N))
        xyz_h = np.vstack([xyz_lidar, ones])  # (4, N)
        
        uv_h = lidar2img @ xyz_h  # (3, N)
        uv = uv_h[:2] / (uv_h[2:3] + 1e-8)
        depth = uv_h[2]
        
        mask = depth > 0.1
        return uv, depth, mask
    
    # 手动计算: lidar -> ego -> camera
    # 从info中提取
    l2e_r = Quaternion(cam_info.get('lidar2ego_rotation', [1,0,0,0])).rotation_matrix
    l2e_t = np.array(cam_info.get('lidar2ego_translation', [0,0,0])).reshape(3, 1)
    e2c_r = Quaternion(cam_info.get('ego2cam_rotation', cam_info.get('sensor2ego_rotation', [1,0,0,0]))).rotation_matrix
    e2c_t = np.array(cam_info.get('ego2cam_translation', cam_info.get('sensor2ego_translation', [0,0,0]))).reshape(3, 1)
    intrinsic = np.array(cam_info.get('cam_intrinsic', np.eye(3)))
    
    # lidar -> ego
    xyz_ego = l2e_r.T @ (xyz_lidar - l2e_t)
    # ego -> camera
    xyz_cam = e2c_r.T @ (xyz_ego - e2c_t)
    
    # 投影
    depth = xyz_cam[2, :]
    uv_h = intrinsic @ xyz_cam
    uv = uv_h[:2] / (uv_h[2:3] + 1e-8)
    
    mask = depth > 0.1
    return uv, depth, mask


def visualize(sample_idx=0, data_set='train'):
    """可视化雷达点云"""
    dataroot = '/home/radardepth/data/nuscenes/'
    
    # 加载info
    info_path = os.path.join(
        dataroot, f'nuscenes_infos_{data_set}_4d_interval3_max60_wradar.pkl')
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    
    info = data['infos'][sample_idx]
    print(f"Sample token: {info['token']}")
    
    # 提取相机图像路径
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 加载所有雷达点云并转换到LIDAR坐标系
    all_xyz_lidar = []
    all_rcs = []
    if 'radars' in info:
        print("\n加载雷达点云:")
        for radar_name, radar_info in info['radars'].items():
            if 'data_path' not in radar_info:
                continue
            data_path = radar_info['data_path']
            if not os.path.exists(data_path):
                print(f"  {radar_name}: 文件不存在")
                continue
            
            pc = RadarPointCloud.from_file(data_path)
            points = pc.points  # (18, N)
            
            if points.shape[1] == 0:
                continue
            
            print(f"  {radar_name}: {points.shape[1]} 个点")
            
            # 转换到LIDAR坐标
            if 'sensor2lidar_rotation' in radar_info:
                rot = np.array(radar_info['sensor2lidar_rotation']).reshape(3, 3)
                trans = np.array(radar_info['sensor2lidar_translation']).reshape(3, 1)
                xyz_lidar = rot @ points[:3, :] + trans
                all_xyz_lidar.append(xyz_lidar)
                all_rcs.append(points[5, :])
    
    if len(all_xyz_lidar) == 0:
        print("没有有效的雷达点!")
        return
    
    all_xyz_lidar = np.concatenate(all_xyz_lidar, axis=1)
    all_rcs = np.concatenate(all_rcs, axis=0)
    print(f"\n总雷达点: {all_xyz_lidar.shape[1]}")
    print(f"LIDAR范围: x[{all_xyz_lidar[0].min():.1f}, {all_xyz_lidar[0].max():.1f}] "
          f"y[{all_xyz_lidar[1].min():.1f}, {all_xyz_lidar[1].max():.1f}] "
          f"z[{all_xyz_lidar[2].min():.1f}, {all_xyz_lidar[2].max():.1f}]")
    
    # 投影到各个相机图像
    print("\n投影到相机图像:")
    for cam in camera_types:
        if cam not in info['cams']:
            continue
        
        cam_info = info['cams'][cam]
        img_path = cam_info['data_path']
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 投影
        uv, depth, mask = project_lidar_to_image(all_xyz_lidar, cam_info)
        valid_pts = mask & (uv[0] >= 0) & (uv[0] < img.shape[1]) & \
                              (uv[1] >= 0) & (uv[1] < img.shape[0])
        
        print(f"  {cam}: {valid_pts.sum().astype(int)} 个点投影到图像上")
        
        # 在图像上画点
        vis = img.copy()
        if valid_pts.any():
            u = uv[0, valid_pts].astype(int)
            v = uv[1, valid_pts].astype(int)
            depths = depth[valid_pts]
            rcs = all_rcs[valid_pts]
            
            # 按距离着色 (近红远蓝)
            for i in range(len(u)):
                d = depths[i]
                # 颜色: 近(0-20m)=红, 中(20-50m)=黄绿, 远(>50m)=蓝
                if d < 20:
                    color = (0, 0, 255)  # 红
                elif d < 50:
                    t = (d - 20) / 30
                    color = (int(t * 255), int((1-t) * 255), 255 - int(t * 100))
                else:
                    color = (255, 128, 0)  # 蓝绿
                cv2.circle(vis, (u[i], v[i]), 3, color, -1)
        
        # 保存到文件
        out_dir = 'radar_vis'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'sample{sample_idx}_{cam}_radar_proj.jpg')
        cv2.imwrite(out_path, vis)
        print(f"  已保存: {out_path}")
    
    # 生成鸟瞰图
    bev_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # 范围: x[-50, 50], y[-50, 50]
    for i in range(all_xyz_lidar.shape[1]):
        x = all_xyz_lidar[0, i]
        y = all_xyz_lidar[1, i]
        if abs(x) > 50 or abs(y) > 50:
            continue
        px = int((x + 50) / 100 * 1000)
        py = int((50 - y) / 100 * 1000)  # y轴翻转
        px = np.clip(px, 0, 999)
        py = np.clip(py, 0, 999)
        rcs_val = all_rcs[i]
        # 颜色: RCS(-20~+30dBsm)
        intensity = np.clip((rcs_val + 20) / 50, 0, 1)
        color = (0, int(255 * intensity), int(255 * (1-intensity)))
        cv2.circle(bev_img, (px, py), 2, color, -1)
    
    # 画GT框
    if 'gt_boxes' in info:
        for i in range(len(info['gt_boxes'])):
            box = info['gt_boxes'][i]
            cx, cy = box[0], box[1]
            if abs(cx) > 50 or abs(cy) > 50:
                continue
            px = int((cx + 50) / 100 * 1000)
            py = int((50 - cy) / 100 * 1000)
            wl, hl = box[3], box[4]
            pw = int(wl / 100 * 1000)
            ph = int(hl / 100 * 1000)
            cv2.rectangle(bev_img, 
                         (px - pw//2, py - ph//2),
                         (px + pw//2, py + ph//2), 
                         (0, 255, 255), 1)
    
    
    bev_path = os.path.join(out_dir, f'sample{sample_idx}_bev.jpg')
    cv2.imwrite(bev_path, bev_img)
    print(f"\nBEV鸟瞰图已保存: {bev_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--set', type=str, default='train')
    args = parser.parse_args()
    
    visualize(args.sample_idx, args.set)