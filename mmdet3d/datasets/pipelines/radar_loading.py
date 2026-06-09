# -*- coding: utf-8 -*-
"""
加载nuScenes毫米波雷达点云

将多个雷达（RADAR_FRONT, RADAR_FRONT_RIGHT, RADAR_FRONT_LEFT,
RADAR_BACK, RADAR_BACK_LEFT, RADAR_BACK_RIGHT）的.pcd文件加载
并转换到LIDAR坐标系下，合并为一个点云。
"""
import numpy as np
import struct
import torch

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type


def load_pcd_file(filepath):
    """手动加载.pcd文件，不依赖nuscenes-devkit（兼容DataLoader多进程）

    支持的PCD格式：
        - 二进制 (DATA ascii 或 binary)

    Returns:
        points: (N, 18) numpy array with nuScenes radar fields
    """
    with open(filepath, 'rb') as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                break
            line = line.decode('utf-8').strip()
            if line.startswith('#'):
                continue
            if line == 'DATA ascii':
                data_format = 'ascii'
                break
            if line == 'DATA binary':
                data_format = 'binary'
                break
            if ' ' in line:
                key, value = line.split(' ', 1)
                header[key] = value

        n_points = int(header.get('POINTS', 0))
        if n_points == 0:
            return np.zeros((0, 18), dtype=np.float32)

        if data_format == 'binary':
            # 读取二进制数据
            # nuScenes雷达点云是18个float32字段
            raw_data = f.read()
            points = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 18)
        else:
            # ascii格式
            lines = f.read().decode('utf-8').strip().split('\n')
            points = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(-1, 18)
    return points


@PIPELINES.register_module()
class LoadRadarPointsFromFile(object):
    """加载并合并多个nuScenes雷达点云

    从info中的radars字段读取所有雷达的数据路径和位姿信息，
    将每个雷达点云转换到LIDAR坐标系下，合并为一个点云。

    雷达点云字段:
        [0]: x, [1]: y, [2]: z
        [3]: dyn_prop (动态属性)
        [4]: id (目标ID)
        [5]: rcs (雷达散射截面积, dBsm)
        [6]: vx (多普勒速度x, m/s)
        [7]: vy (多普勒速度y, m/s)
        [8]: vx_comp (补偿速度x)
        [9]: vy_comp (补偿速度y)
        [10]: is_quality_valid
        [11]: ambig_state (模糊状态)
        [12]: x_rms, [13]: y_rms, [14]: z_rms (位置精度)
        [15]: vx_rms, [16]: vy_rms (速度精度)
        [17]: pdh0, [18]: invalid_state

    Args:
        use_dim (list[int]): 使用的维度。默认使用 [0,1,2,3,5,6,7]
            (x,y,z,dyn_prop,rcs,vx,vy)
        load_dim (int): 加载的维度数（RadarPointCloud默认18维）
        coord_type (str): 坐标类型
        max_points (int): 最大点数，超过则随机采样
        min_points (int): 最小点数，少于则复制填充
    """

    def __init__(self,
                 use_dim=[0, 1, 2, 3, 5, 6, 7],
                 load_dim=18,
                 coord_type='LIDAR',
                 max_points=None,
                 min_points=None):
        self.use_dim = use_dim
        self.load_dim = load_dim
        self.coord_type = coord_type
        self.max_points = max_points
        self.min_points = min_points

    def __call__(self, results):
        """加载并合并所有雷达点云

        期望 results['radars'] 是一个dict:
            {'RADAR_FRONT': {'data_path': ..., 'sensor2lidar_rotation': ...,
                             'sensor2lidar_translation': ...},
             ...}

        Returns:
            results: 包含 'radar_points' 字段
        """
        radars = results.get('radars', {})

        all_points = []
        if radars:
            for radar_name, radar_info in radars.items():
                data_path = radar_info['data_path']

                # 加载.pcd文件 (使用内联函数以避免DataLoader多进程问题)
                points = load_pcd_file(data_path)  # (N, 18)

                if len(points) == 0:
                    continue

                # 转换到LIDAR坐标系
                sensor2lidar_rotation = np.array(radar_info['sensor2lidar_rotation']).reshape(3, 3)
                sensor2lidar_translation = np.array(radar_info['sensor2lidar_translation']).reshape(1, 3)

                # 应用旋转
                points_xyz = points[:, :3] @ sensor2lidar_rotation.T
                # 应用平移
                points_xyz += sensor2lidar_translation

                # 替换坐标
                points[:, :3] = points_xyz

                # 提取需要的维度
                if max(self.use_dim) < points.shape[1]:
                    points = points[:, self.use_dim]

            all_points.append(points)

        if len(all_points) == 0:
            # 没有有效点，生成虚拟点云（仍写入points字段避免后续transform报错）
            dummy_points = np.zeros((1, len(self.use_dim)), dtype=np.float32)
            points_class = get_points_type(self.coord_type)
            results['points'] = points_class(
                dummy_points, points_dim=dummy_points.shape[-1], attribute_dims=None)
            results['radar_points'] = dummy_points
            return results

        # 合并所有雷达的点
        merged_points = np.concatenate(all_points, axis=0).astype(np.float32)

        # 点云数量控制
        if self.max_points is not None and merged_points.shape[0] > self.max_points:
            indices = np.random.choice(merged_points.shape[0], self.max_points, replace=False)
            merged_points = merged_points[indices]

        if self.min_points is not None and merged_points.shape[0] < self.min_points:
            n_repeat = (self.min_points // merged_points.shape[0]) + 1
            merged_points = np.tile(merged_points, (n_repeat, 1))[:self.min_points]

        # 同时也写入points字段（转换为BasePoints），让现有的3D增强（RandomFlip3D、GlobalRotScaleTrans）处理
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            merged_points, points_dim=merged_points.shape[-1], attribute_dims=None)
        results['radar_points'] = merged_points
        return results


@PIPELINES.register_module()
class CollectRadarPoints(object):
    """收集雷达点云到data字典"""

    def __init__(self):
        pass

    def __call__(self, results):
        if 'radar_points' in results:
            results['radar_points'] = torch.from_numpy(results['radar_points']).float()
        return results