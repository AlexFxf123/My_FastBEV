# -*- coding: utf-8 -*-
"""
为现有的nuscenes info文件添加雷达点云路径信息

基于已有的 nuscenes_infos_*_4d_interval3_max60.pkl 文件，
为每个info添加 radars 字段，包含6个雷达的路径和位姿信息。

用法:
    python tools/data_converter/add_radar_info.py
"""
import pickle
import numpy as np
import os
from nuscenes import NuScenes
from pyquaternion import Quaternion


def add_radar_info():
    """为已有的info文件添加雷达路径信息"""
    dataroot = '/home/radardepth/data/nuscenes/'
    
    # 处理train和val
    for data_set in ['train', 'val']:
        # 输入文件: 已有的4D时序info
        input_path = os.path.join(
            dataroot, f'nuscenes_infos_{data_set}_4d_interval3_max60.pkl')
        
        if not os.path.exists(input_path):
            print(f'{input_path} 不存在，跳过')
            continue
            
        print(f'\n处理 {data_set} 集...')
        print(f'输入: {input_path}')
        
        # 加载已有info
        with open(input_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f'加载了 {len(dataset["infos"])} 个样本')
        
        # 初始化nuscenes (使用mini版，因为只安装了mini)
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
        
        radar_cameras = [
            'RADAR_FRONT',
            'RADAR_FRONT_RIGHT',
            'RADAR_FRONT_LEFT',
            'RADAR_BACK',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT',
        ]
        
        # 处理每个info
        for id, info in enumerate(dataset['infos']):
            if id % 100 == 0:
                print(f'  处理中: {id}/{len(dataset["infos"])}')
            
            sample_token = info['token']
            sample = nusc.get('sample', sample_token)
            
            radars = {}
            for radar_cam in radar_cameras:
                if radar_cam not in sample['data']:
                    continue
                    
                radar_token = sample['data'][radar_cam]
                sd_rec = nusc.get('sample_data', radar_token)
                cs_rec = nusc.get('calibrated_sensor',
                                 sd_rec['calibrated_sensor_token'])
                pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                
                data_path = str(nusc.get_sample_data_path(radar_token))
                
                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                
                # 计算sensor到lidar的变换
                l2e_r_s_mat = Quaternion(cs_rec['rotation']).rotation_matrix
                e2g_r_s_mat = Quaternion(pose_rec['rotation']).rotation_matrix
                e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
                
                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (np.array(cs_rec['translation']) @ e2g_r_s_mat.T + np.array(pose_rec['translation'])) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= np.array(info['ego2global_translation']) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                ) + np.array(l2e_t) @ np.linalg.inv(l2e_r_mat).T
                
                radars[radar_cam] = {
                    'data_path': data_path,
                    'sample_data_token': sd_rec['token'],
                    'sensor2lidar_rotation': R.T.tolist(),
                    'sensor2lidar_translation': T.tolist(),
                    'timestamp': sd_rec['timestamp'],
                    'sensor_modality': sd_rec['sensor_modality'],
                }
            
            dataset['infos'][id]['radars'] = radars
        
        # 保存
        output_path = os.path.join(
            dataroot, f'nuscenes_infos_{data_set}_4d_interval3_max60_wradar.pkl')
        print(f'保存到: {output_path}')
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f'{data_set} 集处理完成!')


def check_radar_info():
    """检查添加的雷达信息"""
    dataroot = '/home/radardepth/data/nuscenes/'
    
    for data_set in ['train', 'val']:
        input_path = os.path.join(
            dataroot, f'nuscenes_infos_{data_set}_4d_interval3_max60_wradar.pkl')
        
        if not os.path.exists(input_path):
            print(f'{input_path} 不存在')
            continue
            
        with open(input_path, 'rb') as f:
            dataset = pickle.load(f)
        
        info = dataset['infos'][0]
        print(f'\n{data_set} 集检查:')
        print(f'  info中有 radars: {"radars" in info}')
        if 'radars' in info:
            print(f'  雷达数量: {len(info["radars"])}')
            for radar_name, radar_info in info['radars'].items():
                print(f'    {radar_name}: {radar_info["data_path"]}')
                print(f'      点数文件存在: {os.path.exists(radar_info["data_path"])}')


if __name__ == '__main__':
    add_radar_info()
    check_radar_info()