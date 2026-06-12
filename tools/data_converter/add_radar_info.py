# -*- coding: utf-8 -*-
"""
为 nuscenes_infos_*_4d_interval3_max60.pkl 的每个 info 添加 radars 字段
（radar -> lidar 外参 + 相对路径）

依赖:
  pip install nuscenes-devkit pyquaternion

目录约定:
  dataroot/
    samples/RADAR_FRONT/...
    samples/RADAR_FRONT_LEFT/...
    ...
    sweeps/RADAR_FRONT/...
    nuscenes_infos_train_4d_interval3_max60.pkl
    nuscenes_infos_val_4d_interval3_max60.pkl
"""

import os
import pickle
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

# ---------- 配置 ----------
DATAROOT = '/home/radardepth/data/nuscenes/'

# nuScenes 只有 5 个雷达通道（FIX: 去掉不存在的 RADAR_BACK）
RADAR_CHANNELS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]

# 你要处理的 info pkl 名（不含路径），按你的命名来
INFO_PKL_NAMES = {
    'train': 'nuscenes_infos_train_4d_interval3_max60.pkl',
    'val':   'nuscenes_infos_val_4d_interval3_max60.pkl',
}

# 输出后缀
OUT_SUFFIX = '_wradar.pkl'

# 你 info 里存的 lidar2ego_rotation 是什么格式？
#  pyquaternion Quaternion 默认构造用 (w, x, y, z)
#  如果你的 info 也是 (w,x,y,z) list/tuple -> 设为 'wxyz'
#  极少情况下有人存成 (x,y,z,w) -> 改成 'xyzw'
LIDAR_QUAT_ORDER = 'wxyz'
# --------------------------


def _quat_from_info(q, order='wxyz'):
    """把 info 里的 lidar2ego_rotation 变成 pyquaternion.Quaternion"""
    if isinstance(q, Quaternion):
        return q
    q = list(q)
    if order == 'wxyz':
        return Quaternion(q[0], q[1], q[2], q[3])
    else:
        return Quaternion(q[3], q[0], q[1], q[2])


def _rt_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def add_radar_info():
    # 只初始化一次（SAFER）
    print('Init NuScenes ...')
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=DATAROOT,
        verbose=True
    )

    for split in ['train', 'val']:
        in_name = INFO_PKL_NAMES.get(split)
        if in_name is None:
            print(f'[WARN] skip split={split}: no pkl name mapped')
            continue

        in_path = os.path.join(DATAROOT, in_name)
        if not os.path.exists(in_path):
            print(f'[SKIP] not exist: {in_path}')
            continue

        out_name = in_name.replace('.pkl', OUT_SUFFIX)
        if not out_name.endswith('.pkl'):
            out_name = in_name.replace(
                '_4d_interval3_max60.pkl',
                '_4d_interval3_max60_wradar.pkl'
            )
        out_path = os.path.join(DATAROOT, out_name)

        print('\n========================================')
        print(f'[{split}]')
        print(f'  input:  {in_path}')
        print(f'  output: {out_path}')

        with open(in_path, 'rb') as f:
            dataset = pickle.load(f)

        infos = dataset['infos']
        print(f'  total infos: {len(infos)}')

        for idx, info in enumerate(infos):
            if idx % 100 == 0:
                print(f'  [{split}] progress {idx}/{len(infos)}')

            tok = info['token']
            try:
                sample = nusc.get('sample', tok)
            except Exception as e:
                # 如果 token 不对应 sample（理论上不应发生），保底留空
                print(f'[WARN] sample not found for token={tok}: {e}')
                info['radars'] = {}
                continue

            # -------- lidar2ego (来自 info) --------
            # 这里用 double 做矩阵运算，最后再转 float32 存
            q_l2e = _quat_from_info(info['lidar2ego_rotation'], order=LIDAR_QUAT_ORDER)
            R_l2e = q_l2e.rotation_matrix.astype(np.float64)          # lidar->ego
            t_l2e = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
            T_lidar2ego = _rt_to_T(R_l2e, t_l2e)                      # lidar->ego (4x4)

            # 获取 LIDAR_TOP 的 ego_pose（用于 lidar global 对齐）
            lidar_token = sample['data']['LIDAR_TOP']
            sd_lidar = nusc.get('sample_data', lidar_token)
            cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
            pose_lidar = nusc.get('ego_pose', sd_lidar['ego_pose_token'])

            # lidar sensor -> lidar ego
            R_lse = Quaternion(cs_lidar['rotation']).rotation_matrix.astype(np.float64)
            t_lse = np.asarray(cs_lidar['translation'], dtype=np.float64)
            # lidar ego -> global
            R_leg = Quaternion(pose_lidar['rotation']).rotation_matrix.astype(np.float64)
            t_leg = np.asarray(pose_lidar['translation'], dtype=np.float64)

            # 构建 lidar 的 4x4：sensor -> ego -> global
            T_lidar_s2e = _rt_to_T(R_lse, t_lse)
            T_lidar_e2g = _rt_to_T(R_leg, t_leg)
            T_lidar_s2g = T_lidar_e2g @ T_lidar_s2e  # lidar sensor -> global

            radars = {}

            for ch in RADAR_CHANNELS:
                if ch not in sample['data']:
                    continue

                radar_token = sample['data'][ch]
                sd = nusc.get('sample_data', radar_token)
                cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                pose = nusc.get('ego_pose', sd['ego_pose_token'])

                # ---- 路径：存相对路径 ----
                data_relpath = str(sd['filename'])

                # ---- 雷达 sensor -> radar ego -> radar global ----
                R_s2e = Quaternion(cs['rotation']).rotation_matrix.astype(np.float64)
                t_s2e = np.asarray(cs['translation'], dtype=np.float64)
                T_radar_s2e = _rt_to_T(R_s2e, t_s2e)

                R_e2g = Quaternion(pose['rotation']).rotation_matrix.astype(np.float64)
                t_e2g = np.asarray(pose['translation'], dtype=np.float64)
                T_radar_e2g = _rt_to_T(R_e2g, t_e2g)

                T_radar_s2g = T_radar_e2g @ T_radar_s2e  # radar sensor -> global

                # ---- sensor -> lidar: 先都转到 global 再相减 ----
                # T_sensor2lidar = inv(T_lidar_s2g) @ T_radar_s2g
                T_sl = np.linalg.inv(T_lidar_s2g) @ T_radar_s2g

                R_out = T_sl[:3, :3].astype(np.float32)
                t_out = T_sl[:3, 3].astype(np.float32)

                radars[ch] = {
                    'data_path': data_relpath,
                    'sample_data_token': sd['token'],
                    'sensor2lidar_rotation': R_out.tolist(),
                    'sensor2lidar_translation': t_out.tolist(),
                    'timestamp': sd['timestamp'],
                    'sensor_modality': sd['sensor_modality'],
                }

            info['radars'] = radars

        with open(out_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f'[{split}] done -> {out_path}')


def check_radar_info():
    """检查：相对路径能否拼回真实文件 + 看一眼外参是否合理"""
    for split in ['train', 'val']:
        in_name = INFO_PKL_NAMES.get(split)
        if in_name is None:
            continue
        # 试着找 _wradar 输出
        out_name = in_name.replace('.pkl', OUT_SUFFIX)
        if not out_name.endswith('.pkl'):
            out_name = in_name.replace(
                '_4d_interval3_max60.pkl',
                '_4d_interval3_max60_wradar.pkl'
            )
        p = os.path.join(DATAROOT, out_name)
        if not os.path.exists(p):
            print(f'[CHECK SKIP] {p} not exist')
            continue

        with open(p, 'rb') as f:
            dataset = pickle.load(f)

        info = dataset['infos'][0]
        print(f'\n=== {split} check ===')
        print('has radars:', 'radars' in info)
        if not info.get('radars'):
            print('  radars empty!')
            continue

        print('  num radars:', len(info['radars']))
        for ch, r in info['radars'].items():
            real = os.path.join(DATAROOT, r['data_path'])
            ok = os.path.exists(real)
            print(f'  [{ch}]')
            print(f'    relpath : {r["data_path"]}')
            print(f'    exists  : {ok}  (abs guess: {real}')
            R = np.asarray(r['sensor2lidar_rotation'], dtype=np.float32)
            t = np.asarray(r['sensor2lidar_translation'], dtype=np.float32)
            det = np.linalg.det(R)
            print(f'    R det  : {det:.6f}  (should ~1.0)')
            print(f'    t[:3]  : {t}')


if __name__ == '__main__':
    add_radar_info()
    check_radar_info()