# Copyright (c) Phigent Robotics. All rights reserved.

import pickle
from nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import ipdb

def add_adj_info():
    import os

    interval = 3
    max_adj = 60
    sample_num = None

    base_dir = '/home/radardepth/data/nuscenes'

    # ============================================================
    # ★ 只 init 一次 NuScenes，不再每个 set 重新 Loading 全量表
    # ============================================================
    nuscenes_version = 'v1.0-trainval'
    dataroot = base_dir
    print('[INIT] NuScenes once, version=', nuscenes_version)
    nuscenes = NuScenes(nuscenes_version, dataroot, verbose=True)
    print('[INIT] Done. scenes:', len(nuscenes.scene))

    # 预建 token→id（跨 set 也用同一份 nuscenes 对象）
    # 先收集所有 pkl 里出现过的 token，但我们还不知道有哪些 pkl
    # → 所以改为：先决定要跑哪些 set，再统一用同一个 nuscenes 对象处理

    target_sets = ['val', 'train']
    # target_sets = ['train']   # 如果还炸，就一次只跑一个

    results = {}

    for set_ in target_sets:
        pkl_path = os.path.join(base_dir, f'nuscenes_infos_{set_}.pkl')
        if not os.path.exists(pkl_path):
            print(f'[SKIP] pkl not found: {pkl_path}')
            continue

        print(f'\n===== [{set_}] loading pkl: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)

        # ---- token → id ----
        map_token_to_id = {}
        total = len(dataset['infos'])
        for _id in range(total):
            map_token_to_id[dataset['infos'][_id]['token']] = _id
            if sample_num is not None and _id >= sample_num:
                break

        # ---- 主处理（你原来的逻辑，几乎原样搬） ----
        for _id in range(total):
            if _id % 10 == 0:
                print(f'[{set_}] {_id}/{total}')
            if sample_num is not None and _id >= sample_num:
                break

            info = dataset['infos'][_id]
            sample = nuscenes.get('sample', info['token'])

            # === next / prev sweeps ===
            for adj in ('next', 'prev'):
                sweeps = []
                adj_list = {}
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                    adj_list[cam] = []
                    sd = nuscenes.get('sample_data', sample['data'][cam])
                    count = 0
                    while count < max_adj:
                        if sd[adj] == '':
                            break
                        sd = nuscenes.get('sample_data', sd[adj])
                        adj_list[cam].append(dict(
                            data_path=os.path.join(base_dir, sd['filename']),
                            timestamp=sd['timestamp'],
                            ego_pose_token=sd['ego_pose_token'],
                        ))
                        count += 1

                front_len = len(adj_list['CAM_FRONT'])
                for cnt in range(interval - 1, min(max_adj, front_len), interval):
                    ts_front = adj_list['CAM_FRONT'][cnt]['timestamp']
                    pose_rec = nuscenes.get(
                        'ego_pose', adj_list['CAM_FRONT'][cnt]['ego_pose_token']
                    )
                    cam_infos = {
                        'CAM_FRONT': dict(
                            data_path=adj_list['CAM_FRONT'][cnt]['data_path']
                        )
                    }
                    for cam in ['CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                                 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                        ts_arr = np.array(
                            [t['timestamp'] for t in adj_list[cam]], dtype=np.long
                        )
                        sel = int(np.argmin(np.abs(ts_arr - ts_front)))
                        cam_infos[cam] = dict(
                            data_path=adj_list[cam][sel]['data_path']
                        )
                    sweeps.append(dict(
                        timestamp=ts_front,
                        cams=cam_infos,
                        ego2global_translation=pose_rec['translation'],
                        ego2global_rotation=pose_rec['rotation'],
                    ))

                info[adj] = sweeps if sweeps else None

            # === ego velocity ===
            prev_id = _id
            if sample['prev'] != '':
                s_prev = nuscenes.get('sample', sample['prev'])
                prev_id = map_token_to_id[s_prev['token']]

            next_id = _id
            if sample['next'] != '':
                s_next = nuscenes.get('sample', sample['next'])
                next_id = map_token_to_id[s_next['token']]

            t_prev = 1e-6 * dataset['infos'][prev_id]['timestamp']
            t_next = 1e-6 * dataset['infos'][next_id]['timestamp']
            dt = t_next - t_prev

            p_prev = np.array(dataset['infos'][prev_id]['ego2global_translation'], dtype=np.float32)
            p_next = np.array(dataset['infos'][next_id]['ego2global_translation'], dtype=np.float32)
            vel_global = (p_next - p_prev) / dt

            e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
            l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            vel_global_3 = np.array([vel_global[0], vel_global[1], 0.0], dtype=np.float32)
            vel_lidar = vel_global_3 @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            vel_lidar = vel_lidar[:2]

            info['velo'] = vel_lidar
            if set_ in ('train', 'val'):
                if 'gt_velocity' in info:
                    info['gt_velocity'] = info['gt_velocity'] - vel_lidar.reshape(1, 2)

        # ---- dump ----
        out_name = f'nuscenes_infos_{set_}_4d_interval{interval}_max{max_adj}.pkl'
        out_path = os.path.join(base_dir, out_name)
        if sample_num is not None:
            stem, ext = os.path.splitext(out_name)
            out_path = os.path.join(base_dir, f'{stem}_sample{sample_num}{ext}')

        print(f'[{set_}] writing {out_path}')
        with open(out_path, 'wb') as fid:
            pickle.dump(dataset, fid)
        print(f'[{set_}] DONE.')

    print('\n====== ALL DONE ====')

if __name__ == '__main__':
    add_adj_info()
