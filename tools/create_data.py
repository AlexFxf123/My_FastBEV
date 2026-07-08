# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
from functools import partial

from tools.data_converter import nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database


def _norm_ver(version: str) -> str:
    v = (version or '').strip()
    if v in ('', 'v1.0'):
        return 'v1.0-trainval'
    if v == 'trainval':
        return 'v1.0-trainval'
    if v == 'test':
        return 'v1.0-test'
    if v == 'mini':
        return 'v1.0-mini'
    if v in {'v1.0-trainval', 'v1.0-test', 'v1.0-mini'}:
        return v
    raise ValueError(
        f'Unsupported --version {v!r}. '
        f'Use v1.0-trainval / v1.0-test / v1.0-mini'
    )


def nuscenes_data_prep(root_path, info_prefix, version, out_dir, max_sweeps=10):
    """
    root_path: 必须是包含 v1.0-trainval/ 的那层
      e.g. /home/radardepth/data/nuscenes
    """

    print(f'[nuscenes_data_prep] version={version}, root={root_path}')

    # 1) 生成 infos pkl（你的 trainval01 已在 nuscenes_converter 里按 exist scene 切分）
    nuscenes_converter.create_nuscenes_infos(
        root_path=root_path,
        info_prefix=info_prefix,
        version=version,
        max_sweeps=max_sweeps,
    )

    # 2) export_2d_annotation / create_groundtruth_database
    #    对 FastBEV 多数配置并不需要，且容易触发二次内存峰值 → 先可跳过
    SKIP_2D_AND_GTDB = True  # ← 设为 False 仅在你需要它们时再开

    if version == 'v1.0-test':
        if not SKIP_2D_AND_GTDB:
            info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
            nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        return

    if not SKIP_2D_AND_GTDB:
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path   = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val_path,   version=version)
        create_groundtruth_database(
            'NuScenesDataset',
            root_path,
            info_prefix,
            f'{out_dir}/{info_prefix}_infos_train.pkl',
        )
    else:
        print('[INFO] Skip export_2d_annotation & create_groundtruth_database '
              '(set SKIP_2D_AND_GTDB=False if your pipeline really needs them).')


def main():
    parser = argparse.ArgumentParser(description='Data converter')
    parser.add_argument(
        'dataset',
        choices=['nuscenes', 'kitti', 'waymo', 'lyft', 'scannet', 's3dis', 'sunrgbd'],
        help='dataset name'
    )
    parser.add_argument(
        '--root-path',
        type=str,
        default='/home/radardepth/data/nuscenes',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        help='v1.0-trainval | v1.0-test | v1.0-mini'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='where to write pkls (default same as root-path)'
    )
    parser.add_argument('--extra-tag', type=str, default='nuscenes')
    parser.add_argument('--max-sweeps', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.root_path

    if args.dataset == 'nuscenes':
        ver = _norm_ver(args.version)

        # 完整版：trainval 然后 test；你只有 trainval01 就先跑 trainval 也完全 OK
        if ver in ('v1.0-trainval',):
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version='v1.0-trainval',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
            )
            # 如果你之后补全 v1.0-test 数据，再单独加一步：
            # nuscenes_data_prep(..., version='v1.0-test', ...)

        elif ver == 'v1.0-mini':
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version='v1.0-mini',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
            )
        else:
            raise ValueError(f'Unhandled ver={ver}')
    else:
        raise NotImplementedError(
            f'dataset={args.dataset} branch not refactored yet.'
        )


if __name__ == '__main__':
    main()