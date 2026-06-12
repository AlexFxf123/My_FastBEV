# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from os import path as osp

mmdet3d_root = os.environ.get('MMDET3D')
if mmdet3d_root is not None and osp.exists(mmdet3d_root):
    import sys
    sys.path.insert(0, mmdet3d_root)
    print(f"using mmdet3d: {mmdet3d_root}")

from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import  set_random_seed
from mmdet.datasets import replace_ImageToTensor
from IPython import embed
import ipdb


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        '--vis',
        action='store_true',
        help='whether vis')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='whether debug')
    parser.add_argument('--debug_num', type=int, default=50)
    parser.add_argument('--extrinsic-noise', '-n', type=float, default=0)

    # =================================================================
    # ★ INSERT-1 START：注册 --save-dir（必须放 return args 之前）
    # =================================================================
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='REQUIRED in practice: root folder for ALL outputs '
             '(pkl / format submission / eval tmp). '
             'If not set, falls back to cfg.work_dir or ./output_eval.')
    # =================================================================
    # ★ INSERT-1 END
    # =================================================================

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options

    # =================================================================
    # ★ 路径规范化（放 return 之前/之后都行；这里放 return 前最清晰）
    # =================================================================
    # ① 锁 save_dir
    if args.save_dir is not None:
        args.save_dir = osp.abspath(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        # fallback：先试着从 cfg.work_dir，再兜底
        _wd = getattr(args, '_work_dir_cache', None)
        if not _wd and 'cfg' in dir():
            pass  # 下面 main() 里再拿 cfg.work_dir
        if _wd is None:
            _wd = './output_eval'
        args.save_dir = osp.abspath(_wd)
        os.makedirs(args.save_dir, exist_ok=True)

    # ② --out：如果没给，自动落 save_dir；如果给了，保证父目录存在
    if args.out is not None:
        args.out = osp.abspath(args.out)
        os.makedirs(osp.dirname(args.out), exist_ok=True)
    else:
        args.out = osp.join(args.save_dir, 'results.pkl')
        # 先别 mkdir 这里也行，rank==0 再建；但为了安全：
        os.makedirs(args.save_dir, exist_ok=True)

    # ③ --tmpdir：拽出 /tmp，防 OOM-killer
    if args.tmpdir is None:
        args.tmpdir = osp.join(args.save_dir, 'eval_tmp')
    args.tmpdir = osp.abspath(args.tmpdir)
    os.makedirs(args.tmpdir, exist_ok=True)

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    if args.vis:
        nms_thr = 0.0001
        try:
            cfg.model.test_cfg.nms_thr = nms_thr
        except:
            print('### imvoxelnet except in train.py ###')
            cfg.test_cfg.nms_thr = nms_thr

    if args.extrinsic_noise > 0:
        for i in range(3):
            print('### test camera extrinsic robustness ###')
        cfg.model.extrinsic_noise = args.extrinsic_noise

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # =================================================================
    # ★ INSERT-2 START：把 eval tmp 拽出 /tmp + 保证 save_dir 一定存在
    # （在 distributed 判断之后，推理之前/之后都行；这里放推理前最稳）
    # =================================================================
    # 如果 save-dir 仍没初始化（例如 fallback 走 cfg.work_dir 的情况）：
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        _wd = cfg.work_dir if hasattr(cfg, 'work_dir') and cfg.work_dir else './output_eval'
        args.save_dir = osp.abspath(_wd)
        os.makedirs(args.save_dir, exist_ok=True)

    # 把 show-dir 也收进 save-dir
    if args.show_dir is None:
        args.show_dir = osp.join(args.save_dir, 'vis')
    args.show_dir = osp.abspath(args.show_dir)
    os.makedirs(args.show_dir, exist_ok=True)

    # tmpdir 二次保险（防 format_results / nusc_eval 暗开 /tmp）
    if not hasattr(args, 'tmpdir') or args.tmpdir is None:
        args.tmpdir = osp.join(args.save_dir, 'eval_tmp')
    args.tmpdir = osp.abspath(args.tmpdir)
    os.makedirs(args.tmpdir, exist_ok=True)
    # =================================================================
    # ★ INSERT-2 END
    # =================================================================

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # 你原始 single_gpu_test 原型是 4 个实参：(model, data_loader, show, show_dir)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  debug=args.debug)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, debug=args.debug,
                                 debug_num=args.debug_num)

    rank, _ = get_dist_info()
    if rank == 0:
        # outputs = mmcv.load(args.out)
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        kwargs = {} if args.eval_options is None else args.eval_options

        if args.format_only:
            # 输出到工程目录下的 test_results/
            _out_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'test_results')
            _fmt_root = osp.join(_out_dir, 'results')
            os.makedirs(_out_dir, exist_ok=True)

            kwargs['jsonfile_prefix'] = _fmt_root

            print(f'[SAVE-DIR] format_results → {_out_dir}')
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval','tmpdir','start','gpu_collect','save_best','rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(f'[SAVE-DIR] eval tmp = {args.tmpdir}')
            print(dataset.evaluate(outputs, vis_mode=args.vis, **eval_kwargs))


if __name__ == '__main__':
    main()