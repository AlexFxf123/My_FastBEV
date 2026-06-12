import os
from mmcv.runner import HOOKS, Hook
from .radar_loading import save_visual_debug


@HOOKS.register_module()
class VisualDebugHook(Hook):
    """
    训练过程中每 interval 次迭代保存一组可视化结果（6相机+GT投影、BEV雷达点云+GT框）
    通过配置文件 custom_hooks 启用
    """

    def __init__(self, interval=100, data_root='', enabled=True):
        self.interval = interval
        self.data_root = data_root
        self.enabled = enabled

    def before_run(self, runner):
        if not self.enabled:
            return
        self.out_dir = os.path.join(runner.work_dir, 'vis_debug')
        os.makedirs(self.out_dir, exist_ok=True)
        runner.logger.info(f'[VisualDebugHook] enabled, save to {self.out_dir}')

    def after_train_iter(self, runner):
        if not self.enabled:
            return
        if runner.iter == 0 or runner.iter % self.interval != 0:
            return

        try:
            # 获取当前 batch 的 sample_idx
            batch = runner.data_loader._data_batch
            if batch is None:
                return

            img_metas = batch.get('img_metas', None)
            if img_metas is None:
                return

            # 获取第一个样本的 sample_idx
            if hasattr(img_metas[0], '_data'):
                metas = img_metas[0]._data
            elif isinstance(img_metas[0], dict):
                metas = img_metas[0]
            else:
                return

            sample_idx = metas.get('sample_idx', None)
            if sample_idx is None:
                return

            dataset = runner.data_loader.dataset
            if hasattr(dataset, 'dataset'):  # CBGSDataset wrapper
                dataset = dataset.dataset

            if hasattr(dataset, 'data_infos') and sample_idx < len(dataset.data_infos):
                info = dataset.data_infos[sample_idx]
                tag = f'_iter{runner.iter}'
                save_visual_debug(info, self.data_root, self.out_dir, tag=tag)
                runner.logger.info(f'[VisualDebugHook] saved vis at iter {runner.iter}')
        except Exception as e:
            runner.logger.warning(f'[VisualDebugHook] error at iter {runner.iter}: {e}')
