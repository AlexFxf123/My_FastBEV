import os
import pickle
from mmcv.runner import HOOKS, Hook
from .radar_loading import save_visual_debug


@HOOKS.register_module()
class VisualDebugHook(Hook):
    """
    训练过程中每 interval 次迭代保存一组可视化结果。
    直接从数据集的 data_infos 中取对应索引的数据。
    """

    def __init__(self, interval=100, data_root='', enabled=True):
        self.interval = interval
        self.data_root = data_root
        self.enabled = enabled
        self._count = 0

    def before_run(self, runner):
        if not self.enabled:
            return
        self.out_dir = os.path.join(runner.work_dir, 'vis_debug')
        os.makedirs(self.out_dir, exist_ok=True)
        runner.logger.info(f'[VisualDebugHook] enabled, save to {self.out_dir}')

    def after_train_iter(self, runner):
        if not self.enabled:
            return
        if runner.iter == 0:
            return
        if runner.iter % self.interval != 0:
            return

        try:
            # 从数据集中取 info
            dataset = runner.data_loader.dataset
            # CBGSDataset wrapper
            if hasattr(dataset, 'dataset'):
                dataset = dataset.dataset

            data_infos = getattr(dataset, 'data_infos', [])
            if len(data_infos) == 0:
                runner.logger.warning('[VisualDebugHook] no data_infos')
                return

            # 取当前索引：用 runner.iter 取模
            idx = self._count % len(data_infos)
            self._count += 1
            info = data_infos[idx]

            tag = f'_iter{runner.iter}'
            save_visual_debug(info, self.data_root, self.out_dir, tag=tag)
            runner.logger.info(f'[VisualDebugHook] saved vis at iter {runner.iter} (idx={idx})')
        except Exception as e:
            import traceback
            runner.logger.warning(f'[VisualDebugHook] error at iter {runner.iter}: {e}\n{traceback.format_exc()}')
