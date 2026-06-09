# mmdet3d/utils/progressbar_hook.py
import logging
from tqdm import tqdm
from mmcv.runner import HOOKS, LoggerHook


@HOOKS.register_module()
class ProgressBarLoggerHook(LoggerHook):
    """Custom LoggerHook that shows a tqdm progress bar."""

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.pbar = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.iters_per_epoch = 0

    def before_run(self, runner):
        super().before_run(runner)
        self.total_epochs = runner.max_epochs

    def before_epoch(self, runner):
        self.current_epoch = runner.epoch + 1
        # 计算每 epoch 的迭代数（取数据加载器中最小的长度）
        self.iters_per_epoch = len(runner.data_loader)
        
        if self.pbar is not None:
            self.pbar.close()
        
        self.pbar = tqdm(
            total=self.iters_per_epoch,
            desc=f'Epoch [{self.current_epoch}/{self.total_epochs}]',
            position=0,
            leave=True,
            bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.pbar.set_postfix_str('starting...')

    def after_epoch(self, runner):
        if self.pbar is not None:
            self.pbar.update(self.iters_per_epoch - self.pbar.n)
            self.pbar.close()
            self.pbar = None

    def log(self, runner):
        if self.pbar is None:
            return
        
        # 更新进度条
        self.pbar.update(self.every_n_interval)
        
        # 获取关键指标
        tags = self.get_loggable_tags(runner)
        if tags:
            # 过滤并格式化指标
            metrics = {}
            for k, v in tags.items():
                if isinstance(v, float):
                    metrics[k] = f'{v:.4f}'
                else:
                    metrics[k] = str(v)
            self.pbar.set_postfix(metrics)

    def after_run(self, runner):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None