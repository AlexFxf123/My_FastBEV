import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv import Config
from mmdet3d.models import build_model

cfg_path = 'configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'
cfg = Config.fromfile(cfg_path)
model = build_model(cfg['model'])
model.eval()

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_output.txt')
with open(out_path, 'w') as f:
    f.write(str(model))
print(f'模型结构已保存至 {out_path}')
