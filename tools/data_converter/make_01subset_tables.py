#!/usr/bin/env python3
"""
从 v1.0-trainval 的 JSON 表中，只抽磁盘上存在 LIDAR_TOP 的 scene 子集，
输出到 v1.0-trainval-01subset/。

内存策略：
  - 用 NuScenes 仅收集 token_set（它在 __init__ 内会吃 ~7GB，但收集完立刻 del+gc）
  - 过滤 sample_data / annotation 时用 json 读入但尽快写盘、不做多余拷贝
  - annotation 表最大，用 "读一条判一条写一条" 的 writer 模式省峰值
"""

import os
import sys
import json
import gc

ROOT = '/home/radardepth/data/nuscenes'
SRC  = os.path.join(ROOT, 'v1.0-trainval')
DST  = os.path.join(ROOT, 'v1.0-trainval-01subset')

os.makedirs(DST, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# STEP 1：用 NuScenes 只收集 keep_scene_tokens + sample_token_set
#        收集完立刻 del nusc 并 gc.collect()，把那 ~7GB 还回去
# ───────────────────────────────────────────────────────────────
print('[1/5] NuScenes init (will eat ~7GB briefly)...')
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=ROOT, verbose=True)

keep_scene_tokens = set()
missing = 0
for s in nusc.scene:
    try:
        sample = nusc.get('sample', s['first_sample_token'])
        sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        if os.path.exists(os.path.join(ROOT, sd['filename'])):
            keep_scene_tokens.add(s['token'])
        else:
            missing += 1
    except Exception:
        missing += 1

print(f'keep scenes: {len(keep_scene_tokens)} / {len(nusc.scene)}  (missing_on_disk={missing})')
assert keep_scene_tokens, 'keep_scene_tokens 为空！检查 blobs 是否解压到 samples/LIDAR_TOP/'

# 串起每个 keep scene 的 sample token 链表
sample_token_set = set()
for s in nusc.scene:
    if s['token'] not in keep_scene_tokens:
        continue
    tok = s['first_sample_token']
    while tok and tok != '':
        sample_token_set.add(tok)
        st = nusc.get('sample', tok)
        tok = st['next'] if st['next'] else ''
print(f'sample_token_set size: {len(sample_token_set)}')

# ★★★ 关键：把结果序列化到 tmp，然后毁掉 nusc 把内存还回去 ★★★
_tmp_token_pkl = os.path.join(DST, '._tmp_sample_token_set.json')
with open(_tmp_token_pkl, 'w') as f:
    json.dump({'keep_scene_tokens': list(keep_scene_tokens),
               'sample_token_set': list(sample_token_set)}, f)

print('Releasing NuScenes tables from memory...')
del nusc
gc.collect()
print('gc done.')

# ───────────────────────────────────────────────────────────────
# STEP 2：重新读 tmp，做 JSON 过滤（此时没有 devkit 的 ~7GB 表了）
# ───────────────────────────────────────────────────────────────
with open(_tmp_token_pkl, 'r') as f:
    _t = json.load(f)
keep_scene_tokens = set(_t['keep_scene_tokens'])
sample_token_set = set(_t['sample_token_set'])
os.remove(_tmp_token_pkl)

print(f'[2/5] Filtering scene.json ...')
scene_all = json.load(open(os.path.join(SRC, 'scene.json'), 'r'))
scenes_sub = [s for s in scene_all if s['token'] in keep_scene_tokens]
with open(os.path.join(DST, 'scene.json'), 'w') as f:
    json.dump(scenes_sub, f, indent=2)
print(f'  scene.json {len(scene_all)} -> {len(scenes_sub)}')

print(f'[3/5] Filtering sample.json ...')
sample_all = json.load(open(os.path.join(SRC, 'sample.json'), 'r'))
samples_sub = [s for s in sample_all if s['token'] in sample_token_set]
with open(os.path.join(DST, 'sample.json'), 'w') as f:
    json.dump(samples_sub, f, indent=2)
print(f'  sample.json {len(sample_all)} -> {len(samples_sub)}')

# ───────────────────────────────────────────────────────────────
# STEP 4：sample_data.json  — 最大表之一，过滤 + 尽早释放
# ───────────────────────────────────────────────────────────────
print(f'[4/5] Filtering sample_data.json ...')
sd_all = json.load(open(os.path.join(SRC, 'sample_data.json'), 'r'))
print(f'  loaded {len(sd_all)} entries, filtering ...')
sd_sub = [x for x in sd_all if x.get('sample_token') in sample_token_set]
del sd_all
gc.collect()

with open(os.path.join(DST, 'sample_data.json'), 'w') as f:
    json.dump(sd_sub, f, indent=2)
print(f'  -> {len(sd_sub)} entries kept')

# ───────────────────────────────────────────────────────────────
# STEP 5：sample_annotation.json — 最大表（116万条），用流式写法更安全
#         如果这里还炸，就启用 ijson 分支（见注释）
# ───────────────────────────────────────────────────────────────
print(f'[5/5] Filtering sample_annotation.json ...')

ann_path = os.path.join(SRC, 'sample_annotation.json')
dst_ann  = os.path.join(DST, 'sample_annotation.json')

# --- 快速试探：json.load 能不能扛住 ---
try:
    ann_all = json.load(open(ann_path, 'r'))
    ann_sub = [x for x in ann_all if x.get('sample_token') in sample_token_set]
    del ann_all
    gc.collect()
    with open(dst_ann, 'w') as f:
        json.dump(ann_sub, f, indent=2)
    print(f'  -> {len(ann_sub)} annotations kept (direct load)')
except MemoryError:
    print('  direct json.load OOM, falling back to streaming parse...')
    # 流式：sample_annotation.json = "[ {...}, {...}, ... ]"
    # 逐块读、遇到完整对象就 dumps 到输出数组
    import re
    _buf = ''
    _out = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        _buf = f.read(2)  # skip leading '[\n'
    with open(ann_path, 'r', encoding='utf-8') as f:
        f.read(2)  # skip [
        depth = 0
        cur = ''
        in_str = False
        esc = False
        written = 0
        skipped = 0
        for ch in f.read():
            if ch == '"' and not esc:
                in_str = not in_str
            if not in_str:
                if ch == '{': depth += 1
                if ch == '}' and depth == 1:
                    cur += '}'
                    obj = json.loads(cur)
                    if obj.get('sample_token') in sample_token_set:
                        _out.append(obj)
                        written += 1
                    else:
                        skipped += 1
                    cur = ''
                    depth = 0
                    # skip whitespace/commas/newline until next {
                    continue
            if depth >= 1:
                cur += ch
            esc = (ch == '\\' and not esc)
        with open(dst_ann, 'w') as fw:
            json.dump(_out, fw, indent=2)
        print(f'  -> stream parsed: written={written} skipped={skipped}')

# ───────────────────────────────────────────────────────────────
# 复制小表 + symlink 大数据目录
# ───────────────────────────────────────────────────────────────
print('Copying small tables...')
for fname in ['category.json', 'attribute.json', 'visibility.json',
              'instance.json', 'sensor.json', 'calibrated_sensor.json',
              'ego_pose.json', 'log.json', 'map.json']:
    src_f = os.path.join(SRC, fname)
    dst_f = os.path.join(DST, fname)
    if not os.path.exists(dst_f):
        import shutil
        shutil.copy2(src_f, dst_f)

print('Symlinking samples/ sweeps/ maps/ ...')
for dname in ['samples', 'sweeps', 'maps']:
    src_d = os.path.join(ROOT, dname)
    dst_ln = os.path.join(DST, dname)
    if os.path.islink(dst_ln):
        os.remove(dst_ln)
    if os.path.exists(dst_ln) and not os.path.islink(dst_ln):
        print(f'  WARNING: {dst_ln} exists and is not a symlink, skip')
        continue
    os.symlink(src_d, dst_ln)

print('\n✅  Done.')
print('  DST:', DST)
print('\nVerify with:')
print("  python -c \"from nuscenes import NuScenes; n=NuScenes('v1.0-trainval-01subset', '/home/radardepth/data/nuscenes'); print(len(n.scene))\"")