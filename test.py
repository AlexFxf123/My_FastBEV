import pickle
pkl = '/home/radardepth/data/nuscenes/nuscenes_infos_train_4d_interval3_max60_wradar.pkl'
with open(pkl, 'rb') as f:
    data = pickle.load(f)
infos = data['infos'] if 'infos' in data else data
print('总样本:', len(infos))
print('第0个样本 radars keys:', list(infos[0].get('radars', {}).keys()))