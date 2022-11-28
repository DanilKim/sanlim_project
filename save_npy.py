import os
import numpy as np

num_samples = 1652
num_crops = 100

nss = str(num_samples)
ncs = str(num_crops)
with open('datalist/sanlim_'+nss+'samples/sanlim_'+nss+'samples_trainval_crop'+ncs+'.csv', 'r') as f:
    fl = f.readlines()

for i, fn in enumerate(fl):
    fn = '/data/sanlim_crop_'+ncs+'/' + fn.strip()
    path = fn.split('/')
    save_dir = '/'.join(['', path[1], path[2]+'_npy', path[3]])
    os.makedirs(save_dir, exist_ok=True)
    save_fn = os.path.join(save_dir, path[4][:-4] + '.npy')
    if os.path.exists(save_fn):
        continue
    l = []
    with open(fn, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            try:
                x, y, z = list(map(float, line.split()))
            except:
                continue
            l.append([x, y, z])
    
    xyz = np.array(l)
    np.save(save_fn, xyz)

    if i % 1000 == 0:
        print('[ {} / {} ]'.format(i+1, len(fl)))