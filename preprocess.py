import laspy
import numpy as np
import pickle
import glob
import os
import math
import pickle
from sklearn.neighbors import KDTree
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
  if (features is None) and (labels is None): return cpp_subsampling.subsample(points, sampleDl=sampleDl, verbose=verbose)
  elif (labels is None): return cpp_subsampling.subsample(points, features=features, sampleDl=sampleDl, verbose=verbose)
  elif (features is None): return cpp_subsampling.subsample(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
  else: return cpp_subsampling.subsample(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)
  
sampleDl = 0.01
leafsize = 4096
theta_grid = 10
phi_grid = 10
theta_threshold = 2 / 3
n_points = 4096

root_dir = '/data/sanlim_las'
tree_dir = '/data/sanlim_pickle'
#save_dir = './temp'
save_dir = '/data/sanlim_crop_npy'
filelist = glob.glob(os.path.join(root_dir, '*/*.las'))

for i, file in enumerate(filelist):
  _, _, classname, filename = file.split('/')
  name = filename.split('.')[0]
  tdir = os.path.join(tree_dir, classname)

  #if os.path.exists(os.path.join(tdir, name+'.pickle')):
  #  print('[{}/{}] Already processed {} - skip...'.format(i+1, len(filelist), name))
  #  continue

  print('[{}/{}] Processing {}...'.format(i+1, len(filelist), name))
  sdir = os.path.join(save_dir, classname)
  if not os.path.isdir(sdir): os.mkdir(sdir)
  if not os.path.isdir(tdir): os.mkdir(tdir)
  
  # get data points from las
  data = laspy.read(file)
  data = np.dstack((data.x, data.y, data.z, data.red, data.blue, data.green))[0].astype(np.float32)
  points, features = grid_subsampling(data[:, :3], features=data[:,3:], sampleDl=sampleDl)
  
  l = points[points[:, 2].argsort()]

  # get rtp from xyz
  xyz = np.array(l)
  mid = np.percentile(xyz, 50, axis=0)
  xyz2 = xyz - mid
  xyzrtp = np.hstack((xyz, np.zeros(xyz.shape)))
  xy = xyz2[:, 0]**2 + xyz2[:, 1]**2
  xyzrtp[:, 3] = np.sqrt(xy + xyz2[:, 2]**2)
  xyzrtp[:, 4] = np.arctan2(np.sqrt(xy), xyz2[:, 2])
  xyzrtp[:, 5] = np.arctan2(xyz2[:, 1], xyz2[:, 0])
  xyzrtp = xyzrtp[xyzrtp[:, 4].argsort()]
  
  # get boundary points
  boundary = np.zeros((theta_grid*phi_grid, 6))
  theta_step = (math.pi * theta_threshold) / (2 * theta_grid)
  phi_step = math.pi / phi_grid
  cur_theta = theta_step
  idx = 0
  for i in range(theta_grid):
    lt = xyzrtp[abs(xyzrtp[:, 4] - cur_theta) < theta_step]
    lt = lt[lt[:, 5].argsort()]
    cur_phi = -math.pi + phi_step
    idx = 0
    for j in range(phi_grid):
      max_val = 0
      while idx < lt.shape[0] and cur_phi - phi_step < lt[idx, 5] < cur_phi + phi_step:
        if max_val < lt[idx, 3]:
          max_val = lt[idx, 3]
          boundary[i*phi_grid+j, :] = lt[idx, :]
        idx += 1
      cur_phi += (2 * phi_step) 
    cur_theta += (2 * theta_step)

  boundary = boundary[~np.all(boundary == 0, axis=1)]
  boundary_points = boundary[:, 0:3]
  boundary_points += mid
  
  # make kdtree
  try:
    with open('{}.pickle'.format(os.path.join(tdir, name)), 'rb') as f:
      tree = pickle.load(f)
    continue
  except:
    tree = KDTree(xyz, leaf_size=leafsize)
    with open('{}.pickle'.format(os.path.join(tdir, name)), 'wb') as f:
      pickle.dump(tree, f)

  # get crop pcd
  for i in range(boundary_points.shape[0]):
    cur = boundary_points[i, :].reshape(1, -1)
    dist, idx = tree.query(cur, k=n_points)
    idx = idx.flatten().tolist()
    points = xyz[idx]
    np.save('{}/{}_{}'.format(sdir, name, i), points)

    
ns = 2500
splits = ['train', 'val', 'test']

ns = str(ns)
crop_dir = '/data/sanlim_crop_npy'
root = './datalist'
for sp in splits:
    with open(os.path.join(root, sp + '.csv'), 'r') as rf:
        f_list = rf.readlines()

    with open(os.path.join(root, sp + '_crop.csv'), 'w') as wf:
        for i, fn in enumerate(f_list):
            wf.writelines(['/'.join(f.split('/')[-2:])+'\n' for f in glob.glob(crop_dir + '/' + fn.strip() + '*')])
            if i % 100 == 0:
                print('{} Split [{}/{}] {}'.format(sp, i+1, len(f_list), fn.strip()))