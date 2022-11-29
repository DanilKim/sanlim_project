import os
import numpy as np
import numpy.random as random
import transforms3d

import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree

from labelmap import label_to_kind
from utils import LloydPoolingPyramid, GeometricAdjacencyCompander

class TreePCDDataset(Dataset):
    def __init__(self, cfg, split='train', mode='train'):
        super().__init__()

        self.split = split
        self.mode = mode
        self.n_points = cfg.num_points
        self.data_root = cfg.data_root
        self.K = cfg.K
        self.aug = not cfg.no_aug
        self.pooler = LloydPoolingPyramid(4, GeometricAdjacencyCompander, [0.5]*4)

        self.data_list_file = os.path.join(cfg.list_root, self.split + '_crop.csv')
        self.data_dir = os.path.join(self.data_root, 'sanlim_crop_npy')

        with open(self.data_list_file, 'r') as df:
            self.data_list = df.readlines()

        self.kind_label_map = label_to_kind()
        self.label_map = {'nl':0, 'bl':1, 'bb':2}
        self.inverse_label_map = {0:'침엽수', 1:'활엽수', 2:'기타수종'}

        print('Tree PCD {} dataset with {} cropped samples loaded'.format(self.split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        sample_cls, sample_name = self.data_list[idx].strip().split('/')
        sample_path = os.path.join(self.data_dir, sample_cls, sample_name)

        tree_name = '_'.join(sample_name.split('.')[0].split('_')[:-1])

        cloud = np.load(sample_path)

        label = self.label_map[self.kind_label_map[sample_cls]]

        indices = np.random.choice(cloud.shape[0], size=self.n_points, replace=False)
        xyz = cloud[indices, :3]
        
        ## Augmentation ##
        if self.mode == 'train' and self.split in ['train', 'trainval'] and self.aug:
            M = np.eye(3)
            s = random.uniform(1/1.1, 1.1)
            M = np.dot(transforms3d.zooms.zfdir2mat(s), M)            
            if random.random() < 0.5/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
            if random.random() < 0.5/2:
                M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
            xyz = np.dot(xyz, M.T)
        
        kd = KDTree(xyz, leaf_size=self.K)
        sqr_distances, indices = kd.query(xyz, self.K) # K = 2 gives itself and other point from cloud which is closest

        #Jiggle the model a little bit if it is perfectly aligned with the axes
        vertexMean = np.mean(xyz, axis=0)
        vertexStd = np.std(xyz, axis=0)
        if not vertexStd.all():
            M = np.eye(3)
            angle = random.uniform(0.01,0.1,size=3)
            sign = random.choice([-1,1],size=3,replace=True)
            M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], sign[0] * angle[0]), M) 
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], sign[1] * angle[1]), M) 
            M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], sign[2] * angle[2]), M)
            xyz = np.dot(xyz,M.T)
            vertexMean = np.mean(xyz, axis=0)
            vertexStd = np.std(xyz, axis=0)

        xyz = (xyz - vertexMean)/vertexStd 

        sqr_distances[:,0] += 1 #includes self-loops
        valid = np.logical_or(indices > 0, sqr_distances>1e-10)
        rowi, coli = np.nonzero(valid)
        idx = indices[(rowi,coli)]
        
        edges = np.vstack([idx, rowi]).transpose()
        A = np.zeros(shape=(self.n_points, 8, self.n_points))
        zindices = np.dot([4, 2, 1], np.greater((xyz[edges[:,0],:] - xyz[edges[:,1],:]).transpose(), np.zeros((3, edges.shape[0]))))
        edgeLen = 1
        A[edges[:,0], zindices, edges[:,1]] = edgeLen
        A[edges[:,1], zindices, edges[:,0]] = edgeLen

        Plist = self.pooler.makeP(A, xyz)

        return torch.Tensor(xyz), torch.Tensor(A), label, [torch.Tensor(P) for P in Plist], tree_name