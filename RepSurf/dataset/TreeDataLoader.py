from torch.utils.data import Dataset
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


class TreeDataLoader(Dataset):
  def __init__(self, root, split='training', bg=True):
    assert (split == 'training' or split == 'test')
    root_dir = '../tree'
    label_dict = {'Buche': 0, 'Douglasie': 1, 'Eiche': 2, 'Esche': 3, 'Fichte': 4, 'Kiefer': 5, 'Roteiche': 6}
    if split == 'training':
      csv = './data/Tree/train0.csv'
    elif split == 'test':
      csv = './data/Tree/test0.csv'
    self.data = []
    self.label = []
    f = open(csv, 'r', newline='\n')
    files = f.read().split('\n')[:-1]
    f.close()
    
    for file in files:
      l, _ = file.split('/')
      l = label_dict[l]
      dir = os.path.join(root_dir, file)
      with open(dir, 'r') as f:
        text = np.array(list(map(lambda u: list(map(np.float32, u.split()[:3])), f.read().split('\n')[:-1])))
        self.data.append(text)
        self.label.append(l)
        
    self.data = np.array(self.data)
    self.label = np.array(self.label)
    
    print('The size of %s data is %d' % (split, self.data.shape[0]))

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index):
    idx = np.random.randint(len(self.data[index]), size=4096)
    return self.data[index][idx, :].T, self.label[index]


