from torch.utils.data import Dataset
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

class SanLimDataLoader(Dataset):
  def __init__(self, root, split='train'):
    self.root_dir = root
    self.list_dir = '/sanlim_project/datalist'
    self.data_dir = os.path.join(self.root_dir, 'sanlim_crop_npy')
    txt = os.path.join(self.list_dir, '{}_crop.csv'.format(split))
    with open(txt, 'r') as f:
      files = f.read().split('\n')[:-1]
    
    self.files = files
    # 0: nl, 1: bl, 2: other
    classes = {
      'AC': 1, 'AG': 1, 'AH': 0, 'AP': 1, 'CA': 1, 'CC': 1, 'CEJ': 1, 'CHP': 0,
      'CK': 1, 'CMP': 1, 'CO': 1, 'CP': 0, 'STJ': 1, 'PR': 0,
      'QQ': 1, 'LL': 0, 'TB': 0, 'GB': 0, 'ZS': 1, 'QV': 1, 'MP': 1, 'PK': 0, 'PD': 0,
      'AA': 1, 'AK': 0, 'AN': 1, 'APY': 1, 'AVE': 0, 'CHAP': 0, 'CJ': 0, 'COK': 1,
      'CPV': 0, 'DIK': 1, 'FR': 1, 'HPD': 0, 'JU': 0, 'KP': 1, 
      'MAG': 1, 'PAU': 1, 'PC': 1, 'PDM': 0, 'PIS': 0, 
      'PLO': 0, 'PTA': 1, 'QA': 1, 'QS': 1, 'QV': 1, 'SA': 1, 'SP': 0,
      'STR': 0, 'UDV': 1, 'ZZ': 1, 'ALJ': 0, 'BP': 1, 'CAJ': 1,
      'CD': 0, 'CES': 1, 'CR': 1, 'EA': 1, 'JR': 1, 'KAS': 1, 'LI': 1,
      'LT': 1, 'MG': 0, 'PA': 0, 'PB': 0, 'PE': 0, 'PPU': 0, 'PS': 1, 'PY': 1, 
      'QUA': 1, 'RP': 1, 'SJ': 1, 'STY': 1, 'TX': 0, 'ZS': 1, 'AT': 1,
      'QY': 1, 'HE': 1, 'FS': 1, 'Ul': 1, 'AHI': 1, 'UP': 1, 'EU': 1,
      'PQ': 1, 'ABN': 0, 'CPP': 1, 'AJ': 1, 'BD': 1, 'SOA': 1, 'SO': 1,
      'COR': 1, 'CAO': 1, 'QD': 1, 'QM': 1, 'HD': 1,
      'TD': 1, 'CAT': 1, 'SB': 1, 'POD': 1, 'ACM': 1,
      'CL': 1, 'TI': 1, 'IP': 1, 'AM': 1, 'CU': 0, 'BB': 2, 
      'JM': 1, 'ACD': 1, 'CB': 1, 'AE': 1, 'PO': 1, 'DM': 1, 
      'FM': 1, 'TV': 1, 'MAU': 1, 'PSE': 1, 'IC': 1,
      'BS': 1, 'POC': 1, 'TN': 0, 'PHA': 1, 'BK': 1, 'PRS': 1, 'TKM':1,
      'STO': 1, 'STP': 1, 'HC': 1, 'MT': 1, 'VA': 1, 'TSM': 1, 'CN': 1,
      'EJ': 1, 'TS': 1
    }
    
    self.classes = classes
    
    print('The size of %s data is %d' % (split, len(self.files)))

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, index):
    label, name = self.files[index].split('/')
    label = int(self.classes[label])
    
    text = np.load('{}/{}'.format(self.data_dir, self.files[index])).astype(np.float32)
    return text.T, label, name
