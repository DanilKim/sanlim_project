import math
import torch
import torch.nn as nn
from repsurf_utils import UmbrellaSurfaceConstructor


class GCNLayer(nn.Module):
  def __init__(self, no, no_A, no_features, no_filters):
    super(GCNLayer, self).__init__()
    self.no = no
    self.no_A = no_A
    self.no_features = no_features
    self.no_filters = no_filters

    self.hi = nn.Linear(no_features, no_filters)
    self.h = nn.Linear(no_features * no_A, no_filters, bias=False)

    nn.init.trunc_normal_(self.hi.weight.data, std=math.sqrt(1.0/(no_features*(no_A+1))))
    nn.init.constant_(self.hi.bias.data, 0.1)
    nn.init.trunc_normal_(self.h.weight.data, std=math.sqrt(1.0/(no_features*(no_A+1))))
    
  def forward(self, V, A):
    result = self.hi(V)
    A = torch.reshape(A, (-1, self.no * self.no_A, self.no))
    n = torch.bmm(A, V)
    n = torch.reshape(n, (-1, self.no, self.no_A * self.no_features))
    result = self.h(n)
    
    return result


class GraphPoolingLayer(nn.Module):
  def __init__(self, index):
    super(GraphPoolingLayer, self).__init__()
    self.index = index
  
  def forward(self, V, A, P):
    P_i = P[self.index]
    V_out = torch.matmul(P_i.permute(0, 2, 1), V)
    A_shape = A.shape
    P_rep = torch.unsqueeze(P_i, dim=2).repeat(1, 1, A_shape[2], 1)
    P_transpose = P_rep.permute(0, 2, 3, 1)
    P_nottranspose = P_rep.permute(0, 2, 1, 3)
    A_batched = A.permute(0, 2, 1, 3)
    leftMultiply = torch.matmul(P_transpose, A_batched)
    rightMultiply = torch.matmul(leftMultiply, P_nottranspose)
    A_out = rightMultiply.permute(0, 2, 1, 3)
    return V_out, A_out


class Block(nn.Module):
  def __init__(self, no, no_A, no_features, no_filters, Pindex):
    super(Block, self).__init__()
    self.no_filters = no_filters
    self.Pindex = Pindex
    self.c1 = GCNLayer(no, no_A, no_features, no_filters)
    self.bn1 = nn.BatchNorm1d(no)
    self.c2 = GCNLayer(no, no_A, no_filters, no_filters)
    self.bn2 = nn.BatchNorm1d(no)
    self.c3 = GCNLayer(no, no_A, no_filters, no_filters)
    self.bn3 = nn.BatchNorm1d(no)
    self.relu = nn.ReLU(inplace=True)
    self.gp = GraphPoolingLayer(Pindex)
  
  def forward(self, V, A, P):
    V = self.c1(V, A)
    V = self.bn1(V)
    V = self.relu(V)
    identity = V
    V = self.c2(V, A)
    V = self.bn2(V)
    V = self.relu(V)
    V = self.c3(V, A)
    V = self.bn3(V)
    V = self.relu(V)
    V = V + identity
    V, A = self.gp(V, A, P)
    return V, A


class G3DNet18(nn.Module):
  def __init__(self, no, no_A, no_features=3, dropout=0):
    super(G3DNet18, self).__init__()
    self.no = no
    self.no_A = no_A
    self.no_features = no_features
    self.Block_128 = Block(no, no_A, no_features, 128, 0)
    self.Block_256 = Block(no//2, no_A, 128, 256, 1)
    self.Block_512 = Block(no//4, no_A, 256, 512, 2)
    self.Block_1024 = Block(no//8, no_A, 512, 1024, 3)
    self.flatten = nn.Flatten()
    self.fc_2048 = nn.Linear((no//16)*1024, 2048)
    self.bn = nn.BatchNorm1d(2048)
    self.relu = nn.ReLU()
    self.do = nn.Dropout(dropout)
    self.fc_3 = nn.Linear(2048, 3)

  def forward(self, V, A, P=None):
    
    V, A = self.Block_128(V, A, P)
    V, A = self.Block_256(V, A, P)
    V, A = self.Block_512(V, A, P)
    V, A = self.Block_1024(V, A, P)
    logits = self.flatten(V)
    logits = self.fc_2048(logits)
    logits = self.bn(logits)
    logits = self.relu(logits)
    logits = self.do(logits)
    logits = self.fc_3(logits)
    
    return logits
  

class G3DNet26(nn.Module):
  def __init__(self, no, no_A, no_features=3, dropout=0):
    super(G3DNet26, self).__init__()
    self.no = no
    self.no_A = no_A
    self.no_features = no_features
    self.Block_128 = Block(no, no_A, no_features, 128, 0)
    self.Block_256 = Block(no//2, no_A, 128, 256, 1)
    self.Block_512 = Block(no//4, no_A, 256, 512, 2)
    self.Block_1024 = Block(no//8, no_A, 512, 1024, 3)
    self.Block_2048 = Block(no//16, no_A, 1024, 2048, 4)
    self.Block_4096 = Block(no//32, no_A, 2048, 4096, 5)
    self.flatten = nn.Flatten()
    self.fc_2048 = nn.Linear((no//64)*4096, 4096)
    self.bn = nn.BatchNorm1d(4096)
    self.relu = nn.ReLU()
    self.do = nn.Dropout(dropout)
    self.fc_3 = nn.Linear(4096, 3)

  def forward(self, V, A, P=None):
    
    V, A = self.Block_128(V, A, P)
    V, A = self.Block_256(V, A, P)
    V, A = self.Block_512(V, A, P)
    V, A = self.Block_1024(V, A, P)
    V, A = self.Block_2048(V, A, P)
    V, A = self.Block_4096(V, A, P)
    logits = self.flatten(V)
    logits = self.fc_2048(logits)
    logits = self.bn(logits)
    logits = self.relu(logits)
    logits = self.do(logits)
    logits = self.fc_3(logits)
    
    return logits


class SurfG3D18(G3DNet18):
  def __init__(self, no, no_A, no_features=10, dropout=0):
    super(SurfG3D18, self).__init__(no, no_A, 10, dropout)
    self.surface_constructor = UmbrellaSurfaceConstructor(9, 10, return_dist=True, aggr_type='sum', cuda=True)

  def forward(self, V, A, P=None):
    V = self.surface_constructor(V)
    logits = super().forward(V, A, P)
    
    return logits