import os
import torch
import torch.nn as nn
import pyamg
import scipy.sparse
import numpy as np

def voting(preds, labels, names, votes, gts):
    assert len(preds) > 0
    assert len(preds) == len(labels) == len(names)
    nClass = 3

    for p, l, n in zip(preds, labels, names):
        if n in votes:
            votes[n][p] += 1
        else:
            votes[n] = np.zeros(nClass)
            votes[n][p] += 1
            gts[n] = l
    
    return votes, gts


def save_model(model, epoch, best_acc, save_path):
    torch.save({
        'epoch': epoch,
        'best_acc': best_acc,
        'model': model,
        #'model_state_dict': model.state_dict()
    }, save_path)


class LloydPoolingPyramid():

    def __init__(self,numRepresentations,companderConstructor, ratios):
        self.numRepresentations = numRepresentations
        self.companderConstructor = companderConstructor
        self.ratios = ratios

    def makeP(self, A, V=None):
        Plist = []
        companderInstance = self.companderConstructor(V, A)
        for pIndex in range(self.numRepresentations):
            #print(companderInstance.contractA().shape)
            #print((companderInstance.contractA() != 0).sum())
            P = pyamg.aggregation.aggregate.lloyd_aggregation(\
            scipy.sparse.csr_matrix(companderInstance.contractA()), ratio=self.ratios[pIndex], distance='same', maxiter=10)[0]
            P = P.todense()
            Pcolsum = np.tile(np.count_nonzero(np.array(P), axis=0), (P.shape[0], 1))
            Pcolsum[Pcolsum == 0] = 1
            P = np.divide(P, Pcolsum.astype(np.float64))
            Plist.append(P.astype(np.float32))
            companderInstance.update(P)
            A = companderInstance.expandA()
            V = companderInstance.V
        return Plist

    def write(self,Ps,As):
        AsparseList = []
        for A in As:
            currentA = A.tolist()
            pass


class GeometricAdjacencyCompander():

    def __init__(self, V, A):
        self.V = V
        self.A = A
        self.N = V.shape[0]
        self.numDirs = 8
        self.flatA = 0

    def contractA(self):
        self.flatA = self.A.sum(axis=1)
        return self.flatA

    def expandA(self):
        expandedA = np.zeros((self.N,self.numDirs,self.N))
        #print(self.N)
        #print(self.flatA.shape)
        (iVals,jVals) = np.nonzero(self.flatA)
        zindex = np.dot([4, 2, 1], np.greater((self.V[iVals,:] - self.V[jVals,:]).transpose(), np.zeros((3,iVals.shape[0]))));
        edgeLen = np.linalg.norm(self.V[iVals,:] - self.V[jVals,:],axis=1)
            # print('From {0} to {1}: Len {2}',i,j,edgeLen)
        expandedA[iVals, zindex, jVals] = edgeLen
        expandedA[jVals, zindex, iVals] = edgeLen
        self.A = expandedA

        return expandedA

    def update(self,P):
        self.flatA = np.dot(np.dot(P.transpose(),self.flatA),P)
        self.V = np.dot(P.transpose(),self.V)
        self.N = self.V.shape[0]

