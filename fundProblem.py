import numpy as np
import scipy.io
import random
import pickle
import glob
import os
import natsort
import torch
from torch_geometric.data import Data
from torch_cluster import knn_graph

class fundProblem():
    def __init__(self, x, y, x0, threshold, numOut, x1p = None, x2p = None, gt = None):
        self.x = x
        self.y = y
        self.x0 = x0
        self.numOut = numOut
        self.threshold = threshold
        self.x1p = x1p
        self.x2p = x2p
        self.gt = gt
        self.graph = self.build_graph()

    def build_graph(self):
        #MPNN = 1
        node_features = torch.cat((self.x[:, :-1], self.y), dim = 1) 
        th = self.threshold.view(1)
        #if MPNN:
        #    edge_index = knn_graph(node_features, k=10, batch=None, loop=False)
        #    graph = Data(x = node_features, edge_index = edge_index, posx = self.x, posy = self.y, th = th, x0 = self.x0)
        #else:
        graph = Data(x = node_features, posx = self.x, posy = self.y, th = th, x0 = self.x0)
        return graph   

def readMatDataset(path):
    dataset = []
    for element in natsort.natsorted(glob.glob(path + "*.mat")):
        print(element)
        mat = scipy.io.loadmat(element)
        x = torch.tensor(mat['x'], dtype = torch.float64)
        y = torch.tensor(mat['y'], dtype = torch.float64)
        th = torch.tensor(mat['epsilonFund'], dtype = torch.float64).view(1, 1)
        x0 = torch.rand(8, 1).type(torch.float64)

        numOut = torch.tensor(mat['numOut'], dtype = torch.int64)
        x1p = torch.tensor(mat['x1p'], dtype = torch.float64)
        x2p = torch.tensor(mat['x2p'], dtype = torch.float64)
        gt = torch.tensor(mat['sol'], dtype = torch.float64)

        instance = fundProblem(x, y, x0, th, numOut, x1p, x2p, gt)
        dataset.append(instance)
    return dataset

def readMatInstance(filename):
    mat = scipy.io.loadmat(filename)
    x = torch.tensor(mat['x'], dtype = torch.float64)
    y = torch.tensor(mat['y'], dtype = torch.float64)
    x0 = torch.rand(8, 1).type(torch.float64)
    th = torch.tensor(mat['epsilonFund'], dtype = torch.float64).view(1, 1)
    numOut = torch.tensor(mat['numOut'], dtype = torch.int64)
    #print("x0: ", x0)
    instance = fundProblem(x, y, x0, th, numOut)
    return instance

if __name__ == '__main__':
    randomFile = "/home/giang/Documents/Improvement/FundMatrix/Data/SynDataTrain/01.mat"
    #randomFile = "/home/giang/Documents/Improvement/FundMatrix/Data/SynDataTest/Testing10/00.mat"
    instance = readMatInstance(randomFile)
    print(instance.x.shape)
    print(instance.numOut)
    #print(instance.solution.shape)