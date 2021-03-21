from state import State
from minMaxSolver.minimaxLP import cvxLP
import torch
from torch_geometric.data import Data
import numpy as np

class MyData(Data):
    def __init__(self, x, basis, th):
        super(MyData, self).__init__()
        self.x = x
        self.basis = basis
        self.th = th
    def __inc__(self, key, value):
        if key == 'basis':
            return self.x.size(0)
        else:
            return super(MyData, self).__inc__(key, value)

class MyDataMPNN(Data):
    def __init__(self, x, edge_index, basis, th):
        super(MyDataMPNN, self).__init__()
        self.x = x
        self.basis = basis
        self.th = th
        self.edge_index = edge_index
    def __inc__(self, key, value):
        if key == 'basis':
            return self.x.size(0)
        else:
            return super(MyDataMPNN, self).__inc__(key, value)

class Environment:
    def __init__(self, orginal_graph):
        self.orginal_graph = orginal_graph
        self.N = orginal_graph['x'].size(0)
 
    def get_initial_environment(self):
        x0 = self.orginal_graph['x0']
        xn, val, bs = cvxLP(self.orginal_graph['posx'], self.orginal_graph['posy'], x0)
        violation_set = torch.zeros(size = (0, 1), dtype = torch.int64)
        avail_set = bs
        last_visited = torch.zeros(size = (0, 1), dtype = torch.int64)

        return State(self.orginal_graph, violation_set, avail_set, last_visited, xn, val)

    def make_nn_input(self, cur_state, instance = None):
        # Set avail features
        avail_feature = -1 * torch.ones(size = (self.N, 1), dtype = torch.float64)
        avail_feature[cur_state.avail_set] = 1
        # Set last visited features
        last_visited_feature = -1 * torch.ones(size = (self.N, 1), dtype = torch.float64)
        last_visited_feature[cur_state.violation_set] = 1
        #Get basis
        avail = self.get_valid_actions(cur_state)
        basis = np.argwhere(avail == 1).reshape(-1)
        
        node_features = torch.cat((cur_state.orginal_graph['x'], avail_feature, last_visited_feature), dim = 1)

        if 0:
            distance = torch.mm(instance.x, cur_state.xn) - instance.y
            node_features = torch.cat((cur_state.orginal_graph['x'], avail_feature, last_visited_feature, distance), dim = 1)

        nn_input_graph = MyData(x = node_features, basis = basis, th = cur_state.orginal_graph['th'])

        #MPNN = 1
        #if MPNN:
        #    nn_input_graph = MyDataMPNN(x = node_features, edge_index = cur_state.orginal_graph['edge_index'], basis = basis, th = cur_state.orginal_graph['th'])
        return nn_input_graph

    def get_next_state_with_reward(self, cur_state, action):
        new_state = cur_state.step(action)
        reward = -1
        #reward = -1/100
        return new_state, reward

    def get_valid_actions(self, cur_state):
        available = torch.zeros(self.N, dtype = torch.int64)
        available[cur_state.avail_set] = 1
        return available

