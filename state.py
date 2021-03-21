import torch
from minMaxSolver.minimaxLP import cvxLP
import math
import time

class State():
    def __init__(self, orginal_graph, violation_set, avail_set, last_visited, xn, val):
        self.orginal_graph = orginal_graph
        self.violation_set = violation_set
        self.avail_set = avail_set
        self.last_visited = last_visited
        self.xn = xn
        self.val = val
        self.N = self.orginal_graph['x'].size(0)

    def step(self, action):
        new_last_visited = torch.tensor(action)
        if len(self.violation_set) == 0:
            new_violation_set = new_last_visited.view(1)
        else:
            new_violation_set = torch.cat((self.violation_set, new_last_visited.view(1)))

        H = torch.arange(self.N) 
        for i in range(len(new_violation_set)):
            H = H[H!=new_violation_set[i]]   
        xn, val, bs = cvxLP(self.orginal_graph['posx'][H, :], self.orginal_graph['posy'][H], self.xn)
        new_avail_set = H[bs]

        newState = State(self.orginal_graph, new_violation_set, new_avail_set, new_last_visited, xn, val)
        return newState

    def is_done(self):
        return self.val <= self.orginal_graph['th']

    def is_success(self):
        return 0