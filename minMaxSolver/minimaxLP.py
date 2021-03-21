import cvxpy as cp
import numpy as np
import torch

def cvxLP(X, Y, xn):
    # solve min gamma
    # s.t. |x_i theta - y_i | <= gamma
    # <=> -gamma <= x_i theta - y_i <= gamma
    # <=> x_i theta - gamma <= y_i
    #     -x_i theta - gamma <= -y_i
    X = X.numpy()
    Y = Y.numpy()
    n = X.shape[0]
    dim = X.shape[1]
    XX = np.concatenate((X, -X), 0)
    YY = np.concatenate((Y, -Y), 0).squeeze(1)

    #set up LP
    theta = cp.Variable(dim+1)
    XX = np.concatenate((XX, -np.ones((2*n,1))), axis=1)
    c = np.zeros((dim+1,1))
    c[-1] = 1.0

    prob = cp.Problem(cp.Minimize(c.T @ theta),
                 [XX @ theta <= YY])
    prob.solve()
    theta = theta.value
    gamma = theta[-1]
    theta = theta[0:-1]

    r = np.matmul(X, theta.reshape(dim,1)) - Y
    r = abs(r).squeeze()
    Midx = np.argsort(r)[-(dim+1):]

    # Solutions 
    M = Midx[0:dim+1]
    d = np.max(r)

    #convert to torch
    theta = torch.tensor(theta, dtype = torch.float64).view(-1, 1)
    d = torch.tensor(d, dtype = torch.float64)
    M = torch.tensor(M, dtype = torch.int64).view(1, -1)
    return theta, d, M
