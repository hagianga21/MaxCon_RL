import os
import sys
import glob
import time
import torch
import scipy.io
import argparse
import numpy as np
from PIL import Image
from itertools import count
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from DQNet import DQNet
from replayMemory import ReplayMemory
from environment import Environment
from collections import namedtuple
from minMaxSolver.minimaxLP import cvxLP
from fundProblem import fundProblem, readMatInstance, readMatDataset
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter

def select_action(args, graph, avail, avail_idx):
    graph_batch = Batch.from_data_list([graph]).to(device)
    out = brain.predict(graph_batch, target = False).view(-1, graph['x'].size(0))
    out = out * avail.to(device)
    out = out[out != 0]
    action = avail_idx[torch.argmax(out)].view(1, 1).to(device)
    return action

def evaluate_model(instance):
    env = Environment(instance.graph)
    cur_state = env.get_initial_environment()  

    for t in count():
        graph = env.make_nn_input(cur_state, instance)
        avail = env.get_valid_actions(cur_state)
        avail_idx = np.argwhere(avail == 1).reshape(-1)

        #Select action
        action = select_action(args, graph, avail, avail_idx)
        next_state, reward = env.get_next_state_with_reward(cur_state, action.item())
        reward = torch.tensor([reward], device = device)
        cur_state = next_state
        if next_state.is_done():
            real_reward = t + 1
            break
    
    distance = torch.abs(torch.mm(cur_state.orginal_graph['posx'], cur_state.xn) - cur_state.orginal_graph['posy'])
    outlierBefore = np.asarray(distance > cur_state.orginal_graph['th']).nonzero()[0]
    outlierBefore = torch.tensor(outlierBefore)
    OutlierAfter = OutlierRefine(0, cur_state, outlierBefore)
    print("Outliers: ", OutlierAfter.size(0))
    return outlierBefore, OutlierAfter

def OutlierRefine(itera, cur_state, violation_set):
    for i in range(itera, len(violation_set)):
        now_i = i
        temp_violation = torch.cat([violation_set[0:i], violation_set[i+1:]])
        H = torch.arange(cur_state.orginal_graph['x'].size(0)) 
        for i in range(len(temp_violation)):
            H = H[H!=temp_violation[i]]
        xn, val, bs = cvxLP(cur_state.orginal_graph['posx'][H, :], cur_state.orginal_graph['posy'][H], cur_state.xn)
        if val <= cur_state.orginal_graph['th']:
            final_violation = OutlierRefine(now_i, cur_state, temp_violation)
            return final_violation
    return violation_set  

def plot_Matches(randomFile, outlier_idx):
    plt.clf()
    mat = scipy.io.loadmat(randomFile)
    im1name = mat['im1name'][0]
    im2name = mat['im2name'][0]
    x1 = torch.tensor(mat['x1'])
    x2 = torch.tensor(mat['x2'])
    img1 = torch.tensor(mpimg.imread(im1name))
    img2 = torch.tensor(mpimg.imread(im2name))
    imgfinal = torch.cat([img1, img2])
    imgplot = plt.imshow(imgfinal)

    #Plot inliers
    inlier_idx = torch.arange(x1.shape[1])
    for i in range(len(outlier_idx)):
        inlier_idx = inlier_idx[inlier_idx!=outlier_idx[i]]
    #for i in outlier_idx:
    #    plt.plot([x1[0, i], x2[0, i]], [x1[1, i], x2[1, i] + 376], 'r', linewidth=0.8)  
    for i in inlier_idx:
        plt.plot([x1[0, i], x2[0, i]], [x1[1, i], x2[1, i] + 376], 'g', linewidth=0.8)
    plt.axis("off")
    savedName = os.path.basename(randomFile).split(".")[0]
    plt.savefig('./Data/results/' + savedName + '.png', bbox_inches='tight')
    #plt.show()

def parse_arguments():
    """
    Parse general args for the problem
    """
    parser = argparse.ArgumentParser()
    # Instances parameters
    parser.add_argument('--dimension', type=int, default=8+2)
    parser.add_argument('--threshold', type=float, default=0.1)
    # Hyper parameters
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='./pretrained_model')
    parser.add_argument('--saved_model_name', type=str, default='iter_14000_model.pth.tar')
    return parser.parse_args()

if __name__ == '__main__':
    print("************************************************")
    print("[INFO] Linearized Fundamental Matrix Estimation")
    print("[INFO] Reinforcement learning model")
    print("************************************************")
    args = parse_arguments()
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    brain = DQNet(args)
    brain.load_saved_model(args.save_dir, args.saved_model_name)

    for data in sorted(glob.glob('./Data/Demo/' + "*.mat")):
        print(data)
        instance = readMatInstance(data)
        outlierbefore, outliers = evaluate_model(instance)
        plot_Matches(data, outliers)
    
    print("Create gif results...")
    fp_in = "./Data/results/*.png"
    fp_out = "./Data/results/demo.gif"
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=500, loop=0)
    print("Done, saved all results in folder Data/results")
    