from fundProblem import fundProblem, readMatInstance, readMatDataset
from replayMemory import ReplayMemory
from environment import Environment
from DQNet import DQNet
import torch
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from collections import namedtuple
import argparse
import sys
from itertools import count
import math
import os
import glob

def select_action(args, graph, avail, avail_idx):
    """
    Select action for the state
    Balance expolration and exploitation
    """
    global steps_done
    sample = random.random()
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100000
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        #Convert State to Batch to use max pooling
        graph_batch = Batch.from_data_list([graph]).to(device)
        out = brain.predict(graph_batch, target = False).view(-1, graph['x'].size(0))
        out = out * avail.to(device)
        out = out[out != 0]
        action = avail_idx[torch.argmax(out)].view(1, 1).to(device)
    else:
        action = random.choice(avail_idx)
        action = torch.tensor([[action]], device = device)
    return action

def optimize_model(args):
    """
    Optimize model using Q function
    """
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    ###########################################
    #Compute a mask of non-final states
    ###########################################
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_list = list(data[0] for data in batch.next_state if data is not None)
    non_final_next_states_batch = Batch.from_data_list(non_final_next_states_list).to(device)
    non_final_next_avail_list = list(data[1] for data in batch.next_state if data is not None)
    non_final_next_avail_batch = torch.cat(non_final_next_avail_list).to(device)

    state_batch_list = list(data[0] for data in batch.state)
    state_batch = Batch.from_data_list(state_batch_list).to(device)

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    ###########################################
    #Predict state_action_values
    ###########################################
    offset = 0
    for i in range(BATCH_SIZE):
        action_batch[i] = action_batch[i] + offset
        offset = offset + (state_batch.batch == i).sum()
    state_action_values = brain.predict_with_grad(state_batch)[action_batch].view(-1, 1)

    ###########################################
    #Predict next state action values
    ###########################################
    next_state_action_values = torch.zeros(BATCH_SIZE, device=device)
    target_out = brain.predict(non_final_next_states_batch, target = True)
    #Take max state value in valid actions
    res = []
    numberOfNonFinalBatch = len(non_final_next_states_batch.th)
    for i in range(numberOfNonFinalBatch):
        maskBatch = (non_final_next_states_batch.batch == i)
        maskAvailAction = non_final_next_avail_batch * maskBatch
        AvailAction = (maskAvailAction != 0).nonzero().squeeze(1)
        posibleTarget = target_out[AvailAction]
        maxPosibleTarget = posibleTarget.max(0)[0].detach()
        res.append(maxPosibleTarget.item())
    target_out = torch.tensor(res).to(device)
    next_state_action_values[non_final_mask] = target_out

    ###########################################
    # Compute the expected Q values
    ###########################################
    expected_state_action_values = (next_state_action_values * args.gamma) + reward_batch
    expected_state_action_values = expected_state_action_values.float()

    # Compute Huber loss and optimize network
    loss = brain.calc_loss_and_optimize(state_action_values, expected_state_action_values.unsqueeze(1))
    return loss

def run_episode(args, i_episode, instance, memory_initialization = False):
    global steps_done
    total_reward = 0.
    total_loss = 0.
    env = Environment(instance.graph)
    cur_state = env.get_initial_environment()

    for t in count():
        graph = env.make_nn_input(cur_state, instance)
        avail = env.get_valid_actions(cur_state)
        avail_idx = np.argwhere(avail == 1).reshape(-1)

        #Select action
        if memory_initialization:
            action = random.choice(avail_idx)
            action = torch.tensor([[action]], device = device)
        else:
            action = select_action(args, graph, avail, avail_idx)
        writeActions(action.item())

        next_state, reward = env.get_next_state_with_reward(cur_state, action.item())
        next_graph = env.make_nn_input(next_state, instance)
        next_avail = env.get_valid_actions(next_state)
        reward = torch.tensor([reward], device = device)
        total_reward = total_reward + reward.item()
            
        cur_state_features = (graph, avail)
        next_state_features = (next_graph, next_avail)
        
        if next_state.is_done():
            next_state_features = None
        
        # Store the transition in memory
        memory.push(cur_state_features, action, next_state_features, reward)
        
        cur_state = next_state

        if not memory_initialization:
            loss = optimize_model(args)
            total_loss = total_loss + loss
            writer.add_scalar('trainingLoss', loss, steps_done)

        if steps_done % args.target_update == 0 and steps_done >= args.target_update:
            print("Update target - Step done: ", steps_done)
            brain.update_target_model()

        if next_state.is_done():
            total_loss = total_loss / (t + 1)
            real_reward = t + 1
            print("NumOut: ", instance.numOut.item())
            print("Debug: ", t + 1)
            if not memory_initialization:
                writer.add_scalar('trainingRewards', real_reward - instance.numOut.item(), i_episode)
            break
    
    writeLoss(instance.numOut.item(), real_reward, total_loss)
    return total_loss, total_reward, real_reward

def writeLoss(NumOut, predictedOutliers, loss):
    with open('loss.txt', 'a') as score:
        score.write('\n' + str(NumOut) + ' ' + str(predictedOutliers) + ' ' + str(loss) + '\n')

def writeActions(action):
    with open('loss.txt', 'a') as score:
        score.write(str(action) + ', ')

def writeFileName(filename):
    with open('loss.txt', 'a') as score:
        score.write(str(filename) + '\n')

def parse_arguments():
    """
    Parse general args for the problem
    """
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--dimension', type=int, default=8+2)
    parser.add_argument('--threshold', type=float, default=0.1)

    # Reinforcement parameters
    parser.add_argument('--memory_capacity', type=int, default=100000)
    parser.add_argument('--target_update', type=int, default=15000)
    parser.add_argument('--gamma', type=float, default=0.999)

    # Hyper parameters
    parser.add_argument('--validation_size', type=int, default=50)
    parser.add_argument('--test_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print("*******************************************************")
    print("[INFO] LINEARIZED FUNDAMENTAL MATRIX ESTIMATION PROBLEM")
    print("[INFO] dimension: %d" % args.dimension)
    print("*******************************************************")
    sys.stdout.flush()

    #Reproduce results
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #Setup Tensorboard
    tsb_folder = "tensorboard"
    tsb_dirname = "DQN_%d" % (args.target_update)
    tsb_path = os.path.join(tsb_folder, tsb_dirname)
    writer = SummaryWriter(tsb_path)

    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    print("TRAINING ON RANDOM INSTANCES")

    memory = ReplayMemory(args.memory_capacity)
    brain = DQNet(args)
    i_episode = 0
    steps_done = 0
    trainPath = "./Data/MediumData/"
    print("Start Initialization")
    while len(memory) < 5*BATCH_SIZE:
        #generate random dataset
        randomFile = random.choice(os.listdir(trainPath))
        randomFile = os.path.join(trainPath, randomFile)
        print(randomFile)
        instance = readMatInstance(randomFile)
        loss, reward, real_reward = run_episode(args, i_episode, instance, memory_initialization=True)
    print("Finish Initialization: ", len(memory))
    i_episode = 0
    steps_done = 0

    for i in range(7):
        for data in sorted(glob.glob(trainPath + "*.mat")):
            print(data)
            writeFileName(data)
            print("Episode: ", i_episode)
            print("Step done: ", steps_done)
            instance = readMatInstance(data)
            loss, reward, real_reward = run_episode(args, i_episode, instance, memory_initialization=False)
            print("Loss: ", loss)
            print("Rewards: ", reward)
            
            #Save model
            if i_episode % 2000 == 0 and i_episode > 4000:
                folderName = "model_random_saved_%d" % (args.dimension)
                fn = "iter_%d_model.pth.tar" % (i_episode)
                print("model saved")
                brain.save(folderName, filename = fn)
            
            i_episode = i_episode + 1
    
    print("Finish Training")