import os
from datetime import datetime

import torch
import numpy as np

import gym
import pybullet as p
import peg_in_hole_gym
from peg_in_hole_gym.envs.base_env import TASK_LIST
from env import PlugIn
from time import sleep

# import pybullet_envs

from ppo import PPO
from pointnet2 import PointNet2

from torch.utils.tensorboard.writer import SummaryWriter

TASK_LIST['plug-in'] = PlugIn
device = torch.device('cuda:0') 
sub_num = 1

def state_extract(state):
    if type(state) == list:
        state = state[0]
    state = np.expand_dims(state, axis=0)
    state = torch.from_numpy(state)
    state = state.permute(0, 2, 1)
    state = state.to(device)
    return state


################################### Training ###################################

def eval():

    env = gym.make('peg-in-hole-v0', client=p.GUI, task="plug-in", task_num=sub_num, offset = [2.,3.,0.],args=[], is_test=True)
    has_continuous_action_space = True
    # state space dimension
    state_dim = 4  # (x,y,z, angle)

    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving


    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames


    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic



    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n



    model_name = ""



    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")


    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # load pre_trained model

    ppo_agent.load(model_name)

    # track total training time



    # training loop
    for ep in range(1, total_test_episodes+1):

        state = env.reset()
        # state,_ = pointnet2(state_extract(state))
        state = state[0]


        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step([action])
            sleep(1/240)
            # print("state shape: ", state[0].shape)
            # state,_ = pointnet2(state_extract(state))
            state = state[0]
            reward = reward[0]
            done = done[0]
            info = info[0]


            # break; if the episode is over
            if done:
                break
    
    ppo_agent.buffer.clear()


    # log_f.close()
    env.close()










if __name__ == '__main__':

    eval()
