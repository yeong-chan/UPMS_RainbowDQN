import csv
from cfg import get_cfg

import os
import torch
import numpy as np
import random

from agent.rainbow_dqn import DQNAgent
from environment.env import UPMSP
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    cfg = get_cfg()
    
    mode = cfg.mode
    if mode == 'heuristic':
        action_size = 3
    else:
        action_size = 12

    state_size = 104  # feature 1~8 size = 104 / 176
    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    env = UPMSP(log_dir=event_path, num_j=1000, num_m=8, action_number=action_size, min=1, max=11,
                action_mode=mode)

    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # parameters
    num_frames = 10000
    memory_size = 10000
    batch_size = 64
    target_update = 100

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, seed)

    num_episode = 10000

    agent.train(num_episode)
