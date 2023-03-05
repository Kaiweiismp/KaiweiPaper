
import torch
import torch.optim as optim

from NGCF import NGCF
from NGCF import NGCF_twoGraph
from utility.helper import *
from utility.batch_test import *
from utility.nearest_neighbor import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore')
import time

from tensorboardX import SummaryWriter
from os.path import join

BOARD_PATH = "/home/ismp/sda1/kaiwei/NGCF-PyTorch/NGCF/board"

if __name__ == '__main__':
    #print("============gpu_id==============")
    #print(args.gpu_id)
    args.device = torch.device('cuda:' + str(args.gpu_id))
    #print(args.device)
    plain_adj, norm_adj, mean_adj, plain_adj_personality, norm_adj_personality, mean_adj_personality = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    print("=========================================")

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 norm_adj_personality,
                 args).to(args.device)
    print("========create model ====================")
    print("=========================================")


   