import numpy as np
import random
import os
from datetime import datetime
import shutil


import sys
sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')
from environment import SlottedGraphEnvPower
from diamond import DIAMOND, SlottedDIAMOND
from environment import generate_env
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


if __name__ == "__main__":
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)
    #------------------------------------------------------------------------

    # # number of nodes
    # N = 4

    # # Adjacency matrix
    # # create 3x3 mesh graph
    # A = np.array([[0, 1, 1, 1],  #means how connects to who
    #               [1, 0, 1, 1],
    #               [1, 1, 0, 1],
    #               [1, 1, 1, 0]])

    
    # # P = [(0, 0), (0, 1), (0, 2),                #the position of each node
    # #      (1, 0), (1, 1), (1, 2),
    # #      (2, 0), (2, 1), (2, 2)]

    # P = [(0, 0), (0, 1),                 #the position of each node
    #      (1, 0), (1, 1)] 

    # # capacity matrix
    # C = 100 * np.ones((N, N))
    # C = 100 * np.array([[1, 1, 1, 1],  #means how connects to who
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1]])
    #------------------------------------------------------------------------
   
    N = 9

    # Adjacency matrix
    # create 3x3 mesh graph
    A = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1, 0, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    
    
    P = [(0, 0), (0, 100.1), (0, 200.2),
         (100.1, 0), (100.1, 100.1), (100.1, 200.2),
         (200.2, 0), (200.2, 100.1), (200.2, 200.2)]

    # capacity matrix
    C = 100 * np.ones((N, N))
    #------------------------------------------------------------------------


    # number of paths to choose from
    action_size = 4                         #search space limitaions?

    # flow demands
    F = [
        {"source": 0, "destination": 8, "packets": 1000, "time_constrain": 10 , 'flow_idx': 0 },
        {"source": 0, "destination": 8, "packets": 1000, "time_constrain": 10, 'flow_idx': 1}
    ]

    slotted_env = SlottedGraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = True,
                                        direction = 'minimize',
                                        render_mode = True)

    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH)
    
    diamond_paths, slotted_grrl_rates_data, slotted_grrl_delay_data = slotted_diamond(slotted_env, grrl_data=True)

