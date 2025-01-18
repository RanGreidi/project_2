import numpy as np
import random
import os
from datetime import datetime
import shutil

import sys

sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')
from environment import SlottedGraphEnvPower
from diamond_aviv import DIAMOND, SlottedDIAMOND
from environment import generate_env
from environment.data import generate_env
# from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import copy

if __name__ == "__main__":
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)
    # ------------------------------------------------------------------------

    #  number of nodes
    N = 4

    #  Adjacency matrix
    #  create 3x3 mesh graph
    # A = np.array([[0, 1, 1, 1],  #means how connects to who
    #               [1, 0, 1, 1],
    #               [1, 1, 0, 1],
    #               [1, 1, 1, 0]])

    # P = [(0, 0), (0, 1), (0, 2),                #the position of each node
    #      (1, 0), (1, 1), (1, 2),
    #      (2, 0), (2, 1), (2, 2)]

    # P = [(0, 0), (0, 1),                 #the position of each node
    #      (1, 0), (1, 1)]

    #  BW matrix [MHz]
    # C = 1 * np.ones((N, N))
    # C = 1 * np.array([[1, 1, 1, 1],  #means how connects to who
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1]])
    # ------------------------------------------------------------------------

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

    P = [(0.0, 0), (0.0, 0.01), (0.0, 0.02),
         (0.1, 0), (0.1, 0.01), (0.1, 0.02),
         (0.2, 0), (0.2, 0.01), (0.2, 0.02)]

    # capacity matrix
    C = 100 * np.ones((N, N))
    # ------------------------------------------------------------------------

    # number of paths to choose from
    action_size = 8  # search space limitaions?

    # flow demands
    F = [
        {"source": 0, "destination": 8, "packets": 1, "time_constrain": 10, 'flow_idx': 0},  # Packets [MegaBytes]
        {"source": 0, "destination": 8, "packets": 1000, "time_constrain": 10, 'flow_idx': 1}
    ]
    #
    slotted_env = SlottedGraphEnvPower(adjacency_matrix=A,
                                       bandwidth_matrix=C,
                                       flows=F,
                                       node_positions=P,
                                       k=action_size,
                                       reward_weights=reward_weights,
                                       telescopic_reward=True,
                                       direction='minimize',
                                       slot_duration=5,  # [in SEC]
                                       Tot_num_of_timeslots=60,  # [in Minutes]
                                       render_mode=True,
                                       trx_power_mode='gain',
                                       channel_gain=1)

    # -------------------- Try and use "generate_env" for general topology ------------------------ #

    # env, slotted_env = generate_env(num_nodes=9,
    #                                 num_edges=10,
    #
    #                                 num_flows=2,
    #                                 min_flow_demand=100,
    #                                 max_flow_demand=200,
    #
    #                                 num_actions=4,
    #
    #                                 min_capacity=10,
    #                                 max_capacity=10,
    #
    #                                 direction="minimize",
    #                                 reward_balance=0.8,
    #                                 seed=37,
    #                                 graph_mode='random',
    #                                 reward_weights=reward_weights,
    #                                 telescopic_reward=True,
    #                                 slot_duration=60,
    #                                 Tot_num_of_timeslots=60,
    #                                 render_mode=True,
    #                                 trx_power_mode = 'gain',
    #                                 channel_gain = 100)

    run_env = copy.deepcopy(slotted_env)

    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH)

    arrival_matrix = np.random.randint(100, 200, size=(4, len(slotted_env.flows)))

    diamond_paths, slotted_grrl_rates_data, slotted_grrl_delay_data = slotted_diamond(slotted_env, grrl_data=True, use_nb3r=False, arrival_matrix=arrival_matrix)

    _ = slotted_diamond.real_run(env=run_env, actions_recipe=diamond_paths)

''' 
Guide for AVIV:

12:00 - [ 12:01 12:02 12:03 ... 12:59] - 13:00


                12:00 -> 13:00                 This duration is *Tot_num_of_timeslots* in agent.run fucntion (defined within)      

                12:01 -> 12:02                 This duration is *slot_duration* defined in SlottedGraphEnvPower initialization


 To Run a scenario where we decide one on allocation in an hour: 
    define Tot_num_of_timeslots = 1
    define slot_duration  = 60*60 [SEC] (if units units of capacity are in seconds, else 60 if units of capacity arew in minutes)

 Note: 
 if  Tot_num_of_timeslots = 1000 and slot_duration = 60*60, when git a decision making every hour for 1000 hours


In case of a total run of one hour, and we want to make 60 decisions (new route every minutes):
    define Tot_num_of_timeslots = 60
    define slot_duration = 60 [SEC] (if units units of capacity are in seconds, else 60 if units of capacity arew in minutes)



it goes like this: we set slot_duration at the begining, and decide its units (seconds\ miliseonds etc...). Say we decide it is in seconds.
Then we decide slot duration, say we set it to 60. This sets the simulation resulotion to one second, and each time slot in one minute, 
and the simulation will run Tot_num_of_timeslots. if (Tot_num_of_timeslots = 60) it will run for one hour.

'''
