import numpy as np
import random
import os
from datetime import datetime
import shutil
import matplotlib.pyplot as plt


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

    # # node positions
    # P = [(0, 0), (0, 1),                 #the position of each node
    #      (1, 0), (1, 1)] 

    # # BW matrix [MHz]
    # C = 1 * np.ones((N, N))
    # C = 1 * np.array([  [1, 1, 1, 1],  #means how connects to who
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1]])
    #------------------------------------------------------------------------
   
    N = 9

    # Adjacency matrix
    A = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1, 0, 1, 0, 0],
                  [1, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0]])

    
    # P = [(0.0, 0), (0.0, 0.01), (0.0, 0.02),
    #      (0.1, 0), (0.1, 0.01), (0.1, 0.02),
    #      (0.2, 0), (0.2, 0.01), (0.2, 0.02)]
   
    P = [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 1), (1, 2),
         (2, 0), (2, 1), (2, 2)]
    # BW matrix
    C = 1e3 * np.ones((N, N)) # in KiloHertz
    #------------------------------------------------------------------------


    # number of paths to choose from
    action_size = 1000                      #search space limitaions?

    # flow demands in KiloByte
    F = [
        {"source": 0, "destination": 8, "packets": 10 *1e3, "time_constrain": 10 , 'flow_idx': 0 },
        {"source": 0, "destination": 7, "packets": 100 *1e3, "time_constrain": 10, 'flow_idx': 1}, #Packets [MegaBytes]
        {"source": 0, "destination": 6, "packets": 500*1e3, "time_constrain": 10, 'flow_idx': 2}, #Packets [MegaBytes]
        {"source": 0, "destination": 5, "packets": 1000*1e3, "time_constrain": 10, 'flow_idx': 3},
        {"source": 0, "destination": 4, "packets": 700 *1e3, "time_constrain": 10 , 'flow_idx': 4 },
        {"source": 0, "destination": 3, "packets": 300 *1e3, "time_constrain": 10, 'flow_idx': 5}, #Packets [MegaBytes]
        {"source": 0, "destination": 2, "packets": 150*1e3, "time_constrain": 10, 'flow_idx': 6}, #Packets [MegaBytes]
        {"source": 0, "destination": 1, "packets": 50*1e3, "time_constrain": 10, 'flow_idx': 7}
    ]

    slotted_env = SlottedGraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = True,
                                        direction = 'minimize',
                                        slot_duration = int(1 * 1e3),          # [in SEC ]
                                        Tot_num_of_timeslots = 1000,         # [num of time slots]
                                        render_mode = False,
                                        trx_power_mode='gain',
                                        channel_gain = 1,
                                        # channel_manual_gain = [100,200,3,400,500,600],
                                        simualte_residauls = False)

    UNslotted_env = SlottedGraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = True,
                                        direction = 'minimize',
                                        slot_duration=int(1000 * 1e3),          # [in SEC]
                                        Tot_num_of_timeslots = 1, # [in Minutes]
                                        render_mode = False,
                                        trx_power_mode='gain',
                                        channel_gain = 1,
                                        # channel_manual_gain = [100,200,3,400,500,600],
                                        simualte_residauls = False)    

    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH)
    
    Tot_rates_sloted = slotted_diamond(slotted_env, grrl_data=False)
    Tot_rates_UNslotted = slotted_diamond(UNslotted_env, grrl_data=False)


    # plot rates
    time_axis = list(range(len(Tot_rates_sloted)))
    plt.figure()
    plt.plot(time_axis[:], Tot_rates_sloted[:], linestyle='-', color='b', label='Slotted Avg Rate [Avg over all flows]')
    plt.plot(time_axis[:], Tot_rates_UNslotted[:], linestyle='-', color='r', label='UnSlotted Avg Rate [Avg over all flows]')
    # Add labels and title
    plt.xlabel('Time (mili seconds)')
    plt.ylabel('Average Rate [Kbps]')
    plt.title('Average Rates Over Time')
    plt.legend()

    # Show the plot
    plt.savefig('average_rates_over_time.png')


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
