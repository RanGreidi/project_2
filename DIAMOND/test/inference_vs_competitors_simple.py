import numpy as np
import random
import os
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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

    # ------------------------------------------------------------------------
    Simulation_Time_Resolution = 5e-1       # miliseconds (i.e. each time step is a milisecond - this is the duration of each time step in [SEC])
    BW_value_in_Hertz = 1e6                   # wanted BW in Hertz
    slot_duration = 3                     # [SEC] 
    Tot_num_of_timeslots = 200               # [num of time slots]
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
    
    # A = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
    #               [1, 0, 1, 0, 1, 0, 0, 0, 0],
    #               [0, 1, 0, 0, 0, 1, 0, 0, 0],
    #               [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #               [0, 1, 0, 1, 0, 1, 0, 1, 0],
    #               [0, 0, 1, 0, 1, 0, 0, 0, 1],
    #               [0, 0, 0, 1, 0, 0, 0, 1, 0],
    #               [0, 0, 0, 0, 1, 0, 1, 0, 1],
    #               [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    
    A = np.array([[0, 1, 0, 1, 1, 1, 0, 1, 0],
                 [1, 0, 1, 1, 1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 1, 0, 0, 0],
                 [1, 1, 0, 0, 1, 0, 1, 0, 0],
                 [1, 1, 0, 1, 0, 1, 0, 1, 0],
                 [1, 0, 1, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 1, 0, 1, 0, 1],
                 [0, 0, 0, 0, 0, 1, 0, 1, 0]])   
    
    # P = [(0.0, 0), (0.0, 0.01), (0.0, 0.02),
    #      (0.1, 0), (0.1, 0.01), (0.1, 0.02),
    #      (0.2, 0), (0.2, 0.01), (0.2, 0.02)]
   
    P = [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 1), (1, 2),
         (2, 0), (2, 1), (2, 2)]
    
    # BW matrix
    C = BW_value_in_Hertz * np.ones((N, N)) * Simulation_Time_Resolution
    #------------------------------------------------------------------------


    # number of paths to choose from
    action_size = 44                      #search space limitaions?

    # flow demands in KiloByte
    F = [
        {"source": 0, "destination": 8, "packets": 3  *1e6    , "time_constrain": 10 , 'flow_idx': 0 , 'constant_flow_name': 0},
        {"source": 1, "destination": 8, "packets": 10 *1e6    , "time_constrain": 10, 'flow_idx': 1, 'constant_flow_name': 1},         #Packets [in Bits] 
        {"source": 3, "destination": 8, "packets": 30 *1e6    , "time_constrain": 10, 'flow_idx': 2, 'constant_flow_name': 2},         #Packets [in Bits]   
        {"source": 4, "destination": 8, "packets": 1000 *1e6    , "time_constrain": 10, 'flow_idx': 3, 'constant_flow_name': 3},
    ]

    # F = [
    #     {"source": 0, "destination": 8, "packets": 2    *1e6, "time_constrain": 10 , 'flow_idx': 0 , 'constant_flow_name': 0},
    #     {"source": 0, "destination": 7, "packets": 1    *1e6, "time_constrain": 10, 'flow_idx': 1 , 'constant_flow_name': 1},         #Packets [in Bits]
    #     {"source": 0, "destination": 6, "packets": 1  *1e6, "time_constrain": 10, 'flow_idx': 2 , 'constant_flow_name': 2},         #Packets [in Bits]
    #     {"source": 0, "destination": 5, "packets": 18 *1e6, "time_constrain": 10, 'flow_idx': 3 , 'constant_flow_name': 3},
    #     {"source": 0, "destination": 4, "packets": 15  *1e6, "time_constrain": 10 , 'flow_idx': 4 , 'constant_flow_name': 4},
    #     {"source": 0, "destination": 3, "packets": 1  *1e6, "time_constrain": 10, 'flow_idx': 5 , 'constant_flow_name': 5},         #Packets [in Bits]
    #     {"source": 0, "destination": 2, "packets": 1  *1e6, "time_constrain": 10, 'flow_idx': 6 , 'constant_flow_name': 6},         #Packets [in Bits]
    #     {"source": 0, "destination": 1, "packets": 15   *1e6, "time_constrain": 10, 'flow_idx': 7 , 'constant_flow_name': 7}
    # ]

    slotted_env = SlottedGraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = True,
                                        direction = 'minimize',
                                        slot_duration = int(slot_duration / Simulation_Time_Resolution),          # [in SEC ]
                                        Tot_num_of_timeslots = Tot_num_of_timeslots,         # [num of time slots]
                                        render_mode = True,
                                        trx_power_mode='gain',
                                        channel_gain = 1,
                                        # channel_manual_gain = [100,200,3,400,500,600],
                                        simualte_residauls = True,
                                        Simulation_Time_Resolution = Simulation_Time_Resolution,
                                        is_slotted = True)

    UNslotted_env = SlottedGraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = False,
                                        direction = 'minimize',
                                        slot_duration = int( (slot_duration*Tot_num_of_timeslots) / Simulation_Time_Resolution),          # [in SEC]
                                        Tot_num_of_timeslots = 1, # [in Minutes]
                                        render_mode = False,
                                        trx_power_mode='gain',
                                        channel_gain = 1,
                                        # channel_manual_gain = [100,200,3,400,500,600],
                                        simualte_residauls = False,
                                        Simulation_Time_Resolution = Simulation_Time_Resolution,
                                        is_slotted = False)    

    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH)
    
    Tot_rates_sloted = slotted_diamond(slotted_env, grrl_data=False)
    Tot_rates_UNslotted = slotted_diamond(UNslotted_env, grrl_data=False)


    # plot rates
    time_axis_in_resulotion = [i * Simulation_Time_Resolution for i in range(1,len(Tot_rates_sloted)+1)] # This time axis is a samples of each Simulation_Time_Resolution
    # we want to avarge rates so that we have time axis sampled in seconds (this way spike due to the residual will be smoothed)
    time_axis_in_seconds = [i  for i in range(1,int(slot_duration*Tot_num_of_timeslots)+1)]

    interpolator_sloted = interp1d(time_axis_in_resulotion, Tot_rates_sloted, kind='linear')
    Tot_rates_sloted_interpolated = interpolator_sloted(time_axis_in_seconds)

    interpolator_unsloted = interp1d(time_axis_in_resulotion, Tot_rates_UNslotted, kind='linear')
    Tot_rates_Unsloted_interpolated = interpolator_unsloted(time_axis_in_seconds)

    plt.figure()
    plt.plot(time_axis_in_seconds, Tot_rates_sloted_interpolated, linestyle='-', color='b', label='Slotted Avg Rate [Avg over all flows]')
    if np.isnan(Tot_rates_sloted_interpolated).any():
        nan_index = np.where(np.isnan(Tot_rates_sloted_interpolated))[0][0]
    else:
        nan_index = len(Tot_rates_sloted_interpolated)
    plt.axvline(x=nan_index, color='b', linestyle='--', label='Slotted flows are done')

    plt.plot(time_axis_in_seconds, Tot_rates_Unsloted_interpolated, linestyle='-', color='r', label='UnSlotted Avg Rate [Avg over all flows]')
    if np.isnan(Tot_rates_Unsloted_interpolated).any():
        nan_index = np.where(np.isnan(Tot_rates_Unsloted_interpolated))[0][0]
    else:
        nan_index = len(Tot_rates_Unsloted_interpolated)
    plt.axvline(x=nan_index, color='r', linestyle='--', label='UnSlotted flows are done')

    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Rate [bps]')
    plt.title('Average Rates Over Time')
    plt.legend()

    # Show the plot
    plt.savefig('DIAMOND/result_average_rates_over_time.png')


'''
timing guide:
working with resolution of seconds: 
each time step is a second, if BW is in Hertz then capcaity is in bps, than the rate is in bps and time axis is in seconds

working with resolution of miliseconds: 
each time step is a milisecond, if BW is in Hertz need to divide BW with *1e3, then capcaity is in bps, than the rate is in bps and time axis is in miliseconds

working with resolution of micro seoconds: 
each time step is a microsecond, if BW is i Hertz need to devide it by *1e6 then capcaity is in bps, than the rate is in bps and time axis is in microseconds

'''


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
