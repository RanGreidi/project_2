import numpy as np
import random
import os
from datetime import datetime
import shutil
import copy

import sys
sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')
from environment import SlottedGraphEnvPower
from diamond import DIAMOND, SlottedDIAMOND
from environment import generate_env
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR
from environment.Traffic_Probability_Model import Traffic_Probability_Model
from environment.utils import wrap_and_save_Rate_plots

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


if __name__ == "__main__":
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)

    # ------------------------------------------------------------------------
    Simulation_Time_Resolution = 1e-1       # miliseconds (i.e. each time step is a milisecond - this is the duration of each time step in [SEC])
    BW_value_in_Hertz = 1e6                   # wanted BW in Hertz
    slot_duration = 15                     # [SEC] 
    Tot_num_of_timeslots = 60               # [num of time slots]

    #------------------------------------------------------------------------

    # Traffic model
    Trafic_model_flow0 = Traffic_Probability_Model()
    Trafic_model_flow1 = Traffic_Probability_Model()
    Trafic_model_flow2 = Traffic_Probability_Model()
    Trafic_model_flow3 = Traffic_Probability_Model()
    Trafic_model_flow4 = Traffic_Probability_Model()
    Trafic_model_flow5 = Traffic_Probability_Model()
    Trafic_model_flow6 = Traffic_Probability_Model()
    Trafic_model_flow7 = Traffic_Probability_Model()        
    Trafic_model_list = [Trafic_model_flow1 , Trafic_model_flow2 , Trafic_model_flow3 , Trafic_model_flow4 , Trafic_model_flow5 , Trafic_model_flow6 , Trafic_model_flow7]

    # constant_flow_name must start with 0 and be an int!
    F = [
        {"source": 0, "destination": 8, "packets": 10   *1e5, "time_constrain": 10, 'flow_idx':  0 , 'constant_flow_name': 0},
        {"source": 0, "destination": 7, "packets": 100  *1e5, "time_constrain": 10, 'flow_idx':  1 , 'constant_flow_name': 1},         #Packets [in Bits]
        {"source": 0, "destination": 6, "packets": 500  *1e6, "time_constrain": 10, 'flow_idx':  2 , 'constant_flow_name': 2},         #Packets [in Bits]
        {"source": 0, "destination": 5, "packets": 200  *1e6, "time_constrain": 10, 'flow_idx':  3 , 'constant_flow_name': 3},
        {"source": 0, "destination": 4, "packets": 50   *1e6, "time_constrain": 10, 'flow_idx':  4 , 'constant_flow_name': 4},
        {"source": 0, "destination": 3, "packets": 10   *1e6, "time_constrain": 10, 'flow_idx':  5 , 'constant_flow_name': 5},         #Packets [in Bits]
        {"source": 0, "destination": 2, "packets": 30   *1e6, "time_constrain": 10, 'flow_idx':  6 , 'constant_flow_name': 6},         #Packets [in Bits]
        {"source": 0, "destination": 1, "packets": 40   *1e6, "time_constrain": 10, 'flow_idx':  7 , 'constant_flow_name': 7}
    ]

    F = F[0:2]
    Trafic_model_list = Trafic_model_list[0:2]
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
   
    # # number of nodes
    N = 9

    # Adjacency matrix
    A = np.array([[0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 1, 0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0]])

    
    P = [(0.0, 0), (0.0, 0.01), (0.0, 0.02),
         (0.1, 0), (0.1, 0.01), (0.1, 0.02),
         (0.2, 0), (0.2, 0.01), (0.2, 0.02)]
   
    # P = [(0, 0), (0, 10), (0, 20),
    #      (10, 0), (10, 10), (10, 20),
    #      (20, 0), (20, 10), (20, 20)]
    
    # BW matrix
    C = BW_value_in_Hertz * np.ones((N, N)) * Simulation_Time_Resolution 
    #------------------------------------------------------------------------

    # number of paths to choose from
    action_size = 20                      #search space limitaions?

    #------------------------------------------------------------------------
    # create environment
    slotted_env = SlottedGraphEnvPower( adjacency_matrix=A,
                                        bandwidth_matrix=C,
                                        flows=F,
                                        node_positions=P,
                                        k=action_size,
                                        reward_weights=reward_weights,
                                        telescopic_reward = True,
                                        direction = 'minimize',
                                        slot_duration = int(slot_duration / Simulation_Time_Resolution),            # [in SEC ]
                                        Tot_num_of_timeslots = Tot_num_of_timeslots,                                # [num of time slots]
                                        render_mode = False,
                                        trx_power_mode='gain',
                                        channel_gain = 1,
                                        # channel_manual_gain = [100,200,3,400,500,600],
                                        simualte_residauls = True,
                                        Simulation_Time_Resolution = Simulation_Time_Resolution)

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
                                          Simulation_Time_Resolution = Simulation_Time_Resolution)    
    
    slotted_env_real_run = copy.deepcopy(slotted_env)
    UNslotted_env_real_run = copy.deepcopy(UNslotted_env)


    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH, Traffic_Probability_Model_list = Trafic_model_list)
    
    Tot_rates_sloted, action_recipe_slotted = slotted_diamond(slotted_env, grrl_data=False) # get recipe from here
    Tot_rates_sloted_RealRun = slotted_diamond.real_run(slotted_env_real_run, action_recipe_slotted) # run packets in the real world
    
    Tot_rates_UNslotted, action_recipe_Unslotted = slotted_diamond(UNslotted_env, grrl_data=False) # get recipe from here
    Tot_rates_UNsloted_RealRun = slotted_diamond.real_run(UNslotted_env_real_run, action_recipe_Unslotted) # run packets in the real world
 

 ## ----------------------  plots ---------------------- ##
    wrap_and_save_Rate_plots('Average_rates_over_time',
                            Simulation_Time_Resolution,
                            slot_duration,
                            Tot_num_of_timeslots,
                            Tot_rates_sloted,
                            Tot_rates_UNslotted                            
                             )

    wrap_and_save_Rate_plots('RealRun_Average_rates_over_time',
                            Simulation_Time_Resolution,
                            slot_duration,
                            Tot_num_of_timeslots,
                            Tot_rates_sloted_RealRun,
                            Tot_rates_UNsloted_RealRun                            
                             )
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
