import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import shutil
import sys

sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')
from environment import SlottedGraphEnvPower
from diamond_aviv import DIAMOND, SlottedDIAMOND
from environment.Traffic_Probability_Model import Traffic_Probability_Model, generate_traffic_matrix
from environment.data import generate_slotted_env
# from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR
from environment.utils import plot_slotted_vs_not_slotted_graph, save_arguments_to_file, load_arguments_from_file, create_video_from_images, wrap_and_save_Rate_plots

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import copy

if __name__ == "__main__":

    # image_folder = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\costume_topologies\graph_images\9_Nodes_17_Edges\slotted"
    # output_video_path = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\costume_topologies\graph_videos\9_Nodes_17_Edges\slotted\graph_video.mp4"
    #
    # # Call the function to create a video from images
    # create_video_from_images(image_folder, output_video_path, fps=1, file_extension="png")

    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)
    # ------------------------------------------------------------------------
    Simulation_Time_Resolution = 1e-1  # 1e-2  # miliseconds (i.e. each time step is a milisecond - this is the duration of each time step in [SEC])
    BW_value_in_Hertz = 1e6  # 1e6                   # wanted BW in Hertz
    slot_duration = 60 # 8 [SEC] 60
    Tot_num_of_timeslots = 15  # 3 60  # [num of time slots]
    # ------------------------------------------------------------------------

    #  number of nodes
    # N = 4

    #  Adjacency matrix
    #  create 3x3 mesh graph
    # A = np.array([[0, 1, 1, 1],  #means how connects to who
    #               [1, 0, 1, 1],
    #               [1, 1, 0, 1],
    #               [1, 1, 1, 0]])

    # A = np.array([[0, 1, 0, 1],  #means how connects to who
    #               [1, 0, 1, 0],
    #               [0, 1, 0, 1],
    #               [1, 0, 1, 0]])

    # P = [(0, 0), (0, 1), (0, 2),                #the position of each node
    #      (1, 0), (1, 1), (1, 2),
    #      (2, 0), (2, 1), (2, 2)]

    # P = [(1, 1), (1, 2),                 #the position of each node
    #      (2, 2), (2, 1)]
    #
    # P = np.array(P) * 0.00001

    #  BW matrix [MHz]
    # C = 1 * np.ones((N, N))
    # C = 1 * np.array([[1, 1, 1, 1],  #means how connects to who
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1],
    #                     [1, 1, 1, 1]])

    # C = BW_value_in_Hertz * np.ones((N, N)) * Simulation_Time_Resolution

    # F = [
    #     {"source": 0, "destination": 2, "packets": 1000 * 1e6, "time_constrain": 10, 'flow_idx': 0},
    #     {"source": 0, "destination": 2, "packets": 10 * 1e6, "time_constrain": 10, 'flow_idx': 1}
    # ]

    # ------------------------------------------------------------------------

    N = 9
    #
    # # Adjacency matrix
    #
    A = np.array([[0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0, 0],
                  [1, 1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 1, 0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 0, 0, 0, 1, 0],
                  [1, 0, 0, 0, 1, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0, 1, 0]])
    #
    P = [(0.0, 0), (0.0, 0.01), (0.0, 0.02),
         (0.1, 0), (0.1, 0.01), (0.1, 0.02),
         (0.2, 0), (0.2, 0.01), (0.2, 0.02)]
    #
    # P = [(0, 0), (0, 10), (0, 20),
    #      (10, 0), (10, 10), (10, 20),
    #      (20, 0), (20, 10), (20, 20)]
    # # BW matrix
    C = BW_value_in_Hertz * np.ones((N, N)) * Simulation_Time_Resolution
    #
    #  flow demands

    # F = [
    #     {"source": 0, "destination": 8, "packets": 10 * 1e5, "time_constrain": 10, 'flow_idx': 0},         # {"source": 0, "destination": 8, "packets": 900 * 1e6, "time_constrain": 10, 'flow_idx': 0}
    #     {"source": 0, "destination": 7, "packets": 100 * 1e5, "time_constrain": 10, 'flow_idx': 1},
    # ]

    # Traffic model
    Trafic_model_flow0 = Traffic_Probability_Model(source=0, destination=8, constant_flow_name=0)
    Trafic_model_flow1 = Traffic_Probability_Model(source=0, destination=7, constant_flow_name=1)
    Trafic_model_flow2 = Traffic_Probability_Model(source=0, destination=6, constant_flow_name=2)
    Trafic_model_flow3 = Traffic_Probability_Model(source=0, destination=5, constant_flow_name=3)
    Trafic_model_flow4 = Traffic_Probability_Model(source=0, destination=4, constant_flow_name=4)
    Trafic_model_flow5 = Traffic_Probability_Model(source=0, destination=3, constant_flow_name=5)
    Trafic_model_flow6 = Traffic_Probability_Model(source=0, destination=2, constant_flow_name=6)
    Trafic_model_flow7 = Traffic_Probability_Model(source=0, destination=1, constant_flow_name=7)
    Trafic_model_list = [Trafic_model_flow0, Trafic_model_flow1, Trafic_model_flow2, Trafic_model_flow3,
                             Trafic_model_flow4, Trafic_model_flow5, Trafic_model_flow6, Trafic_model_flow7]
    # TODO: Inital demand should be the initial state in the probablity model, if start with 0 packets, then the first state should be 0, else first state should be the first demand

    # Inital flow demand list
    # constant_flow_name must start with 0 and be an int!
    F = [
        {"source": 0, "destination": 8, "packets": 10 * 1e5, "time_constrain": 10, 'flow_idx': 0, 'constant_flow_name': 0},
        {"source": 0, "destination": 7, "packets": 100 * 1e5, "time_constrain": 10, 'flow_idx': 1, 'constant_flow_name': 1},  # Packets [in Bits]
        {"source": 0, "destination": 6, "packets": 500 * 1e6, "time_constrain": 10, 'flow_idx': 2, 'constant_flow_name': 2},  # Packets [in Bits]
        {"source": 0, "destination": 5, "packets": 200 * 1e6, "time_constrain": 10, 'flow_idx': 3, 'constant_flow_name': 3},
        {"source": 0, "destination": 4, "packets": 50 * 1e6, "time_constrain": 10, 'flow_idx': 4, 'constant_flow_name': 4},
        {"source": 0, "destination": 3, "packets": 10 * 1e6, "time_constrain": 10, 'flow_idx': 5, 'constant_flow_name': 5},  # Packets [in Bits]
        {"source": 0, "destination": 2, "packets": 30 * 1e6, "time_constrain": 10, 'flow_idx': 6, 'constant_flow_name': 6},  # Packets [in Bits]
        {"source": 0, "destination": 1, "packets": 40 * 1e6, "time_constrain": 10, 'flow_idx': 7, 'constant_flow_name': 7}
        ]

    F = F[0:2]
    Trafic_model_list = Trafic_model_list[0:2]
    # ------------------------------------------------------------------------

    # -------------------------- Aviv Topology ----------------------------------------------
    # N = 3
    #
    # # Adjacency matrix
    #
    # A = np.array([[0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 0]])
    #
    # P = [(0.0, 0), (0.1, 0.0), (0.05, 0.01)]
    #
    # # P = [(0, 0), (0, 1), (0, 2),
    # # (1, 0), (1, 1), (1, 2),
    # # (2, 0), (2, 1), (2, 2)]
    # # BW matrix
    # # C = 10 * np.ones((N, N))
    # C = 1e3 * np.array([[1, 10, 1],  #means how connects to who
    #                   [10, 1, 5],
    #                   [1, 5, 1]])
    #
    # # flow demands
    # F = [
    #     {"source": 0, "destination": 1, "packets": 10 * 1e3, "time_constrain": 10, 'flow_idx': 0},  # Packets [MegaBytes]
    #     {"source": 0, "destination": 1, "packets": 100 * 1e3, "time_constrain": 10, 'flow_idx': 1}
    # ]

    # -------------------------------- Ran Costume Topologies ----------------------------------------

    # number of paths to choose from
    action_size = 20  # search space limitaions?

    #
    slotted_env = SlottedGraphEnvPower(adjacency_matrix=A,
                                       bandwidth_matrix=C,
                                       flows=F,
                                       node_positions=P,
                                       k=action_size,
                                       reward_weights=reward_weights,
                                       telescopic_reward=False,
                                       direction='minimize',
                                       slot_duration= int(slot_duration / Simulation_Time_Resolution),          # [in SEC ]
                                       Tot_num_of_timeslots=Tot_num_of_timeslots,         # [num of time slots]
                                       render_mode=False,
                                       trx_power_mode='gain',
                                       channel_gain=1,
                                       # channel_manual_gain = [100,200,3,400,500,600],
                                       simulate_residuals=True,
                                       Simulation_Time_Resolution=Simulation_Time_Resolution)

    un_slotted_env = SlottedGraphEnvPower(adjacency_matrix=A,
                                       bandwidth_matrix=C,
                                       flows=F,
                                       node_positions=P,
                                       k=action_size,
                                       reward_weights=reward_weights,
                                       telescopic_reward=True,
                                       direction='minimize',
                                       slot_duration=int((slot_duration*Tot_num_of_timeslots) / Simulation_Time_Resolution),
                                       Tot_num_of_timeslots=1,  # [in Minutes]
                                       render_mode=False,
                                       trx_power_mode='gain',
                                       channel_gain=1,
                                       # channel_manual_gain = [100,200,3,400,500,600],
                                       simulate_residuals=False,
                                       Simulation_Time_Resolution=Simulation_Time_Resolution)

    # --------------------------------------------------------------------------------------------- #
    # -------------------- Try and use "generate_env" for random topology ------------------------ #

    # Function arguments
    slotted_env_args = {
        "num_nodes": 5,
        "num_edges": 5,
        "num_flows": 5,
        "min_flow_demand": 10 * 1e6,
        "max_flow_demand": 1000 * 1e6,
        "delta": 1 * 1e6,
        "num_actions": 20,
        "min_capacity": 1 * 1e6,  # [Hz]
        "max_capacity": 10 * 1e6,
        "direction": "minimize",
        "reward_balance": 0.8,
        "seed": 3544,
        "graph_mode": "random",
        "reward_weights": reward_weights,
        "telescopic_reward": True,
        "Simulation_Time_Resolution": Simulation_Time_Resolution,
        "slot_duration": slot_duration,
        "Tot_num_of_timeslots": Tot_num_of_timeslots,
        "render_mode": False,
        "trx_power_mode": "gain",
        "channel_gain": 1,
        "simulate_residuals": True,
        "given_flows": None,
        "max_position": 1.0,
        "slotted": True
    }

    un_slotted_env_args = {
                        "num_nodes": 5,
                        "num_edges": 5,
                        "num_flows": 5,
                        "min_flow_demand": 10 * 1e6,
                        "max_flow_demand": 1000 * 1e6,
                        "delta": 1 * 1e6,
                        "num_actions": 20,
                        "min_capacity": 1 * 1e6,
                        "max_capacity": 10 * 1e6,
                        "direction": "minimize",
                        "reward_balance": 0.8,
                        "seed": 3544,
                        "graph_mode": "random",
                        "reward_weights": reward_weights,
                        "telescopic_reward": True,
                        "Simulation_Time_Resolution": Simulation_Time_Resolution,
                        "slot_duration": Tot_num_of_timeslots * slot_duration,
                        "Tot_num_of_timeslots": 1,
                        "render_mode": False,
                        "trx_power_mode": "gain",
                        "channel_gain": 1,
                        "simulate_residuals": False,
                        "given_flows": None,
                        "max_position": 1.0,
                        "slotted": False
                    }

    load_arguments = False
    # ---------------- Loading args For Observation -------------- #
    if load_arguments:
        subfolder_path = r'C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\random_topologies\60_Nodes_70_Edges\20250129_174033_10_Flows'
        slotted_file_path = os.path.join(subfolder_path, "generate_slotted_env_args.json")
        slotted_env_args = load_arguments_from_file(filename=slotted_file_path)['args']

        slotted_env_args["simulate_residuals"] = True
        slotted_env_args["render_mode"] = False
        # slotted_env_args["Simulation_Time_Resolution"] = 1e-4
        # slotted_env_args["slot_duration"] = 60
        # slotted_env_args["Tot_num_of_timeslots"] = 2
        # slotted_env_args["max_position"] = 0.2
        # slotted_env_args["min_capacity"] = 10 * 1e6
        # slotted_env_args["num_flows"] = 20
        # slotted_env_args["seed"] = 14235

        # ----------------- Fit un slotted args -------------------- #
        un_slotted_env_args = copy.deepcopy(slotted_env_args)
        un_slotted_env_args["slot_duration"] = slotted_env_args["Tot_num_of_timeslots"] * slotted_env_args["slot_duration"]
        un_slotted_env_args["Tot_num_of_timeslots"] = 1
        un_slotted_env_args["simulate_residuals"] = False
    # ------------------------------------------------------------ #

    # -------------------- Arrival Matrix --------------------- #

    transition_matrix_properties = {
                                    'mice_start_from': 10,
                                    'elephent_start_from': 1000,
                                    'p_mice': 0.05,
                                    'p_elephent': 0.05,
                                    'p_idle': 0.9,
                                    'p_finish': 0.1,
                                    'p_same': 0.5,
                                    'p_minus': 0.2,
                                    'p_plus': 0.2,
                                    'Num_of_states': 101
                                    }

    TM = generate_traffic_matrix(num_flows=slotted_env_args["num_flows"],
                                 transition_matrix_properties=transition_matrix_properties,
                                 num_steps=slotted_env_args["Tot_num_of_timeslots"],)

    # ---------------------------------------------------------- #

    # ------------------ Create envs ------------------------------------------ #

    slotted_env = generate_slotted_env(**slotted_env_args, arrival_matrix=TM)

    un_slotted_env = generate_slotted_env(**un_slotted_env_args, arrival_matrix=TM)

    # ------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------- #
    save_arguments = False
    # --------------- Save function arguments For Analysis -------------------- #
    if save_arguments:
        base_path = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\random_topologies"
        subfolder_name = f"{slotted_env.num_nodes}_Nodes_{slotted_env.num_edges // 2}_Edges"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp
        subfolder_path = os.path.join(base_path, subfolder_name, f"{timestamp}_{slotted_env.num_flows}_Flows")
        # Ensure the directory exists
        os.makedirs(subfolder_path, exist_ok=True)
        # Full file path
        file_path_slotted = os.path.join(subfolder_path, "generate_slotted_env_args.json")
        save_arguments_to_file(filename=file_path_slotted, args=slotted_env_args)

    # ------------------------------------------------------------------------- #

    if save_arguments:
        slotted_env.plot_raw_graph(save_path=os.path.join(subfolder_path, "graph.png"))

    # slotted_env.plot_raw_graph()

    # ---------------------------- Initialize DIAMOND ------------------------- #

    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH, Traffic_Probability_Model_list=Trafic_model_list)

    # --------------------------------------------- Run ---------------------------------------- #
    slotted_env_real_run = copy.deepcopy(slotted_env)
    un_slotted_env_real_run = copy.deepcopy(un_slotted_env)

    # ---------------------- slotted ---------------- #
    diamond_paths_slotted, action_recipe_slotted, Tot_rates_slotted = slotted_diamond(slotted_env,
                                                                              grrl_data=True
                                                                              )
    # Tot_rates_slotted_RealRun = slotted_diamond.real_run(slotted_env_real_run, action_recipe_slotted)  # run packets in the real world

    manual_decisions = [[action[0], action[1]] for action in action_recipe_slotted[0]] # action_recipe_slotted[0]   # action_recipe_slotted[:slotted_env.original_num_flows]

    # --------------------- un slotted -------------- #
    diamond_paths_un_slotted, action_recipe_un_slotted, Tot_rates_un_slotted = slotted_diamond(un_slotted_env,
                                                                                               grrl_data=True,
                                                                                               manual_actions=manual_decisions)
    # Tot_rates_un_slotted_RealRun = slotted_diamond.real_run(un_slotted_env_real_run, action_recipe_un_slotted)  # run packets in the real world

    # ------------------------------------------------ #
    # -------------------------------------------------------------------------------------------- #
    # ------------------------------------------------- Plots ------------------------------------------- #
    # wrap_and_save_Rate_plots('Average_rates_over_time',
    #                                 Simulation_Time_Resolution,
    #                                 slot_duration,
    #                                 Tot_num_of_timeslots,
    #                                 Tot_rates_slotted,
    #                                 Tot_rates_un_slotted
    #                          )
    #
    # wrap_and_save_Rate_plots('RealRun_Average_rates_over_time',
    #                                   Simulation_Time_Resolution,
    #                                   slot_duration,
    #                                   Tot_num_of_timeslots,
    #                                   Tot_rates_slotted_RealRun,
    #                                   Tot_rates_un_slotted_RealRun
    #                          )
    # --------------------------------------------------------------------------------------------------- #
    # plot_slotted_vs_not_slotted_graph(mean_rate_over_all_timesteps, mean_rate_over_all_timesteps_raz)

    # # plot rates
    time_axis_in_resulotion = [i * slotted_env.Simulation_Time_Resolution for i in range(1, len(Tot_rates_slotted)+1)] # This time axis is a samples of each Simulation_Time_Resolution
    # we want to avarge rates so that we have time axis sampled in seconds (this way spike due to the residual will be smoothed)    slot_duration = int(slotted_env.slot_duration * slotted_env.Simulation_Time_Resolution)
    time_axis_in_seconds = [i for i in range(1, int(slotted_env.slot_duration * slotted_env.Simulation_Time_Resolution) * slotted_env.Tot_num_of_timeslots + 1)]  # range(1, slot_duration * slotted_env.Tot_num_of_timeslots + 1)

    interpolator_slotted = interp1d(time_axis_in_resulotion, Tot_rates_slotted, kind='linear')
    Tot_rates_slotted_interpolated = interpolator_slotted(time_axis_in_seconds)

    interpolator_unslotted = interp1d(time_axis_in_resulotion, Tot_rates_un_slotted, kind='linear')
    Tot_rates_unslotted_interpolated = interpolator_unslotted(time_axis_in_seconds)

    plt.figure()
    plt.plot(time_axis_in_seconds, Tot_rates_slotted_interpolated, linestyle='-', color='b', label='Slotted Avg Rate')
    try:
        nan_index = np.where(np.isnan(Tot_rates_slotted_interpolated))[0][0]
    except IndexError:
        nan_index = time_axis_in_seconds[-1]
    plt.axvline(x=nan_index, color='b', linestyle='--', label='Slotted flows are done')

    plt.plot(time_axis_in_seconds, Tot_rates_unslotted_interpolated, linestyle='-', color='r', label='UnSlotted Avg Rate')
    try:
        nan_index = np.where(np.isnan(Tot_rates_unslotted_interpolated))[0][0]
    except IndexError:
        nan_index = time_axis_in_seconds[-1]
    plt.axvline(x=nan_index, color='r', linestyle='--', label='UnSlotted flows are done')

    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Rate [bps]')
    plt.title('Average Rates Over Time')
    plt.legend(loc='best', prop={'size': 8})
    plt.grid(True)

    # Show the plot
    if save_arguments:
        plt.savefig(os.path.join(subfolder_path, 'average_rates_over_time.png'))
    plt.show()

    print('finished')

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
