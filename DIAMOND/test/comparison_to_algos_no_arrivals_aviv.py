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
from environment.data import generate_slotted_env, generate_env
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR
from environment.utils import (plot_slotted_vs_not_slotted_graph, save_arguments_to_file, load_arguments_from_file, create_video_from_images,
                               wrap_and_save_Rate_plots, plot_rates_no_arrivals_aviv, plot_rates_multiple_algorithms)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import copy

if __name__ == "__main__":


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

    load_arguments = True
    # ---------------- Loading args For Observation -------------- #
    if load_arguments:
        subfolder_path = r'C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\random_topologies\10_Nodes_20_Edges\20250209_173523_10_Flows'
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

    # ------------------------- Create envs ----------------------------------- #

    slotted_env = generate_slotted_env(**slotted_env_args)

    un_slotted_env = generate_slotted_env(**un_slotted_env_args)
    ospf_env = copy.deepcopy(un_slotted_env)
    RB_env = copy.deepcopy(un_slotted_env)
    ICAR_env = copy.deepcopy(un_slotted_env)
    # dqn_gnn_env = copy.deepcopy(un_slotted_env)

    # ------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------- #
    save_arguments = True
    # --------------- Save function arguments For Analysis -------------------- #
    if save_arguments:
        # base_path = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\random_topologies"
        base_path = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_competition\random_topologies"
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

    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH)

    # ------------------------------  Competition ------------------------------- #
    ospf = OSPF()
    RB = RandomBaseline(num_trials=1, env=RB_env)
    ICAR = IACR(delta=0.5, alpha=1.3)
    # DQN_GNN = DQN_GNN(k=4)
    # ospf_env = generate_env(**slotted_env_args)
    # ospf_run_env = copy.deepcopy(un_slotted_env)
    # ospf_env.show_graph()

    # _, _, _, _, Tot_rates_ospf = ospf.run(un_slotted_env, seed=un_slotted_env.seed)

    # Sort by the paths by flows
    # ospf_paths = [action[1] for action in sorted(ospf_actions, key=lambda x: x[0])]

    # Get manual decisions
    # ospf_manual_decisions = [[idx, un_slotted_env.possible_actions[idx].index(path)] for idx, path in enumerate(ospf_paths)]
    # --------------------------------------------------------------------------- #
    # --------------------------------------------- Run ---------------------------------------- #

    # --------------------------------------------slotted -------------------------------------- #
    print(f'Started DIAMOND_Slotted Algorithm\n')
    # diamond_paths_slotted, action_recipe_slotted, Tot_rates_slotted = slotted_diamond(slotted_env)

    # manual_decisions = [[action[0], action[1]] for action in action_recipe_slotted[0]]  # action_recipe_slotted[0]   # action_recipe_slotted[:slotted_env.original_num_flows]

    # ------------------------------------------- un slotted and competition ----------------------------------- #
    print(f'Started DIAMOND_UN_Slotted Algorithm\n')
    # diamond_paths_un_slotted, action_recipe_un_slotted, Tot_rates_un_slotted = slotted_diamond(un_slotted_env,
    #                                                                                            grrl_data=True,
    #                                                                                            manual_actions=manual_decisions)

    print(f'Started OSPF Algorithm\n')
    # ospf_actions, ospf_paths, _, Tot_rates_ospf = ospf.run(ospf_env, seed=ospf_env.seed)

    print(f'Started Random Baseline Algorithm\n')
    RB_paths, _, Tot_rates_RB = RB.run(seed=RB_env.seed)

    print(f'Started ICAR Algorithm\n')
    # icar_paths, icar_rewards, Tot_rates_icar = ICAR.run(ICAR_env, seed=ICAR_env.seed)

    # dqn_gnn_paths, dqn_gnn_rewards, Tot_rates_dqn_gnn = DQN_GNN.run(dqn_gnn_env, seed=dqn_gnn_env.seed)

    # ospf_paths, ospf_action_recipe, Tot_rates_ospf = slotted_diamond(ospf_run_env,
    #                                                                  grrl_data=True,
    #                                                                  manual_actions=ospf_manual_decisions)
    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------------- Plots ---------------------------------- #

    # plot_rates_no_arrivals_aviv(Tot_rates_slotted=Tot_rates_slotted,
    #                             Tot_rates_un_slotted=Tot_rates_un_slotted,
    #                             slotted_env=slotted_env,
    #                             subfolder_path=subfolder_path if save_arguments else None,
    #                             save_arguments=save_arguments)

    plot_rates_multiple_algorithms(algo_names=["DIAMOND_Slotted", "DIAMOND_UnSlotted", "OSPF", "RB", "ICAR"],
                                   algo_tot_rates=[Tot_rates_slotted, Tot_rates_un_slotted, Tot_rates_ospf, Tot_rates_RB, Tot_rates_icar],
                                   slotted_env=slotted_env,
                                   subfolder_path=subfolder_path if save_arguments else None,
                                   save_arguments=save_arguments)
    print('finished')

