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
from environment.utils import *

SEED = 123
episode_from = 7500
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import copy


def run(num_episodes=1, num_flows=10):

    data = {
        "GRRL_slotted": 0,  # 'diamond_slotted'
        "GRRL_unslotted": 0,  # 'diamond_unslotted'
        'ospf': 0,
        'icar': 0,
    }

    for episode in range(num_episodes):

        # seed
        seed = SEED + (episode + 1) + episode_from + 1

        reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)
        # ------------------------------------------------------------------------
        Simulation_Time_Resolution = 1e-1  # 1e-2  # miliseconds (i.e. each time step is a milisecond - this is the duration of each time step in [SEC])
        BW_value_in_Hertz = 1e6  # 1e6                   # wanted BW in Hertz
        slot_duration = 1  # 8 [SEC] 60
        Tot_num_of_timeslots = 1800  # 3 60  # [num of time slots]
        # ------------------------------------------------------------------------

        # -------------------- Try and use "generate_env" for random topology ------------------------ #

        # Function arguments
        slotted_env_args = {
            "num_nodes": 40,
            "num_edges": 70,
            "num_flows": num_flows,
            "min_flow_demand": 3 * 1e6,
            "max_flow_demand": 50 * 1e6,
            "delta": 1 * 1e6,
            "num_actions": 40,
            "min_capacity": 1 * 1e6,  # [Hz]
            "max_capacity": 10 * 1e6,
            "direction": "minimize",
            "reward_balance": 0.8,
            "seed": seed,
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
            "is_slotted": True
        }

        un_slotted_env_args = {
                            "num_nodes": 40,
                            "num_edges": 70,
                            "num_flows": num_flows,
                            "min_flow_demand": 3 * 1e6,
                            "max_flow_demand": 50 * 1e6,
                            "delta": 1 * 1e6,
                            "num_actions": 40,
                            "min_capacity": 1 * 1e6,
                            "max_capacity": 10 * 1e6,
                            "direction": "minimize",
                            "reward_balance": 0.8,
                            "seed": seed,
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
                            "is_slotted": False
                        }

        load_arguments = False
        # ---------------- Loading args For Observation -------------- #
        if load_arguments:
            # subfolder_path = path
            subfolder_path = r'C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_competition\random_topologies\8_Nodes_13_Edges\20250216_102825_10_Flows'
            slotted_file_path = os.path.join(subfolder_path, "generate_slotted_env_args.json")
            slotted_env_args = load_arguments_from_file(filename=slotted_file_path)['args']

            slotted_env_args["simulate_residuals"] = True
            slotted_env_args["render_mode"] = False
            # slotted_env_args["Simulation_Time_Resolution"] = 1e-4
            # slotted_env_args["slot_duration"] = 60
            # slotted_env_args["Tot_num_of_timeslots"] = 20
            # slotted_env_args["max_position"] = 0.2
            # slotted_env_args["min_capacity"] = 10 * 1e6
            # slotted_env_args["num_flows"] = 20
            # slotted_env_args["seed"] = seed
            slotted_env_args["is_slotted"] = True

            # ----------------- Fit un slotted args -------------------- #
            un_slotted_env_args = copy.deepcopy(slotted_env_args)
            un_slotted_env_args["slot_duration"] = slotted_env_args["Tot_num_of_timeslots"] * slotted_env_args["slot_duration"]
            un_slotted_env_args["Tot_num_of_timeslots"] = 1
            un_slotted_env_args["simulate_residuals"] = False
            un_slotted_env_args["is_slotted"] = False
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

        # base_path = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_unslotted\random_topologies"
        base_path = r"C:\Users\beaviv\Ran_DIAMOND_Plots\slotted_vs_competition\random_topologies"
        subfolder_name = f"{slotted_env.num_nodes}_Nodes_{slotted_env.num_edges // 2}_Edges"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp
        subfolder_path = os.path.join(base_path, subfolder_name, f"{timestamp}_{slotted_env.num_flows}_Flows")

        if save_arguments:
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
        slotted_diamond = SlottedDIAMOND(grrl_model_path=MODEL_PATH, nb3r_steps=5)

        # ------------------------------  Competition ------------------------------- #
        ospf = OSPF()
        RB = RandomBaseline(num_trials=1, env=RB_env)
        ICAR = IACR(delta=0.5, alpha=1.3)
        # DQN_GNN = DQN_GNN(k=4)

        # --------------------------------------------------------------------------- #
        # --------------------------------------------- Run ---------------------------------------- #

        # --------------------------------------------slotted -------------------------------------- #
        print(f'Started DIAMOND_Slotted Algorithm\n')
        diamond_paths_slotted, action_recipe_slotted, Tot_rates_slotted = slotted_diamond(slotted_env)
        diamond_interpolated_rates, nan_index = interpolate_rates(slotted_env=slotted_env, tot_rates=Tot_rates_slotted)
        data["GRRL_slotted"] += np.mean(diamond_interpolated_rates[:nan_index])

        # ------------------------------------------- un slotted ----------------------------------- #
        print(f'Started DIAMOND_UN_Slotted Algorithm\n')
        manual_decisions = [[action[0], action[1]] for action in action_recipe_slotted[0]]  # action_recipe_slotted[0]   # action_recipe_slotted[:slotted_env.original_num_flows]
        diamond_paths_un_slotted, action_recipe_un_slotted, Tot_rates_un_slotted = slotted_diamond(un_slotted_env,
                                                                                                   grrl_data=True,
                                                                                                   manual_actions=manual_decisions)
        un_slotted_diamond_interpolated_rates, nan_index = interpolate_rates(slotted_env=slotted_env, tot_rates=Tot_rates_un_slotted)
        data["GRRL_unslotted"] += np.mean(un_slotted_diamond_interpolated_rates[:nan_index])
        # ------------------------------------------------ OSPF --------------------------------------------------- #

        print(f'Started OSPF Algorithm\n')
        ospf_actions, ospf_paths, _, Tot_rates_ospf = ospf.run(ospf_env, seed=ospf_env.seed)
        ospf_interpolated_rates, nan_index = interpolate_rates(slotted_env=slotted_env, tot_rates=Tot_rates_ospf)
        data["ospf"] += np.mean(ospf_interpolated_rates[:nan_index])
        # ----------------------------------------------------- RB ------------------------------------------------- #

        # print(f'Started Random Baseline Algorithm\n')
        # RB_paths, _, Tot_rates_RB = RB.run(seed=RB_env.seed)
        # rb_interpolated_rates, nan_index = interpolate_rates(slotted_env=slotted_env, tot_rates=Tot_rates_RB)
        # data["RB"] += np.mean(rb_interpolated_rates[:nan_index])
        # --------------------------------------------------- ICAR -------------------------------------------------- #

        print(f'Started ICAR Algorithm\n')
        icar_paths, icar_rewards, Tot_rates_icar = ICAR.run(ICAR_env, seed=ICAR_env.seed)
        icar_interpolated_rates, nan_index = interpolate_rates(slotted_env=slotted_env, tot_rates=Tot_rates_icar)
        data["icar"] += np.mean(icar_interpolated_rates[:nan_index])
        # --------------------------------------------------- DQN+GNN  --------------------------------------------- #

        # dqn_gnn_paths, dqn_gnn_rewards, Tot_rates_dqn_gnn = DQN_GNN.run(dqn_gnn_env, seed=dqn_gnn_env.seed)

        # ------------------------------------------------------------------------------------------ #
        # ------------------------------------------------- Plots ---------------------------------- #

        plot_rates_multiple_algorithms(algo_names=["DIAMOND_Slotted", "DIAMOND_UnSlotted", "OSPF", "ICAR"],
                                       algo_tot_rates=[Tot_rates_slotted, Tot_rates_un_slotted, Tot_rates_ospf, Tot_rates_icar],
                                       slotted_env=slotted_env,
                                       subfolder_path=subfolder_path if save_arguments else None,
                                       save_arguments=save_arguments)
        print('finished')

    # average
    for d in data:
        data[d] /= num_episodes

    return data, subfolder_path


if __name__ == "__main__":

    flows = [10, 15, 20]  # [10, 20, 30]
    episodes = 3
    algo_rates = []

    for num_flows in flows:
        data, subfolder_path = run(num_episodes=episodes, num_flows=num_flows)
        algo_rates.append(list(data.values()))

    algo_names = list(data.keys())

    plot_algorithm_rates(flows=flows, algo_rates=algo_rates, algo_names=algo_names, save_arguments=True,subfolder_path=subfolder_path)

