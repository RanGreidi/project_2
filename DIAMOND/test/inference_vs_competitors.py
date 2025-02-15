import numpy as np
import random
import os
from datetime import datetime
import shutil
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')

from diamond import DIAMOND, SlottedDIAMOND
from environment import generate_env
from environment import SlottedGraphEnvPower
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


class TestvsCompetitors:
    def __init__(self,
                 num_episodes=100,
                 num_rb_trials=100,
                 **kwargs):

        self.MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
        self.reward_weights = dict(rate_weight=0.5, delay_weight=0, interference_weight=0, capacity_reduction_weight=0)


        self.Simulation_Time_Resolution = 1e-1       # miliseconds (i.e. each time step is a milisecond - this is the duration of each time step in [SEC])
        self.BW_value_in_Hertz = 1e6                   # wanted BW in Hertz
        self.slot_duration = 1                     # [SEC] 
        self.Tot_num_of_timeslots = 20000000               # [num of time slots]

        self.action_size = 100 

    
        self.num_episodes = num_episodes
        self.episode_from = kwargs.get('episode_from', 0)

        self.competitors = {
            #'DQN+GNN': DQN_GNN(k=4),
            #'OSPF': OSPF(),
            #'RandomBL': RandomBaseline(num_trials=num_rb_trials),
            #'DIAR': DIAR(n_iter=2),
            #'IACR': IACR(delta=0.5, alpha=1.3),
        }

    def __call__(self, **kwargs):

        # update variables
        self.num_nodes = kwargs.get('num_nodes', 10)
        self.num_edges = kwargs.get('num_edges', 20)
        self.num_flows = kwargs.get('num_flows', 20)
        # self.num_actions = kwargs.get('num_actions', 4)


        for episode in range(self.num_episodes):

            # seed
            seed = SEED + (episode + 1) + self.episode_from + 1

            # generate env
            adjacency,capacity_matrix,interference_matrix,positions,flows = generate_env(   num_nodes=self.num_nodes,
                                                                                            num_edges=self.num_edges,
                                                                                            num_actions=self.action_size,
                                                                                            num_flows=self.num_flows,
                                                                                            min_flow_demand=kwargs.get('min_flow_demand', int(3* 1e6)),
                                                                                            max_flow_demand=kwargs.get('max_flow_demand', int(100 * 1e6)),
                                                                                            min_capacity=kwargs.get('min_capacity', int(1e5)),
                                                                                            max_capacity=kwargs.get('max_capacity', int(1e6)),
                                                                                            seed=seed,
                                                                                            graph_mode=kwargs.get('graph_mode', 'random'),
                                                                                            trx_power_mode=kwargs.get('trx_power_mode', 'equal'),
                                                                                            rayleigh_scale=kwargs.get('rayleigh_scale'),
                                                                                            max_trx_power=kwargs.get('max_trx_power'),
                                                                                            channel_gain=kwargs.get('channel_gain'))

            slotted_env = SlottedGraphEnvPower( adjacency_matrix=adjacency,
                                                bandwidth_matrix=capacity_matrix,
                                                interference_matrix=interference_matrix,
                                                flows=flows,
                                                node_positions=positions,
                                                k=self.action_size,
                                                reward_weights=self.reward_weights,
                                                telescopic_reward = True,
                                                direction = 'minimize',
                                                slot_duration = int(self.slot_duration / self.Simulation_Time_Resolution),          # [in SEC ]
                                                Tot_num_of_timeslots = self.Tot_num_of_timeslots,         # [num of time slots]
                                                render_mode = True,
                                                trx_power_mode='gain',
                                                channel_gain = 1,
                                                # channel_manual_gain = [100,200,3,400,500,600],
                                                simualte_residauls = True,
                                                Simulation_Time_Resolution = self.Simulation_Time_Resolution,
                                                is_slotted = True)

            UNslotted_env = SlottedGraphEnvPower( adjacency_matrix=adjacency,
                                                bandwidth_matrix=capacity_matrix,
                                                interference_matrix=interference_matrix,
                                                flows=flows,
                                                node_positions=positions,
                                                k=self.action_size,
                                                reward_weights=self.reward_weights,
                                                telescopic_reward = False,
                                                direction = 'minimize',
                                                slot_duration = int( (self.slot_duration*self.Tot_num_of_timeslots) / self.Simulation_Time_Resolution),          # [in SEC]
                                                Tot_num_of_timeslots = 1, # [in Minutes]
                                                render_mode = False,
                                                trx_power_mode='gain',
                                                channel_gain = 1,
                                                # channel_manual_gain = [100,200,3,400,500,600],
                                                simualte_residauls = False,
                                                Simulation_Time_Resolution = self.Simulation_Time_Resolution,
                                                is_slotted = False) 


            slotted_diamond = SlottedDIAMOND(grrl_model_path=self.MODEL_PATH)
            
            Tot_rates_sloted = slotted_diamond(slotted_env, grrl_data=False)
            Tot_rates_UNslotted = slotted_diamond(UNslotted_env, grrl_data=False)


            # plot rates
            time_axis_in_resulotion = [i * self.Simulation_Time_Resolution for i in range(1,len(Tot_rates_sloted)+1)] # This time axis is a samples of each Simulation_Time_Resolution
            # we want to avarge rates so that we have time axis sampled in seconds (this way spike due to the residual will be smoothed)
            time_axis_in_seconds = [i  for i in range(1,int(self.slot_duration*self.Tot_num_of_timeslots)+1)]

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
            plt.savefig(f'DIAMOND/result_average_rates_over_time{seed}.png')


        return


if __name__ == "__main__":

    # BASE_PATH = os.path.join("..", "results", "inference_vs_competitors")
    # MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # script_path = os.path.abspath(__file__)

    # params
    num_nodes = 15  # 30
    num_edges = 25  # 50
    temperature = 1.2
    num_episodes = 1
    episode_from = 7500
    # nb3r_steps = 100

    trx_power_mode = 'equal'
    rayleigh_scale = 1
    max_trx_power = 10
    channel_gain = 1

    for GRAPH_MODE in ['random', 'geant', 'nsfnet']:
        for trx_power_mode in ['equal', 'rayleigh', 'steps']:

            print("----------------------------")
            print(trx_power_mode, GRAPH_MODE)
            print("----------------------------")

            data_rates = []
            data_delay = []

            for num_flows in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200] if GRAPH_MODE == 'random' else \
                             [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                alg = TestvsCompetitors(num_episodes=num_episodes, episode_from=episode_from,
                                        temperature=temperature)

                data, labels = alg(num_nodes=num_nodes, num_edges=num_edges, num_flows=num_flows,
                                   graph_mode=GRAPH_MODE,
                                   trx_power_mode=trx_power_mode, rayleigh_scale=rayleigh_scale, max_trx_power=max_trx_power, channel_gain=channel_gain)

    #             data_rates.append([int(num_flows)] + [data[x] for x in list(filter(lambda x: "rates" in x, data.keys()))])
    #             data_delay.append([int(num_flows)] + [data[x] for x in list(filter(lambda x: "delay" in x, data.keys()))])

    #         curr_path = os.path.join(BASE_PATH, timestamp, GRAPH_MODE, trx_power_mode)
    #         os.makedirs(curr_path)
    #         shutil.copy(src=script_path, dst=os.path.join(curr_path, os.path.split(script_path)[1]))

    #         with open(os.path.join(curr_path, f"{GRAPH_MODE}_{trx_power_mode}_rates.csv"), 'w') as f:
    #             f.writelines("# " + trx_power_mode + '\n')
    #             f.writelines("# " + GRAPH_MODE + '\n')
    #             f.writelines("# " + f"k={num_actions}, V={num_nodes}, E={num_edges}" + '\n')
    #             f.writelines("# " + "rates" + '\n')
    #             f.writelines('\n')
    #             f.writelines(",".join(["N"] + labels) + '\n')
    #             np.savetxt(f, np.array(data_rates), delimiter=',', fmt=','.join(['%i'] + ['%1.3f'] * len(labels)))

    #         with open(os.path.join(curr_path, f"{GRAPH_MODE}_{trx_power_mode}_delay.csv"), 'w') as f:
    #             f.writelines("# " + trx_power_mode + '\n')
    #             f.writelines("# " + GRAPH_MODE + '\n')
    #             f.writelines("# " + f"k={num_actions}, V={num_nodes}, E={num_edges}" + '\n')
    #             f.writelines("# " + "delay" + '\n')
    #             f.writelines('\n')
    #             f.writelines(",".join(["N"] + labels) + '\n')
    #             np.savetxt(f, np.array(data_delay), delimiter=',', fmt=','.join(['%i'] + ['%1.3f'] * len(labels)))

    # print('done')
