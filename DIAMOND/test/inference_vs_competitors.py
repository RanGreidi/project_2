import numpy as np
import random
import os
from datetime import datetime
import shutil

import sys
sys.path.insert(0, 'DIAMOND')
##sys.path.insert(0, '/work_space/project2/DIAMOND-master/DIAMOND-master')

from diamond import DIAMOND
from environment import generate_env
from competitors import OSPF, RandomBaseline, DQN_GNN, DIAR, IACR
from matplotlib import pyplot as plt

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


class TestvsCompetitors:
    def __init__(self,
                 grrl_model_path,
                 num_episodes=2,
                 num_rb_trials=100,
                 **kwargs):

        self.diamond = DIAMOND(grrl_model_path=grrl_model_path, nb3r_tmpr=kwargs.get('nb3r_tmpr', 1),
                               nb3r_steps=kwargs.get('nb3r_tmpr', 100))
        self.num_episodes = num_episodes
        self.episode_from = kwargs.get('episode_from', 0)

        self.competitors = {
            # 'DQN+GNN': DQN_GNN(k=4),
            # 'OSPF': OSPF(),
            # 'RandomBL': RandomBaseline(num_trials=num_rb_trials),
            # 'DIAR': DIAR(n_iter=2),
            # 'IACR': IACR(delta=0.5, alpha=1.3),
        }

    def __call__(self, **kwargs):

        # update variables
        self.num_nodes = kwargs.get('num_nodes', 10)
        self.num_edges = kwargs.get('num_edges', 20)
        self.num_flows = kwargs.get('num_flows', 20)
        self.num_actions = kwargs.get('num_actions', 4)

        data = {
            'grrl_delay': [],
            'grrl_rates': [],
            'Slotted Env grrl_delay': [],
            'Slotted Env grrl_rates': [],            
        }
        for c in self.competitors.keys():
            data[f"{c}_delay"] = 0
            data[f"{c}_rates"] = 0

        for episode in range(self.num_episodes):

            # seed
            seed = SEED + (episode + 1) + self.episode_from + 1

            # generate env
            env,slotted_env = generate_env( num_nodes=self.num_nodes,
                                            num_edges=self.num_edges,
                                            num_actions=self.num_actions,
                                            num_flows=self.num_flows,
                                            min_flow_demand=kwargs.get('min_flow_demand', 100),
                                            max_flow_demand=kwargs.get('max_flow_demand', 200),
                                            min_capacity=kwargs.get('min_capacity', 200),
                                            max_capacity=kwargs.get('max_capacity', 500),
                                            seed=seed,
                                            graph_mode=kwargs.get('graph_mode', 'random'),
                                            trx_power_mode=kwargs.get('trx_power_mode', 'equal'),
                                            rayleigh_scale=kwargs.get('rayleigh_scale'),
                                            max_trx_power=kwargs.get('max_trx_power'),
                                            channel_gain=kwargs.get('channel_gain'))

            # test DIAMOND
            grrl_rates_data, grrl_delay_data = self.diamond(env, grrl_data=True)
            slotted_grrl_rates_data, slotted_grrl_delay_data = self.diamond(slotted_env, grrl_data=True)
            
            data['grrl_delay'].append(grrl_delay_data['delay_per_flow'][0])
            data['grrl_rates'].append(grrl_rates_data['rate_per_flow'][0])

            data['Slotted Env grrl_delay'].append(slotted_grrl_delay_data['end_to_end_delay_per_flow'][0])
            data['Slotted Env grrl_rates'].append(slotted_grrl_rates_data['rate_per_flow'][0])

            # competitors
            for name, comp in zip(self.competitors.keys(), self.competitors.values()):
                paths, reward, delay_data, rates_data = comp.run(env, seed)
                data[f"{name}_delay"] += np.max(delay_data['delay_per_flow'])
                data[f"{name}_rates"] += (rates_data['sum_flow_rates'] / self.num_flows)

        # # average
        # for d in data:
        #     data[d][0] /= self.num_episodes

        # print(f"==========================================================================================")
        # print(
        #     f"(V={num_nodes}, E={num_edges}, N={num_flows}, k={num_actions}) {[f'{c}: {data[c][0]:.3f}' for c in data]}")        
        labels = ["DIAMOND", "GRRL"] + list(self.competitors.keys())

        return data, labels


if __name__ == "__main__":
    
    save_to_file = False
    BASE_PATH = os.path.join("..", "results", "inference_vs_competitors")
    MODEL_PATH = os.path.join("DIAMOND", "pretrained", "model_20221113_212726_480.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.abspath(__file__)

    # params
    num_nodes = 60  # 30
    num_edges = 90  # 50
    num_actions = 4
    temperature = 1.2
    num_episodes = 100
    episode_from = 7500
    nb3r_steps = 100

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

            for num_flows in [5] if GRAPH_MODE == 'random' else \
                             [5]:
                alg = TestvsCompetitors(grrl_model_path=MODEL_PATH, num_episodes=num_episodes, episode_from=episode_from,
                                        temperature=temperature, nb3r_steps=nb3r_steps)

                data, labels = alg(num_nodes=num_nodes, num_edges=num_edges, num_flows=num_flows, num_actions=num_actions,
                                   graph_mode=GRAPH_MODE,
                                   trx_power_mode=trx_power_mode, rayleigh_scale=rayleigh_scale, max_trx_power=max_trx_power, channel_gain=channel_gain)

            plt.figure()
            plt.plot(data['grrl_delay'],label='Orig')
            plt.plot(data['Slotted Env grrl_delay'],label='Slotted')
            plt.title(f'DELAY. num_flows: {num_flows},GRAPH_MODE: {GRAPH_MODE},trx_power_mode {trx_power_mode}' )
            plt.xlabel('episode')
            plt.ylabel('flow 0 delay')   
            plt.legend()         
            plt.savefig(f'{num_flows}_{GRAPH_MODE}_{trx_power_mode}_DELAY')
            plt.close()

            plt.figure()
            plt.plot(data['grrl_rates'],label='Orig')
            plt.plot(data['Slotted Env grrl_rates'],label='Slotted')
            plt.title(f'FLOW RATE. num_flows: {num_flows},GRAPH_MODE: {GRAPH_MODE},trx_power_mode {trx_power_mode}' )
            plt.xlabel('episode')
            plt.ylabel('flow 0 Rate')
            plt.legend()
            plt.savefig(f'{num_flows}_{GRAPH_MODE}_{trx_power_mode}_RATES')
            plt.close()

            if save_to_file:
                curr_path = os.path.join(BASE_PATH, timestamp, GRAPH_MODE, trx_power_mode)
                os.makedirs(curr_path)
                shutil.copy(src=script_path, dst=os.path.join(curr_path, os.path.split(script_path)[1]))  
                with open(os.path.join(curr_path, f"{GRAPH_MODE}_{trx_power_mode}_rates.csv"), 'w') as f:
                    f.writelines("# " + trx_power_mode + '\n')
                    f.writelines("# " + GRAPH_MODE + '\n')
                    f.writelines("# " + f"k={num_actions}, V={num_nodes}, E={num_edges}" + '\n')
                    f.writelines("# " + "rates" + '\n')
                    f.writelines('\n')
                    f.writelines(",".join(["N"] + labels) + '\n')
                    np.savetxt(f, np.array(data_rates), delimiter=',', fmt=','.join(['%i'] + ['%1.3f'] * len(labels)))

                with open(os.path.join(curr_path, f"{GRAPH_MODE}_{trx_power_mode}_delay.csv"), 'w') as f:
                    f.writelines("# " + trx_power_mode + '\n')
                    f.writelines("# " + GRAPH_MODE + '\n')
                    f.writelines("# " + f"k={num_actions}, V={num_nodes}, E={num_edges}" + '\n')
                    f.writelines("# " + "delay" + '\n')
                    f.writelines('\n')
                    f.writelines(",".join(["N"] + labels) + '\n')
                    np.savetxt(f, np.array(data_delay), delimiter=',', fmt=','.join(['%i'] + ['%1.3f'] * len(labels)))

    print('done')
