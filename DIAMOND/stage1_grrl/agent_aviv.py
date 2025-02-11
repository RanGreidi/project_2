import torch
from torch.distributions import Categorical
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join("DIAMOND", "stage1_grrl"))


class GRRL:
    """
    Implementation of the GRRL agent (a.k.a stage_1)
    """

    def __init__(self, path):
        """
        :param path: path to pre-trained model
        """
        # Set the device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.device = torch.device('cpu')
        # set path
        self.path = path
        # init model
        self.model = self._load_model()

    def _load_model(self):
        """
        load pre-trained model
        :return: loaded model
        """
        try:
            model = torch.load(self.path)['model']
        except FileNotFoundError:
            sys.path.insert(0, os.path.join("..", "stage1_grrl"))
            model = torch.load(self.path.replace('DIAMOND', '..'))['model']
        model.eval()
        model.to(self.device)
        return model

    def _select_action(self, state, free_paths):
        """
        select next path to allocate
        :param state: graph state
        :param free_paths: list of available path for allocation
        :return: action as chosen path index
        """
        # prepare state
        adj_matrix, edges, _, free_paths_idx, demand = state
        adj_matrix = torch.from_numpy(adj_matrix).to(self.device).float()
        edges = torch.from_numpy(edges).to(self.device)
        demand = torch.from_numpy(demand).to(self.device)

        # apply model
        probs = self.model(adj_matrix, edges, free_paths, demand)
        # select greedy action
        m = Categorical(probs.view((1, -1)))
        action = m.probs.argmax()

        return int(action.cpu().detach().numpy())

    def run(self, env):
        """
        run GRRL to get flow allocations
        :param env: environment to interact with
        :return: action indices, paths and rewards
        """
        actions = []
        paths = []
        state = env.reset()
        reward = 0
        for step in range(env.num_flows):
            a = self._select_action(state, env.possible_actions[step])
            action = [step, a]
            actions.append(action)
            paths.append(env.possible_actions[action[0]][action[1]])
            state, r = env.step(action)
            reward += r

        return actions, paths, reward


class SlottedGRRL:
    """
    Implementation of the GRRL agent (a.k.a stage_1)
    """

    def __init__(self, path):
        """
        :param path: path to pre-trained model
        """
        # Set the device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.device = torch.device('cpu')
        # set path
        self.path = path
        # init model
        self.model = self._load_model()

    def _load_model(self):
        """
        load pre-trained model
        :return: loaded model
        """
        try:
            model = torch.load(self.path)['model']
        except FileNotFoundError:
            sys.path.insert(0, os.path.join("..", "stage1_grrl"))
            model = torch.load(self.path.replace('DIAMOND', '..'))['model']
        model.eval()
        model.to(self.device)
        return model

    def _select_action(self, state, free_paths):
        """
        select next path to allocate
        :param state: graph state
        :param free_paths: list of available path for allocation
        :return: action as chosen path index
        """
        # prepare state
        adj_matrix, edges, _, free_paths_idx, demand = state
        adj_matrix = torch.from_numpy(adj_matrix).to(self.device).float()
        edges = torch.from_numpy(edges).to(self.device)
        demand = torch.from_numpy(demand).to(self.device)

        # apply model
        probs = self.model(adj_matrix, edges, free_paths, demand)
        # select greedy action
        m = Categorical(probs.view((1, -1)))
        action = m.probs.argmax()

        return int(action.cpu().detach().numpy())

    def run(self, env, arrival_matrix=None, manual_actions=[]):
        """
        run GRRL to get flow allocations
        :param env: environment to interact with
        :param arrival_matrix: future_demand
        :return: action indices, paths and rewards
        """
        actions = []
        # paths = [[] for _ in range(env.num_flows)]
        paths = [[] for _ in range(env.Tot_num_of_timeslots)]
        state = env.reset()
        reward = 0
        Tot_num_of_timeslots = env.Tot_num_of_timeslots
        Tot_rates = []

        # manual_actions = [[0, 0], [1, 1]]

        for timeslot in range(Tot_num_of_timeslots):  # as long there is still flows running (determines the num of time_slotes in one episode)
            slot_actions = []
            for step in range(env.num_flows):
                if manual_actions:
                    action = manual_actions[step]  # action = [step, a] # Todo: not correct if needs to use this line more than once
                else:
                    a = self._select_action(state, env.possible_actions[step])  # Todo: not correct, possible_actions[step] doesnt match the correct flow when some finished
                    action = [step, a]

                slot_actions.append([action[0], action[1], env.flows[step]['constant_flow_name']])  # action format: [step , decision , flow_name]
                paths[timeslot].append(env.possible_actions[action[0]][action[1]])  # Todo, this is not correct!!!, possible_actions is always a list of length original_num_flows, will add paths to flows even if finished
                state, r = env.step(action)
                reward += r

            actions.append(slot_actions)
            # Todo: update flows inside slotted_graph_power not here
            if env.arrival_matrix is not None:
                self.update_flows(env=env, timeslot=timeslot)

            state,SlotRates_AvgOverFlows = env.end_of_slot_update()
            Tot_rates += (SlotRates_AvgOverFlows)
            if env.original_num_flows != env.num_flows:
                print(f'{env.original_num_flows - env.num_flows} / {env.original_num_flows} Finished at timeslot {timeslot}')
            print(f'Finished Timeslot {timeslot+1}/{Tot_num_of_timeslots}\n')

        return actions, paths, reward, Tot_rates

    def update_flows(self, env, timeslot):

        alive_flow_indices = [flow['constant_flow_name'] for flow in env.flows]
        list_of_2flows = []
        for flow_id in range(env.original_num_flows):

            if flow_id in alive_flow_indices:
                flow = env.flows[flow_id]
                flow['packets'] += env.arrival_matrix[timeslot][flow_id]
                list_of_2flows.append(flow)
            else:
                if env.arrival_matrix[timeslot][flow_id] > 0:
                    flow = env.original_flows[flow_id]
                    flow['packets'] = env.arrival_matrix[timeslot][flow_id]
                    list_of_2flows.append(flow)

        env.flows = list_of_2flows
        env.num_flows = len(env.flows)


















