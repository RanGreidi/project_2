import torch
from torch.distributions import Categorical
import sys
import os
import copy
import numpy as np

from stage2_nb3r import nb3r

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

    def run(self, env, arrival_matrix=None, manual_actions=[], use_nb3r=False):
        """
        run GRRL to get flow allocations
        :param env: environment to interact with
        :param arrival_matrix: future_demand
        :return: action indices, paths and rewards
        """
        grrl_actions = []
        # paths = [[] for _ in range(env.num_flows)]
        grrl_paths = [[] for _ in range(env.Tot_num_of_timeslots)]
        state = env.reset()
        reward = 0
        Tot_num_of_timeslots = env.Tot_num_of_timeslots
        Tot_rates_grrl = []
        Tot_delays_grrl = []

        # manual_actions = [[0, 0], [1, 1]]
        nb3r_paths = [[] for _ in range(env.Tot_num_of_timeslots)]
        Tot_rates_nb3r = []
        Tot_delays_nb3r = []
        nb3r_actions = []

        for timeslot in range(Tot_num_of_timeslots):  # as long there is still flows running (determines the num of time_slotes in one episode)

            slot_actions = []
            grrl_slot_actions = []
            nb3r_env = copy.deepcopy(env)  # Todo: ensures that grrl and nb3r gets same env

            for step in range(env.num_flows):

                if manual_actions:
                    action = manual_actions[step]  # action = [step, a] # Todo: not correct if needs to use this line more than once
                else:
                    a = self._select_action(state, env.possible_actions[step])  # Todo: not correct, possible_actions[step] doesnt match the correct flow when some finished
                    action = [step, a]

                slot_actions.append([action[0], action[1], env.flows[step]['constant_flow_name']])  # action format: [step , decision , flow_name]
                grrl_paths[timeslot].append(env.possible_actions[action[0]][action[1]])  # Todo, this is not correct!!!, possible_actions is always a list of length original_num_flows, will add paths to flows even if finished
                state, r = env.step(action)
                reward += r

                grrl_slot_actions.append(action)  # for nb3r initial state

            grrl_actions.append(slot_actions)

            # ------------------------------------------------------ #
            # Todo: update flows inside slotted_graph_power not here
            if env.arrival_matrix is not None:
                self.update_flows(env=env, timeslot=timeslot)
            # ------------------------------------------------------ #

            state,SlotRates_AvgOverFlows, SlotDelays_AvgOverFlows, manual_calculated_delay = env.end_of_slot_update()
            Tot_rates_grrl += (SlotRates_AvgOverFlows)
            Tot_delays_grrl.append(manual_calculated_delay)
            # Tot_delays_grrl.append(SlotDelays_AvgOverFlows)

            # ------------------------------------------------  NB3R ------------------------------------------------- #
            if use_nb3r:
                if nb3r_env.num_flows > 0:  # ensures that doesn't enter when all finished

                    print(f'\n Starting NB3R')
                    grrl_slot_actions.sort(key=lambda x: x[0])  # to keep order of flow index
                    grrl_slot_actions = [x[1] for x in grrl_slot_actions]  # a list of N items every element is the path idx for flow n

                    self.nb3r_steps = 5   # int(env.num_flows * 5)
                    nb3r_action = nb3r(
                        objective=lambda a: -self.rates_objective(nb3r_env, a),
                        # objective=lambda a: -self.reward_objective(env, a),
                        # objective=lambda a: -self.delay_objective(env, a),
                        state_space=nb3r_env.get_state_space(),
                        num_iterations=self.nb3r_steps,  # max(self.nb3r_steps, int(env.num_flows * 5)),
                        initial_state=grrl_slot_actions.copy(),
                        verbose=False,
                        seed=env.seed,
                        return_history=False,
                        initial_temperature=1)

                    # routs
                    nb3r_routs = nb3r_env.get_routs(nb3r_action)
                    nb3r_paths[timeslot].append(nb3r_routs)
                    nb3r_actions.append([[flow_id, action_id] for flow_id, action_id in enumerate(nb3r_action)])
                    nb3r_env.eval_all(nb3r_action)

                _, SlotRates_AvgOverFlows, SlotDelays_AvgOverFlows = nb3r_env.end_of_slot_update()  # Todo: need to check if its okay to use this state, because env can hold different values
                Tot_rates_nb3r += (SlotRates_AvgOverFlows)
                Tot_delays_nb3r.append(SlotDelays_AvgOverFlows)

            # -------------------------------------------------------------------------------------------------------- #

            print(f'{env.num_flows}/{len(env.original_flows)} Flow Alive\n Finished Timeslot {timeslot + 1}/{Tot_num_of_timeslots}\n')

        if not use_nb3r:
            return grrl_actions, grrl_paths, reward, Tot_rates_grrl, Tot_delays_grrl
        else:
            return grrl_actions, nb3r_actions, grrl_paths, nb3r_paths, Tot_rates_nb3r, Tot_delays_nb3r

    def update_flows(self, env, timeslot):

        """
        update flow packets with incoming demand, env.flows is not the same as env.original_flows
        anymore. go over original flows: if flow didn't finish than add packets, if finished and has
        positive incoming demand "bring him back to life"
        """

        # alive_flow_indices = [flow['constant_flow_name'] for flow in env.flows]
        list_of_2flows = []
        for flow_id, flow in enumerate(env.original_flows):

            # Search if flow with flow_id is "alive"
            flow = next((flow for flow in env.flows if flow["constant_flow_name"] == flow_id), None)  # if alive returns flow else None

            if flow is not None:  # flow is alive
                flow['packets'] += env.arrival_matrix[timeslot][flow_id]
                list_of_2flows.append(flow)

            # If not alive, check if arrival is positive, if it is, bring him back to life
            else:
                if env.arrival_matrix[timeslot][flow_id] > 0:
                    flow = copy.deepcopy(env.original_flows[flow_id])
                    flow['packets'] = env.arrival_matrix[timeslot][flow_id]
                    list_of_2flows.append(flow)

        env.flows = list_of_2flows
        env.num_flows = len(env.flows)

    @staticmethod
    def rates_objective(env, actions):
        env.reset()  # Todo: check if needed here, because in end_of_slot_update there is env.reset()
        eval_env = copy.deepcopy(env)  # makes sure it's the same env for evaluation
        eval_env.eval_all(actions)
        state, SlotRates_AvgOverFlows = eval_env.end_of_slot_update()  # TODO: After this env.possible_actions can be different with diffrent actions, need to use the same env every time for evaluation
        try:
            return np.mean(SlotRates_AvgOverFlows)  # np.sum(env.get_rates_data()['sum_flow_rates'])

        except TypeError:  # For the case where finished before all iterations
            return np.mean(SlotRates_AvgOverFlows[:SlotRates_AvgOverFlows.index(None)])















