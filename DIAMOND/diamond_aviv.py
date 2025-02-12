import numpy as np
import os

import sys
sys.path.insert(0, 'DIAMOND')
from stage1_grrl import GRRL, SlottedGRRL
from stage2_nb3r import nb3r

from environment.Traffic_Probability_Model import Traffic_Probability_Model

class DIAMOND:
    def __init__(self,
                 grrl_model_path,
                 nb3r_steps=100,
                 nb3r_tmpr=10):
        if grrl_model_path is None:
            grrl_model_path = os.path.join(".", "pretrained", "model_20221113_212726_480.pt")
        self.grrl = GRRL(path=grrl_model_path)
        self.nb3r_steps = nb3r_steps
        self.nb3r_tmpr = nb3r_tmpr

    def __call__(self, env, grrl_data=False):
        # stage 1
        rl_actions, rl_paths, rl_reward = self.grrl.run(env=env)
        rl_actions.sort(key=lambda x: x[0])
        rl_actions = [x[1] for x in rl_actions]
        rl_delay_data = env.get_delay_data()
        rl_rates_data = env.get_rates_data()

        # stage 2
        self.nb3r_steps = int(env.num_flows * 5)
        nb3r_action = nb3r(
            objective=lambda a: -self.rates_objective(env, a),
            # objective=lambda a: -self.reward_objective(env, a),
            # objective=lambda a: -self.delay_objective(env, a),
            state_space=env.get_state_space(),
            num_iterations=self.nb3r_steps,  # max(self.nb3r_steps, int(env.num_flows * 5)),
            initial_state=rl_actions.copy(),
            verbose=False,
            seed=env.seed,
            return_history=False,
            initial_temperature=self.nb3r_tmpr)
        # routs
        routs = env.get_routs(nb3r_action)

        if grrl_data:
            return routs, rl_rates_data, rl_delay_data
        return routs

    @staticmethod
    def rates_objective(env, actions):
        env.reset()
        env.eval_all(actions)
        return np.sum(env.get_rates_data()['sum_flow_rates'])

    @staticmethod
    def reward_objective(env, actions):
        env.reset()
        return env.eval_all(actions)

    @staticmethod
    def delay_objective(env, actions):
        env.reset()
        env.eval_all(actions)
        return np.sum(env.get_delay_data()['mean_delay'])


class SlottedDIAMOND:
    def __init__(self,
                 grrl_model_path,
                 nb3r_steps=100,
                 nb3r_tmpr=10,
                 Traffic_Probability_Model_list=[]):

        if grrl_model_path is None:
            grrl_model_path = os.path.join(".", "pretrained", "model_20221113_212726_480.pt")
        self.slotted_grrl = SlottedGRRL(path=grrl_model_path)
        self.nb3r_steps = nb3r_steps
        self.nb3r_tmpr = nb3r_tmpr
        self.Traffic_Probability_Model_list = Traffic_Probability_Model_list

    def __call__(self, env, grrl_data=False, use_nb3r=False, arrival_matrix=None, manual_actions=[]):

        #################### central computer in network computes allocations  ####################
        # stage 1
        rl_actions, rl_paths, rl_reward, Tot_rates = self.slotted_grrl.run(env=env, arrival_matrix=arrival_matrix, manual_actions=manual_actions)
        action_recipe = rl_actions
        # rl_actions.sort(key=lambda x: x[0])
        # rl_actions = [x[1] for x in rl_actions]
        # rl_delay_data = env.get_delay_data() # Todo : at the last iteration, after env.flows update flows might not have "path" argument so get_delay_data needs to bu updated, or maybe dont update flows at last iteration
        rl_rates_data = env.get_rates_data()

        # If we don't use nb3r we take grrl paths
        routs = rl_paths

        # stage 2
        if use_nb3r:
            self.nb3r_steps = int(env.num_flows * 5)
            nb3r_action = nb3r(
                objective=lambda a: -self.rates_objective(env, a),
                # objective=lambda a: -self.reward_objective(env, a),
                # objective=lambda a: -self.delay_objective(env, a),
                state_space=env.get_state_space(),
                num_iterations=self.nb3r_steps,  # max(self.nb3r_steps, int(env.num_flows * 5)),
                initial_state=rl_actions.copy(),
                verbose=False,
                seed=env.seed,
                return_history=False,
                initial_temperature=self.nb3r_tmpr)
            # routs
            routs = env.get_routs(nb3r_action)

        ####################  The actual run and evaluation of packets in the network   ####################
        # _ = self.real_run(env, actions_recipe)

        if grrl_data:
            return routs, action_recipe, Tot_rates  # routs, rl_rates_data, rl_delay_data, Tot_rates

        return routs

    def real_run(self, env, actions_recipe=None):
        '''
        This function simulate the Real packet flow after the centarl computer has computed the allocations. Once we get the recipe from the agent.
        each time slot, we can simulate the network and get the results according to the recipe. Here we loop on the time slots, and input to the Env all flows
        according to the receipe (Each time slots, all flows are input to the env).
        '''

        Tot_rates = []
        state = env.reset()

        Tot_num_of_timeslots = env.Tot_num_of_timeslots

        print('Simulating Real Run of packets in the real world')
        for slot_indx in range(Tot_num_of_timeslots):
            '''
            This function get the recipe from the agent, and simulate the actual network after agent provided with the recipe of what
            to alocate each time slot (so essentially, this function is the real run of the network).
            '''
            # Add flows to env according to new flows that ask to join according to reality (the real flows that want to join the network)
            ''' At some point during Tot_num_of_timeslots a flow can ask to join the network, we have two options
            1. we predicted that the flow will join the network and we added it to the recipe
            2. we did not predict that the flow will join the network and we did not add it to the recipe - The flow has to wait for the next time slot
            If the flow waits, it adds a delay to the flow. 
            If we predicted that the flow will join the network, we added it to the recipe and the flow will be added to the network.
            '''
            new_flows_dict = self.generate_new_flows_dict_according_to_probabilty_model(env.flows,  self.Traffic_Probability_Model_list)  # Add flows to env according to new flows that ask to join according to reality (the real flows that want to join the network)
            # env.flows = new_flows_dict
            # env.num_flows = len(new_flows_dict)

            self.simualte_real_time_slot(env, actions_recipe[slot_indx])  # self.simualte_real_time_slot(env, actions_recipe[slot_indx])
            _, SlotRates_AvgOverFlows = env.end_of_slot_update()
            Tot_rates += (SlotRates_AvgOverFlows)

            print('Time Slot: ', slot_indx)

        return Tot_rates

    @ staticmethod
    def generate_new_flows_dict_according_to_probabilty_model(flows_dict, Traffic_Probability_Model_list):
        '''
        This function adds new flows that want to join the network to the given flows dict. The function generates the flows according to the probabilty model that we have.
        '''

        # iterate over all flows, sample from the probabilty distribution (is this flow stays empty? is the flow demand increas? or decrease?)
        for flow in Traffic_Probability_Model_list:
            constant_flow_name = flow.constant_flow_name

            # iterate to check if the flow is in the graph aleady, otherwise add it as a new flow
            matching_flow = None
            for d in flows_dict:
                if d.get('constant_flow_name') == constant_flow_name:
                    matching_flow = d
                    break

            # if in the graph
            if (matching_flow is not None) and ('residual_name' not in matching_flow):  # ignore residuals
                current_demand = matching_flow['packets']
                # sample from the distribution
                new_demand = flow.step()
                # modify its demand
                matching_flow['packets'] = new_demand + current_demand

            # if not in the graph - Data is taken from the predefined posible flows list (Traffic_Probability_Model_list) from the simple script
            else:
                new_demand = flow.step()
                if new_demand > 0:
                    new_flow = dict(source=flow.source,
                                    destination = flow.destination,
                                    packets = new_demand,
                                    time_constrain = 10,
                                    flow_idx = max(d['flow_idx'] for d in flows_dict) + 1,
                                    constant_flow_name = flow.constant_flow_name)
                    flows_dict.append(new_flow)

        # modifiy the flow_dct
        new_flows_dict = flows_dict

        return new_flows_dict

    @staticmethod
    def simualte_real_time_slot(env, slot_actions_from_recipe):

        # remiander: slot_actions_from_recipe is a recipe for each step, in the form of: [step_indx , decision , flow_name]

        # here actions may not be matched to numb of flows (number of flows can be different from the number of actions due to the probabilty model addition)
        # if more there are more flows than action, the added flows have to wait for the agnet to open them a rout

        # if less flows than action, the preditior tought there is an additional flow, but there is not. in this case, route was open unneccesarily

        # TODO Code here for all possible cases
        #
        #
        #

        # Input all flows at once from recipe
        # for action in slot_actions_from_recipe:
        #    action = action[0:2] #action is [step, a, flow_name], we need only [step, a]
        #    env.step(action, real_run=True)
        for flow in range(env.num_flows):
            action = slot_actions_from_recipe[flow] #action = [step, a, flow_name]
            env.step(action, real_run=True)
        return

    @staticmethod
    def rates_objective(env, actions):
        env.reset()
        env.eval_all(actions)
        return np.sum(env.get_rates_data()['sum_flow_rates'])

    @staticmethod
    def reward_objective(env, actions):
        env.reset()
        return env.eval_all(actions)

    @staticmethod
    def delay_objective(env, actions):
        env.reset()
        env.eval_all(actions)
        return np.sum(env.get_delay_data()['mean_delay'])