import numpy as np
import os

from stage1_grrl import GRRL, SlottedGRRL
from stage2_nb3r import nb3r


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
                 nb3r_tmpr=10):
        if grrl_model_path is None:
            grrl_model_path = os.path.join(".", "pretrained", "model_20221113_212726_480.pt")
        self.slotted_grrl = SlottedGRRL(path=grrl_model_path)
        self.nb3r_steps = nb3r_steps
        self.nb3r_tmpr = nb3r_tmpr

    def __call__(self, env, grrl_data=False):
        
        #################### central computer in network computes allocations  #################### 
        # stage 1
        rl_actions, rl_paths, rl_reward, Tot_rates, Tot_delays  = self.slotted_grrl.run(env=env)
        rl_actions.sort(key=lambda x: x[0])
        rl_actions = [x[1] for x in rl_actions]
        rl_delay_data = env.get_delay_data()
        rl_rates_data = env.get_rates_data()

        # stage 2
        # self.nb3r_steps = int(env.num_flows * 5)
        # nb3r_action = nb3r(
        #                    objective=lambda a: -self.rates_objective(env, a),
        #                    # objective=lambda a: -self.reward_objective(env, a),
        #                    # objective=lambda a: -self.delay_objective(env, a),
        #                    state_space=env.get_state_space(),
        #                    num_iterations=self.nb3r_steps,  # max(self.nb3r_steps, int(env.num_flows * 5)),
        #                    initial_state=rl_actions.copy(),
        #                    verbose=False,
        #                    seed=env.seed,
        #                    return_history=False,
        #                    initial_temperature=self.nb3r_tmpr)
        
        # routs
        # routs = env.get_routs(rl_actions)
        
        ####################  The actual run and evaluation of packets in the network   #################### 
        # _ = self.real_run(env, actions_recipe)



        if grrl_data:
            return Tot_rates, Tot_delays #routs, rl_rates_data, rl_delay_data
        return Tot_rates, Tot_delays #routs

    def real_run(self,env, actions_recipe = None):
        '''
        This function simulate the Real packet flow after the centarl computer has computed the allocations. Once we get the recipe from the agent.
        each time slot, we can simulate the network and get the results according to the recipe. Here we loop on the time slots, and input to the Env all flows 
        according to the receipe (Each time slots, all flows are input to the env).
        '''
        actions = []
        paths = []
        state = env.reset()

        Tot_num_of_timeslots = env.Tot_num_of_timeslots

        debug_manual_recipe = [[[0,0],[1,3]] for i in range(Tot_num_of_timeslots)] # example of recipe

        for slot_indx in range(Tot_num_of_timeslots):
            '''
            This function get the recipe from thre agent, and simulate the actual netwrok after agent provided with the recipe of what
            to alocate each time slot (so essentially, this function is the real run of the network).
            '''
            self.simualte_real_time_slot(env, debug_manual_recipe[slot_indx]) #self.simualte_real_time_slot(env, actions_recipe[slot_indx])
            env.end_of_step_update()
            print('Time Slot: ', slot_indx)
        
        return 
        
    def simualte_real_time_slot(self, env, actions):
        # Input all flows at once and run them in the network
        for flow in range(env.num_flows):
            action = actions[flow] #action = [step, a]
            state, r = env.step(action)

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