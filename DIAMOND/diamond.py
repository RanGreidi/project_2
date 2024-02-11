import numpy as np
import os

from stage1_grrl import GRRL
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
