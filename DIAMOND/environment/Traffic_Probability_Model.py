import numpy as np
import random
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'DIAMOND')


class Traffic_Probability_Model:
    def __init__(self,
                 source=None,
                 destination=None,
                 constant_flow_name=None,
                 seed=42,
                 transition_matrix_properties={'mice_start_from' : 10,
                                               'elephent_start_from' : 1000,

                                               'p_mice' : 0.05,
                                               'p_elephent' : 0.05,
                                               'p_idle' : 0.9,

                                               'p_finish' : 0.1,
                                               'p_same' : 0.5,
                                                'p_minus' : 0.2,
                                                'p_plus' : 0.2,
                                                'Num_of_states': 101
                                              }):

        """
        Initializes the Markov chain.

        :param transition_matrix: 2D NumPy array representing the transition probabilities between states.
        :param states: List of possible states.
        :param initial_state: (Optional) Initial state. If None, a random state is chosen.
        """
        if type(transition_matrix_properties) != dict:
            self.transition_matrix = np.array(transition_matrix_properties)
        else:
            self.transition_matrix = self.create_transition_matrix(
                                                                    mice_start_from = transition_matrix_properties['mice_start_from'],
                                                                    elephent_start_from = transition_matrix_properties['elephent_start_from'],

                                                                    p_mice = transition_matrix_properties['p_mice'],
                                                                    p_elephent = transition_matrix_properties['p_elephent'],
                                                                    p_idle = transition_matrix_properties['p_idle'],

                                                                    p_finish = transition_matrix_properties['p_finish'],
                                                                    p_same = transition_matrix_properties['p_same'],
                                                                    p_minus = transition_matrix_properties['p_minus'],
                                                                    p_plus = transition_matrix_properties['p_plus'],

                                                                    Num_of_states = transition_matrix_properties['Num_of_states'])

        self.states = [0] + [i + transition_matrix_properties['mice_start_from'] for i in range(0,int((transition_matrix_properties['Num_of_states']-1)/2))] + [i + transition_matrix_properties['elephent_start_from'] for i in range(0,int((transition_matrix_properties['Num_of_states']-1)/2))]
        self.state = 0

        self.source = source
        self.destination = destination
        self.constant_flow_name = constant_flow_name

    def step(self):
        """
        Takes one step in the Markov Chain by transitioning to the next state based on the transition matrix.
        """
        current_state_index = self.states.index(self.state)
        next_state_index = np.random.choice(len(self.states), p=self.transition_matrix[current_state_index])
        self.state = self.states[next_state_index]
        return self.state

    def simulate(self, num_steps):
        """
        Simulates the Markov Chain for a given number of steps.

        :param num_steps: The number of steps to simulate.
        :return: A list of states visited during the simulation.
        """
        state_history = [self.state]
        for _ in range(num_steps):
            self.step()
            state_history.append(self.state)
        return state_history

    def create_transition_matrix(self,
                                mice_start_from = 10,
                                elephent_start_from = 1000,
                                p_mice = 0.05,
                                p_elephent = 0.05,
                                p_idle = 0.9,

                                p_finish = 0.1,
                                p_same = 0.5,
                                p_minus = 0.2,
                                p_plus = 0.2,

                                Num_of_states = 11
                                ):


        #states
        state_zero = ['State_0']
        state_mice = [f'State_{state}'  for state in range(mice_start_from,int(mice_start_from+(Num_of_states-1)/2))]
        states_elephent = [f'State_{state}'  for state in range(elephent_start_from,int(elephent_start_from+(Num_of_states-1)/2))]
        states = state_zero + state_mice + states_elephent

        # transition_matrix = np.zeros([Num_of_states,Num_of_states])
        center_state_mice = int(len(state_mice)/2 )+1
        center_state_elephent = int(len(state_mice)  + len(states_elephent)/2 )+1

        # fisrt row
        fisrt = np.zeros([1,Num_of_states])
        fisrt[0,center_state_mice] = p_mice
        fisrt[0,center_state_elephent] = p_elephent

        # mice
        mice = np.zeros([len(state_mice),Num_of_states]) * p_same
        mice[:,0] = p_finish
        mice[0,2] = 2*p_plus
        mice[-1,mice.shape[0]-1] = 2*p_minus
        for n in range(1,mice.shape[0]-1):
            mice[n,n] = p_minus
            mice[n,n+2] = p_plus

        #elephent
        elephent = np.zeros([len(state_mice),Num_of_states]) * p_same
        elephent[:,0] = p_finish
        elephent[0,1+mice.shape[0]+1] = 2*p_plus
        elephent[-1,-2] = 2*p_minus
        for n in range(1,elephent.shape[0]-1):
            elephent[n,n+mice.shape[0]] = p_minus
            elephent[n,n+mice.shape[0]+2] = p_plus

        transition_matrix = np.concatenate((fisrt,mice,elephent), axis=0)
        np.fill_diagonal(transition_matrix, p_same)
        transition_matrix[0,0] = p_idle

        # np.sum(np.array(transition_matrix), axis=1) # make sure all rows sum to 1
        return transition_matrix


def generate_flow_traffic(transition_matrix_properties, num_steps=20, seed=42):

    # Define Flow markov chain
    TPM = Traffic_Probability_Model(transition_matrix_properties, seed=seed)

    # Simulate the Markov Chain for num steps
    state_history = TPM.simulate(num_steps)

    return state_history


def generate_traffic_matrix(num_flows, transition_matrix_properties, num_steps, seed=42):

    traffic_matrix = np.zeros((num_steps + 1, num_flows))
    for i in range(num_flows):
        traffic_matrix[:, i] = generate_flow_traffic(transition_matrix_properties, num_steps, seed=seed + (i * 10))

    return traffic_matrix


if __name__ == "__main__":
    '''# Example of usage

    # Define the transition matrix
    # Example: A 3-state Markov chain with the following transition probabilities
    # State 0 -> State 0: 0.6, State 0 -> State 1: 0.4
    # State 1 -> State 0: 0.7, State 1 -> State 1: 0.3
    # State 2 -> State 0: 0.2, State 2 -> State 2: 0.8
    transition_matrix = [
        [0.6, 0.4, 0.0],
        [0.7, 0.3, 0.0],
        [0.2, 0.0, 0.8]
    ]



    p_mice = 0.05
    p_elephent = 0.05
    p_idle = 0.9

    p_finish = 0.1
    p_same = 0.5
    p_minus = 0.2
    p_plus = 0.2

    transition_matrix = np.array([
                                    [ p_idle  ,     0           , p_mice    , 0         , 0         , p_elephent,       0],  # State 0
                                    
                                    [ p_finish,     p_same      , 2*p_plus  , 0         , 0         , 0,                0],  # State 10
                                    [ p_finish,     p_minus     , p_same    , p_plus    , 0         , 0,                0],  # State 11  
                                    [ p_finish,     0           , 2*p_minus , p_same    , 0         , 0,                0],  # State 12
                                    
                                    [ p_finish,     0           , 0         , 0         , p_same    , 2*p_plus,         0],  # State 100
                                    [ p_finish,     0           , 0         , 0         , p_minus   , p_same, p_plus     ],  # State 101
                                    [ p_finish,     0           , 0         , 0         , 0         , 2*p_minus, p_same  ],  # State 102                                                  
    ])

    np.sum(np.array(transition_matrix), axis=1) # make sure all rows sum to 1
    '''

    transition_matrix_properties = {
                                    'mice_start_from' : 10,
                                    'elephent_start_from' : 1000,

                                    'p_mice' : 0.05,
                                    'p_elephent' : 0.05,
                                    'p_idle' : 0.9,

                                    'p_finish' : 0.1,
                                    'p_same' : 0.5,
                                    'p_minus' : 0.2,
                                    'p_plus' : 0.2,
                                    'Num_of_states': 101
    }

    TM = generate_traffic_matrix(num_flows=100,
                                 transition_matrix_properties=transition_matrix_properties,
                                 num_steps=10000)

    save_path = r'C:\Users\beaviv\Datasets\Ran_DIAMOND_generated_traffic\generated_traffic_matrix.npy'
    # np.save(save_path, TM)

    # mc = Traffic_Probability_Model(transition_matrix_properties)
    #
    # # Simulate the Markov Chain for 50 steps
    # num_steps = 100
    # state_history = mc.simulate(num_steps)
    #
    # # Optionally, plot the state transitions
    # plt.figure(figsize=(10, 6))
    # plt.plot(state_history, marker='_')
    # plt.title("State Transitions in Markov Chain")
    # plt.xlabel("Step")
    # plt.ylabel("State")
    # plt.yticks(mc.states)  # Use state names for the y-axis
    # plt.grid(True)
    # # plt.savefig('Probability_Model')
    # plt.show()

    print("finished")
