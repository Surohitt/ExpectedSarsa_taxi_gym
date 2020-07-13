import numpy as np
import random
import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, gamma=1, alpha=0.01):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = gamma
        self.alpha = alpha
    
    def get_policy(self, Q_s, epsilon):
        policy_s = np.ones(self.nA) * epsilon/self.nA
        greedy_action = np.argmax(Q_s)
        policy_s[greedy_action] = 1-epsilon + (epsilon/self.nA)
        return policy_s

    def select_action(self, state, epsilon):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.random.choice(np.arange(self.nA), p=self.get_policy(self.Q[state], epsilon))
        return action

    def step(self, state, action, reward, next_state, done, epsilon):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current_q_estimate = self.Q[state][action]
        policy_s = self.get_policy(self.Q[state], epsilon)
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        target = reward + (self.gamma*Qsa_next)
        new_value = current_q_estimate + (self.alpha*(target-current_q_estimate))
        self.Q[state][action] = new_value