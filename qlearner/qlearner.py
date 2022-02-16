'''
Q Learner Module
Translated to Python 3 from Dr. Patrick Mannion's Java implementation for a course at GMIT, March 2017.

Provides the QLearner class which contains all the behavior of a Q Learning agent.
This agent has
- a Q table (a numpy array)
- a Q learning TD update function
- max value action selction
- epsilon-greedy action selection
- random action selection
'''

import numpy as np
import random
import pickle


class QLearner(object):
    """
    Q Learning Agent Class.
    Contains the Q Table, update functions, and action selection.
    """

    def __init__(self, num_states=0, num_actions=0, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        """
        self.num_states = num_states
        self.num_actions = num_actions
        # Assume that no states means we will dynamically discover the states
        if num_states == 0:
            self._q_table = {}
        else:
            self._q_table = np.zeros((num_states, num_actions))

        # Learning Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def checkState(self, state):
        if type(self._q_table) is dict:
            if state in self._q_table:
                pass
            else:
                self._q_table[state] = [1] * self.num_actions

    def updateQValue(self, previous_state, selected_action, current_state, reward):
        old_q = self._q_table[previous_state][selected_action]
        max_q = self.getMaxQValue(current_state)
        new_q = old_q + self.alpha * (reward + self.gamma * max_q - old_q)
        self._q_table[previous_state][selected_action] = new_q

    def selectAction(self, state):
        """
        Epsilon-greedy action selection.

        Args:
            state: Integer, the state the agent sees.

        Returns: Integer, action the agent has selected
        """
        if random.random() < self.epsilon:
            # Select random action with probabilty epsilon
            # print("Random action taken")
            return self.randomAction()
        else:
            # Select most valueable action
            return self.getMaxValueAction(state)

    def randomAction(self):
        """ Returns a random action from the action space. """
        return random.randint(0, self.num_actions - 1)

    def getMaxValueAction(self, state):
        return np.argmax(self._q_table[state])

    def getMaxQValue(self, state):
        """
        Returns the actual Q Value for the max action at this state.
        Args:
            state: Integer, the state the agent sees.

        Returns: The Max Q value for this state.
        """
        best_action = self.getMaxValueAction(state)
        return self._q_table[state][best_action]

    def saveQTable(self, filename="qtable.pkl"):
        """
        Quick & dirty, saves a Q Table as a pickle file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self._q_table, f)

    def loadQTable(self, tablefile="qtable.pkl"):
        """
        Loads the Q Table into the agent from file (pickle)

        Args: 
            tablefile: a file object pointing to a pickle of the Q Table
        """
        with open(tablefile, 'rb') as f:
            self._q_table = pickle.load(f)
