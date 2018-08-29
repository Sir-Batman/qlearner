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

class QLearner(object):
    """
    Q Learning Agent Class.
    Contains the Q Table, update functions, and action selection.
    """
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self._q_table = np.zeros((num_states, num_actions))

        # Learning Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def updateQvalue(self, previous_state, selected_action, current_state, reward):
        old_q = self._q_table[previous_state][selected_action]
        max_q = self.getMaxQValue(current_state)
        new_q = old_q + alpha*(reward + gamma*max_q - old_q)
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
            return self.randomAction()
        else:
            # Select most valueable action
            return self.getMaxValueAction(state)

    def randomAction(self):
        """ Returns a random action from the action space. """
        return random.randint(0, self.numActions)

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

    def saveQTable(self):
        """
        Saves a Q Table as a numpy, either as a pickle or a numpy csv.
        """
        raise NotImplemented

    def loadQTable(self, tablefile=None, numpyarray=None):
        """
        Loads the Q Table into the agent from file (pickle?) or numpy array

        Args: 
            tablefile: a file object pointing to a pickle of the Q Table
            numpyarray: a numpy array, in memory, of the Q Table
        """
        raise NotImplemented


