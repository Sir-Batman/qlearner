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
    def __init__(self):
        """

        """
        raise NotImplemented

    def initializeQValues(self):
        raise NotImplemented

    def updateQvalue(self, PreviousState, selectedAction, currentState, reward):
        raise NotImplemented

    def selectAction(self, state):
        raise NotImplemented

    def randomAction(self):
        """ Returns a random action from the action space. """
        return random.randint(0, self.numActions)

    def getMaxValueAction(self, state):
        raise NotImplemented

    def getMaxQValue(self, state):
        raise NotImplemented

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


