#!/usr/bin/python

import random
import numpy as np


class Agent:

    # Constructor
    def __init__(self):

        # Define actions and the number of row and column for the grid world
        self.actions = ['right', 'up', 'left', 'down', 'stay']
        self.actions_dict = {'right': np.array([0, 1]), 'up': np.array([-1, 0]), 'left': np.array([0, -1]), 'down': np.array([1, 0]), 'stay': np.array([0, 0])}
        self.num_action = len(self.actions)

        self.num_row = 5
        self.num_col = 10
        
    # Public method
    # Method for moving the agent based on the selected action
    def move_agent(self, action):

        # Set termination criteria as false 
        terminate = False
        move_coord = self.actions_dict[action]
        # Give an immediate reward
        reward = - 1

        pos_curr = self.get_position()
        pos_next = self.get_position() + move_coord
        pos_next[0] = np.clip(pos_next[0], 0, 4)
        pos_next[1] = np.clip(pos_next[1], 0, 9)

        # If statement for avoiding being on the wall
        if ((pos_next[0] >= 2) & (pos_next[0] <= 4) & (pos_next[1] == 7)):
            pos_next = pos_curr
            reward = -1

        # If statement when the agent reaches the goal position
        if (pos_next == self.goal).all():
            terminate = True
            reward = 20

        self.__set_position(pos_next)

        return pos_next, reward, terminate
    
    def reset(self, goal_prev):

        # Randomly apply initial position of the agent
        while True:
            start_row = random.randint(0, 4)
            start_col = random.randint(0, 9)
            # If statement for the wall condition
            if start_row < 2 or start_col != 7:
                if start_row != 3 and start_col != 9:
                    break

        # Set initial position and the goal
        self.pos = np.array([start_row, start_col])

        # Randomly apply the goal position 
        while True:
            nxtgoal = random.randint(1, 4)
            if nxtgoal == 1:
                goal_cnd = (goal_prev[0] - 1, goal_prev[1])
            elif nxtgoal == 2:
                goal_cnd = (goal_prev[0] + 1, goal_prev[1])
            elif nxtgoal == 3:
                goal_cnd = (goal_prev[0], goal_prev[1] - 1)
            else:
                goal_cnd = (goal_prev[0], goal_prev[1] + 1)

            if goal_cnd[0] >= 0 and goal_cnd[0] <= 4:
                if goal_cnd[1] >= 0 and goal_cnd[1] <= 9:
                    if goal_cnd != (2, 7) and goal_cnd != (3, 7) and goal_cnd != (4, 7):
                        self.goal = np.array([goal_cnd[0], goal_cnd[1]])
                        break

    # Private method
    # Method for getting position
    def get_position(self):
        return self.pos

    # Method for setting position
    def __set_position(self, pos_next):

        if type(pos_next) == list:
            pos = np.array(pos_next)
        else:
            pos = pos_next

        assert (pos[0] >= 0 and pos[0] < 5 and pos[1] >= 0 and pos[1] < 10)
        self.pos = pos