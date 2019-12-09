#!/usr/bin/python

import random
import numpy as np
import pandas as pd

import GridWorld_FixedDoor
import GridWorld_MoveDoor
import Plotter

class EpisodeRunner:

    def __init__(self):
        self.num_action = 5
        self.num_row = 5
        self.num_col = 10
        self.actions = ['right', 'up', 'left', 'down', 'stay']
        self.num_episode = 5000

        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 1.0
        self.iters_episode = 20

    # Public method
    # Method for running episodes for learn the optimized policy
    def runner(self, reward_history, index, Agent, goal_stat):

        # Initialize Q value
        Q = np.zeros((self.num_action, self.num_row, self.num_col))
        
        goal = np.array([3, 9])

        # Loop for episode
        for epi in range(self.num_episode):

            # Initialize reward
            rewards = np.array([])

            # Initialize the state of the agent
            done = False
            if goal_stat == 'fix':
                Agent.reset()
            else:
                Agent.reset(goal)
            s = Agent.get_position()
            a = self.__select_action(Q, s, self.epsilon)

            # Set temporary reward
            tmp = 0

            for iters in range(self.iters_episode):

                if done == False:

                    a = self.__select_action(Q, s, self.epsilon)
                    s_next, reward, done = Agent.move_agent(self.actions[a])

                    tmp += reward
                    rewards = np.append(rewards, reward)

                    # Select action based on maximum Q value (Q-learning)
                    a_next = self.__select_action(Q, s_next, 0)

                    # Update Q value
                    Q_next = Q[a_next, s_next[0], s_next[1]]
                    Q[a, s[0], s[1]] += self.alpha * (reward + self.gamma * Q_next - Q[a, s[0], s[1]])
                    s = s_next

            reward_history[index, epi] = tmp

        return reward_history, Q

    # Private method
    # Method for selecting action based on epsilon greedy method
    def __select_action(self, Q, s, epsilon):

        greedy = np.argmax(Q[:, s[0], s[1]])
        greedy_index = np.where(Q[:, s[0], s[1]] == greedy)[0]

        if len(greedy_index) > 1:
            greedy = np.random.choice(greedy_index)

        prob = [(1 - epsilon + epsilon / len(self.actions)) if i == greedy else epsilon / len(self.actions) for i in range(len(self.actions))]

        return np.random.choice(np.arange(len(self.actions)), p=prob)

def main():

    # Call each class for learning
    agent_f = GridWorld_FixedDoor.Agent()
    agent_m = GridWorld_MoveDoor.Agent()

    plotter = Plotter.Plotter()

    # Call episode runner for running RL
    episode_runner = EpisodeRunner()

    reward_history = np.zeros((2, int(episode_runner.num_episode)))
    
    reward_history, Q_f = episode_runner.runner(reward_history, index=0, Agent=agent_f, goal_stat='fix')
    reward_history, Q_m = episode_runner.runner(reward_history, index=1, Agent=agent_m, goal_stat='move')

    plotter.reward_history(reward_history)

    for i, action in enumerate(episode_runner.actions):
        plotter.Q_value_history(Q_f[i], "fixed_"+episode_runner.actions[i])
    
    for j, action in enumerate(episode_runner.actions):
        plotter.Q_value_history(Q_m[j], "moving_"+episode_runner.actions[j])

if __name__ == '__main__':
    main()