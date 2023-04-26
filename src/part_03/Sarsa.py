import numpy as np
import pandas as pd
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

class Sarsa:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values

    def train(self, filename, plotFile):
        rewards_plot = []
        episodes_plot = []
        for i in range(1, self.episodes+1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0
            action = self.select_action(state) # selecione a primeira ação
            while not done:
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.select_action(next_state) # selecione a próxima ação

                old_value = self.q_table[state, action]
                next_value = self.q_table[next_state, next_action]
                new_value = old_value + self.alpha*(reward + self.gamma*next_value - old_value)
                self.q_table[state, action] = new_value

                state = next_state
                action = next_action
                actions=actions+1
                rewards=rewards-1
            if reward > 0:
                rewards += 1000
            else:
                rewards -= 1000  
            rewards_plot.append(rewards)
            episodes_plot.append(i)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=',')
        if (plotFile is not None): self.plotactions(plotFile, rewards_plot, episodes_plot)
        return self.q_table

    def plotactions(self, plotFile, rewards_plot, episodes_plot):
        plt.plot(episodes_plot, rewards_plot)
        dict = {"episodes":episodes_plot,"rewards": rewards_plot}
        df = pd.DataFrame(dict)
        df.to_csv("sarsa_plotdata.csv")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Episodes vs Rewards (Sarsa Algorithm)')
        plt.savefig(plotFile+".jpg")     
        plt.close()
