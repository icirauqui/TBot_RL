"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import random 
import math
import pandas as pd

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from constants import *



#CONST_INIT_BALANCE = 1000.0
#CONST_FEE = 0.00075
#CONST_MAX_STEPS = 10000
#CONST_SIM_STEPS = 100000



def rounddown(number,decimals):
    return math.floor(number*(10**decimals))/(10**decimals)



class TradeEnv(gym.Env):
    """
    Description:

    Source:

    Observation:
        Type:   Box(4)
        Num     Observation     Min     Max
        0       Open            0.0     Inf
        1       High            0.0     Inf
        2       Low             0.0     Inf
        3       Close           0.0     Inf
        4       Volume          0.0     Inf
    Actions:
        Type:   Discrete(3)
        Num     Action
        0       Buy
        1       Sell
        2       Hold

    Reward:
        if profit(market)>0 and profit(bot)<>0:
            profit(bot)/profit(market)
    
    Starting State:
        Funds = 1000.0 €
        Time = random time in starting 50% of the data

    Episode Termination:
        funds < 100.0
        Solved Requirements: considered solved when the average return is greater
        than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,df):

        self.df = df
        self.balance = CONST_INIT_BALANCE
        self.profit = 0.0

        self.max_net_worth = CONST_INIT_BALANCE

        self.shares_held = 0.0
        self.state = 0
        self.buy_price = 0.0
        self.sell_price = 0.0

        self.episode = 1
        self.total_steps = 0

        #self.action_space = spaces.Discrete(3)
        
        self.action_space = spaces.Box(
            low = -1,
            high = 1,
            shape = (1, ),
            dtype=np.float16
        )
        
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = (61,8),
            dtype = np.float32
        )

        # Graph to render
        self.graph_reward = []
        self.graph_profit = []
        self.graph_benchmark = []


    def _next_observation(self):
        # Get the data for the last 10 timestep
        frame = np.array([
            self.df.loc[self.current_step-59:self.current_step,'Open'],
            self.df.loc[self.current_step-59:self.current_step,'High'],
            self.df.loc[self.current_step-59:self.current_step,'Low'],
            self.df.loc[self.current_step-59:self.current_step,'Close'],
            self.df.loc[self.current_step-59:self.current_step,'Volume'],
            self.df.loc[self.current_step-59:self.current_step,'EMA1'],
            self.df.loc[self.current_step-59:self.current_step,'EMA2'],
            self.df.loc[self.current_step-59:self.current_step,'EMA3']
        ])

        # Append additional data
        obs = np.append(
            np.transpose(frame),
            [[
                self.net_worth / self.max_net_worth,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]],
            axis = 0
        )
        return obs


    def _take_action(self, action):

        step_row = self.df.iloc[self.current_step]
        current_price = random.uniform(step_row['Real open'],step_row['Real close'])

        #reward = 0

        if action[0]>0:
            # Buy with all available funds
            amount_to_trade = rounddown(self.balance / ((1.0 + CONST_FEE) * current_price),6)
            self.shares_held += amount_to_trade
            self.balance -= amount_to_trade * current_price
            self.buy_value = amount_to_trade * current_price
            self.state = 1
            
        elif action[0]<0:
            amount_to_trade = rounddown(self.shares_held,6)
            self.shares_held -= amount_to_trade
            self.balance += amount_to_trade*((1.0-CONST_FEE)*current_price)
            self.sell_value = amount_to_trade*((1.0-CONST_FEE)*current_price)
            self.state = 2
            
        self.net_worth = self.balance + self.shares_held*current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth





    def step(self, action, end=True):

        self._take_action(action)

        self.current_step += 1
        self.total_steps += 1



        # Calculus of the reward
        profit = self.net_worth - (CONST_INIT_BALANCE)

        profit_percent = profit / (CONST_INIT_BALANCE) * 100

        benchmark_profit = (self.df.loc[self.current_step, 'Real open'] /
                            self.df.loc[self.start_step, 'Real open'] -
                            1) * 100

        diff = profit_percent - benchmark_profit
        reward = np.sign(diff) * (diff)**2


        #if self.state == 2:
        #    profit = self.buy_value - self.sell_value
        #    if profit < 0:
        #        reward += - profit**2
        #    else:
        #        reward += profit**2
        #    self.state = 0

        #reward = self.buy_value - self.sell_value

        if self.state==2:
            outcome = self.sell_value - self.buy_value
            if np.sign(outcome)>0:
                reward = outcome**4
            else:
                reward = - (outcome**2)

        elif self.state==1:
            reward = 0
        else:
            reward = -1

        
        if self.current_step >= (CONST_MAX_STEPS + self.start_step):
            end = True
        else:
            end = False
        #print(self.current_step,self.start_step,end)
        done = self.net_worth <= 100.0

        if done or end:
            self.episode_reward = reward
            #self._render_episode()
            self.graph_profit.append(profit_percent)
            self.graph_benchmark.append(benchmark_profit)
            self.graph_reward.append(reward)
            self.episode += 1
            self.reset()

        obs = self._next_observation()

        self.render(print_step=True)

        return obs, reward, done, {}




    def reset(self):
        self.balance = CONST_INIT_BALANCE
        self.net_worth = CONST_INIT_BALANCE
        self.shares_held = 0
        self.state = 0
        self.buy_price = 0.0
        self.sell_price = 0.0
        self.episode_reward = 0


        self.current_step =  int(60 + random.uniform(0,self.df.shape[0]-CONST_MAX_STEPS-1))
        self.start_step = self.current_step

        self.buy_value = 0.0
        self.sell_value = 0.0

        return self._next_observation()




    def render(self, print_step=False, graph=False, *args):
        profit = self.net_worth - CONST_INIT_BALANCE

        profit_percent = profit / CONST_INIT_BALANCE * 100

        benchmark_profit = (self.df.loc[self.current_step, 'Real open'] /
                            self.df.loc[self.start_step, 'Real open'] -
                            1) * 100

        if print_step:
            print("--------------------------------------------------")
            print(f'Steps            : {round(100*self.total_steps/CONST_SIM_STEPS,2)}% \t {self.current_step - self.start_step}')
            print(f'Net worth (max)  : {round(self.net_worth, 2)} ({round(self.max_net_worth, 2)})')
            print(f'Profit - Bench   : {round(profit_percent, 1)}% \t {round(benchmark_profit, 0)}')

        # Plot the graph of the reward
        if graph:
            fig = plt.figure()
            fig.suptitle('Training graph')

            high = plt.subplot(2, 1, 1)
            high.set(ylabel='Gain')
            plt.plot(self.graph_profit, label='Bot profit')
            plt.plot(self.graph_benchmark, label='Benchmark profit')
            high.legend(loc='upper left')

            low = plt.subplot(2, 1, 2)
            low.set(xlabel='Episode', ylabel='Reward')
            plt.plot(self.graph_reward, label='reward')

            plt.show()

        return profit_percent

    def _render_episode(self, filename='render/render.txt'):
        file = open(filename, 'a')
        file.write('-----------------------\n')
        file.write(f'Episode numero: {self.episode}\n')
        file.write(f'Profit: {round(self.render()[0], 2)}%\n')
        file.write(f'Benchmark profit: {round(self.render()[1], 2)}%\n')
        file.write(f'Reward: {round(self.episode_reward, 2)}\n')
        file.close()
