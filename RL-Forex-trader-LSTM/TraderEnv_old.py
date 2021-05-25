import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from pathlib import Path

# position constant
LONG = 0
SHORT = 1
FLAT = 2

# action constant
BUY = 0
SELL = 1
HOLD = 2


class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, show_trade=True, trailing_stop=0.02, initial_portfolio=1000.0, fee=0.0005,
                 margin=25):
        self.show_trade = show_trade
        self.path = path
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.fee = fee
        self.initial_portfolio = initial_portfolio
        self.margin = margin
        self.seed()
        self.file_list = []
        # load_csv
        self.load_from_csv()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features + 4)
        self.trailing_stop = trailing_stop
        self.max_profit = 0

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def load_from_csv(self):
        self.df = pd.read_csv('./data/train/5_15_30_1h_timeframes.csv')
        # inputs = np.load("inputs.npy")
        # train = pd.read_csv('train_timeframes.csv').iloc[:inputs.shape[0]]

        self.df.dropna(inplace=True)  # drops Nan rows
        self.closingPrices = self.df['close'].values
        feature_list = [c for c in self.df.columns if 'date' not in c and c != 'time' and c != 'close']
        self.df = self.df[feature_list].values.astype(np.float64)

    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        profit = self.get_profit()
        self.reward += profit
        if action == BUY:  # buy
            if self.position == FLAT:  # if previous position was flat
                self.entry_price = self.closingPrice  # maintain entry price
                self.position = LONG  # update position to long
            elif self.position == SHORT:  # if previous position was short
                self.entry_price = self.closingPrice
                self.n_short += 1  # record number of short
                self.position = LONG  # update position to flat
            elif profit - self.max_profit < -self.trailing_stop:  # check stop
                self.entry_price = 0  # clear entry price
                self.n_short += 1  # record number of short
                self.n_stopped += 1
                self.position = FLAT
            elif self.position == LONG and profit > self.max_profit:
                self.max_profit = profit
        elif action == SELL:  # vice versa for short trade
            if self.position == FLAT:
                self.entry_price = self.closingPrice
                self.position = SHORT
            elif self.position == LONG:
                self.entry_price = self.closingPrice
                self.n_long += 1
                self.position = SHORT
            elif profit - self.max_profit < -self.trailing_stop:  # check stop
                self.entry_price = 0
                self.n_long += 1
                self.n_stopped += 1
                self.position = FLAT
            elif self.position == SHORT and profit > self.max_profit:
                self.max_profit = profit
        else:
            self.position = FLAT

        self.portfolio *= (1.0 + self.reward)
        self.current_tick += 1
        if self.show_trade and self.current_tick % 100 == 0:
            print("Tick: {0}/ Portfolio (USD): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        if self.current_tick > (self.df.shape[0]) - self.window_size - 1:
            self.done = True
            self.reward = self.get_profit()  # return reward at end of the game
        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    "history": self.history,
                                                    "n_trades": {'long': self.n_long, 'short': self.n_short}}

    def get_profit(self):
        if self.position == LONG:
            profit = ((self.closingPrice - self.entry_price) / self.entry_price + 1) * (1 - self.fee) ** 2 - 1
        elif self.position == SHORT:
            profit = ((self.entry_price - self.closingPrice) / self.closingPrice + 1) * (1 - self.fee) ** 2 - 1
        else:
            profit = 0
        return profit * self.margin

    def reset(self):
        # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        self.current_tick = 0
        print("start episode ... at {0}".format(self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0
        self.n_stopped = 0

        # clear internal variables
        self.history = []  # keep buy, sell, hold action history
        self.portfolio = self.initial_portfolio
        self.profit = 0

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.updateState()  # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        return self.state

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]

        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position, 3)
        profit = self.get_profit()
        # append two
        self.state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
        return self.state
