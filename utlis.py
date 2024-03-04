import pickle

import numpy as np
import sys
import pandas as pd

def load_pkl(pkl_name):
    with open(pkl_name, 'rb') as f:
        return pickle.load(f)

def invertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon

def edge_probability(n):
    return 3 * np.log(n) / n

def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)

def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d-1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((num_items, 1))/np.sqrt(2)), axis = 1)
    return x

def fracT(self, T):
    return np.sqrt((1 + np.log(1 + T)) / (1 + T))

def orgniaze_reward(reward_hist, all_reward_hist, seed, budget, num_rum):

    # reward_df = pd.DataFrame()

    # get the results
    to_add = pd.DataFrame({'reward': np.cumsum(reward_hist)})
    to_add['all_reward'] = np.cumsum(all_reward_hist)
    to_add['seed'] = seed
    to_add['algorithm'] = 'gnb'
    to_add['round'] = to_add.index + 1
    to_add['horizon'] = budget
    to_add['episode'] = num_rum
    to_add['ex_time'] = None
    # reward_df = pd.concat([reward_df, to_add])

    return to_add