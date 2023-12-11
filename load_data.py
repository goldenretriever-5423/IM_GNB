import pickle

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import os
import random

import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as hcluster
import time

class twitter_IM:
    def __init__(self, Num_Seeds=5, Budget=100, Num_Runs=5):

        os.chdir('/Users/yuting/PycharmProjects/bandit/IM/data_processing/')

        self.num_user = 50 # cluster number
        self.tweets = pd.read_csv("6_date_sorted_influencer_10context_data_clustered.csv", delimiter=";")
        with open('influencer_embedding.pkl','rb') as f_emb:
            influencer_emb = pickle.load(f_emb)
        self.INFLUENCERS = [int(i) for i in list(influencer_emb.keys())]
        self.influencer_emb = np.array(list(influencer_emb.values()))
        self.n_arm = len(self.INFLUENCERS)
        self.MAX_NA = float(self.tweets.new_activations.max())  # maximum new acticated nodes
        self.MAX_LOG_NA = np.log(self.tweets.new_activations.max())
        # mapping str of set to set
        self.tweets.regular_node_set_unique = self.tweets.regular_node_set_unique.apply(
            lambda txt: eval(txt))
        self.tweets.regular_node_set_grouped = self.tweets.regular_node_set_grouped.apply(
            lambda txt: eval(txt))

        # all the contexts
        self.twitter_contexts = self.tweets.context
        # needed as vector for the regressions
        self.twitter_contexts = list(map(lambda context: np.array(context.split(), dtype=float), self.twitter_contexts))  # list of context array
        self.dim = 2*len(self.twitter_contexts[0])

        np.random.seed(100)
        self.seeds = dict.fromkeys(list(set([np.random.randint(1000) for _ in np.arange(Num_Seeds + 10)]))[:Num_Seeds])

        for seed in self.seeds.keys():
            np.random.seed(seed)
            # select the contexts for the running campaigns
            context_idx = list(set([np.random.randint(0, len(self.twitter_contexts)) for _ in np.arange(Budget*Num_Runs + 100)]))[
                          :Budget*Num_Runs]
            self.seeds[seed] = [self.twitter_contexts[idx] for idx in context_idx]

    def generate(self, seed):
        campaign = []
        contexts = self.seeds[seed]
        for context in contexts:

            campaign_temp = self.tweets.loc[self.tweets.context == ' '.join(context[:self.dim + 1].astype(str)), :]
            # if campaign_temp.size > 0:
            #     campaign_temp = campaign_temp.sample()

            campaign.append(campaign_temp)

        return contexts, campaign


class weibo_IM:
    def __init__(self, Num_Seeds=5, Budget=100, Num_Runs=5):

        os.chdir('/Users/yuting/PycharmProjects/bandit/IM/data_processing/')

        # TODO

        for seed in self.seeds.keys():
            np.random.seed(seed)
            # select the contexts for the running campaigns
            context_idx = list(set([np.random.randint(0, len(self.twitter_contexts)) for _ in np.arange(Budget*Num_Runs + 100)]))[
                          :Budget*Num_Runs]
            self.seeds[seed] = [self.twitter_contexts[idx] for idx in context_idx]

    def generate(self, seed):

        # TODO
        return

