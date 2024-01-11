import pandas as pd
import torch

import utlis
from bandit_algo import User_GNN_Bandit_Per_Arm
from parameters import get_GNB_parameters
import argparse
import numpy as np
import time
from datetime import datetime
import sys
# from User_GNN_packages import *


from load_data import twitter_IM


# sys.stdout = Logger(sys.stdout)

if __name__ == '__main__':
    # check gpu status
    # torch.cuda.set_device(0)
    # -----------------------------
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    ####################
    Num_Seeds = 10 # ->NRUNS
    Num_Runs = 1 # T episode (single campaign or multiple campaign)
    Num_cluster = 20
    Budget = 500 # H hirizon
    L = 3
    # seeds = list(set([np.random.randint(1000) for _ in np.arange(Num_Seeds + 10)]))[:Num_Seeds]
    # dataset_list = ['twitter','weibo']
    dataset_list = ['twitter']
    rewards_df = pd.DataFrame()
    ####################

    for data in dataset_list:

        if data == 'twitter':
            b = twitter_IM(Num_Seeds=Num_Seeds, Budget=Budget, Num_Runs=Num_Runs, Num_cluster=Num_cluster)
        if data == 'weibo':
            #TODO load weibo data
            continue

        count = 0
        for seed in b.seeds:
            count += 1
            print("seed: ", count, "in seeds: ", len(b.seeds))

            contexts, campaign = b.generate(seed)
            influencer_hist_lin = []
            activations_hist = [] # distinctive act
            all_activations_hist = []  # all action

            for p_i in range(Num_Runs):
                print("--- Current run: {}/{}".format(p_i + 1, Num_Runs))
                # load training related parameters
                parser = get_GNB_parameters(dataset=data)
                args = parser.parse_args()

                algo_name = 'GNB'
                model = User_GNN_Bandit_Per_Arm(dim=b.dim, user_n=b.num_user, arm_n=b.n_arm, k=args.k,
                                                GNN_lr=args.GNN_lr, user_lr=args.user_lr,
                                                bw_reward=args.bw_reward, bw_conf_b=args.bw_conf_b,
                                                batch_size=args.batch_size,
                                                GNN_pooling_step_size=args.GNN_pool_step_size,
                                                user_pooling_step_size=args.user_pool_step_size,
                                                arti_explore_constant=args.arti_explore_constant,
                                                num_layer=-1, explore_param=args.explore_param,
                                                separate_explore_GNN=args.separate_explore_GNN,
                                                train_every_user_model=args.train_every_user_model,
                                                device=device)
                print(data, algo_name, args.GNN_lr, args.user_lr, args.bw_reward, args.bw_conf_b, args.k,
                      args.batch_size,
                      args.GNN_pool_step_size, args.user_pool_step_size, args.arti_explore_constant,
                      args.train_every_user_model, args.separate_explore_GNN)


                running_time_sum, rec_time_sum = 0.0, 0.0
                print("Round; Regret; Regret/Round")
                start_t = time.time()

                # exploration_factor = np.sqrt(np.log(2 * BUDGET * K / delta) / 2)  # for now not yet


                prev_activated = set()

                for t in range(Budget):

                    context = contexts[p_i * Budget + t]

                    this_rec_time_s = time.time()

                    # Update user graphs
                    model.update_user_graphs(contexts=np.hstack((b.influencer_emb,np.tile(context.reshape(1,-1), (10,1)))),user_i=1) # TODO This user_i=1 is temporary
                    this_g_update_time = time.time()

                    # Recommendation
                    arm_select, point_est, whole_gradients = model.recommend(np.hstack((b.influencer_emb,np.tile(context.reshape(1,-1), (10,1)))), t, L)

                    #
                    running_time_sum += (time.time() - this_rec_time_s)
                    rec_time_sum += (time.time() - this_g_update_time)

                    # -------------------------------------------------------------------------------------

                    # all_activations = set()
                    #
                    tweet = campaign[p_i * Budget + t][campaign[p_i * Budget + t].influencer.isin([b.INFLUENCERS[a] for a in arm_select])]
                    acts = set()
                    acts_grouped = set()
                    for row in tweet.itertuples():
                        acts.update(row.regular_node_set_unique)
                        acts_grouped.update(row.regular_node_set_grouped)
                    influencer_hist_lin.append(list(tweet.influencer))

                    reward = len(acts - prev_activated)
                    reward2 = len(acts)
                    prev_activated.update(acts)
                    activations_hist.append(reward)
                    all_activations_hist.append(reward2)

                    # -------------------------------------------------------------------------------------
                    # update model
                    for arm in arm_select:
                        # Update model when made false prediction ---------------------------------------------------------
                        if reward == 0:
                            for u in np.arange(b.num_user):
                                # Create additional samples for exploration network -----------------------------
                                # Add artificial exploration info when made false predictions
                                if args.arti_explore_constant > 0:
                                    model.update_artificial_explore_info(t, u, arm, whole_gradients)
                                model.update_info(u_selected=u, a_selected=arm, contexts=np.hstack((b.influencer_emb, np.tile(context.reshape(1, -1), (10, 1)))), reward=0,
                                                  GNN_gradient=whole_gradients[arm],
                                                  GNN_residual_reward=-point_est[arm][u])
                        # Update model when made right prediction ---------------------------------------------------------
                        else:
                            for u in acts_grouped:# TODO update on distinctive activations
                                GNN_residual_reward = 1 / len(arm_select) - point_est[arm][u]
                                model.update_info(u_selected=u, a_selected=arm, contexts=np.hstack((b.influencer_emb,np.tile(context.reshape(1,-1), (10,1)))), reward=1/len(arm_select),
                                                  GNN_gradient=whole_gradients[arm],
                                                  GNN_residual_reward=GNN_residual_reward)


                    # for each spread campaign
                    u_exploit_loss, u_explore_loss = model.train_user_models(u=acts_grouped) #TODO from an individual user to all the users
                    GNN_exploit_loss, GNN_explore_loss = model.train_GNN_models()
                    print("Loss: ", u_exploit_loss, u_explore_loss, GNN_exploit_loss, GNN_explore_loss)

            reward_df = utlis.orgniaze_reward(activations_hist,all_activations_hist, seed,Budget,p_i)
            rewards_df = rewards_df.append(reward_df)

        # rewards_df.to_pickle("gnb_" + str(L) + ".pkl")

        # "gnb" + str(L) + "_" + str(Num_Seeds) + "seeds_round" + str(Budget) + "_single.csv"
        # "gnb" + str(L) + "_" + str(Num_Seeds) + "seeds_round" + str(Budget) + "_single"

