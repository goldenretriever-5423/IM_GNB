import argparse

def get_GNB_parameters(dataset):

    parser = argparse.ArgumentParser(description='GNB')

    # --------------------------------------------------------------
    if dataset == "weibo":
        parser.add_argument('--dataset', default='weibo', type=str, help='mnist_only, yelp, movie_real, shuttle')
        #
        parser.add_argument('--GNN_lr', default=0.0001, type=float, help='Learning rates for GNN models')
        parser.add_argument('--user_lr', default=0.0001, type=float, help='Learning rates for GNN models')
        parser.add_argument('--bw_reward', default=5, type=float, help='Kernel bandwidth for exploitation GNN')
        parser.add_argument('--bw_conf_b', default=5, type=float, help='Kernel bandwidth for exploration GNN')
        parser.add_argument('--k', default=1, type=int, help='k-th user neighborhood over user graphs')
        parser.add_argument('--batch_size', default=-1, type=int, help='Batch size for training')
        parser.add_argument('--GNN_pool_step_size', default=20000, type=int, help='Step size for GNN gradient pooling')
        parser.add_argument('--user_pool_step_size', default=4000, type=int, help='Step size for user gradient pooling')
        parser.add_argument('--arti_explore_constant', default=0.1, type=float, help='Artificial exploration constant')
        parser.add_argument('--train_every_user_model', default=True, type=bool, help='Train every user model')
        parser.add_argument('--explore_param', default=1, type=float, help='Exploration parameter')
        #
        parser.add_argument('--separate_explore_GNN', default=False, type=bool,
                            help='Matrix embedding for GNN exploration')

    elif dataset == "twitter":
        parser.add_argument('--dataset', default='twitter', type=str, help='weibo, twitter')
        #
        parser.add_argument('--GNN_lr', default=0.0001, type=float, help='Learning rates for GNN models')
        parser.add_argument('--user_lr', default=0.0001, type=float, help='Learning rates for GNN models')
        parser.add_argument('--bw_reward', default=5, type=float, help='Kernel bandwidth for exploitation GNN')
        parser.add_argument('--bw_conf_b', default=5, type=float, help='Kernel bandwidth for exploration GNN')
        parser.add_argument('--k', default=1, type=int, help='k-th user neighborhood over user graphs')
        parser.add_argument('--batch_size', default=-1, type=int, help='Batch size for training')
        parser.add_argument('--GNN_pool_step_size', default=1000, type=int, help='Step size for GNN gradient pooling')
        parser.add_argument('--user_pool_step_size', default=100, type=int, help='Step size for user gradient pooling')
        parser.add_argument('--arti_explore_constant', default=0.1, type=float, help='Artificial exploration constant')
        parser.add_argument('--train_every_user_model', default=True, type=bool, help='Train every user model')
        parser.add_argument('--explore_param', default=1, type=float, help='Exploration parameter')
        #
        parser.add_argument('--separate_explore_GNN', default=False, type=bool,
                            help='Matrix embedding for GNN exploration')


    else:
        print("Undefined data set")
        return None

    print(parser)
    return parser























