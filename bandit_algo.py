from User_GNN_packages import *
from User_GNN_network import *
import itertools
from collections import defaultdict


class User_GNN_Bandit_Per_Arm:
    def __init__(self, dim, user_n, arm_n, k=1, GNN_lr=0.0001, user_lr=0.0001, hidden=100, bw_reward=10, bw_conf_b=10,
                 batch_size=-1, GNN_pooling_step_size=500, user_pooling_step_size=500,
                 arti_explore_constant=0.01, num_layer=-1, explore_param=1,
                 neighborhood_size=-1, train_every_user_model=False, separate_explore_GNN=False,
                 last_layer_gradient_flag=False,
                 device=None):
        self.context_list = []
        self.reward = []
        self.GNN_lr = GNN_lr
        self.dim = dim
        self.hidden = hidden
        self.t = 0
        self.k = k
        self.batch_size = batch_size
        self.GNN_pooling_step_size = GNN_pooling_step_size
        self.arti_explore_constant = arti_explore_constant
        self.num_layer = num_layer
        self.model_explore_hidden = 100
        self.explore_param = explore_param
        self.neighborhood_size = neighborhood_size
        self.gpy_rbf_kernel_est = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.gpy_rbf_kernel_CB = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.device = device

        print("Num Layer: ", num_layer)
        print("Explore Param: ", self.explore_param)
        print("Explore hidden size: ", self.model_explore_hidden)

        self.bw_reward = bw_reward
        self.bw_conf_b = bw_conf_b

        self.user_n = user_n
        #
        if neighborhood_size > 0:
            graph_user_n = neighborhood_size
        else:
            graph_user_n = user_n
        print("--- Neighborhood size: ", neighborhood_size)
        self.graph_user_n = graph_user_n

        self.user_select_count = [0 for _ in range(user_n)]
        self.selected_user_period = set()
        self.arm_n = arm_n
        self.u_funcs_f_1 = {}
        self.u_funcs_f_2 = {}
        self.user_ests = None
        self.user_gradients = None
        self.separate_explore_GNN = separate_explore_GNN
        self.train_every_user_model = train_every_user_model
        self.last_layer_gradient_flag = last_layer_gradient_flag
        self.target_user_new_indices_ests, self.target_user_new_indices_CBs = None, None
        self.user_neighborhood_list_est = [[] for _ in range(arm_n)]
        self.selected_user_neighborhood_list = []

        # Dimension reduction operators
        self.GNN_reduced_grad_dim = user_n - 1
        self.user_reduced_grad_dim = arm_n - 1
        self.GNN_grad_op = LocallyLinearEmbedding(n_components=self.GNN_reduced_grad_dim)
        self.user_grad_op = LocallyLinearEmbedding(n_components=self.user_reduced_grad_dim)

        # Two user graphs ---> adjacency matrix with regard to each arm
        self.user_exploitation_graph_dict = {i: np.zeros([graph_user_n, graph_user_n]) for i in range(arm_n)}
        self.user_exploration_graph_dict = {i: np.zeros([graph_user_n, graph_user_n]) for i in range(arm_n)}
        self.arm_to_target_user_dict_est = {}
        self.arm_to_target_user_dict_CB = {}
        # initialize the graph weights
        for a_i in range(arm_n):
            for i in range(graph_user_n):
                for j in range(i, graph_user_n):
                    weight_1 = 1 if i == j else random.random()
                    weight_2 = 1 if i == j else random.random()
                    self.user_exploitation_graph_dict[a_i][i, j] = weight_1
                    self.user_exploration_graph_dict[a_i][i, j] = weight_2
        # buid adjacency matrix
        self.adj_m_exploit = []
        self.adj_m_explore = []
        # what is c_matrix?
        self.embedded_c_matrix = {}
        self.context_tensors = {}

        # Change the input dim with dimension reduction
        self.GNN_exploit_model = Exploitation_GNN(user_n=user_n, input_dim=self.dim,
                                                  reduced_output_dim=self.GNN_reduced_grad_dim, hidden_size=self.hidden,
                                                  lr_rate=GNN_lr, batch_size=batch_size,
                                                  pool_step_size=GNN_pooling_step_size, num_layer=num_layer,
                                                  last_layer_gradient_flag=last_layer_gradient_flag,
                                                  neighborhood_size=neighborhood_size,
                                                  device=device)
        if last_layer_gradient_flag:
            self.GNN_exploit_model.exploitation_model.change_grad_last_layer(predicting=True)

            GNN_total_param_count = sum(param.numel() for param in
                                        self.GNN_exploit_model.exploitation_model.est_module.parameters())
            self.GNN_exploit_model.exploitation_model.change_grad_last_layer(predicting=False)
        else:
            GNN_total_param_count = sum(param.numel() for param in
                                        self.GNN_exploit_model.exploitation_model.parameters())
        if self.GNN_pooling_step_size > 0:
            self.GNN_reduced_grad_dim = (GNN_total_param_count // self.GNN_pooling_step_size) + 1

        #
        self.GNN_explore_model = Exploration_GNN(user_n=user_n, input_dim=self.GNN_reduced_grad_dim,
                                                 hidden_size=self.model_explore_hidden,
                                                 lr_rate=GNN_lr, batch_size=batch_size,
                                                 separate_explore_GNN=self.separate_explore_GNN,
                                                 num_layer=num_layer,
                                                 neighborhood_size=neighborhood_size,
                                                 device=device)

        # ----------------------------------------------------------
        user_total_param_count = utils.getuser_f_1_param_count(dim, user_n, arm_n, self.user_reduced_grad_dim, hidden,
                                                               user_lr, batch_size, 1, device)

        #
        user_explore_grad_dim = int((user_total_param_count // user_pooling_step_size))
        if user_pooling_step_size > 0:
            self.user_reduced_grad_dim = user_explore_grad_dim

        print("GNN param count: ", GNN_total_param_count)
        print("GNN pool step size: ", GNN_pooling_step_size)
        print("GNN reduced gradient size: ", (GNN_total_param_count // self.GNN_pooling_step_size) + 1)
        print("User param count: ", user_total_param_count)
        print("User reduced gradient size: ", user_pooling_step_size)
        print("User reduced gradient size: ", user_explore_grad_dim)

        for i in range(user_n):
            self.u_funcs_f_1[i] = Exploitation_FC(dim, user_n, arm_n=arm_n, reduced_dim=self.user_reduced_grad_dim,
                                                  hidden=hidden, lr=user_lr, batch_size=batch_size,
                                                  pool_step_size=user_pooling_step_size, device=device)
            #
            self.u_funcs_f_2[i] = Exploration_FC(self.user_reduced_grad_dim, hidden=self.model_explore_hidden,
                                                 lr=user_lr, batch_size=batch_size, device=device)
        print("User param count", user_total_param_count)
        self.exploitation_adj_matrix_dict, self.exploration_adj_matrix_dict = None, None
        self.selected_arm = None
        print("Each arm is given two user graphs!")

    def update_info(self, u_selected, a_selected, contexts, reward, GNN_gradient, GNN_residual_reward):
        #
        self.user_select_count[u_selected] += 1  # TODO user another counter for truly activated node
        self.selected_user_period.add(u_selected)

        # Update EE-Net module info
        reward = torch.tensor(reward)
        context = torch.tensor(contexts[a_selected, :])
        # gradient and residual reward for exploit/explore NN network
        user_gradient = self.user_gradients[u_selected][a_selected, :].detach().reshape(-1, )
        user_residual_reward = reward - self.user_ests[u_selected, a_selected].detach()
        # exploit / explore NN
        self.u_funcs_f_1[u_selected].update(context, reward)
        self.u_funcs_f_2[u_selected].update(user_gradient, user_residual_reward)

        # Update GNN module info
        embed_c = self.embedded_c_matrix[a_selected]

        if self.separate_explore_GNN:
            embed_g = torch.tensor(
                utils.generate_matrix_embedding_gradients(source=GNN_gradient)).float()
        else:
            embed_g = GNN_gradient

        GNN_residual_reward = GNN_residual_reward
        exploit_adj_m_tensor = self.exploitation_adj_matrix_dict[a_selected]
        explore_adj_m_tensor = self.exploration_adj_matrix_dict[a_selected]

        #
        if self.neighborhood_size > 0:
            u_selected_tensor = self.target_user_new_indices_ests[a_selected]
        else:
            u_selected_tensor = torch.tensor(np.array([u_selected]))

        self.GNN_exploit_model.update_info(embed_c, reward, u_selected_tensor, exploit_adj_m_tensor,
                                           selected_neighborhood=self.user_neighborhood_list_est[a_selected])
        self.GNN_explore_model.update_info(embed_g, GNN_residual_reward, u_selected_tensor, explore_adj_m_tensor)

    def update_artificial_explore_info(self, t, u_selected, arm_selected, whole_gradients):
        index = 0

        '''set small scores for un-selected arms if the selected arm is 0-reward'''

        c = torch.tensor(np.array([self.arti_explore_constant]))
        for arm_grad in whole_gradients:
            if index != arm_selected:
                explore_adj_m_tensor = self.exploration_adj_matrix_dict[index]
                if self.neighborhood_size > 0:
                    u_selected_tensor = self.target_user_new_indices_ests[index]
                else:
                    u_selected_tensor = torch.tensor(np.array([u_selected]))

                #
                if self.separate_explore_GNN:
                    embed_g = torch.tensor(
                        utils.generate_matrix_embedding_gradients(source=arm_grad)).float()
                else:
                    # embed_g = torch.tensor(arm_grad).float()
                    embed_g = arm_grad

                user_gradient = self.user_gradients[u_selected][index, :].detach().reshape(-1, )

                #
                self.GNN_explore_model.update_info(embed_g, c, u_selected_tensor, explore_adj_m_tensor)
                self.u_funcs_f_2[u_selected].update(user_gradient, c)

            index += 1

    ############################################################################
    def get_top_users_random(self, reward_ests, CB_ests, target_user):
        top_k_est_tensor, top_k_CB_tensor = [], []
        target_user_new_indices_ests, target_user_new_indices_CBs = [], []
        user_neighborhood_list_est = []
        target_user_tensor = torch.ones(1, ) * target_user
        user_range = [*range(target_user), *range(target_user + 1, self.user_n)]

        for a_i in range(self.arm_n):
            #
            # sampled_users = torch.tensor(np.random.choice(user_range, size=self.neighborhood_size-1))
            sampled_users = torch.arange(start=0, end=self.neighborhood_size).long()
            sampled_users = torch.cat([target_user_tensor, sampled_users]).long()
            top_users_combined = torch.unique(sampled_users, sorted=True).reshape(-1, ).to(self.device)
            new_index = (top_users_combined == target_user).nonzero(as_tuple=False).reshape(-1, ).to(self.device)

            # indices combined
            top_k_est_tensor.append(reward_ests[top_users_combined, a_i].reshape(-1, 1))
            top_k_CB_tensor.append(CB_ests[top_users_combined, a_i].reshape(-1, 1))
            user_neighborhood_list_est.append(top_users_combined[:, None].to(self.device))

            #
            target_user_new_indices_ests.append(new_index)
            target_user_new_indices_CBs.append(new_index)

        return top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
               user_neighborhood_list_est

    ############################################################################
    def get_top_users_most_frequent(self, reward_ests, CB_ests, target_user):
        top_k_est_tensor, top_k_CB_tensor = [], []
        target_user_new_indices_ests, target_user_new_indices_CBs = [], []
        user_neighborhood_list_est = []
        target_user_tensor = torch.ones(1, ) * target_user

        user_range = np.array([*range(target_user), *range(target_user + 1, self.user_n)])
        user_count = torch.tensor(np.array(self.user_select_count)[user_range])
        (_, top_user_est_i) = torch.topk(user_count, k=self.neighborhood_size-1, largest=True)
        sampled_users = top_user_est_i

        new_index = torch.zeros(1).long().to(self.device)
        sampled_users = torch.cat([target_user_tensor, sampled_users]).long()
        for a_i in range(self.arm_n):
            # indices combined
            top_users_combined = sampled_users

            #
            top_k_est_tensor.append(reward_ests[top_users_combined, a_i].reshape(-1, 1))
            top_k_CB_tensor.append(CB_ests[top_users_combined, a_i].reshape(-1, 1))
            user_neighborhood_list_est.append(top_users_combined[:, None].to(self.device))

            #
            target_user_new_indices_ests.append(new_index)
            target_user_new_indices_CBs.append(new_index)

        return top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
               user_neighborhood_list_est

    # ############################################################################
    def get_top_users(self, reward_ests, CB_ests, target_user=-1):
        top_k_est_tensor, top_k_CB_tensor = [], []
        target_user_new_indices_ests, target_user_new_indices_CBs = [], []
        user_neighborhood_list_est = []
        for a_i in range(self.arm_n):
            #
            other_user_ests, other_user_CBs = reward_ests[:, a_i], CB_ests[:, a_i]
            diff_ests, diff_CBs \
                = torch.abs(other_user_ests - reward_ests[target_user, a_i]).reshape(-1, ), \
                  torch.abs(other_user_CBs - CB_ests[target_user, a_i]).reshape(-1, )
            (_, top_user_est_i) = torch.topk(diff_ests, k=self.neighborhood_size, largest=False)
            (_, top_user_CB_i) = torch.topk(diff_CBs, k=self.neighborhood_size, largest=False)

            # indices combined
            top_users_combined = torch.cat([top_user_est_i, top_user_CB_i])
            top_users_combined = torch.unique(top_users_combined, sorted=True).reshape(-1, )

            #
            top_k_est_tensor.append(reward_ests[top_users_combined, a_i].reshape(-1, 1))
            top_k_CB_tensor.append(CB_ests[top_users_combined, a_i].reshape(-1, 1))
            user_neighborhood_list_est.append(top_users_combined[:, None].to(self.device))

            #
            new_index = (top_users_combined == target_user).nonzero(as_tuple=False).reshape(-1, )
            target_user_new_indices_ests.append(new_index)
            target_user_new_indices_CBs.append(new_index)

        return top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
               user_neighborhood_list_est

    def update_user_graphs(self, contexts, user_i, random_user_flag=False):
        reward_ests = []
        CB_ests = []
        gradients = []
        n_arms = contexts.shape[0]

        #
        top_k_est_tensor, top_k_CB_tensor = None, None
        if self.neighborhood_size > 0:
            #
            for u_i in range(self.user_n):
                res, grad = self.u_funcs_f_1[u_i].output_and_gradient(context=contexts)
                exp_scores = self.u_funcs_f_2[u_i].output(grad=grad)
                reward_ests.append(res.reshape(-1, ))
                CB_ests.append(exp_scores.reshape(-1, ))
                gradients.append(grad.reshape(n_arms, -1))
            #
            reward_ests = torch.stack(reward_ests, dim=0)
            CB_ests = torch.stack(CB_ests, dim=0)
            #
            top_k_est_tensor, top_k_CB_tensor, target_user_new_indices_ests, target_user_new_indices_CBs, \
                self.user_neighborhood_list_est = self.get_top_users_random(reward_ests, CB_ests, target_user=user_i)
            self.target_user_new_indices_ests, self.target_user_new_indices_CBs = \
                target_user_new_indices_ests, target_user_new_indices_CBs
            self.user_ests = reward_ests
            self.user_gradients = gradients
        else:
            #
            for u_i in range(self.user_n):
                res, grad = self.u_funcs_f_1[u_i].output_and_gradient(context=contexts)
                exp_scores = self.u_funcs_f_2[u_i].output(grad=grad)
                reward_ests.append(res.reshape(-1, ))
                CB_ests.append(exp_scores.reshape(-1, ))
                gradients.append(grad.reshape(n_arms, -1))
            #
            reward_ests = torch.stack(reward_ests, dim=0)
            CB_ests = torch.stack(CB_ests, dim=0)
            gradients = torch.stack(gradients, dim=0)
            self.user_ests = reward_ests
            self.user_gradients = gradients
            reward_ests = reward_ests.detach().cpu().numpy()
            CB_ests = CB_ests.detach().cpu().numpy()

        # Update two graphs
        for a_i in range(self.arm_n):
            if self.neighborhood_size > 0:
                this_reward_ests, this_CB_ests = \
                    top_k_est_tensor[a_i].detach().cpu().numpy(), top_k_CB_tensor[a_i].detach().cpu().numpy()
                #
                self.user_exploitation_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(this_reward_ests, this_reward_ests, self.bw_reward)).to(self.device)
                self.user_exploration_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(this_CB_ests, this_CB_ests, self.bw_conf_b)).to(self.device)
            else:
                self.user_exploitation_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(reward_ests[:,a_i].reshape(-1,1), reward_ests[:,a_i].reshape(-1,1), self.bw_reward)).to(self.device)
                self.user_exploration_graph_dict[a_i] = \
                    torch.tensor(Kernel.rbf_kernel(CB_ests[:,a_i].reshape(-1,1), CB_ests[:,a_i].reshape(-1,1), self.bw_conf_b)).to(self.device) # gaussian kernel

    def get_normalized_adj_m_list_for_user_graphs(self):
        exploitation_adj_matrix_dict = {}
        exploration_adj_matrix_dict = {}

        #
        if self.neighborhood_size > 0:
            for a_i in range(self.arm_n):
                exploitation_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploitation_graph_dict[a_i], k=self.k)
                exploitation_adj_matrix_dict[a_i] = exploitation_adj_matrix_normalized
                #
                exploration_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploration_graph_dict[a_i], k=self.k)
                exploration_adj_matrix_dict[a_i] = exploration_adj_matrix_normalized
        else:
            for a_i in range(self.arm_n):
                exploitation_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploitation_graph_dict[a_i], k=self.k)
                exploitation_adj_matrix_dict[a_i] = exploitation_adj_matrix_normalized
                #
                exploration_adj_matrix_normalized = \
                    utils.get_sym_norm_matrix_torch(adj=self.user_exploration_graph_dict[a_i], k=self.k)
                exploration_adj_matrix_dict[a_i] = exploration_adj_matrix_normalized

        return exploitation_adj_matrix_dict, exploration_adj_matrix_dict

    def train_user_models(self, u):
        # if self.train_every_user_model:
        exploit_loss, explore_loss = 0, 0
        if self.batch_size <= 0:
            for u_i in self.selected_user_period:
                exploit_loss = self.u_funcs_f_1[u_i].train()
                explore_loss = self.u_funcs_f_2[u_i].train()
        else:
            for u_i in self.selected_user_period:
                exploit_loss = self.u_funcs_f_1[u_i].batch_train()
                explore_loss = self.u_funcs_f_2[u_i].batch_train()
        self.selected_user_period = set()

        # ----------------------------
        # for neighbour user training ?

        return exploit_loss, explore_loss

    def train_GNN_models(self):
        exploit_loss = []
        explore_loss = []
        for arm in self.selected_arm:
            exploit_adj_tensor = self.exploitation_adj_matrix_dict[arm]
            explore_adj_tensor = self.exploration_adj_matrix_dict[arm]

            if self.batch_size <= 0:
                exploit_loss_per_arm = self.GNN_exploit_model.train_model(c_adj_m=exploit_adj_tensor)
                explore_loss_per_arm = self.GNN_explore_model.train_model(c_adj_m=explore_adj_tensor)
            else:
                exploit_loss_per_arm = self.GNN_exploit_model.train_model_batch(c_adj_m=exploit_adj_tensor)
                explore_loss_per_arm = self.GNN_explore_model.train_model_batch(c_adj_m=explore_adj_tensor)

            exploit_loss.append(exploit_loss_per_arm)
            explore_loss.append(explore_loss_per_arm)

        return exploit_loss, explore_loss

    def recommend(self, contexts, t, L, u=1):
        self.t = t
        g_list = []
        res_list = []
        overall_ests_dict = defaultdict()
        u_tensor = torch.tensor(np.array([u])).to(self.device)

        # Get adjacency matrices for user graphs
        self.exploitation_adj_matrix_dict, self.exploration_adj_matrix_dict = \
            self.get_normalized_adj_m_list_for_user_graphs()

        # Reward estimation ---------------------------------------------
        reduced_grad_array = []
        for a_i, c in enumerate(contexts):

            exploit_adj_m_tensor = self.exploitation_adj_matrix_dict[a_i]
            this_user_n = exploit_adj_m_tensor.shape[0]
            tensor = utils.generate_matrix_embedding_user(source=c, user_n=this_user_n).to(self.device)
            self.embedded_c_matrix[a_i] = tensor
            self.context_tensors[a_i] = torch.tensor(c).to(self.device)

            # f_1
            users_res, users_g \
                = self.GNN_exploit_model.get_reward_estimate_and_gradients(contexts=tensor, adj_m=exploit_adj_m_tensor,
                                                                           neighborhood_users=self.user_neighborhood_list_est[a_i])
            # reduce the dimension of gradient
            users_g = F.avg_pool1d(users_g.unsqueeze(0), kernel_size=self.GNN_pooling_step_size,
                                   stride=self.GNN_pooling_step_size, ceil_mode=True).squeeze(0)
            #
            if self.neighborhood_size > 0:
                user_i = self.target_user_new_indices_ests[a_i]
            else:
                user_i = u_tensor

            res_list.append(users_res)
            #
            reduced_grad_array.append(users_g)

        #
        for a_i in range(self.arm_n):

            explore_adj_m_tensor = self.exploration_adj_matrix_dict[a_i]

            # for exploration score
            gradients_tensor = reduced_grad_array[a_i]
            #
            if self.neighborhood_size > 0:
                user_i = self.target_user_new_indices_CBs[a_i]
            else:
                user_i = u_tensor

            explore_s = self.GNN_explore_model.get_exploration_scores(gradients=gradients_tensor,
                                                                      adj_m=explore_adj_m_tensor, user_i=user_i,
                                                                      user_neighborhood=self.user_neighborhood_list_est[a_i])

            # f_1 + f_2
            r_est = res_list[a_i]
            sample_r = torch.add(r_est,explore_s.unsqueeze(1),alpha=self.explore_param)
            overall_ests_dict[a_i] = sample_r.norm() # select arms according to the norm

        selected = dict(sorted(overall_ests_dict.items(), key=lambda x: x[1], reverse=True)[:L])
        self.selected_arm = list(selected.keys())
        # for i in selected.keys():
        #     self.exploit_adj_m_normalized = self.exploitation_adj_matrix_dict[i]
        #     self.explore_adj_m_normalized = self.exploration_adj_matrix_dict[i]
        #
        # # return selected_arm, g_list[selected_arm], point_est, g_list
        return self.selected_arm, res_list, reduced_grad_array
