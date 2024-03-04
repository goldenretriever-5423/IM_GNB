from User_GNN_packages import *
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn.init as init
import math


# CUDA_VISIBLE_DEVICES=1
class Aggr_module(nn.Module):
    def __init__(self, A, input_dim, embed_dim):
        super(Aggr_module, self).__init__()

        # Param
        self.A = A
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        #
        self.fc_1 = nn.Linear(input_dim, embed_dim, bias=False)
        self.act = nn.ReLU()


    def weights_init(self, m):
        # Xavier initialization
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj, neighborhood_users):
        # -----------
        # First layer
        # (1, node_num, dim)
        aggr_c = torch.mm(adj, seq)

        # First Layer --- SGC
        out = self.fc_1(aggr_c)
        # Out Activation
        out = self.act(out)

        return out

    def batch_forward(self, seq, adj):
        # -----------
        # First layer
        # (1, node_num, dim)
        aggr_c = torch.bmm(adj, seq)

        # First Layer --- SGC
        out = self.fc_1(aggr_c)
        # Out Activation
        out = self.act(out)

        return out


# --------------------------------------------------------
class Aggr_module_MORE_USERS(nn.Module):
    def __init__(self, ori_dim, embed_dim, total_user_num, device):
        super(Aggr_module_MORE_USERS, self).__init__()

        # Param
        self.ori_dim = ori_dim
        self.embed_dim = embed_dim
        self.ori_dim_tensor = torch.arange(self.ori_dim).to(device)

        #
        self.weight = nn.Parameter(torch.empty([embed_dim, total_user_num * ori_dim]), requires_grad=True)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.xavier_uniform_(self.weight)
        self.act = nn.ReLU()

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj, neighborhood_users):
        weight_i = (self.ori_dim * neighborhood_users + self.ori_dim_tensor).reshape(-1, )

        # -----------
        # First layer
        # (1, node_num, dim)
        aggr_c = torch.mm(adj, seq)

        # First Layer --- SGC
        # out = self.fc_1(aggr_c)
        weight_portion = self.weight[:, weight_i]
        out = F.linear(aggr_c, weight_portion)

        # Out Activation
        out = self.act(out)

        return out


# ------------------------------------
class Aggr_module_GCN(nn.Module):
    def __init__(self, A, input_dim, embed_dim, num_layers=2):
        super(Aggr_module_GCN, self).__init__()

        print("Init GCN module...")

        # Param
        self.A = A
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        print("!!Num layers: ", num_layers)

        #
        if num_layers == 2:
            self.fc_1 = nn.Linear(input_dim, embed_dim, bias=False)
            self.act = nn.ReLU()
            self.fc_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        elif num_layers == 3:
            self.fc_1 = nn.Linear(input_dim, embed_dim, bias=False)
            self.act = nn.ReLU()
            self.fc_2 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.fc_3 = nn.Linear(embed_dim, embed_dim, bias=False)

    def weights_init(self, m):
        # Xavier initialization
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj):
        # -----------
        # First layer
        # (1, node_num, dim)
        aggr_c = torch.mm(adj, seq)

        # First Layer --- SGC
        out = self.fc_1(aggr_c)
        # Out Activation
        out = self.act(out)

        # Second aggr
        out = torch.mm(adj, out)
        # Second linear transformation
        out = self.fc_2(out)
        out = self.act(out)

        if self.num_layers > 2:
            # Third act
            out = torch.mm(adj, out)
            out = self.fc_3(out)
            out = self.act(out)

        return out

    # Shape of seq: (nodes, features)
    def batch_forward(self, seq, adj):
        # -----------
        # First layer
        # (1, node_num, dim)
        aggr_c = torch.bmm(adj, seq)

        # First Layer --- SGC
        out = self.fc_1(aggr_c)
        # Out Activation
        out = self.act(out)

        # Second aggr
        out = torch.bmm(adj, out)
        # Second linear transformation
        out = self.fc_2(out)

        if self.num_layers > 2:
            out = self.act(out)
            out = torch.bmm(adj, out)
            out = self.fc_3(out)

        return out


# =====================================================================================================================
# =====================================================================================================================
# Neural-UCB module

class Est_module(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super(Est_module, self).__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

        # Initialization
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.constant_(self.fc1.bias, 0.0)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        # First layer
        out = self.act(self.fc1(x))
        # Second layer
        out = self.fc2(out)

        return out


# ====================================================================================================================

class GNN_Exploitation_Net(nn.Module):
    def __init__(self, user_n, input_dim, reduced_output_dim, batch_size=-1, hidden_size=100, num_layer=-1,
                 neighborhood_size=-1, device=None, last_layer_gradient_flag=False):
        super(GNN_Exploitation_Net, self).__init__()
        self.fc_hidden_size = hidden_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.last_layer_gradient_flag = last_layer_gradient_flag

        # Aggregation module
        if neighborhood_size > 0:
            # embed_dim, total_user_num, device = None
            self.aggr = Aggr_module_MORE_USERS(input_dim, hidden_size, user_n, device)
        else:
            if num_layer < 2:
                self.aggr = Aggr_module(user_n, input_dim * user_n, hidden_size)
            else:
                self.aggr = Aggr_module_GCN(user_n, input_dim * user_n, hidden_size, num_layers=num_layer)

        # Estimation module  estimate the influence probability?
        self.est_module = Est_module(embed_dim=hidden_size, hidden_size=hidden_size)

        # Gradient embedding
        self.embedding = LocallyLinearEmbedding(n_components=reduced_output_dim)

    def change_grad_last_layer(self, predicting=False):
        if predicting:
            self.aggr.requires_grad_(False)
            self.est_module.fc1.requires_grad_(True)
            self.est_module.fc2.requires_grad_(True)
        else:
            self.aggr.requires_grad_(True)
            self.est_module.requires_grad_(True)

    def forward(self, extended_seq, adj, user_i, neighborhood_users=None):
        # overall_seq -> (node_num, embed_dim)
        h_1 = self.aggr(extended_seq, adj, neighborhood_users)

        # Embedded contexts for the labeled user
        embed_c = h_1.index_select(0, user_i)

        # Point estimations
        point_ests = self.est_module(embed_c)

        # Results and gradients for all users given ONE arm
        return point_ests

    def forward_batch(self, extended_seq, adj, user_i):
        # overall_seq -> (node_num, embed_dim)
        h_1 = self.aggr.batch_forward(extended_seq, adj)

        # Embedded contexts for the labeled user --- Batch selection
        embed_c = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(h_1, user_i)])
        embed_c = embed_c.squeeze(1)

        # Point estimations
        point_ests = self.est_module(embed_c)

        # Results and gradients for all users given ONE arm
        return point_ests

    def predict(self, extended_seq, adj, neighborhood_users=None):

        # Get the current parameters of the f_1 model, for calculating the gradients
        f_1_weights = OrderedDict(self.named_parameters())
        g_list = []
        res_list = []

        h_1 = self.aggr(extended_seq, adj, neighborhood_users)
        # Point estimations
        point_ests = self.est_module(h_1)

        # Calculate gradients for support set
        for fx in point_ests:

            # Calculate the Gradients with autograd.grad()
            this_g_list = []
            grad_tuple = torch.autograd.grad(fx, f_1_weights.values(), create_graph=True)
            for grad in grad_tuple:
                this_g_list.append(grad.detach().reshape(-1, ))
            g = torch.cat(this_g_list)
            g_list.append(g)
            # res_list.append(fx)

            #
            del grad_tuple

        g_list = torch.stack(g_list, dim=0)

        # Results and gradients for all users given ONE arm
        return point_ests, g_list


# ------------
class Exploitation_GNN:
    def __init__(self, user_n, input_dim, reduced_output_dim, hidden_size=100, lr_rate=0.001, batch_size=-1,
                 pool_step_size=-1, num_layer=-1, neighborhood_size=-1,
                 last_layer_gradient_flag=False, device=None):
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.device = device
        self.pool_step_size = pool_step_size
        self.last_layer_gradient_flag = last_layer_gradient_flag

        self.selected_contexts = []
        self.rewards = []
        self.served_users = []
        self.exploit_adj_m = []
        self.selected_neghborhood_list = []

        self.exploitation_model = \
            GNN_Exploitation_Net(user_n, input_dim, reduced_output_dim, batch_size=batch_size,
                                 hidden_size=hidden_size, num_layer=num_layer,
                                 neighborhood_size=neighborhood_size,
                                 device=device,
                                 last_layer_gradient_flag=last_layer_gradient_flag).to(device)

    def get_reward_estimate_and_gradients(self, contexts, adj_m, neighborhood_users):
        res_list, g_list = self.exploitation_model.predict(contexts, adj_m, neighborhood_users)
        return res_list, g_list

    def update_info(self, context, reward, user_i, adj_m, selected_neighborhood=None):
        self.selected_contexts.append(context)
        self.rewards.append(reward)
        self.served_users.append(user_i)
        self.exploit_adj_m.append(adj_m)
        self.selected_neghborhood_list.append(selected_neighborhood)

    def train_model(self, c_adj_m):
        time_length = len(self.served_users)

        optimizer = optim.Adam(self.exploitation_model.parameters(), lr=self.lr_rate)
        index = np.arange(time_length)
        np.random.shuffle(index)

        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.selected_contexts[idx].to(self.device)
                r = self.rewards[idx].to(self.device)
                u = self.served_users[idx].to(self.device)
                s_neighborhood = self.selected_neghborhood_list[idx]
                adj_m = self.exploit_adj_m[idx].to(self.device)
                optimizer.zero_grad()
                loss = (
                               self.exploitation_model(c, adj_m, u, s_neighborhood) - r
                               # self.exploitation_model(c, c_adj_m, u) - r
                        ) ** 2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / time_length <= 1e-3:  # control the loss
                return batch_loss / time_length

    def train_model_batch(self, c_adj_m):
        time_length = len(self.served_users)

        optimizer = optim.Adam(self.exploitation_model.parameters(), lr=self.lr_rate)
        mse_loss = nn.MSELoss(reduction='mean')
        index = np.arange(time_length)
        np.random.shuffle(index)

        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0

            replace_flag = False if time_length >= self.batch_size else True
            indices = np.random.choice(index, self.batch_size, replace=replace_flag)

            # --------------
            c = torch.stack([self.selected_contexts[int(idx)] for idx in indices]).to(self.device)
            r = torch.stack([self.rewards[int(idx)] for idx in indices]).reshape([self.batch_size, 1]).float() \
                .to(self.device)
            u = torch.stack([self.served_users[int(idx)] for idx in indices]).to(self.device)
            adj_m = torch.stack([self.exploit_adj_m[int(idx)] for idx in indices]).to(self.device)

            optimizer.zero_grad()

            logits = self.exploitation_model.forward_batch(c, adj_m, u)
            loss = mse_loss(logits, r)

            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            tot_loss += loss.item()
            cnt += 1

            # if cnt >= (1000 // self.batch_size):
            if cnt >= 1000:
                return tot_loss / cnt
            if batch_loss / time_length <= 1e-3:
                return batch_loss / time_length


# =====================================================================================================================
class GNN_Exploration_Net(nn.Module):
    # NN to estimate the reward for calculating the weights in graphs
    def __init__(self, user_n, input_dim, hidden_size=100, batch_size=-1, num_layer=-1, neighborhood_size=-1,
                 device=None, separate_explore_GNN=False):
        super(GNN_Exploration_Net, self).__init__()
        self.fc_hidden_size = hidden_size
        self.input_dim = input_dim
        self.batch_size = batch_size

        # Aggregation module
        if separate_explore_GNN:
            if num_layer < 2:
                self.aggr = Aggr_module(user_n, input_dim * user_n, hidden_size)
            else:
                self.aggr = Aggr_module_GCN(user_n, input_dim * user_n, hidden_size, num_layers=num_layer)
        else:
            if num_layer < 2:
                self.aggr = Aggr_module(user_n, input_dim, hidden_size)
            else:
                self.aggr = Aggr_module_GCN(user_n, input_dim, hidden_size, num_layers=num_layer)

        # Estimation module
        self.est_module = Est_module(embed_dim=hidden_size, hidden_size=hidden_size)

    def forward(self, extended_seq, adj, user_i, neighborhood_users=None):
        # overall_seq -> (node_num, embed_dim)
        h_1 = self.aggr(extended_seq, adj, neighborhood_users)

        # Embedded contexts for the labeled user
        # embed_c = h_1.index_select(0, user_i) # TODO remove index selection, instead,scale it to all users

        # Point estimations
        point_ests = self.est_module(h_1) # TODO point_est dim =  #users x 1

        # (#user, 1)
        return point_ests

    def forward_batch(self, extended_seq, adj, user_i):
        # overall_seq -> (node_num, embed_dim)
        h_1 = self.aggr.batch_forward(extended_seq, adj)

        # Embedded contexts for the labeled user --- Batch selection
        embed_c = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(h_1, user_i)])
        embed_c = embed_c.squeeze(1)

        # Point estimations
        point_ests = self.est_module(embed_c)

        # Results and gradients for all users given ONE arm
        return point_ests


# -------------------------------

class Exploration_GNN:
    def __init__(self, user_n, input_dim, hidden_size=100, lr_rate=0.001, batch_size=-1, num_layer=-1,
                 neighborhood_size=-1,
                 separate_explore_GNN=False, device=None):
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.device = device

        self.selected_gradients = []
        self.rewards = []
        self.served_users = []
        self.explore_adj_m = []
        self.user_neighborhood_list = []

        self.exploration_model = GNN_Exploration_Net(user_n, input_dim, hidden_size=hidden_size,
                                                     batch_size=batch_size, num_layer=num_layer,
                                                     device=device,
                                                     separate_explore_GNN=separate_explore_GNN).to(self.device)

    def get_exploration_scores(self, gradients, adj_m, user_i, user_neighborhood=None):
        # user_i_tensor = torch.tensor(np.array([user_i])).to(self.device)
        exploration_score = self.exploration_model(gradients, adj_m, user_i, user_neighborhood).reshape(-1, )
        # exploration_score = exploration_score.cpu().detach().numpy()
        return exploration_score

    def update_info(self, gradients, residual_reward, user_i, adj_m):
        self.selected_gradients.append(gradients)
        self.rewards.append(residual_reward)
        self.served_users.append(user_i)
        self.explore_adj_m.append(adj_m)

    def train_model(self, c_adj_m):
        time_length = len(self.served_users)

        optimizer = optim.Adam(self.exploration_model.parameters(), lr=self.lr_rate)
        index = np.arange(time_length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.selected_gradients[idx].to(self.device)
                r = self.rewards[idx].item()
                u = self.served_users[idx].to(self.device)
                adj_m = self.explore_adj_m[idx].to(self.device)
                optimizer.zero_grad()

                # The 'r' here is the residual reward
                loss = (
                               self.exploration_model(c, adj_m, u) - r
                               # self.exploration_model(c, c_adj_m, u) - r
                        ) ** 2
                loss[u].backward()
                optimizer.step()
                batch_loss += loss[u].item()
                tot_loss += loss[u].item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / time_length <= 1e-3:
                return batch_loss / time_length

    def train_model_batch(self, c_adj_m):
        time_length = len(self.served_users)

        optimizer = optim.Adam(self.exploration_model.parameters(), lr=self.lr_rate)
        mse_loss = nn.MSELoss(reduction='mean')
        index = np.arange(time_length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0

            replace_flag = False if time_length >= self.batch_size else True
            indices = np.random.choice(index, self.batch_size, replace=replace_flag)

            # ------------------------------------------
            c = torch.stack([self.selected_gradients[int(idx)] for idx in indices]).to(self.device)
            r = torch.stack([self.rewards[int(idx)] for idx in indices]).reshape([self.batch_size, 1]).float()\
                .to(self.device)
            u = torch.stack([self.served_users[int(idx)] for idx in indices]).to(self.device)
            adj_m = torch.stack([self.explore_adj_m[int(idx)] for idx in indices]).to(self.device)

            optimizer.zero_grad()

            logits = self.exploration_model.forward_batch(c, adj_m, u)
            loss = mse_loss(logits, r)

            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            tot_loss += loss.item()
            cnt += 1
            # if cnt >= (1000 // self.batch_size):
            if cnt >= 1000:
                return tot_loss / cnt

            if batch_loss / time_length <= 1e-3:
                return batch_loss / time_length