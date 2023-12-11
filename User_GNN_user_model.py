from User_GNN_packages import *
from collections import OrderedDict


class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


class Exploitation:
    def __init__(self, dim, user_n, arm_n, reduced_dim, hidden=100, lr=0.001, batch_size=-1,
                 pool_step_size=-1, device=None):
        '''dim: number of dimensions of input'''
        '''n_arm: number of arms'''
        '''lr: learning rate'''
        '''hidden: number of hidden nodes'''

        self.func = Network_exploitation(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []

        '''Embed gradient for exploration'''
        self.embedding = LocallyLinearEmbedding(n_components=reduced_dim)

        self.lr = lr
        self.batch_size = batch_size
        self.pool_step_size = pool_step_size
        self.device = device

    def update(self, context, reward):
        # self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.context_list.append(context.reshape(-1, ).float())
        self.reward.append(reward)

    def output_and_gradient(self, context):

        # Get the current parameters of the f_1 model, for calculating the gradients
        f_1_weights = OrderedDict(self.func.named_parameters())
        g_list = []
        res_list = []

        tensor = torch.from_numpy(context).float().to(self.device)
        point_ests = self.func(tensor) # dim 10*1

        # Calculate gradients for support set
        for fx in point_ests:

            # Calculate the Gradients with autograd.grad()
            this_g_list = []
            grad_tuple = torch.autograd.grad(fx, f_1_weights.values(), create_graph=True)
            for grad in grad_tuple:
                this_g_list.append(grad.detach().reshape(-1, ))
            g = torch.cat(this_g_list)
            g_list.append(g)
            res_list.append(fx)

            #
            del grad_tuple

        g_list = torch.stack(g_list, dim=0)
        res_list = torch.stack(res_list, dim=0)

        # Gradient embeddings -------------------------
        if self.pool_step_size <= 0:
            g_list = self.embedding.fit_transform(g_list)
        else:
            g_list = F.avg_pool1d(g_list.unsqueeze(dim=0), kernel_size=self.pool_step_size, stride=self.pool_step_size)\
                .squeeze(0)
            # g_list = block_reduce(g_list, block_size=(1, self.pool_step_size), func=np.mean)

        return res_list, g_list

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx].to(self.device)
                optimizer.zero_grad()
                loss = (self.func(c.to(self.device)) - r) ** 2
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

    def batch_train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss(reduction='mean')
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            replace_flag = False if length >= self.batch_size else True
            indices = np.random.choice(index, self.batch_size, replace=replace_flag)

            c = torch.stack([self.context_list[int(idx)] for idx in indices]).to(self.device)
            r = torch.stack([self.reward[int(idx)] for idx in indices]).reshape([self.batch_size, 1]).float() \
                .to(self.device)

            optimizer.zero_grad()

            logits = self.func(c.to(self.device))
            loss = mse_loss(logits, r)

            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            tot_loss += loss.item()
            cnt += 1
            # if cnt >= (1000 // self.batch_size):
            if cnt >= 1000:
                return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length


class Exploration:
    def __init__(self, dim, hidden=100, lr=0.01, batch_size=-1, device=None):
        self.func = Network_exploration(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def update(self, context, reward):
        # self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.context_list.append(context.reshape(-1, ).float())
        self.reward.append(reward)

    def output(self, grad):
        # tensor = torch.from_numpy(context).float().to(self.device)
        tensor = grad
        ress = self.func(tensor)
        # res = ress.detach().numpy()
        return ress

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=self.lr)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx].to(self.device)
                # output = self.func(c.to(device))
                optimizer.zero_grad()
                delta = self.func(c.to(self.device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length

    def batch_train(self):
        optimizer = optim.Adam(self.func.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss(reduction='mean')
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            #
            replace_flag = False if length >= self.batch_size else True
            indices = np.random.choice(index, self.batch_size, replace=replace_flag)

            c = torch.stack([self.context_list[int(idx)] for idx in indices]).to(self.device)
            r = torch.stack([self.reward[int(idx)] for idx in indices]).reshape([self.batch_size, 1]).float() \
                .to(self.device)

            optimizer.zero_grad()

            logits = self.func(c.to(self.device))
            loss = mse_loss(logits, r)

            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            tot_loss += loss.item()
            cnt += 1
            # if cnt >= (1000 // self.batch_size):
            if cnt >= (1000 // self.batch_size):
                return tot_loss / cnt
            if batch_loss / length <= 1e-3:
                return batch_loss / length
