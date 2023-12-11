from User_GNN_packages import *
from sklearn.utils.validation import check_symmetric
from scipy.linalg import block_diag


# ===========================================================
def get_sym_norm_matrix(source, k):
    _ = check_symmetric(source, raise_exception=True)
    input_matrix = np.array(source, dtype=np.float)
    D_matrix = np.diag(np.sum(input_matrix, axis=1))
    D_matrix_minus_1_2 = np.zeros(D_matrix.shape)
    np.fill_diagonal(D_matrix_minus_1_2, 1 / (D_matrix.diagonal() ** 0.5))
    #
    S_y = np.matmul(np.matmul(D_matrix_minus_1_2, input_matrix), D_matrix_minus_1_2)

    #
    if k > 1:
        S_y = np.linalg.matrix_power(S_y, k)

    return S_y


# ===========================================================
def get_sym_norm_matrix_torch(adj, k):
    # print(adj.size())
    # if len(adj.size()) == 4:
    #     new_r = torch.zeros(adj.size()).type_as(adj)
    #     for i in range(adj.size(1)):
    #         adj_item = adj[0, i]
    #         rowsum = adj_item.sum(1)
    #         d_inv_sqrt = rowsum.pow_(-0.5)
    #         d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    #         d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    #         r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_item), d_mat_inv_sqrt)
    #         new_r[0, i, ...] = r
    #     return new_r
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow_(-0.5)
    # d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    S_y = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    #
    if k > 1:
        S_y = torch.linalg.matrix_power(S_y, k)

    return S_y


# Generate long vector for contexts
def generate_matrix_embedding_user(source, user_n):
    # (1, d) -> (A, A * d)
    # this_context = source.reshape(1, -1)
    # diag_matrix = block_diag(*np.repeat(this_context, user_n, axis=0))

    this_context = torch.tensor(source).reshape(1, -1)
    diag_matrix = torch.block_diag(*torch.repeat_interleave(this_context, user_n, dim=0)).float()

    return diag_matrix


# Generate long vector for gradients
def generate_matrix_embedding_gradients(source):
    # (A, d) -> (A, A * d)
    diag_matrix = torch.block_diag(*source)

    return diag_matrix


def getuser_f_1_param_count(dim, user_n, arm_n, user_reduced_grad_dim, hidden, user_lr, batch_size, pooling_step_size,
                            device):
    test_FC = Exploitation_FC(dim, user_n, arm_n=arm_n, reduced_dim=user_reduced_grad_dim,
                              hidden=hidden, lr=user_lr, batch_size=batch_size,
                              pool_step_size=pooling_step_size, device=device)
    user_total_param_count = \
        sum(param.numel() for param in test_FC.func.parameters())

    return user_total_param_count


if __name__ == '__main__':
    import scipy
    a = scipy.spatial.distance.squareform(np.arange(1, 11))
    b = torch.tensor(a).float()
    print(get_sym_norm_matrix(a, 1))
    print(get_sym_norm_matrix_torch(b, 1))




