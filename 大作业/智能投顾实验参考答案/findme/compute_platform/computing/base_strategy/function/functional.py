from collections import OrderedDict

import torch
import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

def unzip_portfolio(mask_s, arm_keys, weight):
    i = 0
    j = 0
    portfolio = OrderedDict()
    for stock in arm_keys:
        if mask_s[j] == 1:
            if len(weight) == 1:
                portfolio[str(stock)] = weight
            else:
                portfolio[str(stock)] = weight[i]
                i += 1
        else:
            portfolio[str(stock)] = torch.tensor(0.)
        j += 1
    return portfolio

def gen_mask_via_select_arms(selected_arms, d):
    """
    根据select_arms生成d维的掩码
    :param selected_arms: 应该为1的arm集合
    :param d: 目标维度
    :return: s_mask向量(d)
    """
    mask_s = torch.zeros(d)
    for i in range(d, -1, -1):
        if i in selected_arms:
            mask_s[i] = 1
    return mask_s


def projection_in_norm(b, M, device="cpu"):
    """ Projection of x to simplex indiced by matrix M.
         Uses quadratic programming.
    """
    b = b.cpu().numpy()
    M = M.cpu().numpy()
    m = M.shape[0]

    P = matrix((2 * M).tolist())
    q = matrix((-2 * b).tolist())
    G = matrix(-np.eye(m))
    h = matrix(np.zeros((m, 1)))
    A = matrix(np.ones((1, m)))
    b = matrix(1.)

    sol = solvers.qp(P, q, G, h, A, b)
    return torch.tensor(np.squeeze(sol['x']), device=device)