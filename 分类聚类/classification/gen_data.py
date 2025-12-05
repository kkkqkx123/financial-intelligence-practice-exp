import numpy as np
from numpy.linalg import svd

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def gen_data(N, noisy=None):
    '''
    生成数据集。
    输入:  N:     样本数量。
            noisy: 是否对y添加噪声。
    
    输出: X: 样本特征, P-by-N 矩阵。
          y: 样本标签, 1-by-N 行向量。
          w: 目标函数参数, (P+1)-by-1 列向量。
    '''
    data_range = np.array([-1, 1])
    dim = 2

    X = np.random.random((dim, N)) * (data_range[1] - data_range[0]) + data_range[0]
    while True:
        Xsample = np.vstack((np.ones((1, dim)), np.random.random((dim, dim)) * (data_range[1]-data_range[0]) + data_range[0]))
        w = nullspace(Xsample.T)
        a =  np.vstack((np.ones((1, N)), X))
        y = np.sign(np.matmul(w.T, np.vstack((np.ones((1, N)), X))))
        if np.all(y) and np.unique(y).shape[0] > 1:
            break
    if noisy:
        idx = np.random.choice(N, N//10)
        y[0, idx] = -y[0, idx]

    return X, y, w

