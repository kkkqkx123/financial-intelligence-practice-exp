import numpy as np
from scipy.spatial.distance import cdist

def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    Output:  idx: cluster label, N-by-1 vector
    '''

    N, P = X.shape
    idx = np.zeros(N)
    # YOUR CODE HERE
    # ----------------
    # 随机初始化聚类中心
    np.random.seed(42)
    centers = X[np.random.choice(N, k, replace=False)]
    
    max_iters = 100
    for iter in range(max_iters):
        # 分配每个点到最近的聚类中心
        for i in range(N):
            distances = np.linalg.norm(X[i] - centers, axis=1)
            idx[i] = np.argmin(distances)
        
        # 更新聚类中心
        new_centers = np.zeros((k, P))
        for j in range(k):
            points_in_cluster = X[idx == j]
            if len(points_in_cluster) > 0:
                new_centers[j] = np.mean(points_in_cluster, axis=0)
            else:
                new_centers[j] = centers[j]  # 保持原中心不变
        
        # 检查收敛
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    # ----------------
    return idx

def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]
    idx = np.zeros((N, 1))
    # YOUR CODE HERE
    # ----------------
    # 计算度矩阵D
    row_sums = np.sum(W, axis=1)
    
    # 处理度为0的节点，避免除以零
    row_sums = np.maximum(row_sums, 1e-10)  # 确保最小值为一个很小的正数
    
    D = np.diag(row_sums)
    
    # 计算拉普拉斯矩阵L
    L = D - W
    
    # 计算归一化拉普拉斯矩阵L_sym
    D_sqrt_inv = np.diag(1.0 / np.sqrt(row_sums))
    L_sym = D_sqrt_inv @ L @ D_sqrt_inv
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    
    # 选择前k个最小特征值对应的特征向量（跳过第一个为0的特征值）
    # 对于连通图，第一个特征值应该接近0，对应的特征向量是常数向量
    if k == 2:
        # 对于2类聚类，使用前2个非零特征值对应的特征向量
        X = eigenvectors[:, 1:k+1]  # 跳过第一个特征向量
    else:
        X = eigenvectors[:, :k]
    
    # 对特征向量进行归一化，处理零向量的情况
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # 避免除以零
    X_normalized = X / norms
    # ----------------
    X_normalized = X_normalized.astype(float)  # keep real part, discard imaginary part
    idx = kmeans(X_normalized, k)
    return idx

def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]
        W[i, index] = 1
        W[i, i] = 0  # aj[i,i] = 0
    W[aj >= threshold] = 0
    return W
