import numpy as np
from scipy.optimize import minimize

def func(X, y):
    '''
    分类算法 - 支持向量机 (SVM)。

    输入:  X: 训练样本特征, P-by-N
            y: 训练样本标签, 1-by-N

    输出: w: 学习到的参数, (P+1)-by-1
    '''
    P, N = X.shape
    
    # 向X添加偏置项
    X_bias = np.vstack((np.ones((1, N)), X))
    
    # 转换y以确保标签为-1和1
    y = y.flatten()
    
    # 定义SVM的目标函数
    def objective(w):
        w = w.reshape(-1, 1)
        # 合页损失 + L2正则化
        margins = y * (w.T @ X_bias).flatten()
        hinge_loss = np.maximum(0, 1 - margins)
        return 0.5 * np.sum(w[1:]**2) + np.sum(hinge_loss)
    
    # 初始猜测
    w_init = np.zeros(P + 1)
    
    # 优化
    result = minimize(objective, w_init, method='BFGS')
    
    return result.x.reshape(-1, 1)

def logistic_regression(X, y):
    '''
    逻辑回归分类器。

    输入:  X: 训练样本特征, P-by-N
            y: 训练样本标签, 1-by-N

    输出: w: 学习到的参数, (P+1)-by-1
    '''
    P, N = X.shape
    
    # 向X添加偏置项
    X_bias = np.vstack((np.ones((1, N)), X))
    
    # 转换y为0和1用于逻辑回归
    y_binary = (y.flatten() + 1) // 2
    
    # 定义逻辑回归的目标函数
    def objective(w):
        w = w.reshape(-1, 1)
        z = w.T @ X_bias
        # 防止溢出
        z = np.clip(z, -500, 500)
        predictions = 1 / (1 + np.exp(-z.flatten()))
        # 添加小的epsilon以防止log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y_binary * np.log(predictions) + (1 - y_binary) * np.log(1 - predictions))
        return loss
    
    # 初始猜测
    w_init = np.zeros(P + 1)
    
    # 优化
    result = minimize(objective, w_init, method='BFGS')
    
    return result.x.reshape(-1, 1)

def perceptron(X, y):
    '''
    感知器分类器。

    输入:  X: 训练样本特征, P-by-N
            y: 训练样本标签, 1-by-N

    输出: w: 学习到的参数, (P+1)-by-1
    '''
    P, N = X.shape
    
    # 向X添加偏置项
    X_bias = np.vstack((np.ones((1, N)), X))
    
    # 初始化权重
    w = np.zeros((P + 1, 1))
    y = y.flatten()
    
    # 感知器学习算法
    max_iterations = 1000
    learning_rate = 0.1
    
    for iteration in range(max_iterations):
        converged = True
        for i in range(N):
            prediction = np.sign(w.T @ X_bias[:, i])
            if prediction == 0:
                prediction = 1
            
            if prediction != y[i]:
                w += learning_rate * y[i] * X_bias[:, i:i+1]
                converged = False
        
        if converged:
            break
    
    return w