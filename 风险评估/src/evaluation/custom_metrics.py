import numpy as np
from sklearn.metrics import roc_auc_score

def manual_auc_score(y_true, y_scores):
    """
    手写AUC计算实现
    
    参数:
    y_true: 真实标签
    y_scores: 预测概率
    
    返回:
    auc: AUC值
    """
    # 将数据转换为numpy数组
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 获取正负样本索引
    positive_indices = np.where(y_true == 1)[0]
    negative_indices = np.where(y_true == 0)[0]
    
    # 计算正负样本对数
    positive_count = len(positive_indices)
    negative_count = len(negative_indices)
    
    if positive_count == 0 or negative_count == 0:
        return 0.5  # 如果只有一类样本，返回0.5
    
    # 计算每个正样本的得分大于多少负样本
    correct_pairs = 0
    
    for pos_idx in positive_indices:
        pos_score = y_scores[pos_idx]
        # 计算有多少负样本的得分小于当前正样本
        correct_pairs += np.sum(y_scores[negative_indices] < pos_score)
        # 计算有多少负样本的得分等于当前正样本（算作0.5）
        correct_pairs += 0.5 * np.sum(y_scores[negative_indices] == pos_score)
    
    # 计算AUC
    auc = correct_pairs / (positive_count * negative_count)
    
    return auc

def compare_auc_implementations(y_true, y_pred_proba):
    """
    对比sklearn AUC和手写AUC的结果
    
    参数:
    y_true: 真实标签
    y_pred_proba: 预测概率
    
    返回:
    dict: 包含两种实现的结果和差异
    """
    # sklearn AUC
    sklearn_auc = roc_auc_score(y_true, y_pred_proba)
    
    # 手写AUC
    manual_auc = manual_auc_score(y_true, y_pred_proba)
    
    # 计算差异
    difference = abs(sklearn_auc - manual_auc)
    
    results = {
        'sklearn_auc': sklearn_auc,
        'manual_auc': manual_auc,
        'difference': difference,
        'relative_error': difference / sklearn_auc if sklearn_auc != 0 else 0
    }
    
    print("AUC实现对比:")
    print(f"sklearn AUC: {sklearn_auc:.6f}")
    print(f"手写AUC: {manual_auc:.6f}")
    print(f"绝对差异: {difference:.6f}")
    print(f"相对误差: {results['relative_error']:.6f}")
    
    return results

def manual_logistic_regression(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """
    手写逻辑回归实现
    
    参数:
    X: 特征矩阵 (n_samples, n_features)
    y: 目标变量 (n_samples,)
    learning_rate: 学习率
    max_iterations: 最大迭代次数
    tolerance: 收敛容差
    
    返回:
    weights: 权重向量
    bias: 偏置项
    history: 训练历史
    """
    # 添加偏置项
    n_samples, n_features = X.shape
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    # 初始化权重
    weights = np.zeros(n_features + 1)
    
    # 训练历史
    history = {
        'loss': [],
        'gradients': [],
        'weights': []
    }
    
    def sigmoid(z):
        """sigmoid函数"""
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(X, y, weights):
        """计算损失函数"""
        z = X.dot(weights)
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        
        # 逻辑回归的损失函数
        loss = -np.mean(y * np.log(sigmoid(z) + 1e-15) + (1 - y) * np.log(1 - sigmoid(z) + 1e-15))
        return loss
    
    def compute_gradient(X, y, weights):
        """计算梯度"""
        z = X.dot(weights)
        predictions = sigmoid(z)
        gradient = X.T.dot(predictions - y) / len(y)
        return gradient
    
    # 梯度下降
    for iteration in range(max_iterations):
        # 计算损失
        loss = compute_loss(X_with_bias, y, weights)
        
        # 计算梯度
        gradient = compute_gradient(X_with_bias, y, weights)
        
        # 更新权重
        new_weights = weights - learning_rate * gradient
        
        # 检查收敛
        weight_diff = np.linalg.norm(new_weights - weights)
        
        # 保存历史
        history['loss'].append(loss)
        history['gradients'].append(np.linalg.norm(gradient))
        history['weights'].append(weights.copy())
        
        weights = new_weights
        
        if iteration % 100 == 0:
            print(f"迭代 {iteration}: 损失 = {loss:.6f}, 权重变化 = {weight_diff:.6f}")
        
        if weight_diff < tolerance:
            print(f"在第 {iteration} 次迭代时收敛")
            break
    
    # 分离偏置项和权重
    bias = weights[0]
    feature_weights = weights[1:]
    
    return feature_weights, bias, history

def predict_manual_logistic_regression(X, weights, bias):
    """
    使用手写逻辑回归进行预测
    
    参数:
    X: 特征矩阵
    weights: 权重向量
    bias: 偏置项
    
    返回:
    probabilities: 预测概率
    predictions: 预测类别
    """
    z = X.dot(weights) + bias
    probabilities = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    predictions = (probabilities >= 0.5).astype(int)
    
    return probabilities, predictions