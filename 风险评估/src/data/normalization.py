import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def compare_normalization_methods(X_train, X_test, y_train, y_test, model_func):
    """
    对比不同归一化方法对模型性能的影响
    
    参数:
    X_train: 训练特征
    X_test: 测试特征
    y_train: 训练标签
    y_test: 测试标签
    model_func: 模型训练函数，返回(模型, AUC分数)
    
    返回:
    results: 不同归一化方法的比较结果
    """
    results = {}
    
    # 1. 无归一化
    print("=== 无归一化 ===")
    model_no_norm, auc_no_norm = model_func(X_train, X_test, y_train, y_test)
    results['no_normalization'] = {
        'auc': auc_no_norm,
        'model': model_no_norm,
        'X_train': X_train,
        'X_test': X_test
    }
    
    # 2. 标准化 (Z-score标准化)
    print("\n=== Z-score标准化 ===")
    scaler_standard = StandardScaler()
    X_train_standard = scaler_standard.fit_transform(X_train)
    X_test_standard = scaler_standard.transform(X_test)
    
    model_standard, auc_standard = model_func(X_train_standard, X_test_standard, y_train, y_test)
    results['standardization'] = {
        'auc': auc_standard,
        'model': model_standard,
        'X_train': X_train_standard,
        'X_test': X_test_standard,
        'scaler': scaler_standard
    }
    
    # 3. 最小-最大归一化
    print("\n=== 最小-最大归一化 ===")
    scaler_minmax = MinMaxScaler()
    X_train_minmax = scaler_minmax.fit_transform(X_train)
    X_test_minmax = scaler_minmax.transform(X_test)
    
    model_minmax, auc_minmax = model_func(X_train_minmax, X_test_minmax, y_train, y_test)
    results['minmax_normalization'] = {
        'auc': auc_minmax,
        'model': model_minmax,
        'X_train': X_train_minmax,
        'X_test': X_test_minmax,
        'scaler': scaler_minmax
    }
    
    # 打印比较结果
    print("\n" + "="*50)
    print("归一化方法比较结果:")
    print("="*50)
    print(f"无归一化:     AUC = {auc_no_norm:.6f}")
    print(f"Z-score标准化: AUC = {auc_standard:.6f}")
    print(f"最小-最大归一化: AUC = {auc_minmax:.6f}")
    print("="*50)
    
    # 找出最佳方法
    auc_values = {
        'no_normalization': auc_no_norm,
        'standardization': auc_standard,
        'minmax_normalization': auc_minmax
    }
    
    best_method = max(auc_values, key=auc_values.get)
    print(f"最佳归一化方法: {best_method} (AUC = {auc_values[best_method]:.6f})")
    
    return results, best_method

def visualize_normalization_effect(X_train, y_train, feature_names=None):
    """
    可视化归一化前后的数据分布
    
    参数:
    X_train: 原始训练数据
    y_train: 训练标签
    feature_names: 特征名称列表
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # 创建归一化器
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    # 应用归一化
    X_standard = scaler_standard.fit_transform(X_train)
    X_minmax = scaler_minmax.fit_transform(X_train)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('归一化方法对比', fontsize=16)
    
    # 原始数据分布
    axes[0, 0].boxplot(X_train)
    axes[0, 0].set_title('原始数据')
    axes[0, 0].set_xlabel('特征')
    axes[0, 0].set_ylabel('值')
    axes[0, 0].set_xticklabels(feature_names, rotation=45)
    
    # Z-score标准化
    axes[0, 1].boxplot(X_standard)
    axes[0, 1].set_title('Z-score标准化')
    axes[0, 1].set_xlabel('特征')
    axes[0, 1].set_ylabel('值')
    axes[0, 1].set_xticklabels(feature_names, rotation=45)
    
    # 最小-最大归一化
    axes[1, 0].boxplot(X_minmax)
    axes[1, 0].set_title('最小-最大归一化')
    axes[1, 0].set_xlabel('特征')
    axes[1, 0].set_ylabel('值')
    axes[1, 0].set_xticklabels(feature_names, rotation=45)
    
    # 统计信息对比
    methods = ['原始数据', 'Z-score', 'Min-Max']
    data_list = [X_train, X_standard, X_minmax]
    
    means = [np.mean(data) for data in data_list]
    stds = [np.std(data) for data in data_list]
    
    x_pos = np.arange(len(methods))
    
    axes[1, 1].bar(x_pos - 0.2, means, 0.4, label='均值', alpha=0.7)
    axes[1, 1].bar(x_pos + 0.2, stds, 0.4, label='标准差', alpha=0.7)
    axes[1, 1].set_title('统计信息对比')
    axes[1, 1].set_xlabel('归一化方法')
    axes[1, 1].set_ylabel('值')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("归一化统计信息:")
    print("-" * 40)
    for i, method in enumerate(methods):
        print(f"{method}:")
        print(f"  均值: {means[i]:.4f}")
        print(f"  标准差: {stds[i]:.4f}")
        print(f"  最小值: {np.min(data_list[i]):.4f}")
        print(f"  最大值: {np.max(data_list[i]):.4f}")
        print()

def analyze_feature_importance_with_normalization(X_train, y_train, X_test, y_test, feature_names):
    """
    分析不同归一化方法对特征重要性的影响
    
    参数:
    X_train: 训练特征
    y_train: 训练标签
    X_test: 测试特征
    y_test: 测试标签
    feature_names: 特征名称
    """
    from sklearn.linear_model import LogisticRegression
    
    # 创建归一化器
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    
    # 应用归一化
    X_train_standard = scaler_standard.fit_transform(X_train)
    X_test_standard = scaler_standard.transform(X_test)
    
    X_train_minmax = scaler_minmax.fit_transform(X_train)
    X_test_minmax = scaler_minmax.transform(X_test)
    
    # 训练逻辑回归模型
    models = {}
    
    # 原始数据
    lr_original = LogisticRegression(random_state=42)
    lr_original.fit(X_train, y_train)
    models['原始数据'] = lr_original
    
    # Z-score标准化
    lr_standard = LogisticRegression(random_state=42)
    lr_standard.fit(X_train_standard, y_train)
    models['Z-score标准化'] = lr_standard
    
    # 最小-最大归一化
    lr_minmax = LogisticRegression(random_state=42)
    lr_minmax.fit(X_train_minmax, y_train)
    models['最小-最大归一化'] = lr_minmax
    
    # 可视化特征系数
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (method, model) in enumerate(models.items()):
        coefficients = np.abs(model.coef_[0])
        
        # 排序
        sorted_idx = np.argsort(coefficients)[::-1]
        
        axes[i].barh(range(len(feature_names)), coefficients[sorted_idx])
        axes[i].set_yticks(range(len(feature_names)))
        axes[i].set_yticklabels([feature_names[idx] for idx in sorted_idx])
        axes[i].set_title(f'{method}\n特征重要性')
        axes[i].set_xlabel('系数绝对值')
    
    plt.tight_layout()
    plt.savefig('feature_importance_normalization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细结果
    print("特征系数对比:")
    print("-" * 60)
    
    for method, model in models.items():
        print(f"\n{method}:")
        coefficients = model.coef_[0]
        
        # 排序
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        
        for idx in sorted_idx[:10]:  # 显示前10个重要特征
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    return models