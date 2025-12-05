# -*- coding: utf-8 -*-
import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func, logistic_regression, perceptron
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# 设置中文字体，避免中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

no_iter = 1000  # 迭代次数
no_train = 70  # 训练数据数量（70%用于训练）
no_test = 30   # 测试数据数量（30%用于测试）
no_data = 100  # 总数据数量
assert(no_train + no_test == no_data)

def compute_error(X, y, w):
    """计算分类错误率"""
    X_bias = np.vstack((np.ones((1, X.shape[1])), X))
    predictions = np.sign(w.T @ X_bias)
    # 处理预测值为0的情况
    predictions[predictions == 0] = 1
    errors = np.sum(predictions.flatten() != y.flatten())
    return errors / y.shape[1]

def run_classification_experiment(algorithm_name, algorithm_func):
    """使用指定算法运行分类实验"""
    print(f"\n=== 运行 {algorithm_name} ===")
    
    cumulative_train_err = 0
    cumulative_test_err = 0
    
    for i in range(no_iter):
        X, y, w_gt = gen_data(no_data)
        X_train, X_test = X[:, :no_train], X[:, no_train:]
        y_train, y_test = y[:, :no_train], y[:, no_train:]
        
        # 学习参数
        w_l = algorithm_func(X_train, y_train)
        
        # 计算训练和测试错误率
        train_err = compute_error(X_train, y_train, w_l)
        test_err = compute_error(X_test, y_test, w_l)
        
        cumulative_train_err += train_err
        cumulative_test_err += test_err
    
    avg_train_err = cumulative_train_err / no_iter
    avg_test_err = cumulative_test_err / no_iter
    
    print(f"{algorithm_name} 结果:")
    print(f"平均训练错误率: {avg_train_err:.4f}")
    print(f"平均测试错误率: {avg_test_err:.4f}")
    
    return avg_train_err, avg_test_err

# 使用不同算法运行实验
results = {}

# SVM（原始函数）
results['SVM'] = run_classification_experiment('支持向量机', func)

# 逻辑回归
results['Logistic Regression'] = run_classification_experiment('逻辑回归', logistic_regression)

# 感知器
results['Perceptron'] = run_classification_experiment('感知器', perceptron)

# 比较结果
print("\n=== 算法比较 ===")
print("算法\t\t训练错误率\t测试错误率")
print("-" * 50)
for algo, (train_err, test_err) in results.items():
    print(f"{algo:<20}\t{train_err:.4f}\t\t{test_err:.4f}")

# 保存结果到文件
results_dir = "classification_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, "classification_comparison.txt"), "w", encoding='utf-8') as f:
    f.write("分类算法比较结果\n")
    f.write("=" * 50 + "\n\n")
    for algo, (train_err, test_err) in results.items():
        f.write(f"{algo}:\n")
        f.write(f"  训练错误率: {train_err:.4f}\n")
        f.write(f"  测试错误率: {test_err:.4f}\n\n")

print(f"\n结果已保存到 {results_dir}/classification_comparison.txt")

# 使用其中一个算法生成最终可视化
X, y, w_gt = gen_data(no_data)
X_train, X_test = X[:, :no_train], X[:, no_train:]
y_train, y_test = y[:, :no_train], y[:, no_train:]
w_l = func(X_train, y_train)

# 生成可视化并保存到当前目录
plt.figure(figsize=(10, 8))
plot(X, y, w_gt, w_l, "SVM分类结果")
plt.savefig("SVM分类结果.png", dpi=300, bbox_inches="tight")
plt.close()
print("分类结果图已保存为: SVM分类结果.png")