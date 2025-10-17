# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体，避免中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

def plot(X, y, w_gt, w_l, title):
    '''
    绘制数据集。

    输入:  X: 样本特征, P-by-N 矩阵。
            y: 样本标签, 1-by-N 行向量。
            w_gt: 真实目标函数参数, (P+1)-by-1 列向量。
            w_l: 学习得到的目标函数参数, (P+1)-by-1 列向量。
            title: 图形标题。
    '''

    if X.shape[0] != 2:
        print('这里我们只支持二维X数据')
        return

    plt.plot(X[0, y.flatten() == 1], X[1, y.flatten() == 1], 'o', markerfacecolor='r', \
                                              markersize=10)

    plt.plot(X[0, y.flatten() == -1], X[1, y.flatten() == -1], 'o', markerfacecolor='g', \
                                                markersize=10)
    
    k, b = -w_gt[1] / w_gt[2], -w_gt[0] / w_gt[2]
    max_x = max(min((1 - b) / k, (-1 - b ) / k), -1)
    min_x = min(max((1 - b) / k, (-1 - b ) / k), 1)
    x = np.arange(min_x, max_x, (max_x - min_x) / 100)
    temp_y = k * x + b
    plt.plot(x, temp_y, color='b', linewidth=2, linestyle='-')
    k, b = -w_l[1] / w_l[2], -w_l[0] / w_l[2]
    max_x = max(min((1 - b) / k, (-1 - b ) / k), -1)
    min_x = min(max((1 - b) / k, (-1 - b ) / k), 1)
    x = np.arange(min_x, max_x, (max_x - min_x) / 100)
    temp_y = k * x + b
    plt.plot(x, temp_y, color='b', linewidth=2, linestyle='--')
    plt.title(title)
    # 移除plt.show()，让调用者控制显示和保存