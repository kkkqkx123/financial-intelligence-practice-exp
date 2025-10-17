import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# 加载数据
cluster_data = sio.loadmat('cluster_data.mat')
X = cluster_data['X']

# 计算每个点到原点的距离
distances = np.sqrt(X[:,0]**2 + X[:,1]**2)
print(f'距离分布:')
print(f'  最小距离: {distances.min():.3f}')
print(f'  最大距离: {distances.max():.3f}')
print(f'  平均距离: {distances.mean():.3f}')
print(f'  距离标准差: {distances.std():.3f}')

# 分析距离分布
hist, bins = np.histogram(distances, bins=50)
print(f'\\n距离直方图分析:')
for i in range(min(10, len(hist))):
    print(f'  区间 [{bins[i]:.3f}, {bins[i+1]:.3f}]: {hist[i]} 个点')

# 尝试找到内外圆的分界点
sorted_distances = np.sort(distances)
print(f'\\n排序后的距离（前20个）: {sorted_distances[:20]}')
print(f'排序后的距离（后20个）: {sorted_distances[-20:]}')