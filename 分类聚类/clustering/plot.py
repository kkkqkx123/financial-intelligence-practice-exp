import matplotlib.pyplot as plt
import os

def plot(X, idx, title, save_path=None):
    '''
    Show clustering results

    Input:  X: data point features, n-by-p maxtirx.
            idx: data point cluster labels, n-by-1 vector.
            title: plot title
            save_path: path to save the plot (optional)
    '''
    plt.figure(figsize=(6, 6))
    plt.plot(X[idx==0, 0],X[idx==0, 1],'r.', markersize=5, label='Cluster 1')
    plt.plot(X[idx==1, 0],X[idx==1, 1],'b.', markersize=5, label='Cluster 2')
    plt.title(title)
    plt.legend(loc='upper right')
    
    # 保存图片到当前目录
    if save_path is None:
        # 默认保存到当前目录的clustering_results文件夹
        save_dir = 'clustering_results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"聚类结果图片已保存到: {save_path}")
    
    plt.show()