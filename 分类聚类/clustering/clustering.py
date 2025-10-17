import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from plot import plot
from todo import kmeans
from todo import spectral
from todo import knn_graph
import os
import pandas as pd
from datetime import datetime

def save_clustering_results(X, labels, algorithm_name, params=None):
    """保存聚类结果到文件"""
    results_dir = 'clustering_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存标签结果
    filename = f"{results_dir}/{algorithm_name}_labels.npy"
    np.save(filename, labels)
    
    # 保存参数信息
    if params:
        info_file = f"{results_dir}/{algorithm_name}_params.txt"
        with open(info_file, 'w') as f:
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Data shape: {X.shape}\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    
    print(f"结果已保存到 {filename}")
    return filename

def save_tuning_results_to_csv(results, algorithm_name="spectral"):
    """将参数调优结果保存为CSV文件"""
    if not results:
        return None
    
    # 创建DataFrame
    df_data = []
    for i, result in enumerate(results):
        cluster_sizes = result['cluster_sizes']
        row = {
            'test_id': i + 1,
            'k': result['k'],
            'threshold': result['threshold'],
            'score': result['score'],
            'cluster_0_size': cluster_sizes[0] if len(cluster_sizes) > 0 else 0,
            'cluster_1_size': cluster_sizes[1] if len(cluster_sizes) > 1 else 0,
            'n_clusters': len(cluster_sizes),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # 保存到CSV文件
    csv_filename = f"clustering_results/{algorithm_name}_tuning_results.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    # 同时保存最佳结果
    best_result = max(results, key=lambda x: x['score'])
    best_cluster_sizes = best_result['cluster_sizes']
    best_df = pd.DataFrame([{
        'best_k': best_result['k'],
        'best_threshold': best_result['threshold'],
        'best_score': best_result['score'],
        'cluster_0_size': best_cluster_sizes[0] if len(best_cluster_sizes) > 0 else 0,
        'cluster_1_size': best_cluster_sizes[1] if len(best_cluster_sizes) > 1 else 0,
        'n_clusters': len(best_cluster_sizes),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    best_csv_filename = f"clustering_results/{algorithm_name}_best_params.csv"
    best_df.to_csv(best_csv_filename, index=False, encoding='utf-8')
    
    print(f"参数调优结果已保存到: {csv_filename}")
    print(f"最佳参数已保存到: {best_csv_filename}")
    
    return csv_filename, best_csv_filename

def evaluate_clustering(X, labels):
    """简单的聚类评估指标"""
    n_clusters = len(np.unique(labels))
    cluster_sizes = []
    
    for i in range(n_clusters):
        cluster_size = np.sum(labels == i)
        cluster_sizes.append(cluster_size)
    
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'min_cluster_size': min(cluster_sizes),
        'max_cluster_size': max(cluster_sizes)
    }

def tune_spectral_parameters(X, k_range=None, threshold_range=None):
    """调试谱聚类的参数 - 限制搜索次数"""
    print("开始调试谱聚类参数...")
    
    # 限制搜索次数不超过8次
    if k_range is None:
        k_range = [50, 100, 200]  # 3个k值
    if threshold_range is None:
        threshold_range = [0, 0.2, 0.4]  # 3个阈值
    
    best_params = {}
    best_score = -1
    results = []
    total_tests = 0
    
    print(f"参数搜索空间: k值 {list(k_range)}, 阈值 {list(threshold_range)}")
    print(f"总搜索次数: {len(k_range) * len(threshold_range)}")
    
    for k in k_range:
        for threshold in threshold_range:
            total_tests += 1
            if total_tests > 8:  # 严格限制搜索次数
                print(f"已达到最大搜索次数限制(8次)，停止搜索")
                break
                
            try:
                print(f"测试参数组合 {total_tests}: k={k}, threshold={threshold}")
                
                # 构建KNN图
                W = knn_graph(X, k, threshold)
                
                # 检查图的连通性
                if np.sum(W) == 0:
                    print(f"  图不连通，跳过")
                    continue
                    
                # 运行谱聚类
                labels = spectral(W, 2)
                
                # 评估聚类质量（简单的簇大小平衡性）
                eval_metrics = evaluate_clustering(X, labels)
                
                # 如果聚类结果不是2个簇，直接忽略该结果
                if eval_metrics['n_clusters'] != 2:
                    print(f"  产生 {eval_metrics['n_clusters']} 个簇，期望2个簇，跳过")
                    continue
                
                # 计算评分（簇大小越平衡越好）
                cluster_sizes = eval_metrics['cluster_sizes']
                
                # 使用相对标准差（变异系数）作为平衡性指标
                mean_size = np.mean(cluster_sizes)
                if mean_size > 0:
                    cv = np.std(cluster_sizes) / mean_size  # 变异系数
                    score = 1.0 / (1.0 + cv)  # 变异系数越小，评分越高
                else:
                    score = 0.0
                
                print(f"  簇大小分布: {eval_metrics['cluster_sizes']}")
                print(f"  评分: {score:.4f}")
                
                results.append({
                    'k': k,
                    'threshold': threshold,
                    'score': score,
                    'cluster_sizes': eval_metrics['cluster_sizes']
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'k': k, 'threshold': threshold}
                    print(f"  *** 新的最佳参数 ***")
                    
            except Exception as e:
                print(f"  参数测试失败: {e}")
                continue
        
        if total_tests > 8:
            break
    
    print(f"\n参数调试完成")
    print(f"最佳参数: k={best_params.get('k', 'N/A')}, threshold={best_params.get('threshold', 'N/A'):.2f}")
    print(f"对应评分: {best_score:.4f}")
    
    return best_params, results

def compare_algorithms(X, kmeans_labels, spectral_labels):
    """比较两种聚类算法的结果"""
    print("\n=== 算法比较分析 ===")
    
    # K-means评估
    kmeans_eval = evaluate_clustering(X, kmeans_labels)
    print(f"K-means聚类:")
    print(f"  簇数量: {kmeans_eval['n_clusters']}")
    print(f"  簇大小分布: {kmeans_eval['cluster_sizes']}")
    print(f"  最小簇大小: {kmeans_eval['min_cluster_size']}")
    print(f"  最大簇大小: {kmeans_eval['max_cluster_size']}")
    
    # 谱聚类评估
    spectral_eval = evaluate_clustering(X, spectral_labels)
    print(f"\n谱聚类:")
    print(f"  簇数量: {spectral_eval['n_clusters']}")
    print(f"  簇大小分布: {spectral_eval['cluster_sizes']}")
    print(f"  最小簇大小: {spectral_eval['min_cluster_size']}")
    print(f"  最大簇大小: {spectral_eval['max_cluster_size']}")
    
    # 计算两种结果的差异（只有当两个算法都产生相同数量的簇时才比较）
    if kmeans_eval['n_clusters'] == spectral_eval['n_clusters'] and kmeans_eval['n_clusters'] >= 2:
        # 由于标签可能对应关系不同，我们计算调整后的兰德指数近似
        kmeans_0_size = kmeans_eval['cluster_sizes'][0]
        kmeans_1_size = kmeans_eval['cluster_sizes'][1]
        spectral_0_size = spectral_eval['cluster_sizes'][0]
        spectral_1_size = spectral_eval['cluster_sizes'][1]
        
        size_diff_0 = abs(kmeans_0_size - spectral_0_size)
        size_diff_1 = abs(kmeans_1_size - spectral_1_size)
        
        print(f"\n簇大小差异:")
        print(f"  簇1大小差异: {size_diff_0}")
        print(f"  簇2大小差异: {size_diff_1}")
        print(f"  总差异: {size_diff_0 + size_diff_1}")
    else:
        print(f"\n无法直接比较簇大小差异:")
        print(f"  K-means产生了 {kmeans_eval['n_clusters']} 个簇")
        print(f"  谱聚类产生了 {spectral_eval['n_clusters']} 个簇")

# 主程序
if __name__ == "__main__":
    # 加载数据
    print("加载数据...")
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'cluster_data.mat')
    print(f"尝试加载数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误: 文件 {data_path} 不存在")
        print(f"当前目录内容: {os.listdir(current_dir)}")
        # 尝试其他可能的路径
        alt_paths = [
            os.path.join(current_dir, '..', 'cluster_data.mat'),
            os.path.join(current_dir, 'cluster_data.mat'),
            'cluster_data.mat'
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"找到文件在: {alt_path}")
                data_path = alt_path
                break
        else:
            print("无法找到数据文件，请检查文件位置")
            exit(1)
    
    try:
        cluster_data = sio.loadmat(data_path)
        X = cluster_data['X']
        print(f"数据加载成功，形状: {X.shape}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        print(f"尝试加载的键: {list(cluster_data.keys()) if 'cluster_data' in locals() else 'N/A'}")
        exit(1)
    
    # 1. K-means聚类
    print("\n=== 运行K-means聚类 ===")
    kmeans_labels = kmeans(X, 2)
    plot(X, kmeans_labels, "Clustering-kmeans")
    
    # 保存K-means结果
    kmeans_filename = save_clustering_results(X, kmeans_labels, "kmeans", 
                                             params={"algorithm": "kmeans", "k": 2})
    
    # 保存K-means聚类结果到CSV
    kmeans_eval = evaluate_clustering(X, kmeans_labels)
    kmeans_cluster_sizes = kmeans_eval['cluster_sizes']
    kmeans_df = pd.DataFrame([{
        'algorithm': 'kmeans',
        'n_clusters': kmeans_eval['n_clusters'],
        'cluster_0_size': kmeans_cluster_sizes[0] if len(kmeans_cluster_sizes) > 0 else 0,
        'cluster_1_size': kmeans_cluster_sizes[1] if len(kmeans_cluster_sizes) > 1 else 0,
        'min_cluster_size': kmeans_eval['min_cluster_size'],
        'max_cluster_size': kmeans_eval['max_cluster_size'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    kmeans_csv = "clustering_results/kmeans_results.csv"
    kmeans_df.to_csv(kmeans_csv, index=False, encoding='utf-8')
    print(f"K-means结果已保存到CSV: {kmeans_csv}")
    
    # 2. 谱聚类参数调试
    print("\n=== 调试谱聚类参数 ===")
    best_params, tuning_results = tune_spectral_parameters(X)
    
    # 保存参数调优结果到CSV
    if tuning_results:
        save_tuning_results_to_csv(tuning_results, "spectral")
    
    # 使用最佳参数运行谱聚类
    print("\n=== 使用最佳参数运行谱聚类 ===")
    if best_params:
        W = knn_graph(X, best_params['k'], best_params['threshold'])
        spectral_labels = spectral(W, 2)
        plot(X, spectral_labels, f"Clustering-Spectral (k={best_params['k']}, threshold={best_params['threshold']:.2f})")
        
        # 保存谱聚类结果
        spectral_filename = save_clustering_results(X, spectral_labels, "spectral", 
                                                    params={"algorithm": "spectral", "k": 2, 
                                                           "knn_k": best_params['k'], 
                                                           "threshold": best_params['threshold']})
        
        # 保存谱聚类结果到CSV
        spectral_eval = evaluate_clustering(X, spectral_labels)
        spectral_cluster_sizes = spectral_eval['cluster_sizes']
        spectral_df = pd.DataFrame([{
            'algorithm': 'spectral',
            'n_clusters': spectral_eval['n_clusters'],
            'cluster_0_size': spectral_cluster_sizes[0] if len(spectral_cluster_sizes) > 0 else 0,
            'cluster_1_size': spectral_cluster_sizes[1] if len(spectral_cluster_sizes) > 1 else 0,
            'min_cluster_size': spectral_eval['min_cluster_size'],
            'max_cluster_size': spectral_eval['max_cluster_size'],
            'knn_k': best_params['k'],
            'threshold': best_params['threshold'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        spectral_csv = "clustering_results/spectral_results.csv"
        spectral_df.to_csv(spectral_csv, index=False, encoding='utf-8')
        print(f"谱聚类结果已保存到CSV: {spectral_csv}")
    else:
        # 使用默认参数
        print("使用默认参数")
        W = knn_graph(X, 15, 1.45)
        spectral_labels = spectral(W, 2)
        plot(X, spectral_labels, "Clustering-Spectral (default)")
        spectral_filename = save_clustering_results(X, spectral_labels, "spectral", 
                                                    params={"algorithm": "spectral", "k": 2, 
                                                           "knn_k": 15, "threshold": 1.45})
        
        # 保存谱聚类结果到CSV（默认参数）
        spectral_eval = evaluate_clustering(X, spectral_labels)
        spectral_cluster_sizes = spectral_eval['cluster_sizes']
        spectral_df = pd.DataFrame([{
            'algorithm': 'spectral',
            'n_clusters': spectral_eval['n_clusters'],
            'cluster_0_size': spectral_cluster_sizes[0] if len(spectral_cluster_sizes) > 0 else 0,
            'cluster_1_size': spectral_cluster_sizes[1] if len(spectral_cluster_sizes) > 1 else 0,
            'min_cluster_size': spectral_eval['min_cluster_size'],
            'max_cluster_size': spectral_eval['max_cluster_size'],
            'knn_k': 15,
            'threshold': 1.45,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        spectral_csv = "clustering_results/spectral_results.csv"
        spectral_df.to_csv(spectral_csv, index=False, encoding='utf-8')
        print(f"谱聚类结果已保存到CSV: {spectral_csv}")
    
    # 3. 比较分析
    compare_algorithms(X, kmeans_labels, spectral_labels)
    
    # 保存算法比较结果到CSV
    kmeans_eval = evaluate_clustering(X, kmeans_labels)
    spectral_eval = evaluate_clustering(X, spectral_labels)
    
    comparison_df = pd.DataFrame([
        {
            'algorithm': 'kmeans',
            'n_clusters': kmeans_eval['n_clusters'],
            'cluster_0_size': kmeans_eval['cluster_sizes'][0],
            'cluster_1_size': kmeans_eval['cluster_sizes'][1],
            'min_cluster_size': kmeans_eval['min_cluster_size'],
            'max_cluster_size': kmeans_eval['max_cluster_size'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'algorithm': 'spectral',
            'n_clusters': spectral_eval['n_clusters'],
            'cluster_0_size': spectral_eval['cluster_sizes'][0],
            'cluster_1_size': spectral_eval['cluster_sizes'][1],
            'min_cluster_size': spectral_eval['min_cluster_size'],
            'max_cluster_size': spectral_eval['max_cluster_size'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    ])
    comparison_csv = "clustering_results/algorithms_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8')
    print(f"算法比较结果已保存到CSV: {comparison_csv}")
    
    print(f"\n=== 结果保存完成 ===")
    print(f"K-means结果: {kmeans_filename}")
    print(f"谱聚类结果: {spectral_filename}")
    print("所有结果已保存到 clustering_results 文件夹")
    print("CSV文件已生成，包含详细的参数调优和算法比较结果")