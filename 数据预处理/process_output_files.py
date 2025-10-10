import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime

def analyze_output_files():
    """分析输出目录中的所有文件并生成综合报告"""
    
    output_dir = "./output"
    
    # 获取所有输出文件
    csv_files = glob.glob(f"{output_dir}/*.csv")
    txt_files = glob.glob(f"{output_dir}/*.txt")
    png_files = glob.glob(f"{output_dir}/*.png")
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print(f"找到 {len(txt_files)} 个文本文件")
    print(f"找到 {len(png_files)} 个图像文件")
    
    # 分析相关矩阵
    correlation_files = [f for f in csv_files if 'correlation_matrix' in f]
    if correlation_files:
        corr_matrix = pd.read_csv(correlation_files[0], index_col=0)
        print(f"\n相关矩阵形状: {corr_matrix.shape}")
        
        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.8:  # 相关系数大于0.8
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
            high_corr_df.to_csv(f"{output_dir}/high_correlation_pairs.csv", index=False)
            print(f"找到 {len(high_corr_pairs)} 对高度相关的特征")
        
        # 计算每个特征的平均绝对相关性
        avg_correlations = corr_matrix.abs().mean().sort_values(ascending=False)
        avg_correlations.to_csv(f"{output_dir}/average_correlations.csv", header=['平均绝对相关性'])
        print("已保存特征平均相关性分析")
    
    # 分析缺失值模式
    missing_analysis_files = [f for f in txt_files if 'missing_values_analysis' in f]
    if missing_analysis_files:
        with open(missing_analysis_files[0], 'r', encoding='utf-8') as f:
            missing_content = f.read()
        print("\n缺失值分析概览:")
        print(missing_content[:500] + "..." if len(missing_content) > 500 else missing_content)
    
    # 分析填充方法比较
    fill_comparison_files = [f for f in csv_files if 'fill_methods_comparison' in f]
    if fill_comparison_files:
        fill_df = pd.read_csv(fill_comparison_files[0])
        best_method = fill_df.loc[fill_df['剩余缺失值总数'].idxmin()]
        print(f"\n最佳填充方法: {best_method['填充方法']} (剩余缺失值: {best_method['剩余缺失值总数']})")
    
    # 分析最终处理的数据
    final_data_files = [f for f in csv_files if 'final_processed_data' in f]
    if final_data_files:
        final_data = pd.read_csv(final_data_files[0])
        print(f"\n最终数据形状: {final_data.shape}")
        print(f"数据类型分布:")
        print(final_data.dtypes.value_counts())
        
        # 分析新增的分箱特征
        bin_features = [col for col in final_data.columns if '_bin' in col]
        if bin_features:
            print(f"\n分箱特征: {bin_features}")
            
            # 生成分箱特征统计
            bin_stats = {}
            for feature in bin_features:
                bin_stats[feature] = {
                    '唯一值数量': final_data[feature].nunique(),
                    '最频繁类别': final_data[feature].mode().iloc[0] if not final_data[feature].mode().empty else 'N/A',
                    '最频繁类别占比': (final_data[feature].value_counts().iloc[0] / len(final_data)) * 100 if not final_data[feature].value_counts().empty else 0
                }
            
            bin_stats_df = pd.DataFrame(bin_stats).T
            bin_stats_df.to_csv(f"{output_dir}/binning_features_statistics.csv")
            print("已保存分箱特征统计")
    
    # 生成特征重要性分析（如果有目标变量）
    # 假设最后一列是目标变量
    if final_data_files and len(final_data_files) > 0:
        final_data = pd.read_csv(final_data_files[0])
        
        # 检查是否有目标变量（假设列名包含'label'或'target'或'Label'）
        target_cols = [col for col in final_data.columns if any(term in col.lower() for term in ['label', 'target', 'y'])]
        
        if target_cols:
            target_col = target_cols[0]
            feature_cols = [col for col in final_data.columns if col != target_col and final_data[col].dtype in ['int64', 'float64']]
            
            if len(feature_cols) > 0:
                # 计算与目标变量的相关性
                target_correlations = final_data[feature_cols].corrwith(final_data[target_col]).abs().sort_values(ascending=False)
                target_correlations.to_csv(f"{output_dir}/feature_target_correlations.csv", header=['与目标变量的相关性'])
                print(f"\n已保存特征与目标变量的相关性分析")
                
                # 可视化前20个最相关的特征
                top_features = target_correlations.head(20)
                plt.figure(figsize=(10, 8))
                top_features.plot(kind='barh')
                plt.title(f'与 {target_col} 最相关的特征 (前20个)')
                plt.xlabel('绝对相关系数')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/top_feature_correlations.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    # 生成综合报告
    generate_comprehensive_report(output_dir)

def generate_comprehensive_report(output_dir):
    """生成综合分析报告"""
    
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""
数据处理综合分析报告
生成时间: {report_time}
=====================================

1. 文件概览
-----------------
"""
    
    # 获取所有文件信息
    all_files = glob.glob(f"{output_dir}/*")
    file_types = {}
    for file in all_files:
        ext = os.path.splitext(file)[1]
        if ext not in file_types:
            file_types[ext] = 0
        file_types[ext] += 1
    
    for ext, count in file_types.items():
        report_content += f"- {ext} 文件: {count} 个\n"
    
    report_content += """
2. 关键发现
-----------------
"""
    
    # 读取高相关性分析
    if os.path.exists(f"{output_dir}/high_correlation_pairs.csv"):
        high_corr = pd.read_csv(f"{output_dir}/high_correlation_pairs.csv")
        if len(high_corr) > 0:
            report_content += f"- 发现 {len(high_corr)} 对高度相关的特征 (相关系数 > 0.8)\n"
            report_content += "  前5对最相关的特征:\n"
            for _, row in high_corr.head().iterrows():
                report_content += f"    {row['feature1']} - {row['feature2']}: {row['correlation']:.3f}\n"
    
    # 读取特征重要性
    if os.path.exists(f"{output_dir}/feature_target_correlations.csv"):
        feat_importance = pd.read_csv(f"{output_dir}/feature_target_correlations.csv", index_col=0)
        if len(feat_importance) > 0:
            report_content += f"- 分析了 {len(feat_importance)} 个特征与目标变量的相关性\n"
            top_feature = feat_importance.index[0]
            top_correlation = feat_importance.iloc[0, 0]
            report_content += f"- 最相关的特征: {top_feature} (相关性: {top_correlation:.3f})\n"
    
    # 读取分箱特征统计
    if os.path.exists(f"{output_dir}/binning_features_statistics.csv"):
        bin_stats = pd.read_csv(f"{output_dir}/binning_features_statistics.csv", index_col=0)
        report_content += f"- 创建了 {len(bin_stats)} 个分箱特征\n"
    
    report_content += """
3. 建议
-----------------
- 考虑移除高度相关的特征以减少冗余
- 优先使用与目标变量高度相关的特征进行建模
- 分箱特征可以作为类别特征使用
- 标准化数据适用于基于距离的算法
- 归一化数据适用于神经网络等算法

4. 下一步操作
-----------------
- 基于特征重要性进行特征选择
- 考虑使用降维技术 (PCA, LDA等)
- 创建特征交互项
- 进行特征编码 (独热编码, 标签编码等)
"""
    
    # 保存报告
    report_file = f"{output_dir}/comprehensive_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n综合分析报告已保存到: {report_file}")

def create_feature_summary():
    """创建特征摘要表"""
    
    output_dir = "./output"
    
    # 尝试读取最终处理的数据
    final_data_files = glob.glob(f"{output_dir}/final_processed_data_*.csv")
    
    if not final_data_files:
        print("未找到最终处理的数据文件")
        return
    
    final_data = pd.read_csv(final_data_files[0])
    
    # 创建特征摘要
    feature_summary = []
    
    for col in final_data.columns:
        col_info = {
            '特征名': col,
            '数据类型': str(final_data[col].dtype),
            '唯一值数量': final_data[col].nunique(),
            '缺失值数量': final_data[col].isnull().sum(),
            '缺失值比例': (final_data[col].isnull().sum() / len(final_data)) * 100,
            '均值': final_data[col].mean() if final_data[col].dtype in ['int64', 'float64'] else 'N/A',
            '标准差': final_data[col].std() if final_data[col].dtype in ['int64', 'float64'] else 'N/A',
            '最小值': final_data[col].min() if final_data[col].dtype in ['int64', 'float64'] else 'N/A',
            '最大值': final_data[col].max() if final_data[col].dtype in ['int64', 'float64'] else 'N/A',
            '是否分箱特征': '是' if '_bin' in col else '否'
        }
        feature_summary.append(col_info)
    
    feature_summary_df = pd.DataFrame(feature_summary)
    feature_summary_df.to_csv(f"{output_dir}/feature_summary.csv", index=False)
    
    print(f"特征摘要已保存到: {output_dir}/feature_summary.csv")
    print(f"总共分析了 {len(feature_summary)} 个特征")

if __name__ == "__main__":
    print("开始分析输出文件...")
    analyze_output_files()
    create_feature_summary()
    print("分析完成！")