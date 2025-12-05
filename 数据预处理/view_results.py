import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_generated_images():
    """显示生成的图像文件"""
    
    output_dir = "./output"
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    print("生成的图像文件:")
    for i, img_file in enumerate(image_files, 1):
        print(f"{i}. {img_file}")
    
    print("\n图像说明:")
    print("1. correlation_heatmap_*.png - 特征相关性热力图")
    print("2. correlation_heatmap_rdbu_*.png - 特征相关性热力图(RdBu色图)")
    print("3. original_data_distribution_*.png - 原始数据分布图")
    print("4. normalized_data_distribution_*.png - 标准化数据分布图")
    print("5. binning_visualization_*.png - 分箱特征可视化")
    print("6. top_feature_correlations.png - 与目标变量最相关的特征")
    
    print("\n要查看特定图像，请使用以下代码:")
    print("```python")
    print("import matplotlib.pyplot as plt")
    print("import matplotlib.image as mpimg")
    print("")
    print("# 例如查看相关性热力图")
    print("img = mpimg.imread('./output/correlation_heatmap_20251010_143306.png')")
    print("plt.imshow(img)")
    print("plt.axis('off')")
    print("plt.show()")
    print("```")

def list_all_output_files():
    """列出所有输出文件及其用途"""
    
    output_dir = "./output"
    files_info = {
        '文本文件 (.txt)': {
            'basic_info_*.txt': '数据集基本信息',
            'missing_values_analysis_*.txt': '缺失值分析结果',
            'data_cleaning_info_*.txt': '数据清洗信息',
            'binning_results_*.txt': '分箱结果',
            'processing_summary_*.txt': '处理摘要',
            'comprehensive_analysis_report.txt': '综合分析报告'
        },
        'CSV文件 (.csv)': {
            'correlation_matrix_*.csv': '特征相关矩阵',
            'fill_methods_comparison_*.csv': '填充方法比较',
            'scaled_data_minmax_*.csv': 'MinMax标准化数据',
            'normalized_data_standard_*.csv': 'Z-score标准化数据',
            'final_processed_data_*.csv': '最终处理的数据',
            'high_correlation_pairs.csv': '高度相关的特征对',
            'average_correlations.csv': '特征平均相关性',
            'feature_target_correlations.csv': '特征与目标变量相关性',
            'binning_features_statistics.csv': '分箱特征统计',
            'feature_summary.csv': '特征摘要表'
        },
        '图像文件 (.png)': {
            'correlation_heatmap_*.png': '相关性热力图',
            'correlation_heatmap_rdbu_*.png': '相关性热力图(RdBu)',
            'original_data_distribution_*.png': '原始数据分布',
            'normalized_data_distribution_*.png': '标准化数据分布',
            'binning_visualization_*.png': '分箱可视化',
            'top_feature_correlations.png': '最相关特征'
        }
    }
    
    print("输出文件详细说明:")
    print("=" * 50)
    
    for category, files in files_info.items():
        print(f"\n{category}:")
        print("-" * 30)
        for pattern, description in files.items():
            matching_files = [f for f in os.listdir(output_dir) if pattern.replace('*', '') in f]
            if matching_files:
                print(f"  {matching_files[0]}: {description}")
    
    print(f"\n所有文件都保存在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    display_generated_images()
    print("\n" + "="*50 + "\n")
    list_all_output_files()