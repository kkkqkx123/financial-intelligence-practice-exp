import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

def analyze_dataset_features():
    """基于生成的文件分析数据集特征"""
    
    output_dir = "./output"
    
    # 检查输出目录是否存在
    if not os.path.exists(output_dir):
        print("输出目录不存在，请先运行main.py生成分析文件")
        return
    
    # 获取最新的分析文件
    analysis_files = glob.glob(f"{output_dir}/*.csv") + glob.glob(f"{output_dir}/*.txt")
    
    if not analysis_files:
        print("未找到分析文件")
        return
    
    print("开始分析数据集特征...")
    print("=" * 60)
    
    # 1. 分析数据基本信息
    print("\n1. 数据基本信息分析")
    print("-" * 40)
    
    basic_info_files = glob.glob(f"{output_dir}/basic_info_*.txt")
    if basic_info_files:
        with open(basic_info_files[0], 'r', encoding='utf-8') as f:
            basic_info = f.read()
        print(basic_info)
    
    # 2. 分析数据形状和清洗情况
    print("\n2. 数据清洗分析")
    print("-" * 40)
    
    cleaning_files = glob.glob(f"{output_dir}/data_cleaning_info_*.txt")
    if cleaning_files:
        with open(cleaning_files[0], 'r', encoding='utf-8') as f:
            cleaning_info = f.read()
        print(cleaning_info)
    
    # 3. 分析特征相关性
    print("\n3. 特征相关性分析")
    print("-" * 40)
    
    corr_matrix_files = glob.glob(f"{output_dir}/correlation_matrix_*.csv")
    if corr_matrix_files:
        corr_matrix = pd.read_csv(corr_matrix_files[0], index_col=0)
        print(f"相关矩阵形状: {corr_matrix.shape}")
        print(f"特征数量: {len(corr_matrix.columns)}")
        
        # 分析高度相关特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        print(f"高度相关特征对数量 (相关系数 > 0.8): {len(high_corr_pairs)}")
        if high_corr_pairs:
            print("前5对最相关的特征:")
            for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)[:5]:
                print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
    
    # 4. 分析特征与目标变量的关系
    print("\n4. 特征与目标变量相关性分析")
    print("-" * 40)
    
    target_corr_files = glob.glob(f"{output_dir}/feature_target_correlations.csv")
    if target_corr_files:
        target_corr = pd.read_csv(target_corr_files[0], index_col=0)
        print(f"分析的特征数量: {len(target_corr)}")
        
        # 找出最相关的特征
        top_features = target_corr.nlargest(10, target_corr.columns[0])
        print("与目标变量最相关的10个特征:")
        for idx, (feature, corr) in enumerate(top_features.iterrows(), 1):
            print(f"  {idx}. {feature}: {corr.iloc[0]:.3f}")
    
    # 5. 分析特征统计信息
    print("\n5. 特征统计信息分析")
    print("-" * 40)
    
    feature_summary_files = glob.glob(f"{output_dir}/feature_summary.csv")
    if feature_summary_files:
        feature_summary = pd.read_csv(feature_summary_files[0])
        
        # 数据类型分布
        dtype_dist = feature_summary['数据类型'].value_counts()
        print("数据类型分布:")
        for dtype, count in dtype_dist.items():
            print(f"  {dtype}: {count}个")
        
        # 缺失值分析
        missing_stats = feature_summary['缺失值比例'].describe()
        print(f"\n缺失值比例统计:")
        print(f"  平均值: {missing_stats['mean']:.2f}%")
        print(f"  最大值: {missing_stats['max']:.2f}%")
        print(f"  最小值: {missing_stats['min']:.2f}%")
        
        # 分箱特征分析
        bin_features = feature_summary[feature_summary['是否分箱特征'] == '是']
        print(f"\n分箱特征数量: {len(bin_features)}")
        if len(bin_features) > 0:
            print("分箱特征列表:")
            for feature in bin_features['特征名']:
                print(f"  {feature}")
    
    # 6. 分析特征重要性
    print("\n6. 特征重要性分析")
    print("-" * 40)
    
    avg_corr_files = glob.glob(f"{output_dir}/average_correlations.csv")
    if avg_corr_files:
        avg_corr = pd.read_csv(avg_corr_files[0], index_col=0)
        
        # 分析特征的平均相关性
        print("平均相关性最高的10个特征:")
        top_avg_corr = avg_corr.nlargest(10, avg_corr.columns[0])
        for idx, (feature, avg_corr_value) in enumerate(top_avg_corr.iterrows(), 1):
            print(f"  {idx}. {feature}: {avg_corr_value.iloc[0]:.3f}")
    
    # 7. 生成综合分析报告
    generate_comprehensive_analysis()

def generate_comprehensive_analysis():
    """生成综合分析报告"""
    
    output_dir = "./output"
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 收集所有分析结果
    analysis_results = {
        "timestamp": report_time,
        "basic_info": "",
        "cleaning_info": "",
        "correlation_analysis": "",
        "target_correlation": "",
        "feature_stats": "",
        "feature_importance": ""
    }
    
    # 读取基本数据信息
    basic_info_files = glob.glob(f"{output_dir}/basic_info_*.txt")
    if basic_info_files:
        with open(basic_info_files[0], 'r', encoding='utf-8') as f:
            analysis_results["basic_info"] = f.read()
    
    # 读取数据清洗信息
    cleaning_files = glob.glob(f"{output_dir}/data_cleaning_info_*.txt")
    if cleaning_files:
        with open(cleaning_files[0], 'r', encoding='utf-8') as f:
            analysis_results["cleaning_info"] = f.read()
    
    # 生成详细分析报告
    report_content = f"""
数据集特征综合分析报告
生成时间: {report_time}
=====================================

一、数据集概览
-----------------
{analysis_results["basic_info"]}

二、数据质量分析
-----------------
{analysis_results["cleaning_info"]}

三、特征工程建议
-----------------
"""
    
    # 添加特征相关性建议
    corr_matrix_files = glob.glob(f"{output_dir}/correlation_matrix_*.csv")
    if corr_matrix_files:
        corr_matrix = pd.read_csv(corr_matrix_files[0], index_col=0)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        report_content += f"1. 特征相关性分析\n"
        report_content += f"   - 发现 {len(high_corr_pairs)} 对高度相关的特征 (相关系数 > 0.8)\n"
        if high_corr_pairs:
            report_content += "   - 建议移除以下高度相关的特征以减少冗余:\n"
            for pair in sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)[:10]:
                report_content += f"     {pair['feature1']} - {pair['feature2']} (相关性: {pair['correlation']:.3f})\n"
    
    # 添加特征重要性建议
    target_corr_files = glob.glob(f"{output_dir}/feature_target_correlations.csv")
    if target_corr_files:
        target_corr = pd.read_csv(target_corr_files[0], index_col=0)
        top_features = target_corr.nlargest(10, target_corr.columns[0])
        
        report_content += f"\n2. 特征重要性分析\n"
        report_content += f"   - 与目标变量最相关的10个特征:\n"
        for idx, (feature, corr) in enumerate(top_features.iterrows(), 1):
            report_content += f"     {idx}. {feature}: {corr.iloc[0]:.3f}\n"
        
        report_content += "   - 建议优先使用这些特征进行建模\n"
    
    # 添加数据预处理建议
    report_content += f"""
3. 数据预处理建议
   - 已完成的预处理:
     * 缺失值处理: 使用众数填充
     * 数据标准化: 提供MinMax和Z-score两种标准化方法
     * 特征分箱: 对X65、X1、X66进行了分箱处理
   
   - 进一步建议:
     * 考虑特征选择: 基于相关性分析选择重要特征
     * 特征编码: 对分箱特征进行独热编码
     * 特征交互: 创建特征交互项提升模型表现

四、建模策略建议
-----------------
1. 模型选择:
   - 逻辑回归(LR): 适合线性关系，需要特征标准化
   - GBDT: 适合非线性关系，对特征尺度不敏感
   - LightGBM: 高效集成树模型，适合大规模数据

2. 特征工程策略:
   - 对LR模型: 使用标准化特征，移除高度相关特征
   - 对树模型: 可直接使用原始特征，保留分箱特征

3. 评估指标:
   - 主要指标: AUC (Area Under ROC Curve)
   - 辅助指标: 准确率、精确率、召回率、F1分数
"""
    
    # 保存报告
    report_file = f"{output_dir}/dataset_feature_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n综合分析报告已保存到: {report_file}")
    
    return report_content

def create_implementation_plan():
    """根据分析结果创建风险评估实现方案"""
    
    # 读取特征分析结果
    analysis_results = generate_comprehensive_analysis()
    
    # 创建实现方案
    plan_content = f"""# 风险评估模型实现方案

## 基于数据集特征分析的风控模型实施方案

### 一、数据准备阶段

#### 1.1 数据预处理（已完成）
- 缺失值处理：使用众数填充
- 数据标准化：提供MinMax和Z-score两种方法
- 特征分箱：对X65、X1、X66进行分箱处理
- 特征选择：基于相关性分析选择重要特征

#### 1.2 特征工程策略
**针对逻辑回归模型：**
- 使用标准化特征（Z-score或MinMax）
- 移除高度相关的特征对（相关系数>0.8）
- 对分箱特征进行独热编码

**针对树模型（GBDT/LightGBM）：**
- 可直接使用原始数值特征
- 保留分箱特征作为类别特征
- 考虑特征交互项的创建

### 二、模型设计与实现

#### 2.1 模型选择
1. **逻辑回归（LR）**
   - 优点：解释性强，计算效率高
   - 适用场景：线性关系明显的数据
   - 实现：sklearn.linear_model.LogisticRegression

2. **梯度提升决策树（GBDT）**
   - 优点：处理非线性关系能力强
   - 适用场景：复杂特征交互
   - 实现：sklearn.ensemble.GradientBoostingClassifier

3. **LightGBM**
   - 优点：训练速度快，内存占用低
   - 适用场景：大规模数据集
   - 实现：lightgbm.LGBMClassifier

#### 2.2 模型训练策略
- **数据划分**：70%训练集，15%验证集，15%测试集
- **交叉验证**：使用5折交叉验证调参
- **超参数优化**：网格搜索或随机搜索

### 三、评估指标与结果分析

#### 3.1 主要评估指标
- **AUC（主要指标）**：衡量模型整体分类能力
- 准确率、精确率、召回率、F1分数

#### 3.2 可选实现内容
1. **手写AUC实现**：对比sklearn结果差异
2. **手写逻辑回归**：理解算法原理
3. **归一化影响分析**：研究不同模型对特征归一化的敏感性

### 四、实施步骤

#### 阶段一：基础模型实现（第1周）
1. 环境配置：安装sklearn、lightgbm等库
2. 数据加载与预处理
3. 实现基础LR和GBDT模型
4. 基础AUC评估

#### 阶段二：模型优化（第2周）
1. 特征工程优化
2. 超参数调优
3. LightGBM模型实现
4. 模型对比分析

#### 阶段三：高级功能（第3周，可选）
1. 手写AUC函数实现
2. 手写逻辑回归
3. 归一化影响分析报告

### 五、预期成果

#### 5.1 必做内容
- 三种模型的AUC结果对比
- 测试集预测结果（CSV文件）
- 实验报告文档

#### 5.2 可选内容
- 自定义AUC实现与对比
- 自定义逻辑回归实现
- 归一化技术分析报告

### 六、技术要点

#### 6.1 特征处理要点
- 基于特征相关性分析进行特征选择
- 针对不同模型采用不同的特征预处理策略
- 利用分箱特征提升模型表现

#### 6.2 模型调优要点
- 使用交叉验证避免过拟合
- 针对不同模型特点设置合适的超参数搜索范围
- 关注模型在验证集上的表现

### 七、参考资料

1. sklearn官方文档
2. LightGBM官方文档
3. 特征工程最佳实践
4. 模型评估指标详解

---
*本方案基于数据集特征分析结果制定，可根据实际实验进展进行调整。*
"""
    
    # 保存实现方案
    plan_file = "../风险评估/implementation-plan.md"
    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write(plan_content)
    
    print(f"实现方案已保存到: {plan_file}")
    
    return plan_content

if __name__ == "__main__":
    # 分析数据集特征
    analyze_dataset_features()
    
    # 创建实现方案
    print("\n" + "="*60)
    print("开始创建风险评估实现方案...")
    create_implementation_plan()
    print("分析完成！")