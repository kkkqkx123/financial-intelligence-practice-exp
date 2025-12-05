import pandas as pd   #导入pandas重命名为pd，后面如果要使用就是pd.xxxxx,如果没有重命名及时pandas.xxxxx
import numpy as np    #导入numpy重命名为np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 创建输出目录
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# 获取当前时间戳用于文件命名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

datas = pd.read_csv("./data/train_new.csv") # 将数据文件读入自定义的变量datas
datas_bk = datas.copy()  #复制原数据备用

# 保存基本数据信息到文件
basic_info = f"""
数据集基本信息:
- 数据形状: {datas.shape}
- 列名: {list(datas.columns)}
- 数据类型:\n{datas.dtypes}
"""
with open(f"{output_dir}/basic_info_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(basic_info)

datas.head(10)   #查看前10行

datas = datas.iloc[:, :-1] #调用iloc函数去掉最后一列的id
datas.head(10)

feature = pd.read_csv("./data/feature_x.csv",index_col = 0) # 将数据文件读入自定义的变量datas
feature

# data.isnull()会用bool值填充每个数据，数据不为空填充False，数据为空填充True
ct_nan = datas.isnull()
ct_nan.head(10)

# 先isnull查找缺失值，再sum求和
ct_nan=datas.isnull().sum()
ct_nan

# 先sinul查找缺失值，再sum求，最后排序
datas.isnull().sum().sort_values(ascending=False)

# 计算特征之间、特征与Label的相关度
correlation_matrix = datas.corr()  #直接调用，默认使用皮尔森相关系数

# 保存相关矩阵到CSV文件
correlation_matrix.to_csv(f"{output_dir}/correlation_matrix_{timestamp}.csv")

# 生成并保存热力图
plt.figure(figsize=(15,15)) #图片大小
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.savefig(f"{output_dir}/correlation_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(15,15)) #图片大小
sns.heatmap(correlation_matrix, cmap='RdBu', center=0)
plt.title('特征相关性热力图 (RdBu色图)')
plt.savefig(f"{output_dir}/correlation_heatmap_rdbu_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()

# 分析缺失值并保存结果
missing_values_col = datas.isnull().sum().sort_values(ascending=False)  #按列
missing_values_row = datas.isnull().sum(axis=1).sort_values(ascending=False)  #按行

# 保存缺失值分析结果
missing_analysis = f"""
缺失值分析结果:

按列统计的缺失值数量:
{missing_values_col}

按行统计的缺失值数量 (前20行):
{missing_values_row.head(20)}
"""
with open(f"{output_dir}/missing_values_analysis_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(missing_analysis)

# 保留至少含有50个非NaN feature的样本点
original_shape = datas.shape
datas = datas.dropna(thresh=50)
new_shape = datas.shape

# 保存数据清洗结果
cleaning_info = f"""
数据清洗结果:
- 原始数据形状: {original_shape}
- 清洗后数据形状: {new_shape}
- 删除的行数: {original_shape[0] - new_shape[0]}
"""
with open(f"{output_dir}/data_cleaning_info_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(cleaning_info)

datas.head()

datas.fillna(-1).head()   #先执行fillna函数，然后head

datas.fillna(method='backfill').fillna(method='ffill').head()

datas.fillna(datas.median()).head()

datas.fillna(datas.mean()).head()

datas.mode()

datas.fillna(datas.mode().iloc[0]).head()

# 比较不同的缺失值填充方法
fill_methods = {
    'fill_minus1': datas.fillna(-1),
    'fill_bfill_ffill': datas.fillna(method='backfill').fillna(method='ffill'),
    'fill_median': datas.fillna(datas.median()),
    'fill_mean': datas.fillna(datas.mean()),
    'fill_mode': datas.fillna(datas.mode().iloc[0])
}

# 保存填充方法比较
fill_comparison = {}
for method_name, filled_data in fill_methods.items():
    remaining_missing = filled_data.isnull().sum().sum()
    fill_comparison[method_name] = remaining_missing

fill_comparison_df = pd.DataFrame(list(fill_comparison.items()), columns=['填充方法', '剩余缺失值总数'])
fill_comparison_df.to_csv(f"{output_dir}/fill_methods_comparison_{timestamp}.csv", index=False)

# 选择最佳填充方法（模式填充）
data_fillna = fill_methods['fill_mode']

# 找到所有数值属性，本数据集中，由于已经做过数字映射，因此特征均为数字属性
X_scale = data_fillna.copy()
numeriic_feats = X_scale.dtypes[X_scale.dtypes != 'object'].index

# 使用最大最小值规范
X_scale[numeriic_feats] = X_scale[numeriic_feats].apply(lambda x: (x-x.min())/(x.max()-x.min()))

# 标准正规化
X_norm = data_fillna.copy()
X_norm[numeriic_feats] = X_norm[numeriic_feats].apply(lambda x:(x-x.mean())/(x.std())) 

# 保存标准化后的数据
X_scale.to_csv(f"{output_dir}/scaled_data_minmax_{timestamp}.csv", index=False)
X_norm.to_csv(f"{output_dir}/normalized_data_standard_{timestamp}.csv", index=False)

# 生成数据分布对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 原始数据分布（选择几个特征作为示例）
sample_features = numeriic_feats[:4] if len(numeriic_feats) >= 4 else numeriic_feats
for i, feat in enumerate(sample_features):
    row, col = i // 2, i % 2
    data_fillna[feat].hist(bins=30, alpha=0.7, ax=axes[row, col])
    axes[row, col].set_title(f'原始数据 - {feat}')
    axes[row, col].set_xlabel('值')
    axes[row, col].set_ylabel('频次')

plt.tight_layout()
plt.savefig(f"{output_dir}/original_data_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()

# 标准化数据分布对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, feat in enumerate(sample_features):
    row, col = i // 2, i % 2
    X_norm[feat].hist(bins=30, alpha=0.7, ax=axes[row, col], color='orange')
    axes[row, col].set_title(f'标准化数据 - {feat}')
    axes[row, col].set_xlabel('值')
    axes[row, col].set_ylabel('频次')

plt.tight_layout()
plt.savefig(f"{output_dir}/normalized_data_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()

X = data_fillna.copy()

# 分箱操作并保存结果
binning_results = {}

# X65分箱 - 等频分箱
X['X65_bin'] = pd.qcut(X["X65"], q=10, duplicates='drop')
binning_results['X65'] = X['X65_bin'].value_counts().sort_index()

# X1分箱 - 等宽分箱
X['X1_bin'] = pd.cut(X.X1, bins=[2,4,6,8,10,12])
binning_results['X1'] = X['X1_bin'].value_counts().sort_index()

# X66分箱 - 自定义分箱
X['X66_bin'] = pd.cut(X.X66, bins=[-3,-2,-1,0,1,2,3])
binning_results['X66'] = X['X66_bin'].value_counts().sort_index()

# 保存分箱结果到文件
with open(f"{output_dir}/binning_results_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write("特征分箱结果:\n\n")
    for feature, counts in binning_results.items():
        f.write(f"{feature} 分箱结果:\n")
        f.write(f"{counts}\n\n")

# 生成分箱可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# X65分箱可视化
binning_results['X65'].plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('X65 等频分箱分布')
axes[0].set_xlabel('分箱区间')
axes[0].set_ylabel('样本数量')
axes[0].tick_params(axis='x', rotation=45)

# X1分箱可视化
binning_results['X1'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('X1 等宽分箱分布')
axes[1].set_xlabel('分箱区间')
axes[1].set_ylabel('样本数量')
axes[1].tick_params(axis='x', rotation=45)

# X66分箱可视化
binning_results['X66'].plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('X66 自定义分箱分布')
axes[2].set_xlabel('分箱区间')
axes[2].set_ylabel('样本数量')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/binning_visualization_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()

# 保存最终处理后的数据
X.to_csv(f"{output_dir}/final_processed_data_{timestamp}.csv", index=False)

# 生成最终数据摘要
final_summary = f"""
数据处理完成摘要:
- 原始数据形状: {datas_bk.shape}
- 清洗后数据形状: {datas.shape}
- 最终处理数据形状: {X.shape}
- 使用的填充方法: 众数填充
- 标准化方法: MinMax标准化和Z-score标准化
- 分箱特征: X65(等频), X1(等宽), X66(自定义)

输出文件列表:
- 基本数据信息: basic_info_{timestamp}.txt
- 相关矩阵: correlation_matrix_{timestamp}.csv
- 相关性热力图: correlation_heatmap_{timestamp}.png, correlation_heatmap_rdbu_{timestamp}.png
- 缺失值分析: missing_values_analysis_{timestamp}.txt
- 数据清洗信息: data_cleaning_info_{timestamp}.txt
- 填充方法比较: fill_methods_comparison_{timestamp}.csv
- 标准化数据: scaled_data_minmax_{timestamp}.csv, normalized_data_standard_{timestamp}.csv
- 数据分布图: original_data_distribution_{timestamp}.png, normalized_data_distribution_{timestamp}.png
- 分箱结果: binning_results_{timestamp}.txt, binning_visualization_{timestamp}.png
- 最终处理数据: final_processed_data_{timestamp}.csv
"""

with open(f"{output_dir}/processing_summary_{timestamp}.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)

print(f"数据处理完成！所有结果已保存到 {output_dir} 目录中")
print(f"处理摘要已保存到: processing_summary_{timestamp}.txt")