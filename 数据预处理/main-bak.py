import pandas as pd   #导入pandas重命名为pd，后面如果要使用就是pd.xxxxx,如果没有重命名及时pandas.xxxxx
import numpy as np    #导入numpy重命名为np
import matplotlib.pyplot as plt
import seaborn as sns

datas = pd.read_csv("./data/train_new.csv") # 将数据文件读入自定义的变量datas
datas_bk = datas.copy()  #复制原数据备用

datas.shape  #查看数据大小

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
datas.corr()  #直接调用，默认使用皮尔森相关系数

plt.figure(figsize=(15,15)) #图片大小
sns.heatmap(datas.corr())
plt.show()

plt.figure(figsize=(15,15)) #图片大小
sns.heatmap(datas.corr(), cmap='RdBu')
plt.show()

# 先sinul查找缺失值，再sum求，最后排序
datas.isnull().sum().sort_values(ascending=False)  #按列

datas.isnull().sum(axis=1).sort_values(ascending=False)  #按行

# 保留至少含有50个非NaN feature的样本点
datas = datas.dropna(thresh=50)
datas.shape    #原来的shap是50000，73（删掉了第74列的id）

datas.head()

datas.fillna(-1).head()   #先执行fillna函数，然后head

datas.fillna(method='backfill').fillna(method='ffill').head()

datas.fillna(datas.median()).head()

datas.fillna(datas.mean()).head()

datas.mode()

datas.fillna(datas.mode().iloc[0]).head()

data_fillna = datas.fillna(datas.mode().iloc[0])

data_fillna.head()  #先查看一下原数据

# 找到所有数值属性，本数据集中，由于已经做过数字映射，因此特征均为数字属性
X_scale = data_fillna.copy()
numeriic_feats = X_scale.dtypes[X_scale.dtypes != 'object'].index

# 使用最大最小值规范
X_scale[numeriic_feats] = X_scale[numeriic_feats].apply(lambda x: (x-x.min())/(x.max()-x.min()))

X_scale.head()

X_norm = data_fillna.copy()
# 标准正规化
X_norm[numeriic_feats] = X_norm[numeriic_feats].apply(lambda x:(x-x.mean())/(x.std())) 

X_norm.head()

data_fillna.head()   #查看原始数据便于对比

X = data_fillna.copy()
X.X65_bin = pd.qcut(X["X65"], q=10, duplicates='drop')  #以X65属性分箱
X.X65_bin

X.X65_bin.value_counts()

X.X1_bin = pd.cut(X.X1, bins=[2,4,6,8,10,12])
X.X1_bin

X.X1_bin.value_counts()

X['X66_bin'] = pd.cut(X.X66, bins=[-3,-2,-1,0,1,2,3])
X.X66_bin.value_counts()