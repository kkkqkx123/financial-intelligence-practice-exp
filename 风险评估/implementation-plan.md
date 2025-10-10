# 风险评估模型实现方案

## 基于数据集特征分析的风控模型实施方案

### 📁 项目目录结构
```
风险评估/
├── data/                           # 数据文件
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后数据
│   └── features/                  # 特征工程结果
├── models/                        # 模型文件
│   ├── lr/                        # 逻辑回归模型
│   ├── gbdt/                      # GBDT模型
│   └── lightgbm/                  # LightGBM模型
├── results/                       # 实验结果
│   ├── predictions/               # 预测结果
│   ├── metrics/                   # 评估指标
│   └── reports/                   # 实验报告
├── logs/                          # 日志文件
│   ├── training/                  # 训练日志
│   ├── evaluation/                # 评估日志
│   └── experiments/               # 实验日志
├── src/                           # 源代码
│   ├── data/                      # 数据处理模块
│   ├── models/                    # 模型实现模块
│   ├── utils/                     # 工具函数
│   └── evaluation/                # 评估模块
└── notebooks/                     # Jupyter笔记本
    ├── experiments/               # 实验记录
    └── analysis/                  # 分析结果
```

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
- **超参数优化**：网格搜索（参数组合≤20个）

#### 2.3 超参数搜索空间（网格优化）
**逻辑回归（LR）：**
```python
param_grid_lr = {
    'C': [0.1, 1, 10],                    # 正则化强度（3个值）
    'penalty': ['l1', 'l2'],              # 正则化类型（2个值）
    'solver': ['liblinear']               # 求解器（1个值）
}
# 总组合数：3 × 2 × 1 = 6个
```

**GBDT：**
```python
param_grid_gbdt = {
    'n_estimators': [50, 100],            # 基学习器数量（2个值）
    'learning_rate': [0.1, 0.3],          # 学习率（2个值）
    'max_depth': [3, 5]                   # 最大深度（2个值）
}
# 总组合数：2 × 2 × 2 = 8个
```

**LightGBM：**
```python
param_grid_lgb = {
    'n_estimators': [50, 100],             # 基学习器数量（2个值）
    'learning_rate': [0.1, 0.3],           # 学习率（2个值）
    'num_leaves': [15, 31]                 # 叶子节点数（2个值）
}
# 总组合数：2 × 2 × 2 = 8个
```

### 三、评估指标与结果分析

#### 3.1 主要评估指标
- **AUC（主要指标）**：衡量模型整体分类能力
- 准确率、精确率、召回率、F1分数

#### 3.2 可选实现内容
1. **手写AUC实现**：对比sklearn结果差异
2. **手写逻辑回归**：理解算法原理
3. **归一化影响分析**：研究不同模型对特征归一化的敏感性

### 四、日志系统

#### 4.1 日志文件结构
```
logs/
├── training/
│   ├── lr_training_YYYYMMDD_HHMMSS.log     # 逻辑回归训练日志
│   ├── gbdt_training_YYYYMMDD_HHMMSS.log   # GBDT训练日志
│   └── lgb_training_YYYYMMDD_HHMMSS.log    # LightGBM训练日志
├── evaluation/
│   ├── lr_eval_YYYYMMDD_HHMMSS.log        # 逻辑回归评估日志
│   ├── gbdt_eval_YYYYMMDD_HHMMSS.log      # GBDT评估日志
│   └── lgb_eval_YYYYMMDD_HHMMSS.log       # LightGBM评估日志
└── experiments/
    ├── feature_engineering_YYYYMMDD_HHMMSS.log  # 特征工程日志
    └── hyperparameter_YYYYMMDD_HHMMSS.log       # 超参数调优日志
```

#### 4.2 日志记录内容
- **训练日志**：模型参数、训练时间、交叉验证结果、最佳参数
- **评估日志**：评估指标、预测结果统计、模型性能对比
- **实验日志**：特征工程步骤、参数搜索过程、实验结论

#### 4.3 日志实现方式
```python
import logging
import datetime

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 五、实施步骤

#### 阶段一：基础模型实现（第1周）
1. 环境配置：安装sklearn、lightgbm等库
2. 创建项目目录结构和日志系统
3. 数据加载与预处理
4. 实现基础LR和GBDT模型
5. 基础AUC评估与日志记录

#### 阶段二：模型优化（第2周）
1. 特征工程优化与日志记录
2. 超参数调优（网格搜索）与日志记录
3. LightGBM模型实现与日志记录
4. 模型对比分析与结果汇总

#### 阶段三：高级功能（第3周，可选）
1. 手写AUC函数实现
2. 手写逻辑回归
3. 归一化影响分析报告

### 六、预期成果

#### 6.1 必做内容
- **模型文件**：保存最优模型到对应目录（models/lr/, models/gbdt/, models/lightgbm/）
- **预测结果**：测试集预测结果保存到 results/predictions/
- **评估指标**：AUC等评估指标保存到 results/metrics/
- **实验报告**：详细实验报告保存到 results/reports/
- **日志文件**：完整的训练和评估日志保存到 logs/

#### 6.2 可选内容
- 自定义AUC实现与对比
- 自定义逻辑回归实现
- 归一化技术分析报告

### 七、文件管理规范

#### 7.1 命名规范
- **模型文件**：`{model_name}_best_model_{timestamp}.pkl`
- **预测结果**：`{model_name}_predictions_{timestamp}.csv`
- **评估指标**：`{model_name}_metrics_{timestamp}.json`
- **日志文件**：`{type}_{model_name}_{timestamp}.log`

#### 7.2 清理策略
- 定期清理临时文件和中间结果
- 保留最终模型和最佳结果
- 日志文件保留最近30天

### 八、技术要点

#### 8.1 特征处理要点
- 基于特征相关性分析进行特征选择
- 针对不同模型采用不同的特征预处理策略
- 利用分箱特征提升模型表现

#### 8.2 模型调优要点
- 使用交叉验证避免过拟合
- 针对不同模型特点设置合适的超参数搜索范围（组合数≤20）
- 关注模型在验证集上的表现

#### 8.3 日志管理要点
- 统一日志格式和时间戳
- 分离不同类型的日志（训练、评估、实验）
- 定期清理过期日志文件
- 关键步骤必须记录日志

### 九、参考资料

1. sklearn官方文档
2. LightGBM官方文档
3. 特征工程最佳实践
4. 模型评估指标详解
5. Python日志模块最佳实践

---
*本方案基于数据集特征分析结果制定，优化了目录结构、参数搜索策略和日志系统，可根据实际实验进展进行调整。*
