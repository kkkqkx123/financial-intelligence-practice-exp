# CSV数据实体抽取框架

基于 `financial_ie_scheme.py` 重构的CSV数据实体抽取框架，专门处理金融投资领域的多个数据集。

## 架构设计

### 1. 基类设计 (`base_extractor.py`)
- **BaseEntityExtractor**: 抽象基类，提供通用的CSV读取和实体抽取框架
- **TextProcessor**: 文本处理工具类，提供通用的文本处理功能

### 2. 具体抽取器
- **InvestmentStructureExtractor**: 投资结构数据抽取器
- **InvestmentEventsExtractor**: 投资事件数据抽取器  
- **CompanyDataExtractor**: 公司数据抽取器

### 3. 主程序 (`main.py`)
- **EntityExtractionPipeline**: 统一的实体抽取管道，支持批量处理和单个数据集处理

## 功能特性

### 支持的实体类型

#### 投资结构数据
- 投资机构
- 行业
- 投资轮次
- 投资规模
- 投资时间
- 地区

#### 投资事件数据
- 投资方
- 融资方
- 投资轮次
- 投资金额
- 投资时间
- 事件资讯

#### 公司数据
- 公司名称
- 行业
- 地区
- 法人代表
- 注册资金
- 成立时间
- 公司介绍

### 核心功能

1. **智能文本解析**
   - 年份提取
   - 金额识别
   - 地点识别
   - 行业关键词匹配

2. **数据标准化**
   - 投资轮次标准化
   - 行业分类标准化
   - 金额单位统一

3. **批量处理**
   - 支持多个数据集同时处理
   - 统一的输出格式
   - 详细的处理统计

4. **错误处理**
   - 健壮的错误处理机制
   - 详细的错误日志
   - 部分失败不影响整体处理

## 使用方法

### 1. 处理所有数据集
```bash
cd src/csv_processing
python main.py --dataset-dir ../dataset --output-dir ../extraction_results/entities
```

### 2. 处理单个数据集
```bash
# 处理投资结构数据
python main.py --dataset investment_structure

# 处理投资事件数据
python main.py --dataset investment_events

# 处理公司数据
python main.py --dataset company_data
```

### 3. 自定义路径
```bash
python main.py --dataset-dir /path/to/datasets --output-dir /path/to/output
```

## 输出格式

### CSV文件格式
所有输出文件遵循统一的CSV格式：
```csv
entity_name,entity_type,entity_name
"红杉资本","投资机构","红杉资本"
"企业服务","行业","企业服务"
"A轮","投资轮次","A轮"
```

### 统计报告
生成详细的处理汇总报告，包含：
- 各数据集处理状态
- 实体类型分布统计
- 处理耗时统计
- 错误信息汇总

## 扩展指南

### 添加新的抽取器

1. 继承 `BaseEntityExtractor` 基类
2. 实现必要的方法：
   - `get_entity_types()`: 返回支持的实体类型列表
   - `extract_entities()`: 实现具体的实体抽取逻辑
   - `get_output_filename()`: 返回输出文件名

3. 在主程序中注册新的抽取器

### 示例代码

```python
from base_extractor import BaseEntityExtractor

class MyCustomExtractor(BaseEntityExtractor):
    def get_entity_types(self) -> List[str]:
        return ['实体类型1', '实体类型2']
    
    def get_output_filename(self) -> str:
        return 'my_custom_entities.csv'
    
    def extract_entities(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 实现具体的抽取逻辑
        # ...
        
        return entities
```

## 性能优化

1. **批量处理**: 每100条记录输出一次进度
2. **内存管理**: 流式处理大文件
3. **错误恢复**: 单条记录失败不影响整体处理
4. **并行处理**: 支持多数据集并行处理

## 质量保证

1. **数据验证**: 严格的输入数据验证
2. **实体去重**: 自动去除重复实体
3. **标准化处理**: 统一的实体命名规范
4. **完整性检查**: 确保所有字段都被正确处理

## 依赖要求

- Python 3.7+
- pandas >= 1.3.0
- 标准库：re, os, sys, argparse, datetime, abc, typing

## 错误处理

框架提供了完善的错误处理机制：
- 文件读取错误
- 数据格式错误
- 编码问题
- 权限问题

所有错误都会被记录，并继续处理其他数据集。