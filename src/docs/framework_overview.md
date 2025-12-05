# 金融数据实体抽取框架文档

## 概述

本项目提供了两个互补的金融数据实体抽取框架，用于从CSV格式的金融数据中提取结构化实体信息。

## 模块说明

### 1. CSV处理框架 (`src/csv_processing/`)

**作用**: 提供全新的实体抽取架构，采用面向对象设计，支持高度定制化的实体抽取需求。

**核心组件**:
- `BaseEntityExtractor`: 抽象基类，定义标准接口
- `InvestmentStructureExtractor`: 投资结构数据专用抽取器
- `InvestmentEventsExtractor`: 投资事件数据专用抽取器  
- `CompanyDataExtractor`: 公司数据专用抽取器

**特点**:
- 模块化设计，易于扩展和维护
- 每个抽取器针对特定数据类型优化
- 支持复杂的实体抽取规则
- 详细的统计报告和错误处理

### 2. 实体抽取框架 (`src/entity_extraction/`)

**作用**: 从金融文本数据中识别和提取结构化实体信息，返回实体名称到类别列表的映射，通过适配器模式复用现有的`financial_ie_scheme.py`中的实体抽取逻辑，保持向后兼容性。

**核心组件**:
- `FinancialEntityExtractor`: 复用现有抽取逻辑的核心类
- `FinancialEntityAdapter`: 适配器，将现有逻辑集成到新框架，统一的实体抽取适配器，自动识别数据类型并调用相应的抽取器
- `EntityExtractionPipeline`: 处理管道，管理整个抽取流程
- 实体标准化：对提取的实体进行格式化和标准化处理

**输出格式**: 实体名称 → 类别列表的映射字典
```python
{
    "红杉资本中国基金": ["投资机构"],
    "企业服务": ["行业"], 
    "A轮": ["投资轮次"],
    "1000万美元": ["投资规模"],
    "北京": ["地区"]
}
```

**特点**:
- 复用经过验证的实体抽取算法
- 无缝集成到新的CSV处理架构
- 自动识别数据类型并调用相应方法
- 保持与原有代码的兼容性

## 实体类型支持

两个框架都支持以下实体类型的抽取：

| 实体类型 | 说明 | 示例 |
|---------|------|------|
| 投资机构 | 投资方、投资机构名称 | 红杉资本、腾讯投资 |
| 被投企业 | 融资方、公司名称 | 阿里巴巴、字节跳动 |
| 投资轮次 | 融资轮次信息 | A轮、B轮、C轮 |
| 投资规模 | 投资金额、融资规模 | 1000万美元、5亿人民币 |
| 投资时间 | 融资时间、成立时间 | 2023年、2021年6月 |
| 地区 | 公司地址、所在地 | 北京、上海、深圳 |
| 行业 | 公司所属行业 | 科技、金融、医疗 |

## 快速开始

### 环境验证

```bash
# 验证数据集文件是否存在
cd src/entity_extraction
python main.py --validate-only
```

### 处理所有数据集

```bash
# 方法1: 使用CSV处理框架
cd src/csv_processing
python main.py --dataset all

# 方法2: 使用实体抽取框架（复用现有逻辑）
cd src/entity_extraction
python main.py --dataset all
```

### 处理单个数据集

```bash
# 处理投资结构数据
python main.py --dataset investment_structure

# 处理投资事件数据
python main.py --dataset investment_events

# 处理公司数据
python main.py --dataset company_data
```

## 输出格式

所有输出文件都遵循统一的CSV格式：

```python
{
    "红杉资本": ["投资机构"],
    "阿里巴巴": ["被投企业"],
    "A轮": ["投资轮次"],
    "1000万美元": ["投资规模"],
    "2023年": ["投资时间"],
    "北京": ["地区"],
    "科技": ["行业"]
}
```

## 输出目录

**CSV处理框架**输出：
- 实体抽取结果：`output/[extractor_name]/[extractor_name]_entities.pkl`
- 格式：pickle格式的字典，键为实体名称，值为类别列表
- 统计报告：`output/[extractor_name]/statistics_[timestamp].json`

**实体抽取框架**输出：
- 实体抽取结果：`output/financial_entities.pkl`
- 格式：pickle格式的字典，键为实体名称，值为类别列表
- 处理日志：`output/financial_extraction_[timestamp].log`

## 选择建议

### 使用CSV处理框架的情况：
- 需要高度定制化的实体抽取规则
- 计划添加新的实体类型或数据集
- 希望采用面向对象的设计模式
- 需要更灵活的扩展性

### 使用实体抽取框架的情况：
- 希望复用现有的、经过验证的抽取逻辑
- 需要快速部署和使用
- 重视与原有代码的兼容性
- 主要处理标准的三种数据集类型

## 扩展开发

### 扩展开发

#### 添加新的实体类型

1. **CSV处理框架**: 在相应的抽取器类中添加新的抽取方法，返回实体名称→类别列表的映射
2. **实体抽取框架**: 在`FinancialEntityExtractor`中添加新的抽取逻辑，确保输出格式为实体名称→类别列表的字典

#### 添加新的数据集支持

1. **CSV处理框架**: 创建新的抽取器类继承`BaseEntityExtractor`
   - 实现`extract_entities()`方法，返回实体名称到类别列表的映射
   - 实现`get_entity_types()`方法，返回支持的实体类型列表
   - 实现`get_output_filename()`方法，定义输出文件名（应为.pkl后缀）
2. **实体抽取框架**: 在`FinancialEntityExtractor`中实现新的抽取方法，确保输出格式为实体名称→类别列表的字典，并使用pickle格式保存结果

## 注意事项

1. **数据格式**: 确保输入CSV文件使用UTF-8编码
2. **文件路径**: 默认数据集文件应在`src/dataset/`目录下
3. **输出目录**: 确保`src/extraction_results/entities/`目录有写入权限
4. **性能优化**: 对于大文件，考虑分批处理或增加内存

## 故障排除

### 故障排除

1. **常见问题**

1. **文件不存在错误**: 检查数据集文件是否在正确位置
2. **编码错误**: 确认CSV文件使用UTF-8编码
3. **权限错误**: 确保有输出目录的写入权限
4. **内存不足**: 考虑分批处理大文件
5. **pickle文件问题**: 确保有读取pickle文件的权限，使用合适的Python版本

**调试技巧**:
- 使用`--debug`参数启用详细日志
- 检查`output/`目录下的日志文件
- 验证输入CSV文件的格式和编码
- 测试pickle文件是否正确生成和加载

**输出格式验证**:
- 确认输出为pickle格式的字典文件
- 验证字典键为实体名称，值为类别列表
- 检查实体名称是否已去重和清理

### 日志信息

程序运行时会生成详细的日志信息，包括处理进度、错误信息和统计结果，便于问题排查和性能分析。

## 相关文档

- [CSV处理框架详细文档](../csv_processing/README.md)
- [实体抽取框架详细文档](../entity_extraction/README.md)
- [项目实现总结](../project_summary.md)