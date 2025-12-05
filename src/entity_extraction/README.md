# 金融实体抽取框架

基于 `financial_ie_scheme.py` 构建的金融数据实体抽取框架，复用现有实体抽取逻辑，适配新的CSV处理架构。

## 架构设计

### 核心组件

1. **FinancialEntityExtractor** (`financial_entity_extractor.py`)
   - 复用现有 `financial_ie_scheme.py` 的实体抽取逻辑
   - 提供三个主要抽取方法：
     - `extract_from_investment_structure()`: 处理投资结构数据
     - `extract_from_investment_events()`: 处理投资事件数据
     - `extract_from_company_data()`: 处理公司数据

2. **FinancialEntityAdapter** (`financial_entity_adapter.py`)
   - 适配器模式，将现有的 `FinancialEntityExtractor` 适配到新的CSV处理框架
   - 继承自 `BaseEntityExtractor`，实现标准接口
   - 自动识别数据类型并调用相应的抽取方法

3. **EntityExtractionPipeline** (`main.py`)
   - 实体抽取管道，管理整个抽取流程
   - 支持批量处理所有数据集或单个数据集处理
   - 提供错误处理和汇总报告功能

### 支持的实体类型

- **投资机构**: 投资方、投资机构名称
- **被投企业**: 融资方、公司名称
- **投资轮次**: 种子轮、天使轮、A轮、B轮、C轮等
- **投资规模**: 投资金额、融资规模
- **投资时间**: 融资时间、成立时间
- **地区**: 公司地址、所在地
- **行业**: 公司所属行业

## 使用方法

### 1. 处理所有数据集

```bash
# 进入实体抽取目录
cd src/entity_extraction

# 处理所有数据集
python main.py --dataset all

# 或者简写
python main.py
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

### 3. 使用自定义输入文件

```bash
# 处理自定义的投资结构数据
python main.py --dataset investment_structure --input-file /path/to/your/file.csv

# 处理自定义的公司数据
python main.py --dataset company_data --input-file /path/to/your/file.csv
```

### 4. 指定输出目录

```bash
# 指定自定义输出目录
python main.py --dataset all --output-dir /path/to/output/dir
```

### 5. 环境验证

```bash
# 仅验证环境，不执行处理
python main.py --validate-only
```

## 输出格式

### 输出文件

处理结果将保存到 `src/extraction_results/entities/` 目录下：

- `investment_structure_entities.csv`: 投资结构数据实体抽取结果
- `investment_events_entities.csv`: 投资事件数据实体抽取结果
- `company_data_entities.csv`: 公司数据实体抽取结果
- `extraction_summary_report.txt`: 处理汇总报告

### CSV文件格式

所有输出文件都遵循统一的格式：

```csv
entity_name,entity_type,entity_name
红杉资本,投资机构,红杉资本
腾讯,投资机构,腾讯
阿里巴巴,被投企业,阿里巴巴
A轮,投资轮次,A轮
10亿美元,投资规模,10亿美元
2023年,投资时间,2023年
北京,地区,北京
科技,行业,科技
```

### 实体类型标准化

- **投资轮次标准化**:
  - 种子轮、天使轮、Pre-A、A轮、B轮、C轮、D轮及以后
  - IPO、上市后、并购、战略投资

- **行业标准化**:
  - 科技、金融、医疗、教育、电商、制造、房地产、汽车、文娱、餐饮
  - 其他（无法识别的行业）

## 扩展指南

### 添加新的实体类型

1. 在 `FinancialEntityExtractor` 中添加新的抽取方法
2. 更新 `get_supported_entity_types()` 方法
3. 在相应的抽取函数中实现新的实体识别逻辑

### 添加新的数据集支持

1. 在 `EntityExtractionPipeline` 的 `dataset_configs` 中添加新配置
2. 在 `FinancialEntityExtractor` 中实现对应的抽取方法
3. 在 `FinancialEntityAdapter` 的 `extract_entities()` 中添加数据类型识别逻辑

### 自定义实体抽取规则

可以通过继承 `FinancialEntityExtractor` 并重写相应的方法来实现自定义的实体抽取规则：

```python
class CustomFinancialEntityExtractor(FinancialEntityExtractor):
    def _parse_industries(self, industries_str: str) -> List[str]:
        # 自定义行业解析逻辑
        pass
    
    def _standardize_industry(self, industry_name: str) -> str:
        # 自定义行业标准化逻辑
        pass
```

## 性能优化

### 批量处理优化

- 使用 pandas 的向量化操作处理大批量数据
- 实现增量处理机制，支持断点续处理
- 提供进度条显示处理进度

### 内存优化

- 使用分块读取处理大文件
- 及时清理不再使用的中间结果
- 优化正则表达式匹配效率

## 质量保证

### 数据验证

- 输入数据格式验证
- 实体抽取结果完整性检查
- 输出文件格式验证

### 错误处理

- 完善的异常捕获和处理机制
- 详细的错误日志记录
- 处理失败时的恢复机制

### 测试覆盖

- 单元测试覆盖核心抽取逻辑
- 集成测试验证端到端流程
- 性能测试确保处理效率

## 依赖要求

```txt
pandas>=1.3.0
numpy>=1.21.0
```

## 错误处理

### 常见错误及解决方案

1. **文件不存在错误**
   - 确保数据集文件存在于 `src/dataset/` 目录下
   - 检查文件路径和名称是否正确

2. **编码错误**
   - 确保CSV文件使用UTF-8编码
   - 如遇编码问题，可尝试使用其他编码格式

3. **内存不足错误**
   - 对于大文件，考虑分批处理
   - 增加系统内存或使用更高配置的环境

4. **实体抽取失败**
   - 检查数据格式是否符合预期
   - 查看日志文件了解详细错误信息

### 日志信息

程序运行时会生成详细的日志信息，包括：

- 处理开始和结束时间
- 每个数据集的处理进度
- 实体抽取统计信息
- 错误和警告信息

日志信息同时输出到控制台和日志文件，便于问题排查和性能分析。