# 金融知识图谱构建系统

一个完整的金融知识图谱构建系统，支持从数据加载到知识图谱构建、验证、增强和导出的全流程处理。

## 系统架构

### 核心组件

1. **数据加载模块** (`src/loaders/`)
   - `csv_loader.py`: CSV文件加载器
   - `json_loader.py`: JSON文件加载器
   - `api_loader.py`: API数据加载器
   - `unified_loader.py`: 统一数据加载接口

2. **数据处理模块** (`src/processors/`)
   - `data_parser.py`: 数据解析器
   - `entity_matcher.py`: 实体匹配器
   - `hybrid_kg_builder.py`: 混合知识图谱构建器
   - `data_validator.py`: 数据验证器
   - `batch_optimizer.py`: 批处理优化器
   - `llm_client.py`: LLM客户端接口

3. **Neo4j集成** (`src/integrations/`)
   - `neo4j_exporter.py`: Neo4j知识图谱导出器

4. **主流程** (`src/main.py`)
   - `KnowledgeGraphPipeline`: 知识图谱构建流水线

## 功能特性

### 数据处理能力
- ✅ 支持多种数据源（CSV、JSON、API）
- ✅ 智能数据清洗和预处理
- ✅ 实体识别和抽取
- ✅ 关系抽取和分类
- ✅ 数据验证和质量检查

### 知识图谱构建
- ✅ 混合知识图谱构建
- ✅ 实体消歧和合并
- ✅ 关系推理和补全
- ✅ 知识图谱一致性验证

### LLM增强功能
- ✅ 实体描述增强（预留接口）
- ✅ 实体冲突解决（预留接口）
- ✅ 关系提取增强（预留接口）
- ✅ 公司行业分类（预留接口）
- ✅ 投资方名称标准化（预留接口）

### Neo4j集成
- ✅ 知识图谱导出到Neo4j
- ✅ 支持公司和投资方实体
- ✅ 支持投资和行业关系
- ✅ 图统计和分析功能

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行完整测试

```bash
# 运行完整流程测试
python run_complete_test.py
```

### 3. 运行主流程

```bash
# 基本运行
python -m src.main

# 启用Neo4j导出
python -m src.main --enable-neo4j --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password password
```

### 4. 运行单元测试

```bash
# 运行组件测试
python test_pipeline.py
```

## 数据格式

### 公司数据格式
```json
{
  "company_name": "公司名称",
  "industry": "行业",
  "founded_year": 成立年份,
  "description": "公司描述",
  "website": "网站",
  "employees": 员工数量
}
```

### 投资事件数据格式
```json
{
  "event_id": "事件ID",
  "company_name": "被投资公司",
  "investor_name": "投资方",
  "investment_amount": "投资金额",
  "investment_round": "投资轮次",
  "investment_date": "投资日期",
  "valuation": "估值"
}
```

### 投资方数据格式
```json
{
  "investor_name": "投资方名称",
  "investor_type": "投资方类型",
  "founded_year": 成立年份,
  "aum": "管理资产规模",
  "headquarters": "总部所在地",
  "description": "投资方描述"
}
```

## 配置说明

### 基本配置 (`src/processors/config.py`)
```python
# 日志配置
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# 批处理配置
BATCH_PROCESSING_CONFIG = {
    'default_batch_size': 50,
    'max_batch_size': 100,
    'min_batch_size': 10,
    'retry_attempts': 3,
    'retry_delay': 1,
    'enable_caching': True,
    'cache_ttl': 3600
}
```

### Neo4j配置
```python
# Neo4j连接配置
NEO4J_CONFIG = {
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password',
    'database': 'neo4j'
}
```

## 输出文件

系统运行后会生成以下输出文件：

- `output/validation_results.json`: 数据验证结果
- `output/enhancement_results.json`: LLM增强结果
- `output/final_knowledge_graph.json`: 最终知识图谱
- `output/complete_test_results.json`: 完整测试结果
- `output/pipeline_statistics.json`: 流水线统计信息

## 性能指标

系统提供详细的性能分析：

- 实体处理速度（实体/秒）
- 关系处理速度（关系/秒）
- 数据质量评分
- LLM增强需求统计
- Neo4j导出状态

## 扩展开发

### 添加新的数据加载器

1. 在 `src/loaders/` 目录下创建新的加载器类
2. 继承 `BaseDataLoader` 基类
3. 实现 `load_data()` 方法
4. 在 `UnifiedDataLoader` 中注册新的加载器

### 添加新的验证规则

1. 在 `src/processors/data_validator.py` 中添加验证方法
2. 在 `validate_company_data()` 等方法中调用新规则
3. 更新验证结果格式

### 集成新的LLM服务

1. 在 `src/processors/llm_client.py` 中实现新的LLM客户端
2. 继承 `LLMClientInterface` 抽象基类
3. 实现所有必需的抽象方法

## 故障排除

### 常见问题

1. **Neo4j连接失败**
   - 检查Neo4j服务是否启动
   - 验证连接配置是否正确
   - 检查防火墙设置

2. **数据验证失败**
   - 检查输入数据格式
   - 验证数据完整性
   - 查看验证结果文件

3. **性能问题**
   - 调整批处理大小
   - 启用缓存功能
   - 优化数据预处理

### 调试模式

```bash
# 启用调试日志
export LOG_LEVEL=DEBUG
python run_complete_test.py
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进系统。