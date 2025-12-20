# 金融知识图谱构建完整工作流程

## 概述

本项目旨在通过大模型技术，自动化生成针对特定金融领域的知识图谱。系统从金融文本中抽取实体和关系，基于大模型实现知识图谱的构建，并将知识图谱存储在Neo4j数据库中。

## 系统架构

### 核心组件

1. **数据加载器** (`src/main.py` - `Pipeline.load_data_files`)
   - 支持CSV和Markdown格式的数据加载
   - 加载公司数据、投资事件和投资结构数据

2. **数据解析器** (`src/processors/data_parser.py`)
   - 解析原始数据，提取关键信息
   - 标准化数据格式

3. **知识图谱构建器** (`src/processors/kg_builder.py`)
   - 构建公司实体
   - 构建投资方实体
   - 构建投资关系
   - 构建投资结构关系

4. **LLM处理器** (`src/processors/llm_processor.py`)
   - 实体识别和抽取
   - 关系抽取
   - 属性抽取
   - 批量处理和优化

5. **Neo4j集成** (`src/integrations/neo4j_exporter.py`)
   - 创建节点（公司、投资方、行业）
   - 创建关系（投资关系、行业关系、结构关系）
   - 数据导出和统计

## 完整工作流程

### 1. 数据准备阶段

```
输入数据：
- src/dataset/company_data.csv (公司数据)
- src/dataset/investment_events.csv (投资事件数据)
- src/dataset/investment_structure.csv (投资结构数据)
```

### 2. 数据加载与解析阶段

1. **加载数据文件**
   - 使用`Pipeline.load_data_files()`方法加载所有CSV文件
   - 验证数据完整性和格式

2. **数据解析**
   - 使用`DataParser`解析原始数据
   - 提取关键信息并标准化格式
   - 处理缺失值和异常数据

### 3. 知识图谱构建阶段

1. **实体构建**
   - 使用`KGBuilder.build_company_entities()`构建公司实体
   - 使用`KGBuilder.build_investor_entities()`构建投资方实体
   - 生成唯一ID和标准化属性

2. **关系构建**
   - 使用`KGBuilder.build_investment_relationships()`构建投资关系
   - 使用`KGBuilder.build_investment_structure_relationships()`构建投资结构关系
   - 处理实体链接和关系属性

3. **LLM增强处理**
   - 使用`SimplifiedLLMProcessor`进行实体识别和关系抽取
   - 批量处理和优化API调用
   - 结果验证和错误处理

### 4. 质量检查阶段

1. **数据质量评估**
   - 计算实体和关系的置信度
   - 检测数据不一致和缺失
   - 生成质量报告

2. **数据清洗**
   - 去重实体和关系
   - 修复不一致数据
   - 标准化属性值

### 5. 结果保存阶段

1. **保存中间结果**
   - 保存解析后的数据
   - 保存构建的知识图谱
   - 保存质量报告

2. **导出Neo4j**
   - 使用`KnowledgeGraphExporter`导出数据到Neo4j
   - 创建节点和关系
   - 生成导出统计

### 6. 验证与分析阶段

1. **结果验证**
   - 验证实体数量是否符合预期（约13000条）
   - 检查关系完整性和一致性
   - 分析数据质量指标

2. **问题分析**
   - 识别LLM标注失效的情况
   - 分析节点/边残缺的原因
   - 提出修复方案

## 执行命令

### 完整流程执行

```bash
# 执行完整知识图谱构建流程（启用Neo4j集成）
python src/main.py --data-dir src/dataset --output-dir output --enable-neo4j

# 指定Neo4j连接参数
python src/main.py --data-dir src/dataset --output-dir output --enable-neo4j --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password 1234567kk

# 跳过Neo4j集成（仅生成知识图谱文件）
python src/main.py --data-dir src/dataset --output-dir output

# 显示详细日志和配置信息
python src/main.py --data-dir src/dataset --output-dir output --enable-neo4j --verbose --show-config

# 使用自定义环境配置文件
python src/main.py --data-dir src/dataset --output-dir output --env-file .env.custom

# 不保存中间结果（仅保存最终结果）
python src/main.py --data-dir src/dataset --output-dir output --no-intermediate

# 使用run_main.py简化执行（推荐）
python src/run_main.py
```

### 命令行选项说明

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--data-dir` | 数据文件目录 | `src/dataset` |
| `--output-dir` | 输出目录 | `output` |
| `--enable-neo4j` | 启用Neo4j集成 | 不启用 |
| `--neo4j-uri` | Neo4j URI | `bolt://localhost:7687` |
| `--neo4j-user` | Neo4j用户名 | `neo4j` |
| `--neo4j-password` | Neo4j密码 | `password` |
| `--env-file` | 自定义.env文件路径 | 使用默认.env |
| `--show-config` | 显示配置信息 | 不显示 |
| `--no-intermediate` | 不保存中间结果 | 保存中间结果 |
| `--verbose` | 详细日志输出 | 标准日志输出 |

### Neo4j配置

系统使用Docker运行的Neo4j数据库：
- 端口映射：7473(HTTP)、7474(HTTPS)、7687(Bolt)
- 数据库名称：neo4j
- 用户名：neo4j
- 密码：1234567kk

### 清空Neo4j数据库

```bash
# 清空Neo4j数据库（删除所有数据）
docker exec neo4j rm -rf /data/databases/neo4j /data/transactions/neo4j
```

## 预期结果

### 数据规模

- 公司实体：约8000-10000个
- 投资方实体：约2000-3000个
- 投资关系：约10000-12000条
- 行业实体：约100-200个
- 结构关系：约500-1000条

### 质量指标

- 实体识别准确率：>85%
- 关系抽取准确率：>80%
- 数据完整性：>90%

## 常见问题与解决方案

### 1. LLM标注失效

**症状**：
- 实体识别结果为空或明显错误
- 关系抽取结果不准确
- 属性提取缺失

**解决方案**：
- 检查LLM API配置和密钥
- 验证提示词模板是否合适
- 调整批量处理大小
- 增加重试机制和错误处理

### 2. 节点/边大量残缺

**症状**：
- 知识图谱中实体数量明显少于预期
- 关系缺失或不完整
- 数据质量报告显示大量问题

**解决方案**：
- 检查数据源文件是否完整
- 验证数据解析逻辑是否正确
- 调整实体匹配和链接算法
- 优化数据清洗和去重逻辑

### 3. Neo4j集成失败

**症状**：
- 无法连接到Neo4j数据库
- 数据导出失败
- 节点/关系创建错误

**解决方案**：
- 检查Neo4j服务是否正常运行
- 验证连接配置和认证信息
- 清空数据库并重新尝试
- 检查数据格式和模型定义

## 性能优化

### 1. LLM处理优化

- 使用批量处理减少API调用次数
- 基于相似度分组处理相似请求
- 实现请求优先级排序
- 添加缓存机制避免重复处理

### 2. 数据处理优化

- 使用并行处理加速数据解析
- 优化实体匹配算法
- 实现增量更新机制
- 优化内存使用和垃圾回收

### 3. Neo4j优化

- 使用批量创建提高导入效率
- 优化索引和约束定义
- 实现增量更新减少全量导入
- 监控和优化查询性能

## 扩展与定制

### 1. 数据源扩展

- 支持更多数据格式（JSON、XML等）
- 添加实时数据流处理
- 集成外部数据API

### 2. 算法优化

- 实现更先进的实体识别算法
- 添加关系推理和补全功能
- 集成领域特定的知识库

### 3. 可视化与分析

- 添加知识图谱可视化界面
- 实现数据分析和统计功能
- 支持自定义查询和报告

## 总结

本工作流程提供了一个完整的金融知识图谱构建解决方案，从数据加载到最终验证的全过程。通过大模型技术的应用，系统能够自动化地从金融文本中提取实体和关系，构建高质量的知识图谱，并将其存储在Neo4j数据库中以便后续分析和应用。

系统设计考虑了可扩展性、可靠性和性能优化，能够处理大规模数据（约13000条记录），并提供了完善的错误处理和质量保证机制。