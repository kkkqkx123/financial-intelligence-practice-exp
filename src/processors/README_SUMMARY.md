# LLM实体抽取优化实现

本项目已完成LLM实体抽取模块的优化，实现了简化的LLM处理器、优化的关系提取流程和重构的批量处理逻辑。

## 已完成的优化

### 1. 简化的LLM处理器 (`llm_processor.py`)

实现了专注于核心功能的LLM处理器，包括：

- **实体提取**：从文本中提取公司、投资方等实体
- **关系提取**：从文本中提取投资关系
- **属性提取**：从文本中提取实体的属性信息
- **批量处理**：支持高效的批量文本处理
- **结果解析**：结构化解析LLM返回的结果

主要特点：
- 移除了多余的实体描述增强、冲突解决等功能
- 专注于核心的实体抽取任务
- 提供了简洁的API接口
- 支持动态批量大小调整
- 实现了文本相似度分组

### 2. 优化的关系提取流程 (`relation_extractor.py`)

结合规则和LLM方法的关系提取流程，包括：

- **规则提取器**：基于正则表达式和关键词的快速提取
- **LLM增强提取器**：在规则提取失败时使用LLM增强
- **投资事件处理**：专门处理投资事件的结构化信息
- **置信度计算**：为每个提取结果计算置信度
- **结果合并**：合并规则和LLM提取的结果

主要特点：
- 先使用规则快速提取，降低LLM调用次数
- 对低置信度结果使用LLM增强
- 支持多种投资关系模式
- 自动提取投资金额、轮次、日期等属性
- 实现了结果去重和合并

### 3. 重构的批量处理逻辑 (`batch_processor.py`)

优化了LLM查询的批量处理，提高了效率并降低了成本：

- **文本相似度计算**：使用TF-IDF和余弦相似度计算文本相似度
- **动态批量大小**：根据token限制动态调整批量大小
- **优先级管理**：根据实体类型和关系类型设置优先级
- **错误处理**：实现了重试机制和错误隔离
- **统计跟踪**：详细的处理统计信息

主要特点：
- 仅合并相似度高的请求，避免无关请求混合
- 根据token数量动态调整批量大小
- 支持优先级处理，重要请求优先处理
- 实现了部分成功提交，单条失败不影响其他请求
- 提供了详细的处理统计和错误分析

### 4. 测试套件 (`test_processors.py`)

实现了完整的测试套件，包括：

- **简化LLM处理器测试**：测试实体、关系和属性提取功能
- **优化关系提取流程测试**：测试规则和LLM结合的提取流程
- **优化批量处理逻辑测试**：测试批量处理的效率和准确性
- **性能测试**：测量各组件的处理时间和成功率
- **结果保存**：将测试结果保存为JSON文件

## 使用方法

### 1. 简化LLM处理器

```python
from src.processors.llm_processor import get_llm_processor

# 获取处理器实例
processor = get_llm_processor()

# 提取实体
entities = await processor.extract_entities(["红杉资本投资了字节跳动"])

# 提取关系
relations = await processor.extract_relations(["红杉资本投资了字节跳动"])

# 提取属性
attributes = await processor.extract_attributes(["红杉资本投资了字节跳动"])
```

### 2. 优化关系提取流程

```python
from src.processors.relation_extractor import get_relation_extractor

# 获取提取器实例
extractor = get_relation_extractor()

# 提取实体
entities = await extractor.extract_entities_from_texts(["红杉资本投资了字节跳动"])

# 提取关系
relations = await extractor.extract_relations_from_texts(["红杉资本投资了字节跳动"])

# 获取统计信息
stats = extractor.get_stats()
```

### 3. 优化批量处理逻辑

```python
from src.processors.batch_processor import get_batch_processor, create_batch_request

# 获取处理器实例
processor = get_batch_processor()

# 创建请求
requests = [
    create_batch_request("req1", "红杉资本投资了字节跳动", "relation"),
    create_batch_request("req2", "腾讯投资了京东", "relation")
]

# 处理请求
results = await processor.process_requests(requests)

# 获取统计信息
stats = processor.get_stats()
```

### 4. 运行测试

```bash
python src/processors/test_processors.py
```

## 性能优化

与原始实现相比，优化后的系统具有以下优势：

1. **减少LLM调用次数**：通过规则预提取和相似度分组，大幅减少LLM调用
2. **提高处理效率**：动态批量大小和优先级处理提高了整体处理效率
3. **降低错误率**：规则和LLM结合的方式提高了提取准确性
4. **增强容错能力**：错误隔离和重试机制提高了系统稳定性
5. **详细统计分析**：提供了全面的处理统计，便于性能分析和优化

## 数据集分析

项目已对三个数据集进行了分析：

- **公司数据集**：约1万条公司数据，包含名称、公司介绍、工商信息等
- **投资事件数据集**：约2千条投资事件数据，包含事件描述、投资方、融资方等
- **投资结构数据集**：约1000条投资机构数据，包含机构介绍、行业、规模等

详细分析结果保存在 `src/dataset/dataset_analysis.md` 文件中。

## 设计方案

完整的设计方案保存在 `src/processors/llm_extraction_design.md` 文件中，包括：

- 设计目标和核心功能
- 系统架构和组件设计
- 数据流程和提示词设计
- 批量处理优化策略
- 错误处理与回退机制
- 性能优化和评估监控
- 实施计划和预期效果

## 总结

通过本次优化，我们实现了：

1. **聚焦核心功能**：移除了多余的实体描述增强等功能，专注于实体抽取
2. **提高处理效率**：通过规则预提取和优化的批量处理，大幅提高处理效率
3. **降低成本**：减少LLM调用次数，降低API使用成本
4. **增强稳定性**：实现错误隔离和重试机制，提高系统稳定性
5. **提供详细统计**：全面的处理统计，便于性能分析和优化

优化后的系统更加轻量、高效、专注，能够更好地服务于知识图谱构建的主链路。