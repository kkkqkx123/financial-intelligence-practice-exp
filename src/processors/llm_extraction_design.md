# LLM实体抽取增强模块设计方案

## 1. 设计目标

基于对现有系统的分析和新需求，设计一个轻量、高效、专注的LLM驱动型实体抽取模块，专注于从金融投资文本中准确提取实体和关系，服务于知识图谱构建。

## 2. 核心功能设计

### 2.1 保留的核心功能

- **关系提取**：从非结构化文本中提取"投资方→被投公司"的二元关系对
- **实体识别**：识别文本中的公司、投资机构等实体
- **属性抽取**：从文本中提取实体的关键属性信息

### 2.2 移除的多余功能

- ~~实体描述增强~~：不再使用LLM增强实体描述
- ~~实体冲突解决~~：通过简单规则或确定性算法处理
- ~~行业分类~~：基于预定义的行业关键词库实现
- ~~名称标准化~~：通过字符串模糊匹配、别名映射表等规则方法完成

## 3. 系统架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM实体抽取增强模块                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   文本预处理    │  │   批量处理管理   │  │   结果后处理     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   实体识别器     │  │   关系提取器     │  │   属性抽取器     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   LLM客户端     │  │   提示词模板     │  │   结果解析器     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 4. 核心组件设计

### 4.1 文本预处理器 (TextPreprocessor)

**功能**：对输入文本进行标准化处理，提高LLM抽取效果

**主要方法**：
- `normalize_text()`: 文本标准化（去除特殊字符、统一标点等）
- `segment_sentences()`: 句子分割，控制输入长度
- `extract_keywords()`: 提取关键词，辅助实体识别

### 4.2 批量处理管理器 (BatchProcessor)

**功能**：智能合并相似请求，优化API调用效率

**主要方法**：
- `group_by_similarity()`: 基于文本相似度分组
- `dynamic_batch_sizing()`: 根据token数量动态调整批量大小
- `prioritize_requests()`: 按重要性排序处理请求

**改进点**：
- 简化分组策略，仅合并高度相似的文本
- 动态调整批量大小，考虑token限制
- 引入优先级机制，重要实体优先处理

### 4.3 实体识别器 (EntityRecognizer)

**功能**：从文本中识别公司、投资机构等实体

**主要方法**：
- `recognize_companies()`: 识别公司实体
- `recognize_investors()`: 识别投资机构实体
- `validate_entities()`: 验证识别结果

### 4.4 关系提取器 (RelationExtractor)

**功能**：从文本中提取"投资方→被投公司"关系

**主要方法**：
- `extract_investment_relations()`: 提取投资关系
- `validate_relations()`: 验证关系有效性
- `resolve_conflicts()`: 解决简单冲突（基于规则）

### 4.5 属性抽取器 (AttributeExtractor)

**功能**：从文本中提取实体的关键属性

**主要方法**：
- `extract_investment_amount()`: 提取投资金额
- `extract_investment_round()`: 提取投资轮次
- `extract_investment_date()`: 提取投资时间

### 4.6 LLM客户端 (LLMClient)

**功能**：与LLM API交互，处理请求和响应

**主要方法**：
- `generate_completion()`: 生成LLM响应
- `handle_errors()`: 处理API错误
- `retry_failed_requests()`: 重试失败请求

### 4.7 提示词模板 (PromptTemplates)

**功能**：提供优化的提示词模板，确保LLM输出结构化结果

**主要模板**：
- `ENTITY_RECOGNITION_TEMPLATE`: 实体识别提示词
- `RELATION_EXTRACTION_TEMPLATE`: 关系提取提示词
- `ATTRIBUTE_EXTRACTION_TEMPLATE`: 属性抽取提示词

### 4.8 结果解析器 (ResultParser)

**功能**：解析LLM输出，转换为结构化数据

**主要方法**：
- `parse_json_response()`: 解析JSON格式响应
- `validate_schema()`: 验证输出格式
- `handle_parse_errors()`: 处理解析错误

## 5. 数据流程设计

```
输入文本 → 文本预处理 → 批量分组 → LLM处理 → 结果解析 → 结构化输出
    ↓           ↓          ↓         ↓         ↓          ↓
  原始数据    标准化文本   请求分组   API调用   原始响应   实体/关系
```

## 6. 提示词设计

### 6.1 实体识别提示词

```
你是一个金融领域的实体识别专家。请从以下文本中识别所有公司名称和投资机构名称。

要求：
1. 只识别公司和投资机构，忽略其他类型的实体
2. 返回JSON格式：{"companies": [...], "investors": [...]}
3. 不要添加解释或额外信息
4. 确保名称准确完整

文本：{text}
```

### 6.2 关系提取提示词

```
你是一个金融投资关系提取专家。请从以下文本中提取"投资方→被投公司"的关系。

要求：
1. 只提取投资关系，格式：(投资方) -> (被投公司)
2. 返回JSON格式：{"relations": [{"investor": "...", "company": "..."}]}
3. 不要添加解释或额外信息
4. 确保关系准确

文本：{text}
```

### 6.3 属性抽取提示词

```
你是一个金融投资属性提取专家。请从以下文本中提取投资金额、轮次和时间信息。

要求：
1. 只提取金额、轮次和时间
2. 返回JSON格式：{"amount": "...", "round": "...", "date": "..."}
3. 如果信息不存在，使用null
4. 不要添加解释或额外信息

文本：{text}
```

## 7. 批量处理优化策略

### 7.1 智能分组策略

```python
def group_by_similarity(texts, threshold=0.8):
    """
    基于文本相似度分组，仅合并高度相似的文本
    """
    # 1. 计算文本嵌入向量
    embeddings = get_text_embeddings(texts)
    
    # 2. 计算相似度矩阵
    similarity_matrix = compute_similarity(embeddings)
    
    # 3. 基于阈值聚类
    groups = cluster_by_threshold(similarity_matrix, threshold)
    
    return groups
```

### 7.2 动态批量大小调整

```python
def dynamic_batch_sizing(texts, max_tokens=4000):
    """
    根据token数量动态调整批量大小
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = count_tokens(text)
        
        if current_tokens + text_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches
```

### 7.3 优先级处理机制

```python
def prioritize_requests(requests):
    """
    按重要性排序处理请求
    """
    def priority_key(request):
        # 1. 知名投资机构优先
        if is_known_investor(request.text):
            return (0, request.text)
        # 2. 大额投资优先
        if is_large_investment(request.text):
            return (1, request.text)
        # 3. 其他
        return (2, request.text)
    
    return sorted(requests, key=priority_key)
```

## 8. 错误处理与回退机制

### 8.1 LLM调用失败处理

```python
def handle_llm_failure(request, error):
    """
    LLM调用失败时的回退机制
    """
    # 1. 记录错误
    log_error(request, error)
    
    # 2. 尝试规则基础抽取
    rule_based_result = rule_based_extraction(request.text)
    
    # 3. 如果规则抽取也失败，返回空结果
    if not rule_based_result:
        return empty_result()
    
    return rule_based_result
```

### 8.2 结果解析失败处理

```python
def handle_parse_failure(response):
    """
    结果解析失败时的处理
    """
    # 1. 尝试修复常见JSON错误
    fixed_response = fix_common_json_errors(response)
    
    # 2. 重新尝试解析
    try:
        return parse_json(fixed_response)
    except:
        # 3. 如果仍然失败，使用正则表达式提取
        return regex_extract(response)
```

## 9. 性能优化设计

### 9.1 缓存机制

- **结果缓存**：缓存相同文本的抽取结果
- **嵌入缓存**：缓存文本嵌入向量，避免重复计算
- **API响应缓存**：缓存API响应，减少重复调用

### 9.2 并发处理

- **异步API调用**：使用异步方式调用LLM API
- **批量并行处理**：并行处理多个批量请求
- **结果流式处理**：流式处理大型结果集

### 9.3 资源管理

- **连接池**：复用HTTP连接，减少连接开销
- **内存管理**：及时释放大型对象，避免内存泄漏
- **限流控制**：控制API调用频率，避免触发限流

## 10. 评估与监控

### 10.1 评估指标

- **实体识别准确率**：正确识别的实体比例
- **关系提取准确率**：正确提取的关系比例
- **属性抽取准确率**：正确抽取的属性比例
- **API调用效率**：每次API调用的平均处理文本数
- **处理延迟**：从请求到响应的平均时间

### 10.2 监控机制

- **实时监控**：实时跟踪处理性能和错误率
- **异常报警**：异常情况自动报警
- **性能分析**：定期分析性能瓶颈
- **质量评估**：定期评估抽取质量

## 11. 实施计划

### 阶段一：核心功能实现（2周）

1. 实现基础实体识别功能
2. 实现关系提取功能
3. 实现属性抽取功能
4. 设计基础提示词模板

### 阶段二：批量处理优化（1周）

1. 实现智能分组策略
2. 实现动态批量大小调整
3. 实现优先级处理机制

### 阶段三：错误处理与回退（1周）

1. 实现错误处理机制
2. 实现规则基础回退
3. 实现结果修复机制

### 阶段四：性能优化（1周）

1. 实现缓存机制
2. 实现并发处理
3. 实现资源管理

### 阶段五：测试与优化（1周）

1. 全面功能测试
2. 性能测试与优化
3. 质量评估与改进

## 12. 预期效果

通过以上设计，预期实现以下效果：

1. **功能聚焦**：专注于实体抽取核心功能，移除多余功能
2. **效率提升**：通过智能批量处理，减少API调用次数50%以上
3. **质量保证**：通过优化的提示词和错误处理，提高抽取准确率
4. **成本降低**：减少不必要的LLM调用，降低处理成本60%以上
5. **易于维护**：模块化设计，便于维护和扩展

## 13. 风险与应对

### 13.1 主要风险

1. **LLM输出不稳定**：模型输出格式可能不一致
2. **API限流**：高频调用可能触发API限流
3. **质量波动**：不同类型文本的抽取质量可能波动较大

### 13.2 应对策略

1. **多重验证**：结合规则验证LLM输出
2. **智能限流**：实现自适应限流机制
3. **质量监控**：建立质量监控体系，及时发现问题

## 14. 总结

本设计方案基于对现有系统的深入分析，针对金融投资领域的实体抽取需求，提出了一个轻量、高效、专注的LLM驱动型实体抽取模块。通过移除多余功能、优化批量处理、改进错误处理等策略，实现了一个更加符合实际需求的解决方案，能够有效支持金融投资知识图谱的构建工作。