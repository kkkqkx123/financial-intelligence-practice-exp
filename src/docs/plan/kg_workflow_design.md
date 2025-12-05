# 基于实际数据集结构的LLM增强知识图谱构建方案

## 1. 实际数据集结构分析

### 1.1 公司数据 (company_data.md)
**数据规模**: 约10,000条记录  
**字段结构**:
- 名称: 公司简称/品牌名
- 公司名称: 完整工商注册名称
- 公司介绍: 详细业务描述文本
- 工商: 工商注册名称（与公司名称可能重复）
- 地址: 注册地址（部分为空）
- 工商注册id: 统一社会信用代码
- 成立时间: 注册日期
- 法人代表: 法定代表人
- 注册资金: 注册资本（格式多样）
- 统一信用代码: 统一社会信用代码（与工商注册id重复）
- 网址: 公司官网（部分为空）

### 1.2 投资事件数据 (investment_events.md)
**数据规模**: 约2,000条记录  
**字段结构**:
- 事件资讯: 完整事件描述
- 投资方: 多个投资机构（空格分隔）
- 融资方: 被投资公司名称
- 融资时间: 投资发生时间
- 轮次: 投资轮次（标准格式）
- 金额: 投资金额（格式多样，含"未披露"）

### 1.3 投资机构数据 (investment_structure.md)
**数据规模**: 约1,000条记录  
**字段结构**:
- 机构名称: 投资机构名称
- 介绍: 机构简介文本
- 行业: 投资领域（多值，空格分隔）
- 规模: 管理规模（多为空）
- 轮次: 投资轮次偏好（多值，空格分隔）

## 2. 基于实际结构的处理策略调整

### 2.1 数据预处理策略

#### 2.1.1 结构化数据解析
```python
class FinancialDataParser:
    """专门处理金融数据集的解析器"""
    
    def parse_company_data(self, text):
        """解析公司数据格式"""
        lines = text.strip().split('\n')
        # 跳过统计行和表头重复
        data_lines = [line for line in lines[2:] if line.strip()]
        
        companies = []
        for line in data_lines:
            fields = line.split(',')
            if len(fields) >= 11:
                company = {
                    'short_name': fields[0],
                    'full_name': fields[1],
                    'description': fields[2],
                    'registration_name': fields[3],
                    'address': fields[4] if fields[4] else None,
                    'registration_id': fields[5],
                    'establish_date': fields[6],
                    'legal_representative': fields[7],
                    'registered_capital': self.normalize_capital(fields[8]),
                    'credit_code': fields[9],
                    'website': fields[10] if fields[10] else None
                }
                companies.append(company)
        return companies
    
    def parse_investment_events(self, text):
        """解析投资事件数据格式"""
        lines = text.strip().split('\n')
        data_lines = [line for line in lines[2:] if line.strip()]
        
        events = []
        for line in data_lines:
            fields = line.split(',')
            if len(fields) >= 6:
                event = {
                    'description': fields[0],
                    'investors': fields[1].split() if fields[1] else [],
                    'investee': fields[2],
                    'investment_date': fields[3],
                    'round': fields[4],
                    'amount': self.normalize_amount(fields[5])
                }
                events.append(event)
        return events
    
    def parse_investment_institutions(self, text):
        """解析投资机构数据格式"""
        lines = text.strip().split('\n')
        data_lines = [line for line in lines[3:] if line.strip()]  # 跳过两行统计信息
        
        institutions = []
        for line in data_lines:
            fields = line.split(',')
            if len(fields) >= 5:
                institution = {
                    'name': fields[0],
                    'description': fields[1] if fields[1] else None,
                    'industries': fields[2].split() if fields[2] else [],
                    'scale': self.normalize_scale(fields[3]),
                    'preferred_rounds': fields[4].split() if fields[4] else []
                }
                institutions.append(institution)
        return institutions
```

#### 2.1.2 数据标准化处理
```python
def normalize_capital(capital_str):
    """标准化注册资本格式"""
    if not capital_str or capital_str == '':
        return None
    
    # 处理各种格式：500万元人民币, 222.2222万元人民币, 10000万元人民币
    if '万元人民币' in capital_str:
        amount = capital_str.replace('万元人民币', '')
        return float(amount) * 10000  # 转换为元
    elif '万美元' in capital_str:
        amount = capital_str.replace('万美元', '')
        return float(amount) * 10000 * 7  # 粗略汇率转换
    else:
        return capital_str

def normalize_amount(amount_str):
    """标准化投资金额格式"""
    if not amount_str or amount_str == '未披露':
        return None
    
    # 处理：3.5亿人民币, 1000万人民币, 800万美元, 数千万人民币
    if '亿人民币' in amount_str:
        amount = amount_str.replace('亿人民币', '')
        return float(amount) * 100000000
    elif '万人民币' in amount_str:
        amount = amount_str.replace('万人民币', '')
        return float(amount) * 10000
    elif '万美元' in amount_str:
        amount = amount_str.replace('万美元', '')
        return float(amount) * 10000 * 7
    elif '百万美元' in amount_str:
        amount = amount_str.replace('百万美元', '')
        return float(amount) * 1000000 * 7
    else:
        return amount_str
```

### 2.2 LLM增强实体抽取策略

#### 2.2.1 公司行业分类增强
```python
def llm_enhance_industry_classification(company_description, company_name):
    """使用LLM从公司描述中提取行业分类"""
    prompt = f"""
    作为金融行业专家，请根据以下信息准确判断公司所属行业：
    
    公司名称：{company_name}
    公司描述：{company_description}
    
    要求：
    1. 从描述中提取主营业务关键词
    2. 判断所属行业（使用标准行业分类）
    3. 提供细分行业（如：人工智能->计算机视觉）
    4. 给出置信度评分（0-1）
    
    输出格式：
    {{
        "primary_industry": "主要行业",
        "sub_industry": "细分行业", 
        "keywords": ["关键词1", "关键词2"],
        "confidence": 0.85,
        "reasoning": "判断理由"
    }}
    """
    return llm_generate(prompt)
```

#### 2.2.2 投资事件信息抽取
```python
def llm_extract_investment_details(event_description):
    """从投资事件描述中提取结构化信息"""
    prompt = f"""
    从以下投资事件描述中提取关键信息：
    
    事件描述：{event_description}
    
    需要提取的信息：
    1. 主要投资方（领投方）
    2. 跟投方列表
    3. 被投公司简介
    4. 投资动机/战略意义
    5. 预计估值（如提及）
    6. 特殊条款（如对赌、回购等）
    
    输出格式：
    {{
        "lead_investor": "领投方",
        "co_investors": ["跟投方1", "跟投方2"],
        "company_brief": "被投公司简介",
        "investment_motivation": "投资动机",
        "valuation": "估值信息",
        "special_terms": "特殊条款",
        "confidence": 0.9
    }}
    """
    return llm_generate(prompt)
```

#### 2.2.3 投资机构画像增强
```python
def llm_enhance_institution_profile(institution_name, description, industries, rounds):
    """增强投资机构画像信息"""
    prompt = f"""
    基于以下信息，完善投资机构画像：
    
    机构名称：{institution_name}
    机构描述：{description}
    投资行业：{industries}
    偏好轮次：{rounds}
    
    需要补充的信息：
    1. 机构类型（VC/PE/战略投资者/政府基金）
    2. 投资阶段偏好详细说明
    3. 单笔投资金额范围
    4. 地域偏好
    5. 知名投资案例
    6. 机构背景（国资/民营/外资）
    7. 投资风格（激进/稳健/保守）
    
    输出格式：
    {{
        "institution_type": "机构类型",
        "investment_stage": "投资阶段偏好",
        "investment_range": "投资金额范围",
        "geographic_preference": "地域偏好",
        "notable_cases": ["案例1", "案例2"],
        "background": "机构背景",
        "investment_style": "投资风格",
        "confidence": 0.8
    }}
    """
    return llm_generate(prompt)
```

### 2.3 实体链接与消歧策略

#### 2.3.1 公司实体链接
```python
def link_company_entities(companies, investment_events):
    """在公司数据和投资事件之间建立实体链接"""
    
    company_dict = {c['full_name']: c for c in companies}
    company_dict.update({c['short_name']: c for c in companies})
    
    linked_events = []
    for event in investment_events:
        investee = event['investee']
        
        # 精确匹配
        if investee in company_dict:
            event['investee_company'] = company_dict[investee]
            event['match_type'] = 'exact'
        else:
            # LLM辅助模糊匹配
            prompt = f"""
            判断以下两个公司名称是否指代同一公司：
            
            投资事件中的名称：{investee}
            公司库中的候选：{list(company_dict.keys())[:10]}
            
            考虑因素：
            1. 名称相似度
            2. 行业相关性  
            3. 成立时间合理性
            4. 常见简称/别名
            
            输出格式：
            {{
                "matched_company": "匹配的公司名称或None",
                "similarity_score": 0.85,
                "match_reasoning": "匹配理由"
            }}
            """
            result = llm_generate(prompt)
            if result['matched_company'] != 'None':
                event['investee_company'] = company_dict[result['matched_company']]
                event['match_type'] = 'fuzzy'
                event['match_confidence'] = result['similarity_score']
        
        linked_events.append(event)
    
    return linked_events
```

#### 2.3.2 投资机构链接
```python
def link_investor_entities(investment_events, institutions):
    """在投资事件和投资机构之间建立链接"""
    
    institution_dict = {inst['name']: inst for inst in institutions}
    
    linked_events = []
    for event in investment_events:
        investors = event['investors']
        linked_investors = []
        
        for investor in investors:
            if investor in institution_dict:
                linked_investors.append({
                    'name': investor,
                    'institution_info': institution_dict[investor],
                    'match_type': 'exact'
                })
            else:
                # 处理投资机构别名
                prompt = f"""
                识别投资机构的规范名称：
                
                事件中的投资方：{investor}
                投资机构库：{list(institution_dict.keys())[:15]}
                
                任务：
                1. 识别是否为已知机构的简称/别名
                2. 识别是否为分支机构（如"红杉中国"vs"红杉资本"）
                3. 识别是否为联合投资中的单个机构
                
                输出格式：
                {{
                    "standard_name": "规范名称或原名称",
                    "is_alias": true/false,
                    "alias_type": "简称/分支机构/其他",
                    "confidence": 0.9
                }}
                """
                result = llm_generate(prompt)
                
                if result['is_alias'] and result['standard_name'] in institution_dict:
                    linked_investors.append({
                        'name': result['standard_name'],
                        'institution_info': institution_dict[result['standard_name']],
                        'match_type': 'alias',
                        'original_name': investor,
                        'confidence': result['confidence']
                    })
                else:
                    linked_investors.append({
                        'name': investor,
                        'match_type': 'unknown',
                        'confidence': 0.0
                    })
        
        event['linked_investors'] = linked_investors
        linked_events.append(event)
    
    return linked_events
```

### 2.4 知识图谱构建策略

#### 2.4.1 实体类型定义
```python
ENTITY_TYPES = {
    'Company': {
        'properties': [
            'name', 'short_name', 'description', 'industry', 'sub_industry',
            'address', 'establish_date', 'legal_representative', 'registered_capital',
            'credit_code', 'website', 'confidence'
        ]
    },
    'InvestmentInstitution': {
        'properties': [
            'name', 'description', 'institution_type', 'investment_stage',
            'investment_range', 'geographic_preference', 'notable_cases',
            'background', 'investment_style', 'industries', 'preferred_rounds'
        ]
    },
    'Person': {
        'properties': ['name', 'role', 'associated_entities']
    },
    'Industry': {
        'properties': ['name', 'category', 'sub_category']
    }
}

RELATIONSHIP_TYPES = {
    'INVESTED_IN': {
        'properties': ['investment_date', 'round', 'amount', 'valuation', 'confidence']
    },
    'LEAD_INVESTED_IN': {
        'properties': ['investment_date', 'round', 'amount', 'co_investors', 'confidence']
    },
    'OPERATES_IN': {
        'properties': ['industry_category', 'confidence']
    },
    'LOCATED_IN': {
        'properties': ['address_type', 'confidence']
    },
    'FOUNDED_BY': {
        'properties': ['found_date', 'role', 'confidence']
    },
    'CO_INVESTED_WITH': {
        'properties': ['cooperation_count', 'cooperation_date', 'confidence']
    }
}
```

#### 2.4.2 图谱构建算法
```python
class FinancialKGBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_resolver = EntityResolver()
        self.confidence_calculator = ConfidenceCalculator()
    
    def build_knowledge_graph(self, companies, institutions, investment_events):
        """构建金融知识图谱"""
        
        # 1. 添加公司实体
        for company in companies:
            company_node = self.create_company_node(company)
            self.graph.add_node(company['full_name'], **company_node)
        
        # 2. 添加投资机构实体  
        for institution in institutions:
            institution_node = self.create_institution_node(institution)
            self.graph.add_node(institution['name'], **institution_node)
        
        # 3. 添加投资关系
        for event in investment_events:
            self.add_investment_relationships(event)
        
        # 4. 添加合作关系
        self.add_cooperation_relationships(investment_events)
        
        # 5. 计算关系置信度
        self.calculate_relationship_confidence()
        
        return self.graph
    
    def create_company_node(self, company):
        """创建公司节点"""
        return {
            'type': 'Company',
            'name': company['full_name'],
            'short_name': company['short_name'],
            'description': company['description'],
            'industry': company.get('industry', 'unknown'),
            'sub_industry': company.get('sub_industry', 'unknown'),
            'address': company.get('address'),
            'establish_date': company.get('establish_date'),
            'legal_representative': company.get('legal_representative'),
            'registered_capital': company.get('registered_capital'),
            'confidence': company.get('confidence', 0.8)
        }
    
    def add_investment_relationships(self, event):
        """添加投资关系"""
        investee = event.get('investee_company', {}).get('full_name', event['investee'])
        
        for investor in event.get('linked_investors', []):
            investor_name = investor.get('name')
            confidence = investor.get('confidence', 0.8)
            
            # 判断是否为领投
            is_lead = self.is_lead_investor(investor_name, event['description'])
            
            relationship_type = 'LEAD_INVESTED_IN' if is_lead else 'INVESTED_IN'
            
            self.graph.add_edge(
                investor_name,
                investee,
                type=relationship_type,
                investment_date=event['investment_date'],
                round=event['round'],
                amount=event['amount'],
                confidence=confidence,
                source='investment_events'
            )
```

## 3. 质量评估与验证

### 3.1 数据质量指标
```python
def calculate_data_quality_metrics(graph):
    """计算知识图谱质量指标"""
    
    metrics = {
        'entity_completeness': calculate_entity_completeness(graph),
        'relationship_accuracy': calculate_relationship_accuracy(graph),
        'temporal_consistency': calculate_temporal_consistency(graph),
        'cross_reference_validation': validate_cross_references(graph),
        'confidence_distribution': analyze_confidence_distribution(graph)
    }
    
    return metrics

def calculate_entity_completeness(graph):
    """计算实体完整度"""
    total_entities = len(graph.nodes())
    complete_entities = 0
    
    for node, data in graph.nodes(data=True):
        required_fields = ['name', 'type']
        if all(data.get(field) for field in required_fields):
            complete_entities += 1
    
    return complete_entities / total_entities if total_entities > 0 else 0
```

### 3.2 业务规则验证
```python
def validate_business_rules(graph):
    """验证业务规则一致性"""
    
    violations = []
    
    # 规则1: 投资时间不能早于公司成立时间
    for investor, investee, data in graph.edges(data=True):
        if data.get('type') in ['INVESTED_IN', 'LEAD_INVESTED_IN']:
            investor_node = graph.nodes[investor]
            investee_node = graph.nodes[investee]
            
            investment_date = data.get('investment_date')
            establish_date = investee_node.get('establish_date')
            
            if investment_date and establish_date:
                if investment_date < establish_date:
                    violations.append({
                        'rule': 'investment_before_establishment',
                        'investor': investor,
                        'investee': investee,
                        'investment_date': investment_date,
                        'establish_date': establish_date
                    })
    
    return violations
```

## 4. 实施建议

### 4.1 分阶段实施计划

**第一阶段（基础构建）**:
1. 实现结构化数据解析器
2. 建立基础实体链接
3. 构建核心知识图谱

**第二阶段（LLM增强）**:
1. 集成LLM进行实体抽取增强
2. 实现关系推理和补全
3. 添加质量评估机制

**第三阶段（优化完善）**:
1. 优化实体消歧算法
2. 增强时序关系处理
3. 建立动态更新机制

### 4.2 技术选型建议

- **图数据库**: Neo4j (支持复杂关系查询)
- **LLM框架**: LangChain + 国产大模型（如文心一言、通义千问）
- **数据处理**: Pandas + 正则表达式 + LLM文本处理
- **质量评估**: 自定义指标 + 人工验证

### 4.3 预期成果

1. **实体规模**: 约13,000个实体（10,000公司 + 1,000机构 + 2,000其他）
2. **关系规模**: 约15,000个关系（投资关系 + 合作关系 + 属性关系）
3. **质量指标**: 实体完整度 >90%，关系准确率 >85%
4. **查询能力**: 支持多跳关系查询、时序分析、路径发现

这个调整后的方案更贴合实际数据结构和业务需求，能够构建出高质量的金融知识图谱。