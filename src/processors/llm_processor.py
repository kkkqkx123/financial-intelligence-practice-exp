"""
简化的LLM实体抽取处理器

专注于从金融投资文本中提取实体和关系，移除了多余的增强功能
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体数据类"""
    id: str
    name: str
    type: str  # 'company' 或 'investor'
    aliases: List[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.properties is None:
            self.properties = {}


@dataclass
class Relation:
    """关系数据类"""
    id: str
    source: str  # 源实体ID
    target: str  # 目标实体ID
    type: str  # 关系类型，如 'invests_in'
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class ExtractionRequest:
    """抽取请求数据类"""
    id: str
    text: str
    request_type: str  # 'entity', 'relation', 'attribute'
    context: Dict[str, Any] = None
    priority: int = 0  # 优先级，数字越大优先级越高
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        # 生成唯一ID
        if not self.id:
            self.id = hashlib.md5(f"{self.text}_{self.request_type}".encode()).hexdigest()


class TextPreprocessor:
    """文本预处理器"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """标准化文本"""
        if not text:
            return ""
        
        # 去除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除特殊字符但保留中文、英文、数字和基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》\-]', '', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    @staticmethod
    def segment_sentences(text: str, max_length: int = 500) -> List[str]:
        """将长文本分割为句子或段落"""
        if len(text) <= max_length:
            return [text]
        
        # 按句号、问号、感叹号分割
        sentences = re.split(r'[。！？]', text)
        
        # 合并短句，分割长句
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_segment + sentence) <= max_length:
                current_segment += sentence + "。"
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = sentence + "。"
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """提取关键词，辅助实体识别"""
        # 投资相关关键词
        investment_keywords = [
            "投资", "融资", "轮", "天使", "种子", "A轮", "B轮", "C轮", "D轮", 
            "Pre-A", "Pre-IPO", "战略融资", "并购", "收购", "股权", "基金",
            "创投", "资本", "风投", "机构", "领投", "跟投", "参投"
        ]
        
        # 公司相关关键词
        company_keywords = [
            "公司", "有限", "集团", "科技", "网络", "股份", "企业", "工作室"
        ]
        
        # 提取包含关键词的片段
        keywords = []
        for keyword in investment_keywords + company_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords


class BatchProcessor:
    """批量处理管理器"""
    
    def __init__(self, max_batch_tokens: int = 3000):
        self.max_batch_tokens = max_batch_tokens
        self.cache = {}  # 简单的内存缓存
    
    def group_by_similarity(self, requests: List[ExtractionRequest], threshold: float = 0.8) -> List[List[ExtractionRequest]]:
        """
        基于文本相似度分组，简化实现
        在实际应用中，可以使用文本嵌入向量计算相似度
        """
        groups = []
        processed = set()
        
        for req in requests:
            if req.id in processed:
                continue
                
            # 创建新组
            current_group = [req]
            processed.add(req.id)
            
            # 查找相似请求
            for other_req in requests:
                if other_req.id in processed:
                    continue
                    
                # 简单相似度计算：共同关键词比例
                similarity = self._calculate_similarity(req.text, other_req.text)
                
                if similarity >= threshold and req.request_type == other_req.request_type:
                    current_group.append(other_req)
                    processed.add(other_req.id)
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的简单相似度"""
        # 提取关键词
        keywords1 = set(TextPreprocessor.extract_keywords(text1))
        keywords2 = set(TextPreprocessor.extract_keywords(text2))
        
        # 计算Jaccard相似度
        if not keywords1 and not keywords2:
            return 1.0
        if not keywords1 or not keywords2:
            return 0.0
            
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def dynamic_batch_sizing(self, requests: List[ExtractionRequest]) -> List[List[ExtractionRequest]]:
        """根据token数量动态调整批量大小"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for request in requests:
            # 简单估算token数（实际应用中应使用tokenizer）
            text_tokens = len(request.text) // 2  # 假设平均2个字符一个token
            
            if current_tokens + text_tokens > self.max_batch_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [request]
                current_tokens = text_tokens
            else:
                current_batch.append(request)
                current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def prioritize_requests(self, requests: List[ExtractionRequest]) -> List[ExtractionRequest]:
        """按重要性排序处理请求"""
        def priority_key(request: ExtractionRequest) -> Tuple[int, str]:
            # 1. 知名投资机构优先
            if self._contains_known_investor(request.text):
                return (0, request.text)
            # 2. 大额投资优先
            if self._contains_large_investment(request.text):
                return (1, request.text)
            # 3. 其他
            return (2, request.text)
        
        return sorted(requests, key=priority_key)
    
    def _contains_known_investor(self, text: str) -> bool:
        """检查是否包含知名投资机构"""
        known_investors = [
            "红杉资本", "IDG资本", "经纬中国", "真格基金", "创新工场", 
            "软银中国", "晨兴资本", "DCM", "高瓴资本", "君联资本"
        ]
        
        return any(investor in text for investor in known_investors)
    
    def _contains_large_investment(self, text: str) -> bool:
        """检查是否包含大额投资"""
        large_investment_patterns = [
            r'\d+亿', r'\d+千万', r'\d+hundred\s+million', 
            r'\d+billion', r'\d+万\s*美元'
        ]
        
        return any(re.search(pattern, text) for pattern in large_investment_patterns)


class PromptTemplates:
    """提示词模板"""
    
    ENTITY_RECOGNITION_TEMPLATE = """你是一个金融领域的实体识别专家。请从以下文本中识别所有公司名称和投资机构名称。

要求：
1. 只识别公司和投资机构，忽略其他类型的实体
2. 返回JSON格式：{{"companies": [...], "investors": [...]}}
3. 不要添加解释或额外信息
4. 确保名称准确完整

文本：{text}"""
    
    RELATION_EXTRACTION_TEMPLATE = """你是一个金融投资关系提取专家。请从以下文本中提取"投资方→被投公司"的关系。

要求：
1. 只提取投资关系，格式：(投资方) -> (被投公司)
2. 返回JSON格式：{{"relations": [{{"investor": "...", "company": "..."}}]}}
3. 不要添加解释或额外信息
4. 确保关系准确

文本：{text}"""
    
    ATTRIBUTE_EXTRACTION_TEMPLATE = """你是一个金融投资属性提取专家。请从以下文本中提取投资金额、轮次和时间信息。

要求：
1. 只提取金额、轮次和时间
2. 返回JSON格式：{{"amount": "...", "round": "...", "date": "..."}}
3. 如果信息不存在，使用null
4. 不要添加解释或额外信息

文本：{text}"""


class ResultParser:
    """结果解析器"""
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """解析JSON格式响应"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试修复常见JSON错误
            fixed_response = ResultParser._fix_common_json_errors(response)
            try:
                return json.loads(fixed_response)
            except json.JSONDecodeError:
                # 如果仍然失败，返回空结果
                logger.warning(f"无法解析JSON响应: {response}")
                return {}
    
    @staticmethod
    def _fix_common_json_errors(response: str) -> str:
        """修复常见JSON错误"""
        # 移除可能的前后缀文本
        response = re.sub(r'^.*?{', '{', response)
        response = re.sub(r'}.*?$', '}', response)
        
        # 替换单引号为双引号
        response = response.replace("'", '"')
        
        # 修复可能的转义问题
        response = response.replace('\\"', '"')
        
        return response
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], expected_keys: List[str]) -> bool:
        """验证输出格式"""
        if not isinstance(data, dict):
            return False
        
        for key in expected_keys:
            if key not in data:
                return False
        
        return True
    
    @staticmethod
    def regex_extract(response: str) -> Dict[str, Any]:
        """使用正则表达式提取信息"""
        result = {}
        
        # 提取公司名称
        companies = re.findall(r'公司[：:]\s*([^,，\n]+)', response)
        if companies:
            result['companies'] = companies
        
        # 提取投资机构
        investors = re.findall(r'投资机构[：:]\s*([^,，\n]+)', response)
        if investors:
            result['investors'] = investors
        
        # 提取关系
        relations = re.findall(r'(\w+)\s*->\s*(\w+)', response)
        if relations:
            result['relations'] = [{'investor': inv, 'company': comp} for inv, comp in relations]
        
        return result


class LLMClient:
    """LLM客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def generate_completion(self, prompt: str) -> str:
        """生成LLM响应"""
        # 这里应该是实际的API调用
        # 为了演示，返回模拟响应
        await asyncio.sleep(0.1)  # 模拟API延迟
        
        # 根据提示词类型返回不同的模拟响应
        if "实体识别" in prompt:
            return '{"companies": ["示例公司"], "investors": ["示例投资机构"]}'
        elif "关系提取" in prompt:
            return '{"relations": [{"investor": "示例投资机构", "company": "示例公司"}]}'
        elif "属性提取" in prompt:
            return '{"amount": "1000万", "round": "A轮", "date": "2023-01-01"}'
        else:
            return "{}"
    
    def handle_errors(self, error: Exception) -> Dict[str, Any]:
        """处理API错误"""
        logger.error(f"LLM API错误: {str(error)}")
        return {"error": str(error)}
    
    async def retry_failed_requests(self, requests: List[Tuple[str, str]], max_retries: int = 3) -> List[Dict[str, Any]]:
        """重试失败请求"""
        results = []
        
        for request_id, prompt in requests:
            success = False
            result = None
            
            for attempt in range(max_retries):
                try:
                    response = await self.generate_completion(prompt)
                    result = ResultParser.parse_json_response(response)
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"请求 {request_id} 第 {attempt + 1} 次尝试失败: {str(e)}")
                    if attempt == max_retries - 1:
                        result = self.handle_errors(e)
            
            results.append(result)
        
        return results


class SimplifiedLLMProcessor:
    """简化的LLM处理器"""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.text_preprocessor = TextPreprocessor()
        self.batch_processor = BatchProcessor()
        self.prompt_templates = PromptTemplates()
        self.result_parser = ResultParser()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "api_calls": 0,
            "entities_extracted": 0,
            "relations_extracted": 0
        }
    
    async def extract_entities(self, texts: List[str]) -> List[Entity]:
        """从文本列表中提取实体"""
        # 创建实体识别请求
        requests = []
        for text in texts:
            normalized_text = self.text_preprocessor.normalize_text(text)
            if normalized_text:
                request = ExtractionRequest(
                    id="",  # 将自动生成
                    text=normalized_text,
                    request_type="entity"
                )
                requests.append(request)
        
        # 优先级排序
        prioritized_requests = self.batch_processor.prioritize_requests(requests)
        
        # 按相似度分组
        groups = self.batch_processor.group_by_similarity(prioritized_requests)
        
        # 处理每个组
        all_entities = []
        for group in groups:
            # 动态批量大小调整
            batches = self.batch_processor.dynamic_batch_sizing(group)
            
            for batch in batches:
                batch_entities = await self._process_entity_batch(batch)
                all_entities.extend(batch_entities)
        
        # 去重
        unique_entities = self._deduplicate_entities(all_entities)
        
        # 更新统计
        self.stats["total_requests"] += len(requests)
        self.stats["entities_extracted"] += len(unique_entities)
        
        return unique_entities
    
    async def extract_relations(self, texts: List[str]) -> List[Relation]:
        """从文本列表中提取关系"""
        # 创建关系提取请求
        requests = []
        for text in texts:
            normalized_text = self.text_preprocessor.normalize_text(text)
            if normalized_text:
                request = ExtractionRequest(
                    id="",
                    text=normalized_text,
                    request_type="relation"
                )
                requests.append(request)
        
        # 优先级排序
        prioritized_requests = self.batch_processor.prioritize_requests(requests)
        
        # 按相似度分组
        groups = self.batch_processor.group_by_similarity(prioritized_requests)
        
        # 处理每个组
        all_relations = []
        for group in groups:
            # 动态批量大小调整
            batches = self.batch_processor.dynamic_batch_sizing(group)
            
            for batch in batches:
                batch_relations = await self._process_relation_batch(batch)
                all_relations.extend(batch_relations)
        
        # 去重
        unique_relations = self._deduplicate_relations(all_relations)
        
        # 更新统计
        self.stats["total_requests"] += len(requests)
        self.stats["relations_extracted"] += len(unique_relations)
        
        return unique_relations
    
    async def extract_attributes(self, texts: List[str]) -> List[Dict[str, Any]]:
        """从文本列表中提取属性"""
        # 创建属性抽取请求
        requests = []
        for text in texts:
            normalized_text = self.text_preprocessor.normalize_text(text)
            if normalized_text:
                request = ExtractionRequest(
                    id="",
                    text=normalized_text,
                    request_type="attribute"
                )
                requests.append(request)
        
        # 优先级排序
        prioritized_requests = self.batch_processor.prioritize_requests(requests)
        
        # 按相似度分组
        groups = self.batch_processor.group_by_similarity(prioritized_requests)
        
        # 处理每个组
        all_attributes = []
        for group in groups:
            # 动态批量大小调整
            batches = self.batch_processor.dynamic_batch_sizing(group)
            
            for batch in batches:
                batch_attributes = await self._process_attribute_batch(batch)
                all_attributes.extend(batch_attributes)
        
        # 更新统计
        self.stats["total_requests"] += len(requests)
        
        return all_attributes
    
    async def _process_entity_batch(self, batch: List[ExtractionRequest]) -> List[Entity]:
        """处理实体抽取批次"""
        # 合并批次中的文本
        combined_text = "\n".join([f"文本{i+1}: {req.text}" for i, req in enumerate(batch)])
        
        # 生成提示词
        prompt = self.prompt_templates.ENTITY_RECOGNITION_TEMPLATE.format(text=combined_text)
        
        # 调用LLM
        response = await self.llm_client.generate_completion(prompt)
        self.stats["api_calls"] += 1
        
        # 解析结果
        result = self.result_parser.parse_json_response(response)
        
        # 验证结果
        if not self.result_parser.validate_schema(result, ["companies", "investors"]):
            self.stats["failed_requests"] += len(batch)
            return []
        
        # 转换为实体对象
        entities = []
        
        # 处理公司实体
        for company_name in result.get("companies", []):
            entity = Entity(
                id=hashlib.md5(f"company_{company_name}".encode()).hexdigest(),
                name=company_name,
                type="company"
            )
            entities.append(entity)
        
        # 处理投资机构实体
        for investor_name in result.get("investors", []):
            entity = Entity(
                id=hashlib.md5(f"investor_{investor_name}".encode()).hexdigest(),
                name=investor_name,
                type="investor"
            )
            entities.append(entity)
        
        self.stats["successful_requests"] += len(batch)
        return entities
    
    async def _process_relation_batch(self, batch: List[ExtractionRequest]) -> List[Relation]:
        """处理关系抽取批次"""
        # 合并批次中的文本
        combined_text = "\n".join([f"文本{i+1}: {req.text}" for i, req in enumerate(batch)])
        
        # 生成提示词
        prompt = self.prompt_templates.RELATION_EXTRACTION_TEMPLATE.format(text=combined_text)
        
        # 调用LLM
        response = await self.llm_client.generate_completion(prompt)
        self.stats["api_calls"] += 1
        
        # 解析结果
        result = self.result_parser.parse_json_response(response)
        
        # 验证结果
        if not self.result_parser.validate_schema(result, ["relations"]):
            self.stats["failed_requests"] += len(batch)
            return []
        
        # 转换为关系对象
        relations = []
        for rel_data in result.get("relations", []):
            investor_name = rel_data.get("investor")
            company_name = rel_data.get("company")
            
            if investor_name and company_name:
                relation = Relation(
                    id=hashlib.md5(f"{investor_name}_{company_name}".encode()).hexdigest(),
                    source=hashlib.md5(f"investor_{investor_name}".encode()).hexdigest(),
                    target=hashlib.md5(f"company_{company_name}".encode()).hexdigest(),
                    type="invests_in",
                    properties={"source_text": combined_text}
                )
                relations.append(relation)
        
        self.stats["successful_requests"] += len(batch)
        return relations
    
    async def _process_attribute_batch(self, batch: List[ExtractionRequest]) -> List[Dict[str, Any]]:
        """处理属性抽取批次"""
        # 合并批次中的文本
        combined_text = "\n".join([f"文本{i+1}: {req.text}" for i, req in enumerate(batch)])
        
        # 生成提示词
        prompt = self.prompt_templates.ATTRIBUTE_EXTRACTION_TEMPLATE.format(text=combined_text)
        
        # 调用LLM
        response = await self.llm_client.generate_completion(prompt)
        self.stats["api_calls"] += 1
        
        # 解析结果
        result = self.result_parser.parse_json_response(response)
        
        # 验证结果
        if not self.result_parser.validate_schema(result, ["amount", "round", "date"]):
            self.stats["failed_requests"] += len(batch)
            return []
        
        # 为每个请求创建属性字典
        attributes = []
        for req in batch:
            attr = {
                "request_id": req.id,
                "text": req.text,
                "amount": result.get("amount"),
                "round": result.get("round"),
                "date": result.get("date")
            }
            attributes.append(attr)
        
        self.stats["successful_requests"] += len(batch)
        return attributes
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体列表"""
        unique_entities = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in unique_entities:
                unique_entities[key] = entity
            else:
                # 合并别名
                existing = unique_entities[key]
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
        
        return list(unique_entities.values())
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系列表"""
        unique_relations = {}
        
        for relation in relations:
            key = (relation.source, relation.target, relation.type)
            if key not in unique_relations:
                unique_relations[key] = relation
        
        return list(unique_relations.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "api_calls": 0,
            "entities_extracted": 0,
            "relations_extracted": 0
        }


# 全局处理器实例
_global_processor = None


def get_llm_processor() -> SimplifiedLLMProcessor:
    """获取全局LLM处理器实例"""
    global _global_processor
    if _global_processor is None:
        _global_processor = SimplifiedLLMProcessor()
    return _global_processor


async def extract_entities_from_texts(texts: List[str]) -> List[Entity]:
    """从文本列表中提取实体的便捷函数"""
    processor = get_llm_processor()
    return await processor.extract_entities(texts)


async def extract_relations_from_texts(texts: List[str]) -> List[Relation]:
    """从文本列表中提取关系的便捷函数"""
    processor = get_llm_processor()
    return await processor.extract_relations(texts)


async def extract_attributes_from_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """从文本列表中提取属性的便捷函数"""
    processor = get_llm_processor()
    return await processor.extract_attributes(texts)