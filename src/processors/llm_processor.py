"""
LLM处理器 - 业务逻辑处理模块
负责处理LLM相关的业务逻辑，与客户端连接分离
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, cast
import logging
import os
from .llm_client import get_llm_client, LLMClientInterface

# 配置日志
logger = logging.getLogger(__name__)


class LLMProcessorInterface(ABC):
    """LLM处理器接口 - 定义业务逻辑处理方法"""
    
    @abstractmethod
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述"""
        pass
    
    @abstractmethod
    def resolve_entity_conflicts(self, conflicting_entities: List[Dict]) -> Dict[str, Any]:
        """解决实体冲突"""
        pass
    
    @abstractmethod
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系"""
        pass
    
    @abstractmethod
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业"""
        pass
    
    @abstractmethod
    def standardize_investor_name(self, investor_name: str, context: Dict) -> str:
        """标准化投资方名称"""
        pass


class MockLLMProcessor(LLMProcessorInterface):
    """模拟LLM处理器（用于开发和测试）"""
    
    def __init__(self):
        self.call_count = 0
        self.mock_responses = {
            'enhance_entity_description': "这是一个经过LLM增强的实体描述。",
            'resolve_entity_conflicts': {'resolved_entity': '统一实体', 'confidence': 0.9},
            'extract_relationships_from_text': [],
            'classify_company_industry': ['科技', '金融'],
            'standardize_investor_name': '标准化投资方名称'
        }
    
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 增强实体描述: {entity_name}")
        
        return f"{entity_name}是一家专注于{context.get('industry', '未知领域')}的公司，成立于{context.get('establish_date', '未知时间')}。"
    
    def resolve_entity_conflicts(self, conflicting_entities: List[Dict]) -> Dict[str, Any]:
        """解决实体冲突（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 解决实体冲突: {len(conflicting_entities)} 个冲突")
        
        if conflicting_entities:
            # 选择置信度最高的实体
            best_entity = max(conflicting_entities, key=lambda x: x.get('confidence', 0))
            return {
                'resolved_entity': best_entity,
                'confidence': best_entity.get('confidence', 0),
                'merged_aliases': [e.get('name') for e in conflicting_entities if e != best_entity]
            }
        
        return cast(Dict[str, Any], self.mock_responses['resolve_entity_conflicts'])
    
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 从文本提取关系: {text[:50]}...")
        
        # 简单的模式匹配模拟
        relationships = []
        
        # 模拟投资关系提取
        if '投资' in text or '融资' in text:
            relationships.append({
                'type': 'investment',
                'confidence': 0.7,
                'description': '模拟投资关系'
            })
        
        return relationships
    
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 公司行业分类: {company_info.get('name', '未知公司')}")
        
        name = company_info.get('name', '')
        description = company_info.get('description', '')
        
        # 基于关键词的简单分类模拟
        industries = []
        
        tech_keywords = ['科技', '技术', '智能', '软件', '互联网', '数据', 'AI', '人工智能']
        finance_keywords = ['金融', '银行', '投资', '理财', '支付', '保险']
        healthcare_keywords = ['医疗', '健康', '生物', '医药']
        
        text = f"{name} {description}"
        
        if any(keyword in text for keyword in tech_keywords):
            industries.append('科技')
        
        if any(keyword in text for keyword in finance_keywords):
            industries.append('金融')
        
        if any(keyword in text for keyword in healthcare_keywords):
            industries.append('医疗健康')
        
        if not industries:
            industries.append('其他')
        
        return industries
    
    def standardize_investor_name(self, investor_name: str, context: Dict) -> str:
        """标准化投资方名称（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 标准化投资方名称: {investor_name}")
        
        # 简单的标准化规则
        standardized = investor_name.strip()
        
        # 移除常见的后缀
        suffixes = ['有限公司', '股份有限公司', '集团', '公司', '基金', '投资']
        for suffix in suffixes:
            if standardized.endswith(suffix) and len(standardized) > len(suffix):
                # 保留部分后缀信息
                break
        
        return standardized
    
    def get_stats(self) -> Dict[str, int]:
        """获取调用统计"""
        return cast(Dict[str, int], {
            'total_calls': self.call_count,
            'mock_responses': len(self.mock_responses)
        })


class OpenAICompatibleProcessor(LLMProcessorInterface):
    """OpenAI兼容处理器 - 使用真实的LLM客户端"""
    
    def __init__(self, llm_client: Optional[LLMClientInterface] = None):
        self.llm_client = llm_client or get_llm_client()
        self.call_count = 0
        logger.info("初始化OpenAI兼容处理器")
    
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述（使用真实LLM）"""
        self.call_count += 1
        logger.info(f"[REAL] 增强实体描述: {entity_name}")
        
        prompt = f"""
        请为以下实体生成详细的描述信息：
        
        实体名称：{entity_name}
        上下文信息：{json.dumps(context, ensure_ascii=False, indent=2)}
        
        请提供一段简洁但全面的描述，包括公司的主要业务、特点和相关信息。
        描述应该专业、准确，长度控制在100-200字之间。
        """
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"增强实体描述失败: {e}")
            # 回退到简单的描述
            return f"{entity_name}是一家专注于{context.get('industry', '未知领域')}的公司。"
    
    def resolve_entity_conflicts(self, conflicting_entities: List[Dict]) -> Dict[str, Any]:
        """解决实体冲突（使用真实LLM）"""
        self.call_count += 1
        logger.info(f"[REAL] 解决实体冲突: {len(conflicting_entities)} 个冲突")
        
        if not conflicting_entities:
            return {'resolved_entity': None, 'confidence': 0.0}
        
        prompt = f"""
        请分析以下实体冲突并提供解决方案：
        
        冲突实体列表：
        {json.dumps(conflicting_entities, ensure_ascii=False, indent=2)}
        
        请确定：
        1. 这些实体是否代表同一个真实实体？
        2. 如果是，哪个是最准确的标准名称？
        3. 其他名称应该如何处理（作为别名还是错误）？
        4. 提供一个统一的实体表示。
        
        请以JSON格式返回结果，包含resolved_entity、confidence和merged_aliases字段。
        """
        
        try:
            response = self.llm_client.generate_response(prompt)
            # 尝试解析JSON响应
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"解决实体冲突失败: {e}")
            # 回退到简单的解决方案
            best_entity = max(conflicting_entities, key=lambda x: x.get('confidence', 0))
            return {
                'resolved_entity': best_entity,
                'confidence': best_entity.get('confidence', 0),
                'merged_aliases': [e.get('name') for e in conflicting_entities if e != best_entity]
            }
    
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系（使用真实LLM）"""
        self.call_count += 1
        logger.info(f"[REAL] 从文本提取关系: {text[:100]}...")
        
        prompt = f"""
        请从以下文本中提取实体之间的关系：
        
        文本内容：
        {text}
        
        请识别文本中提到的所有实体（公司、人物、产品等）以及它们之间的关系。
        关系类型可以包括：投资、合作、竞争、收购、子母公司等。
        
        请以JSON数组格式返回结果，每个关系包含以下字段：
        - type: 关系类型
        - entities: 涉及的实体列表
        - confidence: 置信度（0-1）
        - description: 关系描述
        - metadata: 额外的元数据
        """
        
        try:
            response = self.llm_client.generate_response(prompt)
            # 尝试解析JSON响应
            relationships = json.loads(response)
            return relationships if isinstance(relationships, list) else []
        except Exception as e:
            logger.error(f"提取关系失败: {e}")
            # 回退到简单的模式匹配
            relationships = []
            if any(keyword in text for keyword in ['投资', '融资', '领投', '跟投']):
                relationships.append({
                    'type': 'investment',
                    'confidence': 0.6,
                    'description': '投资关系',
                    'entities': [],
                    'metadata': {}
                })
            return relationships
    
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业（使用真实LLM）"""
        self.call_count += 1
        logger.info(f"[REAL] 公司行业分类: {company_info.get('name', '未知公司')}")
        
        prompt = f"""
        请为以下公司进行行业分类：
        
        公司信息：
        {json.dumps(company_info, ensure_ascii=False, indent=2)}
        
        请基于公司名称、描述、业务范围等信息，确定公司所属的行业类别。
        请提供1-3个最相关的行业分类。
        
        请以JSON数组格式返回行业列表，例如：["科技", "金融", "医疗健康"]
        """
        
        try:
            response = self.llm_client.generate_response(prompt)
            # 尝试解析JSON响应
            industries = json.loads(response)
            return industries if isinstance(industries, list) else []
        except Exception as e:
            logger.error(f"行业分类失败: {e}")
            # 回退到基于关键词的分类
            name = company_info.get('name', '')
            description = company_info.get('description', '')
            text = f"{name} {description}"
            
            industries = []
            keywords_map = {
                '科技': ['科技', '技术', '智能', '软件', '互联网', '数据', 'AI', '人工智能'],
                '金融': ['金融', '银行', '投资', '理财', '支付', '保险'],
                '医疗健康': ['医疗', '健康', '生物', '医药']
            }
            
            for industry, keywords in keywords_map.items():
                if any(keyword in text for keyword in keywords):
                    industries.append(industry)
            
            return industries if industries else ['其他']
    
    def standardize_investor_name(self, investor_name: str, context: Dict) -> str:
        """标准化投资方名称（使用真实LLM）"""
        self.call_count += 1
        logger.info(f"[REAL] 标准化投资方名称: {investor_name}")
        
        prompt = f"""
        请标准化以下投资方名称：
        
        投资方名称：{investor_name}
        上下文信息：{json.dumps(context, ensure_ascii=False, indent=2)}
        
        请提供该投资方的标准名称，移除不必要的后缀，统一格式。
        例如：
        - "深圳市腾讯计算机系统有限公司" -> "腾讯"
        - "阿里巴巴（中国）网络技术有限公司" -> "阿里巴巴"
        
        请只返回标准化后的名称，不要包含其他内容。
        """
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"标准化投资方名称失败: {e}")
            # 回退到简单的标准化
            return investor_name.strip()
    
    def get_stats(self) -> Dict[str, int]:
        """获取调用统计"""
        return cast(Dict[str, int], {
            'total_calls': self.call_count
        })


class BatchLLMProcessor(OpenAICompatibleProcessor):
    """批量LLM处理器 - 支持一次性处理多条数据"""
    
    def __init__(self, batch_size: int = 50, llm_client: Optional[LLMClientInterface] = None):
        super().__init__(llm_client)
        self.batch_size = batch_size
        self.batch_call_count = 0
        logger.info(f"初始化批量LLM处理器，批量大小: {batch_size}")
    
    def batch_enhance_entity_descriptions(self, entities: List[Dict[str, Any]]) -> List[str]:
        """批量增强实体描述"""
        if not entities:
            return []
        
        self.batch_call_count += 1
        logger.info(f"[BATCH] 批量增强实体描述: {len(entities)} 个实体")
        
        # 构建批量提示词
        batch_prompt = self._build_batch_prompt_for_descriptions(entities)
        
        try:
            # 使用真实LLM客户端进行批量处理
            response = self.llm_client.generate_response(
                prompt=batch_prompt,
                system_prompt="你是一个专业的企业信息分析师，请为给定的实体生成详细描述。"
            )
            
            # 解析批量响应
            try:
                result = json.loads(response)
                if 'descriptions' in result and isinstance(result['descriptions'], list):
                    descriptions = result['descriptions']
                    # 确保返回的描述数量与输入实体数量一致
                    if len(descriptions) == len(entities):
                        return descriptions
                    else:
                        logger.warning(f"批量描述数量不匹配: 期望 {len(entities)}, 实际 {len(descriptions)}")
            except json.JSONDecodeError:
                logger.warning("无法解析批量描述的JSON响应，使用回退方法")
            
        except Exception as e:
            logger.error(f"批量增强实体描述失败: {e}")
        
        # 回退到逐个处理
        enhanced_descriptions = []
        for entity in entities:
            name = entity.get('name', '未知实体')
            context = entity.get('context', {})
            description = self.enhance_entity_description(name, context)
            enhanced_descriptions.append(description)
        
        return enhanced_descriptions
    
    def batch_resolve_entity_conflicts(self, conflict_groups: List[List[Dict]]) -> List[Dict[str, Any]]:
        """批量解决实体冲突"""
        if not conflict_groups:
            return []
        
        self.batch_call_count += 1
        logger.info(f"[BATCH] 批量解决实体冲突: {len(conflict_groups)} 个冲突组")
        
        # 构建批量提示词
        batch_prompt = self._build_batch_prompt_for_conflicts(conflict_groups)
        
        try:
            # 使用真实LLM客户端进行批量处理
            response = self.llm_client.generate_response(
                prompt=batch_prompt,
                system_prompt="你是一个专业的数据分析师，请解决给定的实体冲突问题。"
            )
            
            # 解析批量响应
            try:
                result = json.loads(response)
                if 'conflict_resolutions' in result and isinstance(result['conflict_resolutions'], list):
                    resolutions = result['conflict_resolutions']
                    # 确保返回的解决数量与输入冲突组数量一致
                    if len(resolutions) == len(conflict_groups):
                        return resolutions
                    else:
                        logger.warning(f"批量冲突解决数量不匹配: 期望 {len(conflict_groups)}, 实际 {len(resolutions)}")
            except json.JSONDecodeError:
                logger.warning("无法解析批量冲突解决的JSON响应，使用回退方法")
            
        except Exception as e:
            logger.error(f"批量解决实体冲突失败: {e}")
        
        # 回退到逐个处理
        results = []
        for conflict_group in conflict_groups:
            if conflict_group:
                result = self.resolve_entity_conflicts(conflict_group)
                results.append(result)
            else:
                results.append({'resolved_entity': None, 'confidence': 0.0})
        
        return results
    
    def batch_extract_relationships_from_text(self, texts: List[str]) -> List[List[Dict]]:
        """批量从文本中提取关系"""
        if not texts:
            return []
        
        self.batch_call_count += 1
        logger.info(f"[BATCH] 批量提取关系: {len(texts)} 段文本")
        
        relationships_list = []
        for text in texts:
            relationships = self.extract_relationships_from_text(text)
            relationships_list.append(relationships)
        
        return relationships_list
    
    def batch_classify_company_industries(self, companies_info: List[Dict]) -> List[List[str]]:
        """批量分类公司行业"""
        if not companies_info:
            return []
        
        self.batch_call_count += 1
        logger.info(f"[BATCH] 批量行业分类: {len(companies_info)} 家公司")
        
        industries_list = []
        for company_info in companies_info:
            industries = self.classify_company_industry(company_info)
            industries_list.append(industries)
        
        return industries_list
    
    def batch_standardize_investor_names(self, investor_names: List[str], contexts: List[Dict]) -> List[str]:
        """批量标准化投资方名称"""
        if not investor_names or not contexts:
            return []
        
        self.batch_call_count += 1
        logger.info(f"[BATCH] 批量标准化投资方名称: {len(investor_names)} 个名称")
        
        # 确保contexts长度与investor_names一致
        if len(contexts) < len(investor_names):
            contexts.extend([{}] * (len(investor_names) - len(contexts)))
        
        standardized_names = []
        for i, name in enumerate(investor_names):
            context = contexts[i] if i < len(contexts) else {}
            standardized = self.standardize_investor_name(name, context)
            standardized_names.append(standardized)
        
        return standardized_names
    
    def _build_batch_prompt_for_descriptions(self, entities: List[Dict[str, Any]]) -> str:
        """构建批量实体描述的提示词"""
        prompt_parts = []
        prompt_parts.append("请为以下实体生成详细的描述信息：")
        prompt_parts.append("")
        
        for i, entity in enumerate(entities, 1):
            name = entity.get('name', '未知实体')
            context = entity.get('context', {})
            
            prompt_parts.append(f"实体 {i}: {name}")
            if context:
                prompt_parts.append(f"  上下文信息: {json.dumps(context, ensure_ascii=False)}")
            prompt_parts.append("")
        
        prompt_parts.append("请为每个实体生成一段详细的描述，描述应该包含公司的业务、特点、成立时间等信息。")
        prompt_parts.append("请以JSON格式返回结果，格式如下：")
        prompt_parts.append('{"descriptions": ["描述1", "描述2", ...]}')
        
        return "\n".join(prompt_parts)
    
    def _build_batch_prompt_for_conflicts(self, conflict_groups: List[List[Dict]]) -> str:
        """构建批量冲突解决的提示词"""
        prompt_parts = []
        prompt_parts.append("请解决以下实体冲突组，判断每组中的实体是否代表同一个实体：")
        prompt_parts.append("")
        
        for i, conflict_group in enumerate(conflict_groups, 1):
            prompt_parts.append(f"冲突组 {i}:")
            for j, entity in enumerate(conflict_group, 1):
                name = entity.get('name', '未知实体')
                description = entity.get('description', '')
                context = entity.get('context', {})
                
                prompt_parts.append(f"  实体 {j}: {name}")
                if description:
                    prompt_parts.append(f"    描述: {description}")
                if context:
                    prompt_parts.append(f"    上下文: {json.dumps(context, ensure_ascii=False)}")
            prompt_parts.append("")
        
        prompt_parts.append("对于每个冲突组，请判断这些实体是否代表同一个实体。")
        prompt_parts.append("请以JSON格式返回结果，格式如下：")
        prompt_parts.append('{"conflict_resolutions": [{"resolved_entity": {"name": "合并后的名称", "description": "合并后的描述"}, "confidence": 0.9}, ...]}')
        prompt_parts.append("如果实体不冲突，confidence应该较低；如果确定是同一实体，confidence应该较高。")
        
        return "\n".join(prompt_parts)
    
    def process_in_batches(self, items: List[Any], batch_processor: callable) -> List[Any]:
        """通用批量处理函数"""
        if not items:
            return []
        
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = batch_processor(batch)
            results.extend(batch_results)
        
        return results
    
    def get_batch_stats(self) -> Dict[str, int]:
        """获取批量处理统计"""
        base_stats = self.get_stats()
        base_stats['batch_calls'] = self.batch_call_count
        base_stats['batch_size'] = self.batch_size
        return base_stats


# 全局实例
mock_llm_processor = BatchLLMProcessor()


class LLMEnhancementTracker:
    """LLM增强需求跟踪器"""
    
    def __init__(self):
        self.enhancement_requests = []
        self.processed_requests = []
        self.stats = {
            'total_requests': 0,
            'processed_requests': 0,
            'pending_requests': 0
        }
    
    def add_enhancement_request(self, request_type: str, data: Dict, priority: str = 'medium') -> str:
        """添加增强请求"""
        request_id = f"enh_{len(self.enhancement_requests) + 1:06d}"
        
        request = {
            'id': request_id,
            'type': request_type,
            'data': data,
            'priority': priority,
            'status': 'pending',
            'created_at': self._get_timestamp(),
            'result': None
        }
        
        self.enhancement_requests.append(request)
        self.stats['total_requests'] += 1
        self.stats['pending_requests'] += 1
        
        logger.info(f"添加LLM增强请求: {request_id} ({request_type})")
        return request_id
    
    def get_pending_requests(self, request_type: Optional[str] = None, priority: Optional[str] = None) -> List[Dict]:
        """获取待处理请求"""
        pending = [req for req in self.enhancement_requests if req['status'] == 'pending']
        
        if request_type:
            pending = [req for req in pending if req['type'] == request_type]
        
        if priority:
            pending = [req for req in pending if req['priority'] == priority]
        
        return pending
    
    def process_batch_requests(self, llm_client: LLMProcessorInterface, batch_size: int = 50) -> Dict:
        """批量处理请求"""
        pending_requests = self.get_pending_requests()
        
        if not pending_requests:
            return {'processed': 0, 'results': []}
        
        # 按优先级排序
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        pending_requests.sort(key=lambda x: priority_order.get(x['priority'], 1))
        
        # 分批处理
        batch = pending_requests[:batch_size]
        results = []
        
        logger.info(f"批量处理LLM请求: {len(batch)} 个请求")
        
        for request in batch:
            try:
                result = self._process_single_request(request, llm_client)
                request['status'] = 'completed'
                request['result'] = result
                request['processed_at'] = self._get_timestamp()
                
                self.processed_requests.append(request)
                results.append(result)
                
                self.stats['processed_requests'] += 1
                self.stats['pending_requests'] -= 1
                
            except Exception as e:
                logger.error(f"处理LLM请求失败: {request['id']}, 错误: {e}")
                request['status'] = 'failed'
                request['error'] = str(e)
        
        logger.info(f"批量处理完成: {len(results)} 个成功")
        
        return {
            'processed': len(results),
            'results': results,
            'remaining': len(pending_requests) - len(batch)
        }
    
    def process_batch_requests_smart(self, llm_processor: BatchLLMProcessor, batch_size: int = 50) -> Dict:
        """智能批量处理请求 - 多条数据一次性处理"""
        pending_requests = self.get_pending_requests()
        
        if not pending_requests:
            return {'processed': 0, 'results': [], 'batches': 0}
        
        # 按类型分组
        grouped_requests = self._group_requests_by_type(pending_requests)
        
        all_results = []
        total_processed = 0
        batch_count = 0
        
        # 按类型批量处理
        for request_type, requests in grouped_requests.items():
            if not requests:
                continue
            
            # 限制每批处理数量
            batch_requests = requests[:batch_size]
            
            try:
                batch_results = self._process_batch_by_type(batch_requests, request_type, llm_processor)
                all_results.extend(batch_results)
                total_processed += len(batch_results)
                batch_count += 1
                
                # 更新请求状态
                for i, request in enumerate(batch_requests):
                    if i < len(batch_results):
                        request['status'] = 'completed'
                        request['result'] = batch_results[i]
                        request['processed_at'] = self._get_timestamp()
                        
                        self.processed_requests.append(request)
                        self.stats['processed_requests'] += 1
                        self.stats['pending_requests'] -= 1
                
                logger.info(f"智能批量处理 {request_type}: {len(batch_results)} 个请求")
                
            except Exception as e:
                logger.error(f"智能批量处理失败 {request_type}: {e}")
                # 回退到单个处理
                for request in batch_requests:
                    try:
                        result = self._process_single_request(request, llm_processor)
                        request['status'] = 'completed'
                        request['result'] = result
                        request['processed_at'] = self._get_timestamp()
                        
                        self.processed_requests.append(request)
                        all_results.append(result)
                        
                        self.stats['processed_requests'] += 1
                        self.stats['pending_requests'] -= 1
                        
                        total_processed += 1
                    except Exception as single_e:
                        logger.error(f"单个处理也失败: {request['id']}, 错误: {single_e}")
                        request['status'] = 'failed'
                        request['error'] = str(single_e)
        
        return {
            'processed': total_processed,
            'results': all_results,
            'batches': batch_count,
            'remaining': len(pending_requests) - total_processed
        }
    
    def _process_single_request(self, request: Dict, llm_client: LLMProcessorInterface) -> Any:
        """处理单个请求"""
        request_type = request['type']
        data = request['data']
        
        if request_type == 'enhance_description':
            return llm_client.enhance_entity_description(
                data['entity_name'],
                data.get('context', {})
            )
        
        elif request_type == 'resolve_conflict':
            return llm_client.resolve_entity_conflicts(data['conflicting_entities'])
        
        elif request_type == 'extract_relationships':
            return llm_client.extract_relationships_from_text(data['text'])
        
        elif request_type == 'classify_industry':
            return llm_client.classify_company_industry(data['company_info'])
        
        elif request_type == 'standardize_name':
            return llm_client.standardize_investor_name(
                data['investor_name'],
                data.get('context', {})
            )
        
        else:
            raise ValueError(f"未知的请求类型: {request_type}")
    
    def _group_requests_by_type(self, requests: List[Dict]) -> Dict[str, List[Dict]]:
        """按类型分组请求"""
        groups = {
            'enhance_description': [],
            'resolve_conflict': [],
            'extract_relationships': [],
            'classify_industry': [],
            'standardize_name': []
        }
        
        for request in requests:
            request_type = request['type']
            if request_type in groups:
                groups[request_type].append(request)
        
        return groups
    
    def _process_batch_by_type(self, requests: List[Dict], request_type: str, llm_processor: BatchLLMProcessor) -> List[Any]:
        """按类型批量处理"""
        if request_type == 'enhance_description':
            entities = [{
                'name': req['data']['entity_name'],
                'context': req['data'].get('context', {})
            } for req in requests]
            return llm_processor.batch_enhance_entity_descriptions(entities)
        
        elif request_type == 'resolve_conflict':
            conflict_groups = [req['data']['conflicting_entities'] for req in requests]
            return llm_processor.batch_resolve_entity_conflicts(conflict_groups)
        
        elif request_type == 'extract_relationships':
            texts = [req['data']['text'] for req in requests]
            return llm_processor.batch_extract_relationships_from_text(texts)
        
        elif request_type == 'classify_industry':
            companies_info = [req['data']['company_info'] for req in requests]
            return llm_processor.batch_classify_company_industries(companies_info)
        
        elif request_type == 'standardize_name':
            investor_names = [req['data']['investor_name'] for req in requests]
            contexts = [req['data'].get('context', {}) for req in requests]
            return llm_processor.batch_standardize_investor_names(investor_names, contexts)
        
        else:
            raise ValueError(f"不支持的批量处理类型: {request_type}")
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return cast(Dict[str, int], self.stats.copy())
    
    def export_pending_requests(self, filepath: str) -> None:
        """导出待处理请求"""
        pending = self.get_pending_requests()
        
        export_data = {
            'export_time': self._get_timestamp(),
            'total_pending': len(pending),
            'requests': pending
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"导出待处理请求: {len(pending)} 个请求到 {filepath}")
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


# 全局实例
llm_enhancement_tracker = LLMEnhancementTracker()


def get_llm_processor() -> LLMProcessorInterface:
    """获取LLM处理器实例 - 使用OpenAI兼容客户端"""
    llm_client = get_llm_client()
    return OpenAICompatibleProcessor(llm_client)


def get_enhancement_tracker() -> LLMEnhancementTracker:
    """获取LLM增强跟踪器实例"""
    return llm_enhancement_tracker


def get_batch_llm_processor() -> BatchLLMProcessor:
    """获取批量LLM处理器实例 - 使用OpenAI兼容客户端"""
    llm_client = get_llm_client()
    return BatchLLMProcessor(llm_client=llm_client)