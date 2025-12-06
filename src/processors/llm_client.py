"""
LLM客户端接口（预留）
为后续LLM增强功能预留接口
"""

import json
import logging
from typing import Dict, List, Optional, Any, cast
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClientInterface(ABC):
    """LLM客户端接口"""
    
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


class MockLLMClient(LLMClientInterface):
    """模拟LLM客户端（用于开发和测试）"""
    
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
        
        # 模拟LLM调用延迟
        # time.sleep(0.1)
        
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
    
    def process_batch_requests(self, llm_client: LLMClientInterface, batch_size: int = 50) -> Dict:
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
    
    def _process_single_request(self, request: Dict, llm_client: LLMClientInterface) -> Any:
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
mock_llm_client = MockLLMClient()


def get_llm_client() -> LLMClientInterface:
    """获取LLM客户端实例"""
    return mock_llm_client


def get_enhancement_tracker() -> LLMEnhancementTracker:
    """获取增强跟踪器实例"""
    return llm_enhancement_tracker