"""
LLM客户端 - 仅保留客户端连接功能
负责与LLM服务的连接和通信，不包含业务逻辑处理
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, cast

# 配置日志
logger = logging.getLogger(__name__)


class LLMClientInterface(ABC):
    """LLM客户端接口 - 定义与LLM服务交互的基本方法"""
    
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
        logger.info("初始化模拟LLM客户端")
    
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 增强实体描述: {entity_name}")
        
        # 这里应该调用真实的LLM API，现在返回模拟数据
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
        
        return {'resolved_entity': None, 'confidence': 0.0}
    
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 从文本提取关系: {text[:50]}...")
        
        # 这里应该调用真实的LLM API进行关系提取
        relationships = []
        
        # 简单的关键词匹配模拟
        if '投资' in text or '融资' in text:
            relationships.append({
                'type': 'investment',
                'confidence': 0.7,
                'description': '投资关系'
            })
        
        return relationships
    
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业（模拟实现）"""
        self.call_count += 1
        logger.info(f"[MOCK] 公司行业分类: {company_info.get('name', '未知公司')}")
        
        # 这里应该调用真实的LLM API进行行业分类
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
        
        # 这里应该调用真实的LLM API进行名称标准化
        standardized = investor_name.strip()
        
        # 简单的标准化规则
        suffixes = ['有限公司', '股份有限公司', '集团', '公司', '基金', '投资']
        for suffix in suffixes:
            if standardized.endswith(suffix) and len(standardized) > len(suffix):
                break
        
        return standardized
    
    def get_stats(self) -> Dict[str, int]:
        """获取调用统计"""
        return cast(Dict[str, int], {
            'total_calls': self.call_count
        })


# 全局实例
mock_llm_client = MockLLMClient()


def get_llm_client() -> LLMClientInterface:
    """获取LLM客户端实例"""
    return mock_llm_client