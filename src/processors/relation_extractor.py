"""
优化的关系提取流程

专注于从金融投资文本中提取投资关系，结合规则和LLM方法
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib

from .llm_processor import (
    SimplifiedLLMProcessor, 
    Entity, 
    Relation, 
    ExtractionRequest,
    get_llm_processor
)

# 配置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@dataclass
class InvestmentEvent:
    """投资事件数据类"""
    id: str
    description: str  # 事件描述
    investors: List[str]  # 投资方列表
    company: str  # 被投公司
    date: Optional[str] = None  # 投资日期
    round: Optional[str] = None  # 投资轮次
    amount: Optional[str] = None  # 投资金额
    confidence: float = 0.0  # 置信度


class RuleBasedExtractor:
    """基于规则的关系提取器"""
    
    def __init__(self):
        # 投资相关关键词
        self.investment_keywords = [
            "投资", "融资", "参投", "领投", "跟投", "注资", "入股", 
            "并购", "收购", "战略投资", "股权投资", "基金投资"
        ]
        
        # 投资轮次关键词
        self.round_keywords = [
            "种子轮", "天使轮", "Pre-A轮", "Pre-A", "A轮", "A+轮", 
            "B轮", "B+轮", "C轮", "C+轮", "D轮", "E轮", "F轮",
            "Pre-IPO", "IPO", "战略融资", "并购", "新三板"
        ]
        
        # 金额模式
        self.amount_patterns = [
            r'(\d+(?:\.\d+)?)\s*万',  # XX万
            r'(\d+(?:\.\d+)?)\s*千万',  # XX千万
            r'(\d+(?:\.\d+)?)\s*亿',  # XX亿
            r'(\d+(?:\.\d+)?)\s*百亿',  # XX百亿
            r'(\d+(?:\.\d+)?)\s*million',  # XX million
            r'(\d+(?:\.\d+)?)\s*billion',  # XX billion
            r'\$(\d+(?:\.\d+)?)',  # $XX
            r'(\d+(?:\.\d+)?)\s*美元',  # XX美元
            r'(\d+(?:\.\d+)?)\s*人民币',  # XX人民币
        ]
        
        # 日期模式
        self.date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # YYYY年MM月DD日
            r'(\d{4})/(\d{1,2})/(\d{1,2})',  # YYYY/MM/DD
        ]
        
        # 投资机构常见后缀
        self.investor_suffixes = [
            "资本", "投资", "创投", "基金", "资产", "创投基金", 
            "投资管理", "投资集团", "风险投资", "股权投资"
        ]
        
        # 公司常见后缀
        self.company_suffixes = [
            "公司", "有限", "集团", "科技", "股份", "网络", 
            "企业", "工作室", "技术", "信息", "数码"
        ]
    
    def extract_investment_events(self, text: str) -> List[InvestmentEvent]:
        """从文本中提取投资事件"""
        events = []
        
        # 预处理文本
        normalized_text = self._normalize_text(text)
        
        # 尝试多种提取模式
        # 模式1: "投资方 投资了 被投公司"
        pattern1_events = self._extract_pattern1(normalized_text)
        events.extend(pattern1_events)
        
        # 模式2: "被投公司 获得了 投资方 投资"
        pattern2_events = self._extract_pattern2(normalized_text)
        events.extend(pattern2_events)
        
        # 模式3: "投资方、投资方等 投资了 被投公司"
        pattern3_events = self._extract_pattern3(normalized_text)
        events.extend(pattern3_events)
        
        # 模式4: "被投公司 完成了 XX轮 融资，投资方包括 投资方、投资方"
        pattern4_events = self._extract_pattern4(normalized_text)
        events.extend(pattern4_events)
        
        # 为每个事件提取属性
        for event in events:
            self._extract_event_attributes(event, normalized_text)
            # 计算置信度
            event.confidence = self._calculate_confidence(event, normalized_text)
        
        # 去重
        unique_events = self._deduplicate_events(events)
        
        return unique_events
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        if not text:
            return ""
        
        # 去除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 统一标点符号
        text = text.replace('，', ',').replace('。', '.').replace('；', ';')
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def _extract_pattern1(self, text: str) -> List[InvestmentEvent]:
        """模式1: "投资方 投资了 被投公司" """
        events = []
        
        # 构建正则表达式
        investors_pattern = r'([^，,。.]+?)\s*(?:投资|融资|参投|领投|跟投|注资|入股)\s*了?\s*([^，,。.]+?)(?:[,，。.]|$)'
        matches = re.finditer(investors_pattern, text)
        
        for match in matches:
            investors_str = match.group(1).strip()
            company_str = match.group(2).strip()
            
            # 分割多个投资方
            investors = self._split_investors(investors_str)
            
            if investors and company_str:
                event = InvestmentEvent(
                    id=hashlib.md5(f"{investors_str}_{company_str}".encode()).hexdigest(),
                    description=match.group(0),
                    investors=investors,
                    company=company_str
                )
                events.append(event)
        
        return events
    
    def _extract_pattern2(self, text: str) -> List[InvestmentEvent]:
        """模式2: "被投公司 获得了 投资方 投资" """
        events = []
        
        # 构建正则表达式
        pattern = r'([^，,。.]+?)\s*(?:获得|完成|完成)\s*([^，,。.]+?)\s*(?:投资|融资|注资|入股)(?:[,，。.]|$)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            company_str = match.group(1).strip()
            investors_str = match.group(2).strip()
            
            # 分割多个投资方
            investors = self._split_investors(investors_str)
            
            if investors and company_str:
                event = InvestmentEvent(
                    id=hashlib.md5(f"{investors_str}_{company_str}".encode()).hexdigest(),
                    description=match.group(0),
                    investors=investors,
                    company=company_str
                )
                events.append(event)
        
        return events
    
    def _extract_pattern3(self, text: str) -> List[InvestmentEvent]:
        """模式3: "投资方、投资方等 投资了 被投公司" """
        events = []
        
        # 构建正则表达式
        pattern = r'([^，,。.]+?(?:[、,]\s*[^，,。.]+?)*\s*(?:等|等机构)?)\s*(?:投资|融资|参投|领投|跟投|注资|入股)\s*了?\s*([^，,。.]+?)(?:[,，。.]|$)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            investors_str = match.group(1).strip()
            company_str = match.group(2).strip()
            
            # 分割多个投资方
            investors = self._split_investors(investors_str)
            
            if investors and company_str:
                event = InvestmentEvent(
                    id=hashlib.md5(f"{investors_str}_{company_str}".encode()).hexdigest(),
                    description=match.group(0),
                    investors=investors,
                    company=company_str
                )
                events.append(event)
        
        return events
    
    def _extract_pattern4(self, text: str) -> List[InvestmentEvent]:
        """模式4: "被投公司 完成了 XX轮 融资，投资方包括 投资方、投资方" """
        events = []
        
        # 构建正则表达式
        pattern = r'([^，,。.]+?)\s*(?:完成|完成)\s*([^，,。.]*?)\s*(?:融资|投资)\s*(?:[，,]\s*投资方(?:包括|为)\s*([^，,。.]+?))?(?:[,，。.]|$)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            company_str = match.group(1).strip()
            round_str = match.group(2).strip()
            investors_str = match.group(3).strip() if match.group(3) else ""
            
            # 分割多个投资方
            investors = self._split_investors(investors_str) if investors_str else []
            
            if company_str:
                event = InvestmentEvent(
                    id=hashlib.md5(f"{investors_str}_{company_str}_{round_str}".encode()).hexdigest(),
                    description=match.group(0),
                    investors=investors,
                    company=company_str,
                    round=round_str if round_str else None
                )
                events.append(event)
        
        return events
    
    def _split_investors(self, investors_str: str) -> List[str]:
        """分割投资方字符串"""
        if not investors_str:
            return []
        
        # 移除常见的连接词
        investors_str = re.sub(r'\s*(?:等|等机构|和|与|及)\s*$', '', investors_str)
        
        # 按常见分隔符分割
        separators = ['、', ',', ' ', '和', '与', '及']
        investors = [investors_str]
        
        for sep in separators:
            new_investors = []
            for investor in investors:
                parts = investor.split(sep)
                new_investors.extend([part.strip() for part in parts if part.strip()])
            investors = new_investors
        
        # 过滤掉空字符串和明显不是投资机构的词
        filtered_investors = []
        for investor in investors:
            if investor and not self._is_likely_company(investor):
                filtered_investors.append(investor)
        
        return filtered_investors
    
    def _is_likely_company(self, name: str) -> bool:
        """判断名称更可能是公司而非投资机构"""
        # 如果名称包含公司后缀但不包含投资机构后缀，更可能是公司
        has_company_suffix = any(name.endswith(suffix) for suffix in self.company_suffixes)
        has_investor_suffix = any(name.endswith(suffix) for suffix in self.investor_suffixes)
        
        return has_company_suffix and not has_investor_suffix
    
    def _extract_event_attributes(self, event: InvestmentEvent, text: str) -> None:
        """提取事件属性"""
        # 提取金额
        if not event.amount:
            for pattern in self.amount_patterns:
                match = re.search(pattern, text)
                if match:
                    event.amount = match.group(0)
                    break
        
        # 提取轮次
        if not event.round:
            for keyword in self.round_keywords:
                if keyword in text:
                    event.round = keyword
                    break
        
        # 提取日期
        if not event.date:
            for pattern in self.date_patterns:
                match = re.search(pattern, text)
                if match:
                    event.date = match.group(0)
                    break
    
    def _calculate_confidence(self, event: InvestmentEvent, text: str) -> float:
        """计算事件置信度"""
        confidence = 0.0
        
        # 基础分数
        if event.investors and event.company:
            confidence += 0.4
        
        # 投资关键词匹配
        investment_keyword_matches = sum(1 for keyword in self.investment_keywords if keyword in text)
        confidence += min(0.2, investment_keyword_matches * 0.05)
        
        # 投资方数量
        if len(event.investors) > 1:
            confidence += 0.1
        
        # 属性完整性
        if event.amount:
            confidence += 0.1
        if event.round:
            confidence += 0.1
        if event.date:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _deduplicate_events(self, events: List[InvestmentEvent]) -> List[InvestmentEvent]:
        """去重事件列表"""
        unique_events = {}
        
        for event in events:
            # 创建唯一键（投资方集合 + 公司）
            investors_key = tuple(sorted(event.investors))
            key = (investors_key, event.company)
            
            if key not in unique_events or event.confidence > unique_events[key].confidence:
                unique_events[key] = event
        
        return list(unique_events.values())


class LLMEnhancedExtractor:
    """LLM增强的关系提取器"""
    
    def __init__(self, llm_processor: SimplifiedLLMProcessor = None):
        self.llm_processor = llm_processor or get_llm_processor()
        self.rule_extractor = RuleBasedExtractor()
    
    async def extract_investment_events(self, text: str, use_llm_fallback: bool = True) -> List[InvestmentEvent]:
        """从文本中提取投资事件，结合规则和LLM"""
        # 首先尝试规则提取
        rule_events = self.rule_extractor.extract_investment_events(text)
        
        # 如果规则提取结果置信度较低，使用LLM增强
        if use_llm_fallback and (not rule_events or all(e.confidence < 0.5 for e in rule_events)):
            llm_events = await self._extract_with_llm(text)
            
            # 合并结果，优先选择高置信度的事件
            all_events = rule_events + llm_events
            unique_events = self._merge_events(all_events)
            return unique_events
        
        return rule_events
    
    async def _extract_with_llm(self, text: str) -> List[InvestmentEvent]:
        """使用LLM提取投资事件"""
        # 准备文本
        normalized_text = self.rule_extractor._normalize_text(text)
        
        # 使用LLM提取关系
        relations = await self.llm_processor.extract_relations([normalized_text])
        
        # 转换为投资事件
        events = []
        for relation in relations:
            # 这里需要根据relation的source和target获取实体名称
            # 在实际应用中，应该有一个实体名称映射
            # 为了演示，我们假设可以从relation中获取名称
            investor_name = relation.properties.get("investor_name", "未知投资方")
            company_name = relation.properties.get("company_name", "未知公司")
            
            event = InvestmentEvent(
                id=relation.id,
                description=f"{investor_name} 投资 {company_name}",
                investors=[investor_name],
                company=company_name,
                confidence=0.7  # LLM提取的默认置信度
            )
            events.append(event)
        
        return events
    
    def _merge_events(self, events: List[InvestmentEvent]) -> List[InvestmentEvent]:
        """合并规则和LLM提取的事件"""
        # 按投资方和公司分组
        event_groups = {}
        
        for event in events:
            investors_key = tuple(sorted(event.investors))
            key = (investors_key, event.company)
            
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # 为每组选择最佳事件
        merged_events = []
        for key, group in event_groups.items():
            # 选择置信度最高的事件
            best_event = max(group, key=lambda e: e.confidence)
            
            # 合并属性
            for event in group:
                if event.amount and not best_event.amount:
                    best_event.amount = event.amount
                if event.round and not best_event.round:
                    best_event.round = event.round
                if event.date and not best_event.date:
                    best_event.date = event.date
            
            merged_events.append(best_event)
        
        return merged_events


class OptimizedRelationExtractor:
    """优化的关系提取流程"""
    
    def __init__(self, llm_processor: SimplifiedLLMProcessor = None):
        self.llm_processor = llm_processor or get_llm_processor()
        self.rule_extractor = RuleBasedExtractor()
        self.llm_enhanced_extractor = LLMEnhancedExtractor(llm_processor)
        
        # 统计信息
        self.stats = {
            "total_texts": 0,
            "rule_events": 0,
            "llm_events": 0,
            "merged_events": 0,
            "high_confidence_events": 0,
            "llm_calls": 0
        }
    
    async def extract_relations_from_texts(self, texts: List[str], use_llm_fallback: bool = True) -> List[Relation]:
        """从文本列表中提取关系"""
        all_events = []
        
        for text in texts:
            # 提取投资事件
            events = await self.llm_enhanced_extractor.extract_investment_events(text, use_llm_fallback)
            all_events.extend(events)
            
            # 更新统计
            self.stats["total_texts"] += 1
            
            # 统计规则提取和LLM提取的事件数
            rule_events = self.rule_extractor.extract_investment_events(text)
            self.stats["rule_events"] += len(rule_events)
            
            if use_llm_fallback and (not rule_events or all(e.confidence < 0.5 for e in rule_events)):
                self.stats["llm_calls"] += 1
                self.stats["llm_events"] += len(events) - len(rule_events)
        
        # 统计高置信度事件
        self.stats["high_confidence_events"] += sum(1 for e in all_events if e.confidence >= 0.7)
        
        # 转换为关系对象
        relations = []
        for event in all_events:
            for investor in event.investors:
                # 生成实体ID
                investor_id = hashlib.md5(f"investor_{investor}".encode()).hexdigest()
                company_id = hashlib.md5(f"company_{event.company}".encode()).hexdigest()
                
                # 创建关系
                relation = Relation(
                    id=hashlib.md5(f"{investor}_{event.company}".encode()).hexdigest(),
                    source=investor_id,
                    target=company_id,
                    type="invests_in",
                    properties={
                        "investor_name": investor,
                        "company_name": event.company,
                        "amount": event.amount,
                        "round": event.round,
                        "date": event.date,
                        "confidence": event.confidence,
                        "description": event.description
                    }
                )
                relations.append(relation)
        
        # 更新统计
        self.stats["merged_events"] = len(all_events)
        
        return relations
    
    async def extract_entities_from_texts(self, texts: List[str]) -> List[Entity]:
        """从文本列表中提取实体"""
        # 使用LLM处理器提取实体
        entities = await self.llm_processor.extract_entities(texts)
        
        # 补充规则提取的实体
        all_entities = entities.copy()
        
        for text in texts:
            # 使用规则提取投资事件
            events = self.rule_extractor.extract_investment_events(text)
            
            # 从事件中提取实体
            for event in events:
                # 添加投资方实体
                for investor_name in event.investors:
                    investor = Entity(
                        id=hashlib.md5(f"investor_{investor_name}".encode()).hexdigest(),
                        name=investor_name,
                        type="investor",
                        properties={"confidence": event.confidence}
                    )
                    all_entities.append(investor)
                
                # 添加公司实体
                company = Entity(
                    id=hashlib.md5(f"company_{event.company}".encode()).hexdigest(),
                    name=event.company,
                    type="company",
                    properties={"confidence": event.confidence}
                )
                all_entities.append(company)
        
        # 去重
        unique_entities = self._deduplicate_entities(all_entities)
        
        return unique_entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体列表"""
        unique_entities = {}
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in unique_entities:
                unique_entities[key] = entity
            else:
                # 合并属性
                existing = unique_entities[key]
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                
                # 保留更高的置信度
                if "confidence" in entity.properties and "confidence" in existing.properties:
                    if entity.properties["confidence"] > existing.properties["confidence"]:
                        existing.properties["confidence"] = entity.properties["confidence"]
        
        return list(unique_entities.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_texts": 0,
            "rule_events": 0,
            "llm_events": 0,
            "merged_events": 0,
            "high_confidence_events": 0,
            "llm_calls": 0
        }


# 全局关系提取器实例
_global_extractor = None


def get_relation_extractor() -> OptimizedRelationExtractor:
    """获取全局关系提取器实例"""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = OptimizedRelationExtractor()
    return _global_extractor


async def extract_relations_from_texts(texts: List[str]) -> List[Relation]:
    """从文本列表中提取关系的便捷函数"""
    extractor = get_relation_extractor()
    return await extractor.extract_relations_from_texts(texts)


async def extract_entities_and_relations(texts: List[str]) -> Tuple[List[Entity], List[Relation]]:
    """从文本列表中提取实体和关系的便捷函数"""
    extractor = get_relation_extractor()
    entities = await extractor.extract_entities_from_texts(texts)
    relations = await extractor.extract_relations_from_texts(texts)
    return entities, relations