"""
实体链接和匹配系统
基于硬编码优先策略，实现精确匹配、别名匹配、模糊匹配
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from difflib import SequenceMatcher
from .config import ROUND_MAPPING, AMOUNT_UNITS, CAPITAL_UNITS, CONFIDENCE_THRESHOLDS

logger = logging.getLogger(__name__)


class EntityMatcher:
    """实体链接和匹配系统"""
    
    def __init__(self):
        self.company_aliases = {}  # 公司别名映射
        self.investor_aliases = {}  # 投资方别名映射
        self.name_normalization_cache = {}  # 名称标准化缓存
        self.fuzzy_match_cache = {}  # 模糊匹配缓存
        self.entity_linking_stats = {
            'exact_matches': 0,
            'alias_matches': 0,
            'fuzzy_matches': 0,
            'failed_matches': 0,
            'llm_required': 0
        }
        
        # 初始化别名映射
        self._build_alias_mappings()
    
    def _build_alias_mappings(self):
        """构建实体别名映射表"""
        # 公司常见别名映射
        self.company_aliases = {
            '阿里巴巴': ['阿里', 'alibaba', '阿里巴巴', 'Alibaba Group'],
            '腾讯': ['腾讯', 'tencent', 'Tencent Holdings'],
            '百度': ['百度', 'baidu', 'Baidu Inc'],
            '字节跳动': ['字节跳动', 'bytedance', 'ByteDance', '今日头条'],
            '美团': ['美团', 'meituan', 'Meituan-Dianping'],
            '滴滴': ['滴滴', 'didi', 'Didi Chuxing'],
            '小米': ['小米', 'xiaomi', 'Xiaomi Corporation'],
            '京东': ['京东', 'jd', 'JD.com'],
            '网易': ['网易', 'netease', 'NetEase'],
            '华为': ['华为', 'huawei', 'Huawei Technologies']
        }
        
        # 投资机构别名映射
        self.investor_aliases = {
            'IDG资本': ['IDG', 'IDG Capital', 'IDG资本'],
            '红杉资本': ['红杉', 'sequoia', 'Sequoia Capital'],
            '经纬中国': ['经纬', 'matrix', 'Matrix Partners China'],
            '真格基金': ['真格', 'zhenfund', 'ZhenFund'],
            '创新工场': ['创新工场', 'innovation works'],
            '金沙江创投': ['金沙江', 'gsh', 'GSR Ventures'],
            '晨兴资本': ['晨兴', 'morningside', 'Morningside Venture'],
            '高瓴资本': ['高瓴', 'hillhouse', 'Hillhouse Capital'],
            '腾讯投资': ['腾讯投资', 'tencent investment'],
            '阿里巴巴投资': ['阿里投资', 'alibaba investment']
        }
    
    def normalize_name(self, name: str) -> str:
        """标准化实体名称"""
        if not name or not isinstance(name, str):
            return ""
        
        # 使用缓存避免重复处理
        if name in self.name_normalization_cache:
            return self.name_normalization_cache[name]
        
        # 基础清理
        normalized = name.strip()
        
        # 移除常见后缀
        suffixes = [
            '有限公司', '有限责任公司', '股份有限公司', '集团', '公司',
            '投资', '资本', '基金', '创投', '风投', '私募',
            'Partners', 'Capital', 'Ventures', 'Fund', 'Investment'
        ]
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # 统一大小写和空格
        normalized = re.sub(r'\s+', ' ', normalized.lower())
        
        # 缓存结果
        self.name_normalization_cache[name] = normalized
        return normalized
    
    def exact_match(self, name: str, target_entities: Set[str]) -> Optional[str]:
        """精确匹配"""
        if not name or not target_entities:
            return None
        
        normalized_name = self.normalize_name(name)
        
        for entity in target_entities:
            if self.normalize_name(entity) == normalized_name:
                self.entity_linking_stats['exact_matches'] += 1
                return entity
        
        return None
    
    def alias_match(self, name: str, target_entities: Set[str]) -> Optional[str]:
        """别名匹配"""
        if not name:
            return None
        
        # 检查公司别名
        for standard_name, aliases in self.company_aliases.items():
            if name in aliases or self.normalize_name(name) in [self.normalize_name(alias) for alias in aliases]:
                # 在目标实体中查找标准名称
                for entity in target_entities:
                    if self.normalize_name(entity) == self.normalize_name(standard_name):
                        self.entity_linking_stats['alias_matches'] += 1
                        return entity
        
        # 检查投资方别名
        for standard_name, aliases in self.investor_aliases.items():
            if name in aliases or self.normalize_name(name) in [self.normalize_name(alias) for alias in aliases]:
                for entity in target_entities:
                    if self.normalize_name(entity) == self.normalize_name(standard_name):
                        self.entity_linking_stats['alias_matches'] += 1
                        return entity
        
        return None
    
    def fuzzy_match(self, name: str, target_entities: Set[str], threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        """模糊匹配"""
        if not name or not target_entities:
            return None
        
        # 使用缓存
        cache_key = f"{name}_{threshold}"
        if cache_key in self.fuzzy_match_cache:
            return self.fuzzy_match_cache[cache_key]
        
        normalized_name = self.normalize_name(name)
        best_match = None
        best_score = 0.0
        
        for entity in target_entities:
            normalized_entity = self.normalize_name(entity)
            
            # 计算相似度
            similarity = SequenceMatcher(None, normalized_name, normalized_entity).ratio()
            
            # 检查子串匹配
            if normalized_name in normalized_entity or normalized_entity in normalized_name:
                similarity = max(similarity, 0.9)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = entity
        
        if best_match:
            self.entity_linking_stats['fuzzy_matches'] += 1
            result = (best_match, best_score)
        else:
            result = None
        
        self.fuzzy_match_cache[cache_key] = result
        return result
    
    def match_entity(self, name: str, target_entities: Set[str], entity_type: str = "company") -> Dict:
        """
        综合实体匹配
        
        Args:
            name: 待匹配实体名称
            target_entities: 目标实体集合
            entity_type: 实体类型 (company/investor)
            
        Returns:
            匹配结果字典
        """
        if not name or not target_entities:
            return {
                'success': False,
                'matched_entity': None,
                'match_type': None,
                'confidence': 0.0,
                'requires_llm': True,
                'reason': '输入为空或目标实体集为空'
            }
        
        # 1. 精确匹配
        exact_match = self.exact_match(name, target_entities)
        if exact_match:
            return {
                'success': True,
                'matched_entity': exact_match,
                'match_type': 'exact',
                'confidence': 1.0,
                'requires_llm': False,
                'reason': '精确匹配成功'
            }
        
        # 2. 别名匹配
        alias_match = self.alias_match(name, target_entities)
        if alias_match:
            return {
                'success': True,
                'matched_entity': alias_match,
                'match_type': 'alias',
                'confidence': 0.95,
                'requires_llm': False,
                'reason': '别名匹配成功'
            }
        
        # 3. 模糊匹配
        fuzzy_result = self.fuzzy_match(name, target_entities, CONFIDENCE_THRESHOLDS['fuzzy_match'])
        if fuzzy_result:
            matched_entity, confidence = fuzzy_result
            return {
                'success': True,
                'matched_entity': matched_entity,
                'match_type': 'fuzzy',
                'confidence': confidence,
                'requires_llm': False,
                'reason': f'模糊匹配成功，相似度: {confidence:.2f}'
            }
        
        # 4. 需要LLM增强
        self.entity_linking_stats['llm_required'] += 1
        return {
            'success': False,
            'matched_entity': None,
            'match_type': None,
            'confidence': 0.0,
            'requires_llm': True,
            'reason': '硬编码匹配失败，需要LLM增强'
        }
    
    def match_company(self, company_name: str, known_companies: Set[str]) -> Dict:
        """匹配公司实体"""
        return self.match_entity(company_name, known_companies, "company")
    
    def match_investor(self, investor_name: str, known_investors: Set[str]) -> Dict:
        """匹配投资方实体"""
        return self.match_entity(investor_name, known_investors, "investor")
    
    def batch_match_entities(self, entity_list: List[str], target_entities: Set[str], 
                           entity_type: str = "company") -> List[Dict]:
        """批量匹配实体"""
        results = []
        llm_required_items = []
        
        for entity_name in entity_list:
            match_result = self.match_entity(entity_name, target_entities, entity_type)
            results.append(match_result)
            
            # 收集需要LLM增强的项目
            if match_result['requires_llm']:
                llm_required_items.append({
                    'original_name': entity_name,
                    'entity_type': entity_type,
                    'index': len(results) - 1
                })
        
        return {
            'results': results,
            'llm_required_items': llm_required_items,
            'stats': self.entity_linking_stats.copy()
        }
    
    def get_stats(self) -> Dict:
        """获取匹配统计信息"""
        total_attempts = sum(self.entity_linking_stats.values())
        success_rate = 0.0
        if total_attempts > 0:
            success_rate = ((self.entity_linking_stats['exact_matches'] + 
                           self.entity_linking_stats['alias_matches'] + 
                           self.entity_linking_stats['fuzzy_matches']) / total_attempts) * 100
        
        return {
            **self.entity_linking_stats,
            'total_attempts': total_attempts,
            'success_rate': success_rate,
            'cache_size': len(self.name_normalization_cache) + len(self.fuzzy_match_cache)
        }
    
    def clear_cache(self):
        """清理缓存"""
        self.name_normalization_cache.clear()
        self.fuzzy_match_cache.clear()
        logger.info("实体匹配缓存已清理")
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.entity_linking_stats:
            self.entity_linking_stats[key] = 0
        logger.info("实体匹配统计已重置")