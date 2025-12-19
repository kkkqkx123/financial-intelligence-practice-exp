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
from .config import ROUND_MAPPING, CONFIDENCE_THRESHOLDS

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
        """标准化实体名称 - 增强处理，支持全称与简称的不一致问题"""
        if not name or not isinstance(name, str):
            return ""
        
        # 使用缓存避免重复处理
        if name in self.name_normalization_cache:
            cached_result = self.name_normalization_cache[name]
            return cached_result if isinstance(cached_result, str) else ""
        
        # 基础清理
        normalized = name.strip()
        
        # 移除常见的公司后缀
        company_suffixes = [
            '有限公司', '有限责任公司', '股份有限公司', '集团', '公司',
            '科技有限公司', '信息技术有限公司', '网络科技有限公司', '电子商务有限公司'
        ]
        
        # 移除常见的投资机构后缀
        investor_suffixes = [
            '投资', '资本', '基金', '创投', '风投', '私募', '资产', '管理', '合伙企业',
            'Partners', 'Capital', 'Ventures', 'Fund', 'Investment', 'Management'
        ]
        
        # 移除所有后缀
        all_suffixes = company_suffixes + investor_suffixes
        for suffix in all_suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # 处理常见的简称模式
        # 1. 处理"北京/上海/深圳...+公司名"的模式，保留公司名部分
        city_prefixes = ['北京', '上海', '深圳', '广州', '杭州', '成都', '武汉', '南京', '西安', '重庆']
        for city in city_prefixes:
            if normalized.startswith(city):
                normalized = normalized[len(city):].strip()
                break
        
        # 2. 处理"中国+公司名"的模式
        if normalized.startswith('中国'):
            normalized = normalized[2:].strip()
        
        # 3. 处理括号内容，如"(北京)"、"(上海)"等
        normalized = re.sub(r'\([^)]*\)', '', normalized)
        
        # 4. 处理中英文混合情况，如"阿里巴巴(Alibaba)"
        normalized = re.sub(r'[a-zA-Z\(\)]+', '', normalized).strip()
        
        # 5. 移除特殊字符
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', normalized)
        
        # 统一大小写和空格
        normalized = re.sub(r'\s+', ' ', normalized.lower())
        
        # 缓存结果
        self.name_normalization_cache[name] = normalized
        return normalized
    
    def abbreviated_match(self, name: str, target_entities: Set[str]) -> Optional[Tuple[str, float]]:
        """简称匹配 - 处理全称与简称的不一致问题"""
        if not name or not target_entities:
            return None
        
        normalized_name = self.normalize_name(name)
        best_match = None
        best_score = 0.0
        
        for entity in target_entities:
            normalized_entity = self.normalize_name(entity)
            
            # 1. 检查是否一个是另一个的子串
            if normalized_name in normalized_entity:
                # 计算子串长度占比
                score = len(normalized_name) / len(normalized_entity)
                if score > best_score and score >= 0.5:  # 至少占50%长度
                    best_score = score
                    best_match = entity
            elif normalized_entity in normalized_name:
                # 计算子串长度占比
                score = len(normalized_entity) / len(normalized_name)
                if score > best_score and score >= 0.5:  # 至少占50%长度
                    best_score = score
                    best_match = entity
            
            # 2. 检查是否有共同的核心词汇
            # 分割名称为词汇列表
            name_words = set(normalized_name.split())
            entity_words = set(normalized_entity.split())
            
            if name_words and entity_words:
                # 计算共同词汇的比例
                common_words = name_words.intersection(entity_words)
                if common_words:
                    # 计算相似度：共同词汇数量 / 较长词汇列表的长度
                    similarity = len(common_words) / max(len(name_words), len(entity_words))
                    if similarity > best_score and similarity >= 0.6:  # 至少60%共同词汇
                        best_score = similarity
                        best_match = entity
        
        if best_match and best_score >= 0.5:
            self.entity_linking_stats['fuzzy_matches'] += 1
            return (best_match, float(best_score))
        
        return None
    
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
            cached_result = self.fuzzy_match_cache[cache_key]
            if cached_result is None:
                return None
            # 确保缓存结果符合返回类型
            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                match_str, score = cached_result
                if isinstance(match_str, str) and isinstance(score, (int, float)):
                    return (match_str, float(score))
            return None
        
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
            result = (best_match, float(best_score))
        else:
            result = None
        
        self.fuzzy_match_cache[cache_key] = result
        return result
    
    def match_entity(self, name: str, target_entities: Set[str], entity_type: str = "company") -> Dict:
        """
        综合实体匹配 - 增强处理，支持全称与简称的不一致问题
        
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
        
        # 3. 简称匹配 - 降低阈值
        abbreviated_result = self.abbreviated_match(name, target_entities)
        if abbreviated_result:
            matched_entity, confidence = abbreviated_result
            # 降低简称匹配的阈值要求
            if confidence >= 0.4:  # 从原来的0.5降低到0.4
                return {
                    'success': True,
                    'matched_entity': matched_entity,
                    'match_type': 'abbreviated',
                    'confidence': confidence,
                    'requires_llm': False,
                    'reason': f'简称匹配成功，相似度: {confidence:.2f}'
                }
        
        # 4. 模糊匹配 - 降低阈值
        fuzzy_result = self.fuzzy_match(name, target_entities, threshold=0.6)  # 从原来的0.8降低到0.6
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
        
        # 5. 尝试最佳匹配（即使低于阈值）
        best_match = None
        best_similarity = 0.0
        
        for entity in target_entities:
            normalized_name = self.normalize_name(name)
            normalized_entity = self.normalize_name(entity)
            
            # 计算多种相似度指标
            similarity = SequenceMatcher(None, normalized_name, normalized_entity).ratio()
            
            # 检查子串匹配
            if normalized_name in normalized_entity or normalized_entity in normalized_name:
                similarity = max(similarity, 0.7)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entity
        
        # 如果最佳匹配的相似度超过0.4，则返回匹配结果
        if best_match and best_similarity > 0.4:
            return {
                'success': True,
                'matched_entity': best_match,
                'match_type': 'best_match',
                'confidence': best_similarity,
                'requires_llm': False,
                'reason': f'最佳匹配成功，相似度: {best_similarity:.2f}'
            }
        
        # 6. 尝试包含关系匹配
        for entity in target_entities:
            normalized_name = self.normalize_name(name)
            normalized_entity = self.normalize_name(entity)
            
            # 如果一个名称包含另一个名称
            if normalized_name in normalized_entity or normalized_entity in normalized_name:
                # 计算包含比例
                if len(normalized_name) > 0:
                    contain_ratio = min(len(normalized_name), len(normalized_entity)) / max(len(normalized_name), len(normalized_entity))
                    if contain_ratio > 0.4:  # 包含比例超过40%
                        return {
                            'success': True,
                            'matched_entity': entity,
                            'match_type': 'containment',
                            'confidence': contain_ratio,
                            'requires_llm': False,
                            'reason': f'包含关系匹配成功，比例: {contain_ratio:.2f}'
                        }
        
        # 7. 最后才需要LLM增强
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
                           entity_type: str = "company") -> Dict:
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