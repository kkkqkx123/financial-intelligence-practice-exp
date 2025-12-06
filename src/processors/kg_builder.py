"""
混合式知识图谱构建器
结合硬编码规则和LLM增强，实现高效的知识图谱构建
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

from .data_parser import DataParser
from .entity_matcher import EntityMatcher
from .config import ROUND_MAPPING, CONFIDENCE_THRESHOLDS

logger = logging.getLogger(__name__)


class HybridKGBuilder:
    """混合式知识图谱构建器"""
    
    def __init__(self):
        self.parser = DataParser()
        self.matcher = EntityMatcher()
        
        # 知识图谱存储
        self.companies = {}  # 公司实体
        self.investors = {}  # 投资方实体
        self.investment_events = []  # 投资事件
        self.relationships = []  # 关系数据
        
        # LLM增强记录
        self.llm_enhancement_queue = []  # 需要LLM增强的项目队列
        self.llm_results_cache = {}  # LLM结果缓存
        
        # 统计信息
        self.build_stats = {
            'total_companies': 0,
            'total_investors': 0,
            'total_events': 0,
            'successful_links': 0,
            'failed_links': 0,
            'llm_enhancements': 0,
            'processing_time': 0.0
        }
        
        # 实体ID生成器
        self._company_id_counter = 1
        self._investor_id_counter = 1
        self._event_id_counter = 1
    
    def _generate_company_id(self, company_name: str) -> str:
        """生成公司实体ID"""
        company_id = f"company_{self._company_id_counter:06d}"
        self._company_id_counter += 1
        return company_id
    
    def _generate_investor_id(self, investor_name: str) -> str:
        """生成投资方实体ID"""
        investor_id = f"investor_{self._investor_id_counter:06d}"
        self._investor_id_counter += 1
        return investor_id
    
    def _generate_event_id(self) -> str:
        """生成事件ID"""
        event_id = f"event_{self._event_id_counter:06d}"
        self._event_id_counter += 1
        return event_id
    
    def build_company_entities(self, company_data: List[Dict]) -> Dict[str, Dict]:
        """构建公司实体"""
        logger.info(f"开始构建公司实体，共 {len(company_data)} 条数据")
        
        companies = {}
        for company in company_data:
            try:
                company_name = company.get('公司名称', '')
                if not company_name:
                    continue
                
                company_id = self._generate_company_id(company_name)
                
                # 构建公司实体
                company_entity = {
                    'id': company_id,
                    'name': company_name,
                    'aliases': self._generate_company_aliases(company_name),
                    'description': company.get('公司介绍', ''),
                    'registration_info': {
                        'registration_id': company.get('工商注册id', ''),
                        'legal_representative': company.get('法人代表', ''),
                        'registered_capital': self.parser._normalize_capital(company.get('注册资金', '')),
                        'unified_credit_code': company.get('统一信用代码', ''),
                        'establishment_date': self.parser._parse_date(company.get('成立时间', ''))
                    },
                    'contact_info': {
                        'address': company.get('地址', ''),
                        'website': company.get('网址', '')
                    },
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'data_source': 'company_data',
                        'confidence': self._calculate_company_confidence(company)
                    }
                }
                
                companies[company_id] = company_entity
                
            except Exception as e:
                logger.error(f"构建公司实体失败: {company}, 错误: {e}")
                continue
        
        self.build_stats['total_companies'] = len(companies)
        logger.info(f"公司实体构建完成，共 {len(companies)} 个实体")
        return companies
    
    def build_investor_entities(self, investor_data: List[Dict]) -> Dict[str, Dict]:
        """构建投资方实体"""
        logger.info(f"开始构建投资方实体，共 {len(investor_data)} 条数据")
        
        investors = {}
        for investor in investor_data:
            try:
                investor_name = investor.get('机构名称', '')
                if not investor_name:
                    continue
                
                investor_id = self._generate_investor_id(investor_name)
                
                # 解析投资偏好
                investment_preferences = self._parse_investment_preferences(investor)
                
                # 构建投资方实体
                investor_entity = {
                    'id': investor_id,
                    'name': investor_name,
                    'aliases': self._generate_investor_aliases(investor_name),
                    'description': investor.get('介绍', ''),
                    'investment_focus': investment_preferences,
                    'scale': self._parse_scale(investor.get('规模', '')),
                    'preferred_rounds': self._parse_preferred_rounds(investor.get('轮次', '')),
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'data_source': 'investment_structure',
                        'confidence': self._calculate_investor_confidence(investor)
                    }
                }
                
                investors[investor_id] = investor_entity
                
            except Exception as e:
                logger.error(f"构建投资方实体失败: {investor}, 错误: {e}")
                continue
        
        self.build_stats['total_investors'] = len(investors)
        logger.info(f"投资方实体构建完成，共 {len(investors)} 个实体")
        return investors
    
    def build_investment_relationships(self, event_data: List[Dict], 
                                       companies: Dict[str, Dict], 
                                       investors: Dict[str, Dict]) -> List[Dict]:
        """构建投资关系"""
        logger.info(f"开始构建投资关系，共 {len(event_data)} 条事件")
        
        relationships = []
        company_names = {company['name'] for company in companies.values()}
        investor_names = {investor['name'] for investor in investors.values()}
        
        for event in event_data:
            try:
                # 解析投资方
                investor_names_raw = event.get('投资方', '')
                investor_list = self._parse_investor_names(investor_names_raw)
                
                # 解析融资方
                company_name = event.get('融资方', '')
                
                # 解析投资金额
                amount_info = self._parse_investment_amount(event.get('金额', ''))
                
                # 解析投资轮次
                round_info = self._standardize_round(event.get('轮次', ''))
                
                # 解析投资时间
                investment_date = self.parser._parse_date(event.get('融资时间', ''))
                
                # 为每个投资方创建关系
                for investor_name in investor_list:
                    relationship = self._create_investment_relationship(
                        investor_name, company_name, amount_info, round_info, 
                        investment_date, companies, investors, 
                        company_names, investor_names
                    )
                    
                    if relationship:
                        relationships.append(relationship)
                
            except Exception as e:
                logger.error(f"构建投资关系失败: {event}, 错误: {e}")
                continue
        
        self.build_stats['total_events'] = len(relationships)
        logger.info(f"投资关系构建完成，共 {len(relationships)} 个关系")
        return relationships
    
    def _create_investment_relationship(self, investor_name: str, company_name: str,
                                       amount_info: Dict, round_info: Dict, investment_date: str,
                                       companies: Dict[str, Dict], investors: Dict[str, Dict],
                                       company_names: Set[str], investor_names: Set[str]) -> Optional[Dict]:
        """创建单个投资关系"""
        try:
            # 匹配投资方
            investor_match = self.matcher.match_investor(investor_name, investor_names)
            
            # 匹配公司
            company_match = self.matcher.match_company(company_name, company_names)
            
            # 获取实体ID
            investor_id = None
            company_id = None
            
            if investor_match['success']:
                # 找到匹配的投资方实体
                for inv_id, inv_data in investors.items():
                    if inv_data['name'] == investor_match['matched_entity']:
                        investor_id = inv_id
                        break
            else:
                # 创建新的投资方实体
                investor_id = self._generate_investor_id(investor_name)
                new_investor = {
                    'id': investor_id,
                    'name': investor_name,
                    'aliases': self._generate_investor_aliases(investor_name),
                    'description': '',
                    'investment_focus': {'industries': [], 'scales': [], 'rounds': []},
                    'scale': 'unknown',
                    'preferred_rounds': [],
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'data_source': 'investment_event',
                        'confidence': 0.6,
                        'requires_enhancement': True
                    }
                }
                investors[investor_id] = new_investor
                self.build_stats['total_investors'] += 1
            
            if company_match['success']:
                # 找到匹配的公司实体
                for comp_id, comp_data in companies.items():
                    if comp_data['name'] == company_match['matched_entity']:
                        company_id = comp_id
                        break
            else:
                # 创建新的公司实体
                company_id = self._generate_company_id(company_name)
                new_company = {
                    'id': company_id,
                    'name': company_name,
                    'aliases': self._generate_company_aliases(company_name),
                    'description': '',
                    'registration_info': {},
                    'contact_info': {},
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'data_source': 'investment_event',
                        'confidence': 0.6,
                        'requires_enhancement': True
                    }
                }
                companies[company_id] = new_company
                self.build_stats['total_companies'] += 1
            
            # 记录需要LLM增强的项目
            if investor_match['requires_llm']:
                self.llm_enhancement_queue.append({
                    'type': 'investor_enhancement',
                    'entity_id': investor_id,
                    'original_name': investor_name,
                    'match_info': investor_match
                })
            
            if company_match['requires_llm']:
                self.llm_enhancement_queue.append({
                    'type': 'company_enhancement',
                    'entity_id': company_id,
                    'original_name': company_name,
                    'match_info': company_match
                })
            
            # 创建关系
            relationship = {
                'id': self._generate_event_id(),
                'type': 'investment',
                'investor_id': investor_id,
                'company_id': company_id,
                'investor_name': investor_name,
                'company_name': company_name,
                'amount': amount_info,
                'round': round_info,
                'date': investment_date,
                'confidence': min(investor_match.get('confidence', 0.6), 
                               company_match.get('confidence', 0.6)),
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source_event': True
                }
            }
            
            # 更新统计信息
            if investor_match['success'] and company_match['success']:
                self.build_stats['successful_links'] += 1
            else:
                self.build_stats['failed_links'] += 1
            
            return relationship
            
        except Exception as e:
            logger.error(f"创建投资关系失败: {investor_name} -> {company_name}, 错误: {e}")
            return None
    
    def _generate_company_aliases(self, company_name: str) -> List[str]:
        """生成公司别名"""
        aliases = [company_name]
        
        # 移除常见后缀的别名
        suffixes = ['有限公司', '有限责任公司', '股份有限公司', '公司', '集团']
        for suffix in suffixes:
            if company_name.endswith(suffix):
                alias = company_name[:-len(suffix)].strip()
                if alias and alias not in aliases:
                    aliases.append(alias)
        
        return aliases
    
    def _generate_investor_aliases(self, investor_name: str) -> List[str]:
        """生成投资方别名"""
        aliases = [investor_name]
        
        # 常见投资机构缩写
        abbreviations = {
            '红杉资本中国基金': ['红杉中国', 'Sequoia China'],
            'IDG资本': ['IDG'],
            '经纬中国': ['经纬'],
            '真格基金': ['真格'],
            '创新工场': ['创工'],
        }
        
        for full_name, abbr_list in abbreviations.items():
            if investor_name == full_name:
                aliases.extend(abbr_list)
        
        return aliases
    
    def _parse_investment_preferences(self, investor: Dict) -> Dict:
        """解析投资偏好"""
        industries_str = investor.get('行业', '')
        industries = []
        
        if industries_str:
            # 简单的行业解析
            parts = industries_str.split(' ')
            for part in parts:
                if '家' in part and len(part) > 2:
                    industry = part.replace('家', '').strip()
                    if industry:
                        industries.append(industry)
        
        return {
            'industries': industries,
            'scales': [],
            'rounds': []
        }
    
    def _parse_scale(self, scale_str: str) -> str:
        """解析规模信息"""
        if not scale_str:
            return 'unknown'
        
        if '人民币' in scale_str or '美元' in scale_str:
            return 'managed_fund'
        
        return 'unknown'
    
    def _parse_preferred_rounds(self, rounds_str: str) -> List[str]:
        """解析偏好轮次"""
        if not rounds_str:
            return []
        
        rounds = []
        for round_name in ROUND_MAPPING.keys():
            if round_name in rounds_str:
                rounds.append(ROUND_MAPPING[round_name])
        
        return rounds
    
    def _parse_investor_names(self, investor_names_str: str) -> List[str]:
        """解析投资方名称列表"""
        if not investor_names_str:
            return []
        
        # 使用智能分割
        return self.parser._smart_split(investor_names_str)
    
    def _parse_investment_amount(self, amount_str: str) -> Dict:
        """解析投资金额"""
        if not amount_str or amount_str == '未披露':
            return {
                'amount': None,
                'currency': 'unknown',
                'unit': 'unknown',
                'confidence': 0.3
            }
        
        # 使用金额标准化函数
        normalized_amount = self.parser._normalize_amount(amount_str)
        
        return {
            'amount': normalized_amount.get('amount'),
            'currency': normalized_amount.get('currency', 'unknown'),
            'unit': normalized_amount.get('unit', 'unknown'),
            'original': amount_str,
            'confidence': 0.8 if normalized_amount.get('amount') else 0.3
        }
    
    def _standardize_round(self, round_str: str) -> Dict:
        """标准化投资轮次"""
        if not round_str:
            return {'round': 'unknown', 'confidence': 0.3}
        
        # 使用轮次映射
        for key, value in ROUND_MAPPING.items():
            if key in round_str:
                return {'round': value, 'confidence': 0.9}
        
        return {'round': round_str.lower(), 'confidence': 0.6}
    
    def _calculate_company_confidence(self, company: Dict) -> float:
        """计算公司数据置信度"""
        confidence = 0.5
        
        # 检查关键字段
        if company.get('公司名称'):
            confidence += 0.2
        if company.get('工商注册id'):
            confidence += 0.1
        if company.get('统一信用代码'):
            confidence += 0.1
        if company.get('注册资金'):
            confidence += 0.05
        if company.get('网址'):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _calculate_investor_confidence(self, investor: Dict) -> float:
        """计算投资方数据置信度"""
        confidence = 0.5
        
        # 检查关键字段
        if investor.get('机构名称'):
            confidence += 0.2
        if investor.get('介绍'):
            confidence += 0.1
        if investor.get('行业'):
            confidence += 0.1
        if investor.get('规模'):
            confidence += 0.05
        if investor.get('轮次'):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def get_llm_enhancement_batch(self) -> List[Dict]:
        """获取需要LLM增强的批处理任务"""
        return self.llm_enhancement_queue.copy()
    
    def clear_llm_queue(self):
        """清空LLM增强队列"""
        self.llm_enhancement_queue.clear()
    
    def get_knowledge_graph(self) -> Dict:
        """获取完整的知识图谱"""
        return {
            'entities': {
                'companies': self.companies,
                'investors': self.investors
            },
            'relationships': self.relationships,
            'statistics': self.build_stats,
            'llm_enhancement_required': len(self.llm_enhancement_queue),
            'created_at': datetime.now().isoformat()
        }
    
    def save_knowledge_graph(self, filepath: str):
        """保存知识图谱到文件"""
        kg_data = self.get_knowledge_graph()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"知识图谱已保存到: {filepath}")
    
    def get_build_statistics(self) -> Dict:
        """获取构建统计信息"""
        return {
            **self.build_stats,
            'entity_matcher_stats': self.matcher.get_stats(),
            'llm_queue_size': len(self.llm_enhancement_queue),
            'cache_info': {
                'name_normalization_cache': len(self.matcher.name_normalization_cache),
                'fuzzy_match_cache': len(self.matcher.fuzzy_match_cache)
            }
        }