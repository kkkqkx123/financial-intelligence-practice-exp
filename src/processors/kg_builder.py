"""
混合式知识图谱构建器
结合硬编码规则和LLM增强，实现高效的知识图谱构建
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from collections import defaultdict

from .data_parser import DataParser
from .entity_matcher import EntityMatcher
from .config import ROUND_MAPPING, CONFIDENCE_THRESHOLDS

# 配置日志
logger = logging.getLogger(__name__)

# 设置日志级别为WARNING，减少INFO级别的日志输出
logger.setLevel(logging.WARNING)


class HybridKGBuilder:
    """混合式知识图谱构建器"""
    
    def __init__(self):
        """初始化混合知识图谱构建器"""
        self.matcher = EntityMatcher()
        self.parser = DataParser()
        self.knowledge_graph = {
            'companies': [],
            'investors': [],
            'relationships': []
        }
        self.llm_enhancement_queue = []
        self.stats = {
            'companies_processed': 0,
            'investors_processed': 0,
            'relationships_created': 0,
            'failed_events': 0,
            'llm_enhancements_queued': 0
        }
        self.parser: DataParser = DataParser()
        self.matcher: EntityMatcher = EntityMatcher()
        
        # 知识图谱存储
        self.companies = {}  # 公司实体
        self.investors = {}  # 投资方实体
        self.investment_events = []  # 投资事件
        self.relationships = []  # 关系数据
        
        # LLM增强记录
        self.llm_enhancement_queue: List[Dict[str, Any]] = []  # 需要LLM增强的项目队列
        self.llm_results_cache: Dict[str, Any] = {}  # LLM结果缓存
        
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
    
    def build_investment_relationships(self, investment_events: List[Dict]) -> None:
        """构建投资关系 - 增强错误处理"""
        logger.info(f"开始构建 {len(investment_events)} 个投资事件的关系")
        
        for event in investment_events:
            try:
                self._create_investment_relationship(event)
            except Exception as e:
                self.stats['failed_events'] += 1
                logger.error(f"处理投资事件失败: {event.get('description', '')[:50]}... 错误: {e}")
                # 将失败的事件添加到LLM增强队列，以便后续处理
                self.llm_enhancement_queue.append(('event', str(event)))
        
        logger.info(f"投资关系构建完成: 成功={self.stats['relationships_created']}, 失败={self.stats['failed_events']}")
    
    def _create_investment_relationship(self, event: Dict) -> None:
        """创建单个投资关系 - 增强实体链接错误处理"""
        try:
            # 获取投资方和融资方名称
            investor_names = event.get('investors', [])
            investee_name = event.get('investee', '')
            
            # 验证必要字段
            if not investor_names or not investee_name:
                logger.warning(f"投资事件缺少必要字段: 投资方={investor_names}, 融资方={investee_name}")
                self.stats['failed_events'] += 1
                return
            
            # 获取已知实体集合
            known_companies = set([comp['name'] for comp in self.knowledge_graph['companies']])
            known_investors = set([inv['name'] for inv in self.knowledge_graph['investors']])
            
            # 匹配融资方实体
            investee_match = self.matcher.match_company(investee_name, known_companies)
            investee_entity = None
            
            if investee_match['success']:
                # 查找已存在的实体
                for comp in self.knowledge_graph['companies']:
                    if comp['name'] == investee_match['matched_entity']:
                        investee_entity = comp
                        break
            else:
                # 创建新实体并标记需要LLM增强
                investee_entity = {
                    'name': investee_name,
                    'type': 'Company',
                    'properties': {
                        'short_name': investee_name,
                        'full_name': investee_name,
                        'description': '',
                        'needs_llm_enhancement': True,
                        'source_event': event.get('description', ''),
                        'confidence': 0.5
                    }
                }
                self.knowledge_graph['companies'].append(investee_entity)
                self.llm_enhancement_queue.append(('company', investee_name))
                logger.info(f"创建新公司实体: {investee_name} (需要LLM增强)")
            
            # 处理每个投资方
            for investor_name in investor_names:
                if not investor_name or not investor_name.strip():
                    continue
                    
                # 匹配投资方实体
                investor_match = self.matcher.match_investor(investor_name, known_investors)
                investor_entity = None
                
                if investor_match['success']:
                    # 查找已存在的实体
                    for inv in self.knowledge_graph['investors']:
                        if inv['name'] == investor_match['matched_entity']:
                            investor_entity = inv
                            break
                else:
                    # 创建新实体并标记需要LLM增强
                    investor_entity = {
                        'name': investor_name,
                        'type': 'Investor',
                        'properties': {
                            'short_name': investor_name,
                            'full_name': investor_name,
                            'description': '',
                            'needs_llm_enhancement': True,
                            'source_event': event.get('description', ''),
                            'confidence': 0.5
                        }
                    }
                    self.knowledge_graph['investors'].append(investor_entity)
                    self.llm_enhancement_queue.append(('investor', investor_name))
                    logger.info(f"创建新投资方实体: {investor_name} (需要LLM增强)")
                
                # 创建投资关系
                relationship = {
                    'source': investor_entity['name'],
                    'target': investee_entity['name'],
                    'type': 'INVESTED_IN',
                    'properties': {
                        'amount': event.get('amount'),
                        'round': event.get('round'),
                        'date': event.get('investment_date'),
                        'description': event.get('description', ''),
                        'confidence': 0.8
                    }
                }
                
                self.knowledge_graph['relationships'].append(relationship)
                self.stats['relationships_created'] += 1
                
        except Exception as e:
            self.stats['failed_events'] += 1
            logger.error(f"创建投资关系失败: {event.get('description', '')[:50]}... 错误: {e}")
            # 将失败的事件添加到LLM增强队列，以便后续处理
            self.llm_enhancement_queue.append(('event', str(event)))
    
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
    
    def _parse_investment_amount(self, amount_str: str) -> Dict[str, Any]:
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
            'amount': normalized_amount.get('amount') if isinstance(normalized_amount, dict) else normalized_amount,
            'currency': normalized_amount.get('currency', 'unknown') if isinstance(normalized_amount, dict) else 'unknown',
            'unit': normalized_amount.get('unit', 'unknown') if isinstance(normalized_amount, dict) else 'unknown',
            'original': amount_str,
            'confidence': 0.8 if (normalized_amount.get('amount') if isinstance(normalized_amount, dict) else normalized_amount) else 0.3
        }
    
    def _standardize_round(self, round_str: str) -> Dict[str, Any]:
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
    
    def get_llm_enhancement_batch(self) -> List[Dict[str, Any]]:
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