"""
知识图谱构建器
结合硬编码规则和LLM增强，实现高效的知识图谱构建
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime
from collections import defaultdict

from .data_parser import DataParser
from .entity_matcher import EntityMatcher
from .config import ROUND_MAPPING, CONFIDENCE_THRESHOLDS

# 配置日志
logger = logging.getLogger(__name__)

# 设置日志级别为INFO，显示投资方未披露的信息
logger.setLevel(logging.INFO)


class KGBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        """初始化知识图谱构建器"""
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
    
    def add_company(self, company_data: Dict[str, Any]) -> None:
        """添加公司实体到知识图谱
        
        Args:
            company_data: 公司数据字典，包含name、industry、description等字段
        """
        if not company_data or not company_data.get('name'):
            logger.warning("尝试添加空公司数据或缺少名称字段")
            return
        
        # 处理可能为列表的字段
        name = company_data.get('name', '')
        if isinstance(name, list):
            name = ' '.join(str(item) for item in name) if name else ''
        
        industry = company_data.get('industry', '')
        if isinstance(industry, list):
            industry = ' '.join(str(item) for item in industry) if industry else ''
            
        description = company_data.get('description', '')
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description) if description else ''
            
        id_field = company_data.get('id', '')
        if isinstance(id_field, list):
            id_field = ' '.join(str(item) for item in id_field) if id_field else ''
            
        established_date = company_data.get('established_date', '')
        if isinstance(established_date, list):
            established_date = ' '.join(str(item) for item in established_date) if established_date else ''
            
        registered_capital = company_data.get('registered_capital', '')
        if isinstance(registered_capital, list):
            registered_capital = ' '.join(str(item) for item in registered_capital) if registered_capital else ''
            
        legal_representative = company_data.get('legal_representative', '')
        if isinstance(legal_representative, list):
            legal_representative = ' '.join(str(item) for item in legal_representative) if legal_representative else ''
            
        credit_code = company_data.get('credit_code', '')
        if isinstance(credit_code, list):
            credit_code = ' '.join(str(item) for item in credit_code) if credit_code else ''
            
        website = company_data.get('website', '')
        if isinstance(website, list):
            website = ' '.join(str(item) for item in website) if website else ''
        
        # 生成公司ID
        company_id = self._generate_company_id(name)
        
        # 构建公司实体
        company_entity = {
            'id': company_id,
            'name': name,
            'aliases': self._generate_company_aliases(name),
            'description': description,
            'industry': industry,
            'registration_info': {
                'registration_id': id_field,
                'legal_representative': legal_representative,
                'registered_capital': self.parser._normalize_capital(registered_capital) if registered_capital else None,
                'unified_credit_code': credit_code,
                'establishment_date': self.parser._parse_date(established_date) if established_date else None
            },
            'contact_info': {
                'address': company_data.get('address', ''),
                'website': website
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'data_source': 'test_data',
                'confidence': 0.8
            }
        }
        
        # 添加到公司字典
        self.companies[company_id] = company_entity
        self.build_stats['total_companies'] = len(self.companies)
        
        logger.debug(f"已添加公司实体: {company_entity['name']} (ID: {company_id})")
    
    def add_investor(self, investor_data: Dict[str, Any]) -> None:
        """添加投资方实体到知识图谱
        
        Args:
            investor_data: 投资方数据字典，包含name、type、description等字段
        """
        if not investor_data or not investor_data.get('name'):
            logger.warning("尝试添加空投资方数据或缺少名称字段")
            return
        
        # 处理可能为列表的字段
        name = investor_data.get('name', '')
        if isinstance(name, list):
            name = ' '.join(str(item) for item in name) if name else ''
        
        type_field = investor_data.get('type', '')
        if isinstance(type_field, list):
            type_field = ' '.join(str(item) for item in type_field) if type_field else ''
            
        description = investor_data.get('description', '')
        if isinstance(description, list):
            description = ' '.join(str(item) for item in description) if description else ''
            
        industry = investor_data.get('industry', '')
        if isinstance(industry, list):
            industry = ' '.join(str(item) for item in industry) if industry else ''
            
        scale = investor_data.get('scale', '')
        if isinstance(scale, list):
            scale = ' '.join(str(item) for item in scale) if scale else ''
            
        rounds = investor_data.get('rounds', '')
        if isinstance(rounds, list):
            rounds = ' '.join(str(item) for item in rounds) if rounds else ''
        
        # 生成投资方ID
        investor_id = self._generate_investor_id(name)
        
        # 构建投资方实体
        investor_entity = {
            'id': investor_id,
            'name': name,
            'aliases': self._generate_investor_aliases(name),
            'description': description,
            'type': type_field,
            'industry': industry,
            'scale': self._parse_scale(scale) if scale else 'unknown',
            'preferred_rounds': self._parse_preferred_rounds(rounds) if rounds else [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'data_source': 'test_data',
                'confidence': 0.8
            }
        }
        
        # 添加到投资方字典
        self.investors[investor_id] = investor_entity
        self.build_stats['total_investors'] = len(self.investors)
        
        logger.debug(f"已添加投资方实体: {investor_entity['name']} (ID: {investor_id})")
    
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
            # 获取投资方和融资方名称 - 优先使用英文键名（因为DataParser现在返回英文键名）
            investor_names = event.get('investors', event.get('投资方', []))
            investee_name = event.get('investee', event.get('融资方', ''))
            
            # 验证必要字段
            if not investor_names or not investee_name:
                if not investor_names and not investee_name:
                    logger.warning(f"投资事件缺少必要字段: 投资方={investor_names}, 融资方={investee_name}")
                elif not investor_names:
                    logger.info(f"投资事件投资方未披露，跳过关系创建: 融资方={investee_name}, 描述={event.get('description', '')[:50]}...")
                else:
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
    
    def _parse_scale(self, scale_str: Union[str, List[str]]) -> str:
        """解析规模信息"""
        if not scale_str:
            return 'unknown'
        
        # 如果是列表，转换为字符串
        if isinstance(scale_str, list):
            scale_str = ' '.join(str(item) for item in scale_str) if scale_str else ''
        
        if '人民币' in scale_str or '美元' in scale_str:
            return 'managed_fund'
        
        return 'unknown'
    
    def _parse_preferred_rounds(self, rounds_str: Union[str, List[str]]) -> List[str]:
        """解析偏好轮次"""
        if not rounds_str:
            return []
        
        # 如果是列表，转换为字符串
        if isinstance(rounds_str, list):
            rounds_str = ' '.join(str(item) for item in rounds_str) if rounds_str else ''
        
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
    
    def build_investment_structure_relationships(self, investment_structures: List[Dict]) -> None:
        """构建投资结构关系
        
        Args:
            investment_structures: 投资结构数据列表
        """
        logger.info(f"开始构建 {len(investment_structures)} 个投资结构关系")
        
        structure_relationships = []
        
        for structure in investment_structures:
            try:
                # 获取投资方名称（兼容两种字段名）
                investor_name = structure.get('机构名称', structure.get('name', ''))
                if not investor_name:
                    logger.warning(f"投资结构数据缺少机构名称: {structure}")
                    continue
                
                # 解析投资偏好（兼容两种字段名）
                industries = self._parse_structure_industries(structure.get('行业', structure.get('industries', '')))
                rounds = self._parse_structure_rounds(structure.get('轮次', structure.get('rounds', '')))
                
                # 创建投资偏好关系
                for industry in industries:
                    relationship = {
                        'source': investor_name,
                        'target': industry,
                        'type': 'INVESTS_IN_INDUSTRY',
                        'properties': {
                            'preference_strength': 'high',
                            'source_data': 'investment_structure',
                            'confidence': 0.8
                        }
                    }
                    structure_relationships.append(relationship)
                
                # 创建轮次偏好关系
                for round_type in rounds:
                    relationship = {
                        'source': investor_name,
                        'target': round_type,
                        'type': 'PREFERS_ROUND',
                        'properties': {
                            'preference_strength': 'medium',
                            'source_data': 'investment_structure',
                            'confidence': 0.8
                        }
                    }
                    structure_relationships.append(relationship)
                
            except Exception as e:
                logger.error(f"处理投资结构失败: {structure.get('机构名称', '')}, 错误: {e}")
                continue
        
        # 将投资结构关系添加到知识图谱
        if 'structure_relationships' not in self.knowledge_graph:
            self.knowledge_graph['structure_relationships'] = []
        
        self.knowledge_graph['structure_relationships'].extend(structure_relationships)
        logger.info(f"投资结构关系构建完成，共 {len(structure_relationships)} 个关系")
    
    def _parse_structure_industries(self, industries_str: str) -> List[str]:
        """解析投资结构中的行业信息
        
        Args:
            industries_str: 行业字符串，如"企业服务36家 文娱内容游戏9家"
            
        Returns:
            行业列表
        """
        if not industries_str:
            return []
        
        industries = []
        
        # 分割行业信息
        parts = industries_str.split(' ')
        for part in parts:
            # 提取行业名称（去掉数字和"家"）
            if '家' in part and len(part) > 2:
                industry = part.replace('家', '').strip()
                # 移除数字
                industry = ''.join([c for c in industry if not c.isdigit()])
                if industry:
                    industries.append(industry)
        
        return industries
    
    def _parse_structure_rounds(self, rounds_str: str) -> List[str]:
        """解析投资结构中的轮次信息
        
        Args:
            rounds_str: 轮次字符串，如"A轮56家、B轮47家"
            
        Returns:
            轮次列表
        """
        if not rounds_str:
            return []
        
        rounds = []
        
        # 分割轮次信息
        parts = rounds_str.split('、')
        for part in parts:
            # 提取轮次名称（去掉数字和"家"）
            if '轮' in part and len(part) > 2:
                round_type = part.replace('轮', '').strip()
                # 移除数字和"家"
                round_type = ''.join([c for c in round_type if not c.isdigit() and c != '家'])
                if round_type:
                    # 标准化轮次名称
                    standardized_round = self._standardize_round(round_type + '轮')
                    if standardized_round['round'] != 'unknown':
                        rounds.append(standardized_round['round'])
        
        return rounds