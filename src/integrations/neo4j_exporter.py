#!/usr/bin/env python3
"""
Neo4j知识图谱集成模块
将构建的知识图谱导入到Neo4j图数据库
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

try:
    from py2neo import Graph, Node, Relationship, NodeMatcher
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("py2neo库未安装，Neo4j功能将不可用")

try:
    from processors.config import LOGGING_CONFIG
except ImportError:
    # 如果无法导入，使用默认配置
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }

# 配置日志
logging.basicConfig(
    level=LOGGING_CONFIG.get('level', 'INFO'),
    format=LOGGING_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    datefmt=LOGGING_CONFIG.get('datefmt', '%Y-%m-%d %H:%M:%S')
)
logger = logging.getLogger(__name__)

# 设置日志级别为WARNING，减少INFO级别的日志输出
logger.setLevel(logging.WARNING)


@dataclass
class Config:
    """连接配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "1234567kk"  # 更新为正确的密码
    database: str = "neo4j"


class KnowledgeGraphExporter:
    """知识图谱导出器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.graph: Optional[Graph] = None
        self.node_matcher: Optional[NodeMatcher] = None
        self._connect()
    
    def _connect(self):
        """连接到Neo4j数据库"""
        if not NEO4J_AVAILABLE:
            raise RuntimeError("py2neo库未安装，无法连接Neo4j")
        
        try:
            self.graph = Graph(
                self.config.uri, 
                auth=(self.config.username, self.config.password)
            )
            self.node_matcher = NodeMatcher(self.graph)
            logger.info(f"成功连接到Neo4j: {self.config.uri}")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def clear_existing_data(self, entity_types: List[str] = None):
        """清除现有数据"""
        if not entity_types:
            entity_types = ['公司', '投资方', '行业', '概念']
        
        logger.info(f"清除现有数据: {entity_types}")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        try:
            for entity_type in entity_types:
                query = f"MATCH (n:`{entity_type}`) DETACH DELETE n"
                self.graph.run(query)
                logger.info(f"已清除 {entity_type} 实体")
        except Exception as e:
            logger.error(f"清除数据失败: {e}")
            raise
    
    def create_company_nodes(self, companies: Dict[str, Dict]) -> int:
        """创建公司节点"""
        logger.info(f"开始创建公司节点: {len(companies)} 个")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        created_count = 0
        
        try:
            for company_id, company_data in companies.items():
                # 检查是否已存在
                existing = self.node_matcher.match(
                    "公司", 
                    公司ID=company_id
                ).first()
                
                if existing is not None:
                    logger.debug(f"公司已存在，跳过: {company_data.get('name', company_id)}")
                    continue
                
                # 创建节点属性
                node_properties = {
                    '公司ID': company_id,
                    '公司名称': company_data.get('name', ''),
                    'name': company_data.get('name', ''),  # 添加name属性以便查找
                    '股票代码': company_data.get('stock_code', ''),
                    '成立时间': company_data.get('establishment_date', ''),
                    '注册资本': company_data.get('registered_capital', ''),
                    '所属行业': company_data.get('industry', ''),
                    '公司地址': company_data.get('address', ''),
                    '公司简介': company_data.get('description', ''),
                    '统一社会信用代码': company_data.get('unified_social_credit_code', ''),
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                # 添加联系信息
                contact_info = company_data.get('contact_info', {})
                if contact_info:
                    node_properties.update({
                        '联系电话': contact_info.get('phone', ''),
                        '公司网站': contact_info.get('website', ''),
                        '电子邮箱': contact_info.get('email', '')
                    })
                
                company_node = Node("公司", **node_properties)
                self.graph.create(company_node)
                created_count += 1
                
                if created_count % 100 == 0:
                    logger.info(f"已创建 {created_count} 个公司节点")
            
            logger.info(f"公司节点创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建公司节点失败: {e}")
            raise
    
    def create_investor_nodes(self, investors: Dict[str, Dict]) -> int:
        """创建投资方节点"""
        logger.info(f"开始创建投资方节点: {len(investors)} 个")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        created_count = 0
        
        try:
            for investor_id, investor_data in investors.items():
                # 检查是否已存在
                existing = self.node_matcher.match(
                    "投资方", 
                    投资方ID=investor_id
                ).first()
                
                if existing is not None:
                    logger.debug(f"投资方已存在，跳过: {investor_data.get('name', investor_id)}")
                    continue
                
                # 处理投资偏好数据
                investment_focus = investor_data.get('investment_focus', {})
                if isinstance(investment_focus, list):
                    # 如果investment_focus是列表，直接作为投资领域
                    industries = investment_focus
                    stages = []
                elif isinstance(investment_focus, dict):
                    # 如果investment_focus是字典，提取industries和stages
                    industries = investment_focus.get('industries', [])
                    stages = investment_focus.get('stages', [])
                else:
                    industries = []
                    stages = []
                
                # 创建节点属性
                node_properties = {
                    '投资方ID': investor_id,
                    '投资方名称': investor_data.get('name', ''),
                    'name': investor_data.get('name', ''),  # 添加name属性以便查找
                    '投资方类型': investor_data.get('type', ''),
                    '管理资金规模': investor_data.get('managed_capital', ''),
                    '成立时间': investor_data.get('establishment_date', ''),
                    '投资领域': ','.join(industries) if industries else '',
                    '投资阶段': ','.join(stages) if stages else '',
                    '投资方简介': investor_data.get('description', ''),
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                # 添加联系信息
                contact_info = investor_data.get('contact_info', {})
                if contact_info:
                    node_properties.update({
                        '联系电话': contact_info.get('phone', ''),
                        '公司网站': contact_info.get('website', ''),
                        '电子邮箱': contact_info.get('email', ''),
                        '办公地址': contact_info.get('address', '')
                    })
                
                investor_node = Node("投资方", **node_properties)
                self.graph.create(investor_node)
                created_count += 1
                
                if created_count % 100 == 0:
                    logger.info(f"已创建 {created_count} 个投资方节点")
            
            logger.info(f"投资方节点创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建投资方节点失败: {e}")
            raise
    
    def create_industry_nodes(self, industries: List[str]) -> int:
        """创建行业节点"""
        logger.info(f"开始创建行业节点: {len(industries)} 个")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        created_count = 0
        
        try:
            for industry in industries:
                if not industry:
                    continue
                
                # 检查是否已存在
                existing = self.node_matcher.match(
                    "行业", 
                    行业名称=industry
                ).first()
                
                if existing:
                    logger.debug(f"行业已存在，跳过: {industry}")
                    continue
                
                # 创建节点属性
                node_properties = {
                    '行业名称': industry,
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                industry_node = Node("行业", **node_properties)
                self.graph.create(industry_node)
                created_count += 1
            
            logger.info(f"行业节点创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建行业节点失败: {e}")
            raise
    
    def create_structure_relationships(self, structure_relationships: List[Dict]) -> int:
        """创建投资结构关系"""
        logger.info(f"开始创建投资结构关系: {len(structure_relationships)} 个")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        created_count = 0
        
        try:
            for rel in structure_relationships:
                # 获取关系数据
                source = rel.get('source', '')
                target = rel.get('target', '')
                rel_type = rel.get('type', '')
                
                if not source or not target or not rel_type:
                    logger.warning(f"投资结构关系缺少必要字段: {rel}")
                    continue
                
                # 查找投资方节点
                investor_node = self.node_matcher.match(
                    "投资方", 
                    投资方名称=source
                ).first()
                
                # 如果找不到投资方节点，尝试创建一个
                if not investor_node:
                    investor_node = self.node_matcher.match(
                        "投资方", 
                        name=source
                    ).first()
                
                # 根据关系类型处理不同的目标节点
                target_node = None
                if rel_type == 'INVESTS_IN_INDUSTRY':
                    # 查找行业节点
                    target_node = self.node_matcher.match(
                        "行业", 
                        行业名称=target
                    ).first()
                    
                    # 如果找不到行业节点，创建一个
                    if not target_node:
                        target_node = Node("行业", 行业名称=target)
                        self.graph.create(target_node)
                
                elif rel_type == 'PREFERS_ROUND':
                    # 查找轮次节点
                    target_node = self.node_matcher.match(
                        "轮次", 
                        轮次名称=target
                    ).first()
                    
                    # 如果找不到轮次节点，创建一个
                    if not target_node:
                        target_node = Node("轮次", 轮次名称=target)
                        self.graph.create(target_node)
                
                if not investor_node or not target_node:
                    logger.warning(f"找不到对应的节点 - 投资方: {source}, 目标: {target}, 类型: {rel_type}")
                    continue
                
                # 创建关系属性
                rel_properties = {
                    'preference_strength': rel.get('properties', {}).get('preference_strength', 'medium'),
                    'source_data': rel.get('properties', {}).get('source_data', 'investment_structure'),
                    'confidence': rel.get('properties', {}).get('confidence', 0.8),
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                # 创建关系
                relationship = Relationship(
                    investor_node, 
                    '投资行业' if rel_type == 'INVESTS_IN_INDUSTRY' else '偏好轮次', 
                    target_node, 
                    **rel_properties
                )
                self.graph.create(relationship)
                created_count += 1
                
                if created_count % 500 == 0:
                    logger.info(f"已创建 {created_count} 个投资结构关系")
            
            logger.info(f"投资结构关系创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建投资结构关系失败: {e}")
            raise
    
    def create_investment_relationships(self, relationships: List[Dict]) -> int:
        """创建投资关系"""
        logger.info(f"开始创建投资关系: {len(relationships)} 个")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        created_count = 0
        
        try:
            # 获取所有投资方和公司节点
            investor_nodes = list(self.graph.run("MATCH (n:投资方) RETURN n"))
            company_nodes = list(self.graph.run("MATCH (n:公司) RETURN n"))
            
            # 创建名称映射字典
            investor_name_map = {}
            for record in investor_nodes:
                node = record['n']
                name = node.get('name', '')
                alt_name = node.get('投资方名称', '')
                if name:
                    investor_name_map[name] = node
                if alt_name:
                    investor_name_map[alt_name] = node
            
            company_name_map = {}
            for record in company_nodes:
                node = record['n']
                name = node.get('name', '')
                alt_name = node.get('公司名称', '')
                if name:
                    company_name_map[name] = node
                if alt_name:
                    company_name_map[alt_name] = node
            
            for rel in relationships:
                # 获取关系数据 - 使用字典访问方式
                source = rel.get('source')
                target = rel.get('target')
                rel_type = rel.get('type')
                
                if not source or not target or not rel_type:
                    logger.warning(f"投资关系缺少必要字段: source={source}, target={target}, type={rel_type}")
                    continue
                
                # 根据关系类型处理不同的关系
                if rel_type == 'INVESTED_IN':
                    # 投资方投资公司的关系
                    # 直接查找投资方节点
                    investor_node = investor_name_map.get(source)
                    
                    # 如果找不到投资方节点，尝试模糊匹配
                    if not investor_node:
                        investor_node = self._fuzzy_match_node(source, investor_name_map)
                    
                    # 如果仍然找不到，创建新的投资方节点
                    if not investor_node:
                        investor_node = Node(
                            "投资方",
                            name=source,
                            投资方名称=source,
                            数据来源='知识图谱构建系统',
                            创建时间=datetime.now().isoformat()
                        )
                        self.graph.create(investor_node)
                        logger.info(f"创建新的投资方节点: {source}")
                    
                    # 直接查找公司节点
                    company_node = company_name_map.get(target)
                    
                    # 如果找不到公司节点，尝试模糊匹配
                    if not company_node:
                        company_node = self._fuzzy_match_node(target, company_name_map)
                    
                    # 如果仍然找不到，创建新的公司节点
                    if not company_node:
                        company_node = Node(
                            "公司",
                            name=target,
                            公司名称=target,
                            数据来源='知识图谱构建系统',
                            创建时间=datetime.now().isoformat()
                        )
                        self.graph.create(company_node)
                        logger.info(f"创建新的公司节点: {target}")
                    
                    # 创建关系
                    relationship = Relationship(
                        investor_node, 
                        '投资', 
                        company_node, 
                        **self._create_relationship_properties(rel)
                    )
                    self.graph.create(relationship)
                    created_count += 1
                    
                elif rel_type == 'INVESTS_IN_INDUSTRY':
                    # 投资方投资行业的关系
                    # 直接查找投资方节点
                    investor_node = investor_name_map.get(source)
                    
                    # 如果找不到投资方节点，尝试模糊匹配
                    if not investor_node:
                        investor_node = self._fuzzy_match_node(source, investor_name_map)
                    
                    # 如果仍然找不到，创建新的投资方节点
                    if not investor_node:
                        investor_node = Node(
                            "投资方",
                            name=source,
                            投资方名称=source,
                            数据来源='知识图谱构建系统',
                            创建时间=datetime.now().isoformat()
                        )
                        self.graph.create(investor_node)
                        logger.info(f"创建新的投资方节点: {source}")
                    
                    # 查找行业节点
                    industry_node = self.node_matcher.match(
                        "行业", 
                        行业名称=target
                    ).first()
                    
                    # 如果找不到行业节点，创建新的行业节点
                    if not industry_node:
                        industry_node = Node(
                            "行业",
                            行业名称=target,
                            name=target,
                            数据来源='知识图谱构建系统',
                            创建时间=datetime.now().isoformat()
                        )
                        self.graph.create(industry_node)
                        logger.info(f"创建新的行业节点: {target}")
                    
                    # 创建关系
                    relationship = Relationship(
                        investor_node, 
                        '投资行业', 
                        industry_node, 
                        **self._create_relationship_properties(rel)
                    )
                    self.graph.create(relationship)
                    created_count += 1
                    
                elif rel_type == 'PREFERS_ROUND':
                    # 投资方偏好轮次的关系
                    # 直接查找投资方节点
                    investor_node = investor_name_map.get(source)
                    
                    # 如果找不到投资方节点，尝试模糊匹配
                    if not investor_node:
                        investor_node = self._fuzzy_match_node(source, investor_name_map)
                    
                    # 如果仍然找不到，创建新的投资方节点
                    if not investor_node:
                        investor_node = Node(
                            "投资方",
                            name=source,
                            投资方名称=source,
                            数据来源='知识图谱构建系统',
                            创建时间=datetime.now().isoformat()
                        )
                        self.graph.create(investor_node)
                        logger.info(f"创建新的投资方节点: {source}")
                    
                    # 创建或获取轮次节点
                    round_node = self.node_matcher.match(
                        "轮次", 
                        轮次名称=target
                    ).first()
                    
                    if not round_node:
                        # 如果轮次节点不存在，创建它
                        round_node = Node(
                            "轮次",
                            轮次名称=target,
                            name=target,
                            数据来源='知识图谱构建系统',
                            创建时间=datetime.now().isoformat()
                        )
                        self.graph.create(round_node)
                    
                    # 创建关系
                    relationship = Relationship(
                        investor_node, 
                        '偏好轮次', 
                        round_node, 
                        **self._create_relationship_properties(rel)
                    )
                    self.graph.create(relationship)
                    created_count += 1
                
                if created_count % 500 == 0:
                    logger.info(f"已创建 {created_count} 个投资关系")
            
            logger.info(f"投资关系创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建投资关系失败: {e}")
            raise
    
    def _fuzzy_match_node(self, name: str, name_map: Dict) -> Node:
        """模糊匹配节点"""
        # 直接匹配
        if name in name_map:
            return name_map[name]
        
        # 简单的模糊匹配：检查名称是否包含在节点名称中
        for node_name, node in name_map.items():
            if name in node_name or node_name in name:
                return node
        
        # 尝试去除空格和特殊字符后匹配
        cleaned_name = ''.join(c for c in name if c.isalnum())
        for node_name, node in name_map.items():
            cleaned_node_name = ''.join(c for c in node_name if c.isalnum())
            if cleaned_name in cleaned_node_name or cleaned_node_name in cleaned_name:
                return node
        
        # 尝试分割名称并匹配部分
        name_parts = name.split()
        if len(name_parts) > 1:
            for part in name_parts:
                if len(part) > 2:  # 只匹配长度大于2的部分
                    for node_name, node in name_map.items():
                        if part in node_name:
                            return node
        
        # 使用更高级的相似度计算
        best_match = None
        best_score = 0
        for node_name, node in name_map.items():
            # 计算Jaccard相似度
            name_chars = set(name)
            node_name_chars = set(node_name)
            intersection = name_chars & node_name_chars
            union = name_chars | node_name_chars
            
            if len(union) > 0:
                jaccard_similarity = len(intersection) / len(union)
            else:
                jaccard_similarity = 0
            
            # 计算字符序列相似度
            common_substrings = 0
            min_len = min(len(name), len(node_name))
            for i in range(min_len - 1):
                if name[i:i+2] == node_name[i:i+2]:
                    common_substrings += 1
            
            sequence_similarity = common_substrings / (min_len - 1) if min_len > 1 else 0
            
            # 综合相似度
            combined_similarity = 0.7 * jaccard_similarity + 0.3 * sequence_similarity
            
            if combined_similarity > best_score and combined_similarity > 0.4:  # 降低相似度阈值
                best_score = combined_similarity
                best_match = node
        
        return best_match
    
    def _create_relationship_properties(self, rel) -> Dict:
        """创建关系属性"""
        # 处理Relation对象或字典
        if hasattr(rel, 'properties'):
            properties = rel.properties
        elif isinstance(rel, dict):
            properties = rel.get('properties', {})
        else:
            properties = {}
        
        # 确保属性是字典格式
        if not isinstance(properties, dict):
            properties = {}
        
        # 构建标准化的关系属性
        rel_properties = {}
        for key, value in properties.items():
            if value is not None:
                rel_properties[key] = value
        
        # 添加默认属性
        rel_properties['created_at'] = datetime.now().isoformat()
        rel_properties['data_source'] = 'financial_intelligence'
        
        return rel_properties
    
    def create_industry_relationships(self, companies: Dict) -> int:
        """创建行业关系"""
        logger.info(f"开始创建行业关系: {len(companies)} 个")
        
        if self.graph is None or self.node_matcher is None:
            raise RuntimeError("Neo4j连接不可用")
        
        created_count = 0
        
        try:
            for company_id, company_data in companies.items():
                company_name = company_data.get('name', '')
                industry = company_data.get('industry', '')
                
                if not company_name or not industry:
                    continue
                
                # 优先使用name属性查找公司节点
                company_node = self.node_matcher.match(
                    "公司", 
                    name=company_name
                ).first()
                
                # 如果找不到公司节点，尝试用name字段查找
                if not company_node:
                    company_node = self.node_matcher.match(
                        "公司", 
                        name=company_name
                    ).first()
                
                # 查找行业节点
                industry_node = self.node_matcher.match(
                    "行业", 
                    行业名称=industry
                ).first()
                
                if not company_node or not industry_node:
                    continue
                
                # 创建关系属性
                rel_properties = {
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                # 创建关系
                relationship = Relationship(
                    company_node, 
                    '属于', 
                    industry_node, 
                    **rel_properties
                )
                self.graph.create(relationship)
                created_count += 1
            
            logger.info(f"行业关系创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建行业关系失败: {e}")
            raise
    
    def export_knowledge_graph(self, kg_data: Dict, clear_existing: bool = True) -> Dict[str, Any]:
        """导出知识图谱到Neo4j"""
        logger.info("开始导出知识图谱到Neo4j")
        
        start_time = datetime.now()
        export_stats: Dict[str, Union[int, float]] = {}
        
        # 处理不同格式的输入数据
        companies_raw = kg_data.get('companies', [])
        investors_raw = kg_data.get('investors', [])
        relationships_raw = kg_data.get('relationships', [])
        structure_relationships_raw = kg_data.get('structure_relationships', [])
        
        # 如果数据是列表格式，转换为字典格式
        if isinstance(companies_raw, list):
            companies = {}
            for company in companies_raw:
                company_id = company.get('id') or company.get('name', '')
                if company_id:
                    companies[company_id] = company
        else:
            companies = companies_raw
            
        if isinstance(investors_raw, list):
            investors = {}
            for investor in investors_raw:
                investor_id = investor.get('id') or investor.get('name', '')
                if investor_id:
                    investors[investor_id] = investor
        else:
            investors = investors_raw
            
        if isinstance(relationships_raw, list):
            relationships = relationships_raw
        else:
            relationships = relationships_raw
            
        if isinstance(structure_relationships_raw, list):
            structure_relationships = structure_relationships_raw
        else:
            structure_relationships = []
        
        # 清除现有数据（可选）
        if clear_existing:
            self.clear_existing_data()
        
        # 创建实体节点
        export_stats['companies_created'] = self.create_company_nodes(companies)
        export_stats['investors_created'] = self.create_investor_nodes(investors)
        
        # 提取行业信息并创建行业节点
        industries = set()
        for company in companies.values():
            industry = company.get('industry')
            if industry:
                industries.add(industry)
        
        export_stats['industries_created'] = self.create_industry_nodes(list(industries))
        
        # 创建关系
        export_stats['investment_relationships_created'] = self.create_investment_relationships(relationships)
        export_stats['industry_relationships_created'] = self.create_industry_relationships(companies)
        export_stats['structure_relationships_created'] = self.create_structure_relationships(structure_relationships)
        
        # 计算总耗时
        end_time = datetime.now()
        export_stats['export_duration'] = (end_time - start_time).total_seconds()
        export_stats['total_entities'] = int(
            export_stats['companies_created'] + 
            export_stats['investors_created'] + 
            export_stats['industries_created']
        )
        export_stats['total_relationships'] = int(
            export_stats['investment_relationships_created'] + 
            export_stats['industry_relationships_created'] +
            export_stats['structure_relationships_created']
        )
        
        logger.info("知识图谱导出完成")
        logger.info(f"导出统计: {export_stats}")
        
        return export_stats
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """获取导出统计信息"""
        if self.graph:
            try:
                # 实体统计
                company_count = len(list(self.node_matcher.match("公司")))
                investor_count = len(list(self.node_matcher.match("投资方")))
                industry_count = len(list(self.node_matcher.match("行业")))
                
                # 关系统统统
                rel_query = """
                MATCH ()-[r]->() 
                RETURN type(r) as relation_type, count(r) as count
                ORDER BY count DESC
                """
                relationships = self.graph.run(rel_query).data()
                
                return {
                    'entity_counts': {
                        'companies': company_count,
                        'investors': investor_count,
                        'industries': industry_count,
                        'total': company_count + investor_count + industry_count
                    },
                    'relationships': relationships,
                    'connection_status': 'connected'
                }
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}")
                return {'connection_status': 'error', 'error': str(e)}
        else:
            return {'connection_status': 'not_connected', 'error': 'Graph not available'}
    
    def close(self):
        """关闭连接"""
        if self.graph:
            logger.info("关闭Neo4j连接")
            # py2neo会自动管理连接，这里不需要特别处理


class IntegrationManager:
    """集成管理器"""
    
    def __init__(self, neo4j_config: Optional[Config] = None):
        self.neo4j_config = neo4j_config or Config()
        self.exporter: Optional[KnowledgeGraphExporter] = None
        self._initialize_exporter()
    
    def _initialize_exporter(self):
        """初始化导出器"""
        if NEO4J_AVAILABLE:
            try:
                self.exporter = KnowledgeGraphExporter(self.neo4j_config)
                logger.info("Neo4j导出器初始化成功")
            except Exception as e:
                logger.warning(f"Neo4j导出器初始化失败: {e}")
                self.exporter = None
        else:
            logger.warning("py2neo库未安装，Neo4j功能不可用")
            self.exporter = None
    
    def integrate_with_neo4j(self, kg_data: Dict, clear_existing: bool = True) -> Dict:
        """集成知识图谱到Neo4j"""
        if self.exporter:
            # 导出知识图谱
            export_stats = self.exporter.export_knowledge_graph(kg_data, clear_existing)
            
            # 获取统计信息
            statistics = self.exporter.get_export_statistics()
            
            return {
                'success': True,
                'export_statistics': export_stats,
                'current_statistics': statistics,
                'neo4j_available': True
            }
        else:
            return {
                'success': False,
                'error': 'Neo4j导出器不可用',
                'recommendation': '请安装py2neo库并配置Neo4j连接'
            }
    
    def get_integration_status(self) -> Dict:
        """获取集成状态"""
        if self.exporter:
            statistics = self.exporter.get_export_statistics()
            return {
                'neo4j_available': True,
                'connection_status': statistics.get('connection_status', 'unknown'),
                'statistics': statistics
            }
        else:
            return {
                'neo4j_available': False,
                'connection_status': 'not_configured',
                'recommendation': '请安装py2neo库并配置Neo4j连接'
            }
    
    def close(self):
        """关闭管理器"""
        if self.exporter:
            self.exporter.close()


def main():
    """主函数 - 用于测试Neo4j集成"""
    print("Neo4j知识图谱集成测试")
    print("="*50)
    
    # 创建集成管理器
    config = Config(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"  # 请根据实际情况修改
    )
    
    manager = IntegrationManager(config)
    
    # 检查集成状态
    status = manager.get_integration_status()
    print(f"集成状态: {status}")
    
    if status['neo4j_available']:
        print("✓ Neo4j连接成功")
        
        # 加载知识图谱数据（示例）
        kg_file = Path("output/knowledge_graph.json")
        if kg_file.exists():
            with open(kg_file, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            
            print(f"加载知识图谱数据: {len(kg_data.get('companies', {}))} 公司, {len(kg_data.get('investors', {}))} 投资方")
            
            # 执行集成
            result = manager.integrate_with_neo4j(kg_data)
            
            if result['success']:
                print("✓ 知识图谱集成成功")
                print(f"导出统计: {result['export_statistics']}")
            else:
                print(f"✗ 集成失败: {result['error']}")
        else:
            print("未找到知识图谱数据文件，请先运行主流程")
    else:
        print("✗ Neo4j不可用")
        print(f"建议: {status.get('recommendation', '')}")
    
    manager.close()
    print("测试完成")


if __name__ == "__main__":
    main()