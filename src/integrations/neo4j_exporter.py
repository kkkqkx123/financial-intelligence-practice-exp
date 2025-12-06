#!/usr/bin/env python3
"""
Neo4j知识图谱集成模块
将构建的知识图谱导入到Neo4j图数据库
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from py2neo import Graph, Node, Relationship, NodeMatcher
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("py2neo库未安装，Neo4j功能将不可用")

from src.processors.config import LOGGING_CONFIG

# 配置日志
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j连接配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class Neo4jKnowledgeGraphExporter:
    """Neo4j知识图谱导出器"""
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.graph = None
        self.node_matcher = None
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
        
        created_count = 0
        
        try:
            for company_id, company_data in companies.items():
                # 检查是否已存在
                existing = self.node_matcher.match(
                    "公司", 
                    公司ID=company_id
                ).first()
                
                if existing:
                    logger.debug(f"公司已存在，跳过: {company_data.get('name', company_id)}")
                    continue
                
                # 创建节点属性
                node_properties = {
                    '公司ID': company_id,
                    '公司名称': company_data.get('name', ''),
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
        
        created_count = 0
        
        try:
            for investor_id, investor_data in investors.items():
                # 检查是否已存在
                existing = self.node_matcher.match(
                    "投资方", 
                    投资方ID=investor_id
                ).first()
                
                if existing:
                    logger.debug(f"投资方已存在，跳过: {investor_data.get('name', investor_id)}")
                    continue
                
                # 创建节点属性
                node_properties = {
                    '投资方ID': investor_id,
                    '投资方名称': investor_data.get('name', ''),
                    '投资方类型': investor_data.get('type', ''),
                    '管理资金规模': investor_data.get('managed_capital', ''),
                    '成立时间': investor_data.get('establishment_date', ''),
                    '投资领域': ','.join(investor_data.get('investment_focus', {}).get('industries', [])),
                    '投资阶段': ','.join(investor_data.get('investment_focus', {}).get('stages', [])),
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
    
    def create_investment_relationships(self, relationships: List[Dict]) -> int:
        """创建投资关系"""
        logger.info(f"开始创建投资关系: {len(relationships)} 个")
        
        created_count = 0
        
        try:
            for rel in relationships:
                investor_id = rel.get('investor_id')
                company_id = rel.get('company_id')
                
                if not investor_id or not company_id:
                    logger.warning(f"关系缺少必要的ID: {rel}")
                    continue
                
                # 查找投资方节点
                investor_node = self.node_matcher.match(
                    "投资方", 
                    投资方ID=investor_id
                ).first()
                
                # 查找公司节点
                company_node = self.node_matcher.match(
                    "公司", 
                    公司ID=company_id
                ).first()
                
                if not investor_node or not company_node:
                    logger.warning(f"找不到对应的节点 - 投资方: {investor_id}, 公司: {company_id}")
                    continue
                
                # 创建关系属性
                rel_properties = {
                    '投资金额': rel.get('amount', ''),
                    '投资轮次': rel.get('round', ''),
                    '投资时间': rel.get('date', ''),
                    '投资比例': rel.get('ratio', ''),
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                # 创建投资关系（投资方 -> 公司）
                investment_rel = Relationship(
                    investor_node, 
                    '投资', 
                    company_node, 
                    **rel_properties
                )
                self.graph.create(investment_rel)
                created_count += 1
                
                if created_count % 1000 == 0:
                    logger.info(f"已创建 {created_count} 个投资关系")
            
            logger.info(f"投资关系创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建投资关系失败: {e}")
            raise
    
    def create_industry_relationships(self, companies: Dict[str, Dict]) -> int:
        """创建行业关系"""
        logger.info("开始创建行业关系")
        
        created_count = 0
        
        try:
            for company_id, company_data in companies.items():
                industry = company_data.get('industry')
                if not industry:
                    continue
                
                # 查找公司节点
                company_node = self.node_matcher.match(
                    "公司", 
                    公司ID=company_id
                ).first()
                
                # 查找行业节点
                industry_node = self.node_matcher.match(
                    "行业", 
                    行业名称=industry
                ).first()
                
                if not company_node or not industry_node:
                    logger.warning(f"找不到对应的节点 - 公司: {company_id}, 行业: {industry}")
                    continue
                
                # 创建行业关系
                rel_properties = {
                    '数据来源': '知识图谱构建系统',
                    '创建时间': datetime.now().isoformat()
                }
                
                industry_rel = Relationship(
                    company_node, 
                    '属于行业', 
                    industry_node, 
                    **rel_properties
                )
                self.graph.create(industry_rel)
                created_count += 1
                
                if created_count % 500 == 0:
                    logger.info(f"已创建 {created_count} 个行业关系")
            
            logger.info(f"行业关系创建完成: {created_count} 个")
            return created_count
            
        except Exception as e:
            logger.error(f"创建行业关系失败: {e}")
            raise
    
    def export_knowledge_graph(self, kg_data: Dict, clear_existing: bool = True) -> Dict[str, int]:
        """导出知识图谱到Neo4j"""
        logger.info("开始导出知识图谱到Neo4j")
        
        start_time = datetime.now()
        export_stats = {}
        
        try:
            companies = kg_data.get('companies', {})
            investors = kg_data.get('investors', {})
            relationships = kg_data.get('relationships', [])
            
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
            
            # 计算总耗时
            end_time = datetime.now()
            export_stats['export_duration'] = (end_time - start_time).total_seconds()
            export_stats['total_entities'] = (
                export_stats['companies_created'] + 
                export_stats['investors_created'] + 
                export_stats['industries_created']
            )
            export_stats['total_relationships'] = (
                export_stats['investment_relationships_created'] + 
                export_stats['industry_relationships_created']
            )
            
            logger.info("知识图谱导出完成")
            logger.info(f"导出统计: {export_stats}")
            
            return export_stats
            
        except Exception as e:
            logger.error(f"导出知识图谱失败: {e}")
            raise
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """获取导出统计信息"""
        if not self.graph:
            return {}
        
        try:
            # 实体统计
            company_count = len(list(self.node_matcher.match("公司")))
            investor_count = len(list(self.node_matcher.match("投资方")))
            industry_count = len(list(self.node_matcher.match("行业")))
            
            # 关系统计
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
    
    def close(self):
        """关闭连接"""
        if self.graph:
            logger.info("关闭Neo4j连接")
            # py2neo会自动管理连接，这里不需要特别处理


class KnowledgeGraphIntegrationManager:
    """知识图谱集成管理器"""
    
    def __init__(self, neo4j_config: Optional[Neo4jConfig] = None):
        self.neo4j_config = neo4j_config or Neo4jConfig()
        self.exporter = None
        self._initialize_exporter()
    
    def _initialize_exporter(self):
        """初始化导出器"""
        if NEO4J_AVAILABLE:
            try:
                self.exporter = Neo4jKnowledgeGraphExporter(self.neo4j_config)
                logger.info("Neo4j导出器初始化成功")
            except Exception as e:
                logger.warning(f"Neo4j导出器初始化失败: {e}")
                self.exporter = None
        else:
            logger.warning("py2neo库未安装，Neo4j功能不可用")
            self.exporter = None
    
    def integrate_with_neo4j(self, kg_data: Dict, clear_existing: bool = True) -> Dict:
        """集成知识图谱到Neo4j"""
        if not self.exporter:
            return {
                'success': False,
                'error': 'Neo4j导出器不可用',
                'recommendation': '请安装py2neo库并配置Neo4j连接'
            }
        
        try:
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
            
        except Exception as e:
            error_msg = f"集成到Neo4j失败: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'neo4j_available': False
            }
    
    def get_integration_status(self) -> Dict:
        """获取集成状态"""
        if not self.exporter:
            return {
                'neo4j_available': False,
                'connection_status': 'not_configured',
                'recommendation': '请安装py2neo库并配置Neo4j连接'
            }
        
        try:
            statistics = self.exporter.get_export_statistics()
            return {
                'neo4j_available': True,
                'connection_status': statistics.get('connection_status', 'unknown'),
                'statistics': statistics
            }
        except Exception as e:
            return {
                'neo4j_available': True,
                'connection_status': 'error',
                'error': str(e)
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
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"  # 请根据实际情况修改
    )
    
    manager = KnowledgeGraphIntegrationManager(config)
    
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