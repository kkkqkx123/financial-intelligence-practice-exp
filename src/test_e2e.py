"""
端到端测试文件
验证整个知识图谱构建工作流
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from main import Pipeline
from processors import KGBuilder, OptimizedRelationExtractor, get_relation_extractor, DataParser, DataValidator
from integrations.neo4j_exporter import IntegrationManager

# 配置日志
log_file = "D:/Source/torch/financial-intellgience/src/logs/test_e2e.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """端到端测试运行器"""
    
    def __init__(self):
        self.test_results = {
            "data_loading": {},
            "data_processing": {},
            "entity_extraction": {},
            "relation_extraction": {},
            "knowledge_graph_construction": {},
            "neo4j_integration": {},
            "overall": {}
        }
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self):
        """运行所有端到端测试"""
        logger.info("开始端到端测试...")
        self.start_time = datetime.now()
        
        try:
            # 1. 测试数据加载
            await self.test_data_loading()
            
            # 2. 测试数据处理
            await self.test_data_processing()
            
            # 3. 测试实体提取
            await self.test_entity_extraction()
            
            # 4. 测试关系提取
            await self.test_relation_extraction()
            
            # 5. 测试知识图谱构建
            await self.test_knowledge_graph_construction()
            
            # 6. 测试Neo4j集成
            await self.test_neo4j_integration()
            
            # 7. 测试完整工作流
            await self.test_complete_workflow()
            
        except Exception as e:
            logger.error(f"端到端测试失败: {e}")
            self.test_results["overall"] = {
                "success": False,
                "error": str(e)
            }
        
        self.end_time = datetime.now()
        self.print_test_results()
        self.save_test_results()
    
    async def test_data_loading(self):
        """测试数据加载"""
        logger.info("测试数据加载...")
        start_time = datetime.now()
        
        try:
            # 使用Pipeline加载CSV数据
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            raw_data = pipeline.load_data_files()
            
            # 检查数据是否加载成功
            data_status = {}
            for data_type, data in raw_data.items():
                data_status[data_type] = {
                    "loaded": len(data) > 0,
                    "count": len(data),
                    "sample": data[0] if data else None
                }
            
            end_time = datetime.now()
            
            self.test_results["data_loading"] = {
                "success": True,
                "message": "数据加载成功",
                "data_status": data_status,
                "duration": (end_time - start_time).total_seconds()
            }
            logger.info(f"数据加载测试通过，耗时: {self.test_results['data_loading']['duration']:.2f}秒")
            
        except Exception as e:
            end_time = datetime.now()
            self.test_results["data_loading"] = {
                "success": False,
                "message": f"数据加载失败: {str(e)}",
                "duration": (end_time - start_time).total_seconds()
            }
            logger.error(f"数据加载测试失败: {e}")
    
    async def test_data_processing(self):
        """测试数据处理"""
        logger.info("测试数据处理...")
        start_time = datetime.now()
        
        try:
            # 初始化组件
            parser = DataParser()
            validator = DataValidator()
            
            # 使用Pipeline加载CSV数据
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            raw_data = pipeline.load_data_files()
            
            # 处理数据
            company_data = parser.parse_companies(raw_data.get('companies', []))
            event_data = parser.parse_investment_events(raw_data.get('investment_events', []))
            structure_data = parser.parse_investment_institutions(raw_data.get('investors', []))
            
            # 验证数据
            validation_results = {
                "company_data": validator.validate_company_data(company_data),
                "event_data": validator.validate_investment_event_data(event_data),
                "structure_data": validator.validate_investor_data(structure_data)
            }
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.test_results["data_processing"] = {
                "success": True,
                "duration": duration,
                "company_data_count": len(company_data),
                "event_data_count": len(event_data),
                "structure_data_count": len(structure_data),
                "validation_results": validation_results
            }
            
            # 存储数据供后续测试使用
            self.company_data = company_data
            self.event_data = event_data
            self.structure_data = structure_data
            
        except Exception as e:
            logger.error(f"数据处理测试失败: {e}")
            self.test_results["data_processing"] = {
                "success": False,
                "error": str(e)
            }
    
    async def test_entity_extraction(self):
        """测试实体提取"""
        logger.info("测试实体提取...")
        start_time = datetime.now()
        
        try:
            # 使用Pipeline加载CSV数据
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            raw_data = pipeline.load_data_files()
            
            # 初始化关系提取器（它也能提取实体）
            relation_extractor = get_relation_extractor()
            
            # 从投资事件中提取实体
            investment_events = raw_data.get('investment_events', [])
            # 将投资事件转换为文本列表
            texts = []
            for event in investment_events:
                if isinstance(event, dict):
                    # 尝试组合事件描述
                    description = event.get("事件资讯", "")
                    investor = event.get("投资方", "")
                    investee = event.get("融资方", "")
                    round_type = event.get("轮次", "")
                    amount = event.get("金额", "")
                    date = event.get("融资时间", "")
                    
                    # 组合文本
                    text_parts = []
                    if investor:
                        text_parts.append(f"投资方: {investor}")
                    if investee:
                        text_parts.append(f"融资方: {investee}")
                    if description:
                        text_parts.append(f"描述: {description}")
                    if round_type:
                        text_parts.append(f"轮次: {round_type}")
                    if amount:
                        text_parts.append(f"金额: {amount}")
                    if date:
                        text_parts.append(f"时间: {date}")
                    
                    text = " ".join(text_parts)
                    if text:
                        texts.append(text)
            
            # 提取实体
            entities = await relation_extractor.extract_entities_from_texts(texts)
            
            # 统计不同类型的实体
            company_entities = [e for e in entities if e.type == 'company']
            investor_entities = [e for e in entities if e.type == 'investor']
            event_entities = 0  # 没有单独的事件实体类型
            
            end_time = datetime.now()
            
            self.test_results["entity_extraction"] = {
                "success": True,
                "message": "实体提取成功",
                "total_entities": len(entities),
                "company_entities": len(company_entities),
                "investor_entities": len(investor_entities),
                "event_entities": event_entities,
                "duration": (end_time - start_time).total_seconds()
            }
            logger.info(f"实体提取测试通过，耗时: {self.test_results['entity_extraction']['duration']:.2f}秒")
            
        except Exception as e:
            end_time = datetime.now()
            self.test_results["entity_extraction"] = {
                "success": False,
                "message": f"实体提取失败: {str(e)}",
                "duration": (end_time - start_time).total_seconds()
            }
            logger.error(f"实体提取测试失败: {e}")
    
    async def test_relation_extraction(self):
        """测试关系提取"""
        logger.info("测试关系提取...")
        start_time = datetime.now()
        
        try:
            # 使用Pipeline加载CSV数据
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            raw_data = pipeline.load_data_files()
            
            # 初始化关系提取器
            relation_extractor = get_relation_extractor()
            
            # 从投资事件中提取关系
            investment_events = raw_data.get('investment_events', [])
            
            # 将投资事件转换为文本列表
            texts = []
            for event in investment_events:
                if isinstance(event, dict):
                    # 尝试组合事件描述
                    description = event.get("事件资讯", "")
                    investor = event.get("投资方", "")
                    investee = event.get("融资方", "")
                    round_type = event.get("轮次", "")
                    amount = event.get("金额", "")
                    date = event.get("融资时间", "")
                    
                    # 组合文本
                    text_parts = []
                    if investor:
                        text_parts.append(f"投资方: {investor}")
                    if investee:
                        text_parts.append(f"融资方: {investee}")
                    if description:
                        text_parts.append(f"描述: {description}")
                    if round_type:
                        text_parts.append(f"轮次: {round_type}")
                    if amount:
                        text_parts.append(f"金额: {amount}")
                    if date:
                        text_parts.append(f"时间: {date}")
                    
                    text = " ".join(text_parts)
                    if text:
                        texts.append(text)
            
            # 提取关系
            relations = await relation_extractor.extract_relations_from_texts(texts)
            
            # 统计不同类型的关系
            investment_relations = [r for r in relations if r.type == 'invests_in']
            cooperation_relations = [r for r in relations if r.type == 'cooperates_with']
            
            end_time = datetime.now()
            
            self.test_results["relation_extraction"] = {
                "success": True,
                "message": "关系提取成功",
                "total_relations": len(relations),
                "investment_relations": len(investment_relations),
                "cooperation_relations": len(cooperation_relations),
                "duration": (end_time - start_time).total_seconds()
            }
            logger.info(f"关系提取测试通过，耗时: {self.test_results['relation_extraction']['duration']:.2f}秒")
            
        except Exception as e:
            end_time = datetime.now()
            self.test_results["relation_extraction"] = {
                "success": False,
                "message": f"关系提取失败: {str(e)}",
                "duration": (end_time - start_time).total_seconds()
            }
            logger.error(f"关系提取测试失败: {e}")
    
    async def test_knowledge_graph_construction(self):
        """测试知识图谱构建"""
        logger.info("测试知识图谱构建...")
        start_time = datetime.now()
        
        try:
            # 使用Pipeline加载CSV数据
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            raw_data = pipeline.load_data_files()
            
            # 初始化关系提取器（用于提取实体和关系）
            relation_extractor = get_relation_extractor()
            
            # 从投资事件中提取实体
            investment_events = raw_data.get('investment_events', [])
            # 将投资事件转换为文本列表
            texts = []
            for event in investment_events:
                if isinstance(event, dict):
                    # 尝试组合事件描述
                    description = event.get("事件资讯", "")
                    investor = event.get("投资方", "")
                    investee = event.get("融资方", "")
                    round_type = event.get("轮次", "")
                    amount = event.get("金额", "")
                    date = event.get("融资时间", "")
                    
                    # 组合文本
                    text_parts = []
                    if investor:
                        text_parts.append(f"投资方: {investor}")
                    if investee:
                        text_parts.append(f"融资方: {investee}")
                    if description:
                        text_parts.append(f"描述: {description}")
                    if round_type:
                        text_parts.append(f"轮次: {round_type}")
                    if amount:
                        text_parts.append(f"金额: {amount}")
                    if date:
                        text_parts.append(f"时间: {date}")
                    
                    text = " ".join(text_parts)
                    if text:
                        texts.append(text)
            
            # 提取关系
            relations = await relation_extractor.extract_relations_from_texts(texts)
            
            # 初始化知识图谱构建器
            kg_builder = KGBuilder()
            
            # 添加实体到知识图谱
            companies = raw_data.get('companies', [])
            investors = raw_data.get('investors', [])
            
            # 将实体添加到知识图谱
            for company in companies:
                if isinstance(company, dict) and company.get('公司名称'):
                    kg_builder.add_company({
                        'name': company.get('公司名称'),
                        'industry': company.get('行业', ''),
                        'description': company.get('公司介绍', ''),
                        'id': company.get('工商注册id', ''),
                        'established_date': company.get('成立时间', ''),
                        'registered_capital': company.get('注册资金', ''),
                        'legal_representative': company.get('法人代表', ''),
                        'credit_code': company.get('统一信用代码', ''),
                        'website': company.get('网址', '')
                    })
            
            for investor in investors:
                if isinstance(investor, dict) and investor.get('机构名称'):
                    kg_builder.add_investor({
                        'name': investor.get('机构名称'),
                        'type': investor.get('类型', ''),
                        'description': investor.get('介绍', ''),
                        'industry': investor.get('行业', ''),
                        'scale': investor.get('规模', ''),
                        'rounds': investor.get('轮次', '')
                    })
            
            # 添加关系到知识图谱 - 将Relation对象转换为字典格式
            kg_builder.relationships = []
            for relation in relations:
                if hasattr(relation, 'id'):
                    # 将Relation对象转换为字典
                    relation_dict = {
                        'id': relation.id,
                        'source': relation.source,
                        'target': relation.target,
                        'type': relation.type,
                        'properties': relation.properties if hasattr(relation, 'properties') else {}
                    }
                    kg_builder.relationships.append(relation_dict)
                elif isinstance(relation, dict):
                    # 已经是字典格式
                    kg_builder.relationships.append(relation)
            
            end_time = datetime.now()
            
            self.test_results["knowledge_graph_construction"] = {
                "success": True,
                "message": "知识图谱构建成功",
                "companies_count": len(kg_builder.companies),
                "investors_count": len(kg_builder.investors),
                "relations_count": len(kg_builder.relationships),
                "duration": (end_time - start_time).total_seconds()
            }
            logger.info(f"知识图谱构建测试通过，耗时: {self.test_results['knowledge_graph_construction']['duration']:.2f}秒")
            
        except Exception as e:
            end_time = datetime.now()
            self.test_results["knowledge_graph_construction"] = {
                "success": False,
                "message": f"知识图谱构建失败: {str(e)}",
                "duration": (end_time - start_time).total_seconds()
            }
            logger.error(f"知识图谱构建测试失败: {e}")
    
    async def test_neo4j_integration(self):
        """测试Neo4j集成"""
        logger.info("测试Neo4j集成...")
        start_time = datetime.now()
        
        try:
            # 使用Pipeline加载CSV数据
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            raw_data = pipeline.load_data_files()
            
            # 初始化关系提取器（用于提取实体和关系）
            relation_extractor = get_relation_extractor()
            
            # 从投资事件中提取实体
            investment_events = raw_data.get('investment_events', [])
            # 将投资事件转换为文本列表
            texts = []
            for event in investment_events:
                if isinstance(event, dict):
                    # 尝试组合事件描述
                    description = event.get("事件资讯", "")
                    investor = event.get("投资方", "")
                    investee = event.get("融资方", "")
                    round_type = event.get("轮次", "")
                    amount = event.get("金额", "")
                    date = event.get("融资时间", "")
                    
                    # 组合文本
                    text_parts = []
                    if investor:
                        text_parts.append(f"投资方: {investor}")
                    if investee:
                        text_parts.append(f"融资方: {investee}")
                    if description:
                        text_parts.append(f"描述: {description}")
                    if round_type:
                        text_parts.append(f"轮次: {round_type}")
                    if amount:
                        text_parts.append(f"金额: {amount}")
                    if date:
                        text_parts.append(f"时间: {date}")
                    
                    text = " ".join(text_parts)
                    if text:
                        texts.append(text)
            
            # 提取实体和关系
            entities = await relation_extractor.extract_entities_from_texts(texts)
            relations = await relation_extractor.extract_relations_from_texts(texts)
            
            # 构建知识图谱
            kg_builder = KGBuilder()
            
            # 添加实体和关系到知识图谱
            # 注意：KGBuilder使用不同的数据结构，我们需要适配
            kg_builder.companies = {}  # 初始化公司实体字典
            kg_builder.investors = {}  # 初始化投资方实体字典
            
            # 将Relation对象转换为字典格式
            kg_builder.relationships = []
            for relation in relations:
                if hasattr(relation, 'id'):
                    # 将Relation对象转换为字典
                    relation_dict = {
                        'id': relation.id,
                        'source': relation.source,
                        'target': relation.target,
                        'type': relation.type,
                        'properties': relation.properties if hasattr(relation, 'properties') else {}
                    }
                    kg_builder.relationships.append(relation_dict)
                elif isinstance(relation, dict):
                    # 已经是字典格式
                    kg_builder.relationships.append(relation)
            
            # 将提取的实体添加到知识图谱
            for entity in entities:
                if entity.type == 'company':
                    kg_builder.companies[entity.id if hasattr(entity, 'id') else entity.name] = entity
                elif entity.type == 'investor':
                    kg_builder.investors[entity.id if hasattr(entity, 'id') else entity.name] = entity
            
            # 初始化Neo4j集成管理器
            integration_manager = IntegrationManager()
            
            # 获取集成状态
            status = integration_manager.get_integration_status()
            
            if status.get('neo4j_available', False):
                # 准备知识图谱数据
                kg_data = kg_builder.get_knowledge_graph()
                
                # 写入数据到Neo4j
                result = integration_manager.integrate_with_neo4j(kg_data)
                
                end_time = datetime.now()
                
                self.test_results["neo4j_integration"] = {
                    "success": True,
                    "message": "Neo4j集成成功",
                    "entities_count": len(entities),
                    "relations_count": len(relations),
                    "connection_success": True,
                    "write_success": result.get('success', False),
                    "duration": (end_time - start_time).total_seconds()
                }
                logger.info(f"Neo4j集成测试通过，耗时: {self.test_results['neo4j_integration']['duration']:.2f}秒")
            else:
                end_time = datetime.now()
                
                self.test_results["neo4j_integration"] = {
                    "success": False,
                    "message": "Neo4j不可用",
                    "entities_count": len(entities),
                    "relations_count": len(relations),
                    "connection_success": False,
                    "write_success": False,
                    "duration": (end_time - start_time).total_seconds()
                }
                logger.error("Neo4j不可用")
            
            # 关闭集成管理器
            integration_manager.close()
            
        except Exception as e:
            end_time = datetime.now()
            self.test_results["neo4j_integration"] = {
                    "success": False,
                    "message": f"Neo4j集成失败: {str(e)}",
                    "connection_success": False,
                    "write_success": False,
                    "duration": (end_time - start_time).total_seconds()
                }
            logger.error(f"Neo4j集成测试失败: {e}")
    
    async def test_complete_workflow(self):
        """测试完整工作流"""
        logger.info("测试完整工作流...")
        start_time = datetime.now()
        
        try:
            # 创建完整的Pipeline
            pipeline = Pipeline(data_dir="d:/Source/torch/financial-intellgience/src/dataset")
            
            # 执行完整工作流
            result = await pipeline.run_full_pipeline()
            success = result.get('success', False)
            
            end_time = datetime.now()
            
            self.test_results["overall"] = {
                "success": success,
                "message": "完整工作流执行成功" if success else "完整工作流执行失败",
                "duration": (end_time - start_time).total_seconds(),
                "total_duration": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
            }
            logger.info(f"完整工作流测试完成，耗时: {self.test_results['overall']['duration']:.2f}秒")
            
        except Exception as e:
            end_time = datetime.now()
            self.test_results["overall"] = {
                "success": False,
                "message": f"完整工作流测试失败: {str(e)}",
                "duration": (end_time - start_time).total_seconds(),
                "total_duration": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
            }
            logger.error(f"完整工作流测试失败: {e}")
    
    def print_test_results(self):
        """打印测试结果"""
        logger.info("="*50)
        logger.info("端到端测试结果")
        logger.info("="*50)
        
        for test_name, result in self.test_results.items():
            if result.get("success", False):
                logger.info(f"✅ {test_name}: 成功")
                if "duration" in result:
                    logger.info(f"   耗时: {result['duration']:.2f}秒")
                
                # 打印特定测试的额外信息
                if test_name == "data_loading" and result.get("success"):
                    data_status = result.get("data_status", {})
                    for data_type, status in data_status.items():
                        logger.info(f"   {data_type}: {status.get('count', 0)} 条")
                
                elif test_name == "data_processing" and result.get("success"):
                    logger.info(f"   处理后公司数据: {result.get('company_data_count', 0)} 条")
                    logger.info(f"   处理后事件数据: {result.get('event_data_count', 0)} 条")
                    logger.info(f"   处理后结构数据: {result.get('structure_data_count', 0)} 条")
                
                elif test_name == "entity_extraction" and result.get("success"):
                    logger.info(f"   公司实体: {result.get('company_entities', 0)} 个")
                    logger.info(f"   事件实体: {result.get('event_entities', 0)} 个")
                    logger.info(f"   投资者实体: {result.get('investor_entities', 0)} 个")
                    logger.info(f"   总实体数: {result.get('total_entities', 0)} 个")
                
                elif test_name == "relation_extraction" and result.get("success"):
                    logger.info(f"   关系数: {result.get('relations_count', 0)} 条")
                
                elif test_name == "knowledge_graph_construction" and result.get("success"):
                    logger.info(f"   实体数: {result.get('entities_count', 0)} 个")
                    logger.info(f"   关系数: {result.get('relations_count', 0)} 条")
                
                elif test_name == "neo4j_integration" and result.get("success"):
                    logger.info(f"   连接状态: {'成功' if result.get('connection_success', False) else '失败'}")
                    logger.info(f"   写入状态: {'成功' if result.get('write_success', False) else '失败'}")
                    logger.info(f"   实体数: {result.get('entities_count', 0)} 个")
                    logger.info(f"   关系数: {result.get('relations_count', 0)} 条")
                
                elif test_name == "overall" and result.get("success"):
                    logger.info(f"   总耗时: {result.get('total_duration', 0):.2f}秒")
            else:
                logger.error(f"❌ {test_name}: 失败")
                if "error" in result:
                    logger.error(f"   错误: {result['error']}")
                if "message" in result:
                    logger.error(f"   消息: {result['message']}")
        
        total_duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        logger.info(f"总耗时: {total_duration:.2f}秒")
        logger.info("="*50)
    
    def save_test_results(self):
        """保存测试结果"""
        results = {
            "test_time": self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
            "total_duration": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0,
            "results": self.test_results
        }
        
        # 创建输出目录
        output_dir = "output_e2e"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"e2e_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试结果已保存到: {output_path}")


async def main():
    """主函数"""
    runner = E2ETestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())