#!/usr/bin/env python3
"""
金融知识图谱构建主流程
基于硬编码优先策略，结合LLM增强
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置管理器 - 必须在其他模块之前加载
from processors.config_manager import load_configuration, get_config_manager

# 加载配置文件
config_loaded = load_configuration()
if not config_loaded:
    print("⚠️  配置加载失败，将使用默认配置")

from processors import DataParser, EntityMatcher, KGBuilder, DataValidator, OptimizedBatchProcessor
from processors.config import LOGGING_CONFIG

try:
    from integrations.neo4j_exporter import IntegrationManager, Config
    NEO4J_INTEGRATION_AVAILABLE = True
except ImportError:
    NEO4J_INTEGRATION_AVAILABLE = False

# 配置日志 - 使用新的日志模块
from utils.logger import setup_logger, get_logger

# 设置主日志器
setup_logger(
    name='financial_kg',
    level='INFO',
    log_file='D:/Source/torch/financial-intellgience/src/logs/pipeline.log',
    console_output=True,
    file_output=True
)
logger = get_logger('financial_kg')

# 设置日志级别为WARNING，减少INFO级别的日志输出
logger.setLevel(logging.WARNING)


class Pipeline:
    """构建流水线"""
    
    def __init__(self):
        # 硬编码数据目录和输出目录
        self.data_dir = Path("D:/Source/torch/financial-intellgience/src/dataset")
        self.output_dir = Path("D:/Source/torch/financial-intellgience/src/output")
            
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
        
        self.parser = DataParser()
        self.builder = KGBuilder()
        self.validator = DataValidator()
        self.optimizer = OptimizedBatchProcessor()
        
        # Neo4j集成 - 默认启用
        self.enable_neo4j = NEO4J_INTEGRATION_AVAILABLE
        self.neo4j_manager: Optional[IntegrationManager] = None
        if self.enable_neo4j:
            neo4j_config = {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': '1234567kk'
            }
            neo4j_config_obj = Config(**neo4j_config)
            self.neo4j_manager = IntegrationManager(neo4j_config_obj)
            logger.info("Neo4j集成已启用")
        
        # 确保输出目录存在
        self.output_dir.mkdir(exist_ok=True)
        
        # 统计数据
        self.pipeline_stats: Dict[str, Any] = {
            'start_time': None,
            'end_time': None,
            'total_processing_time': 0,
            'stage_stats': {}
        }
    
    def load_data_files(self) -> Dict[str, List[Dict]]:
        """加载数据文件 - 支持CSV和Markdown格式"""
        print("\n" + "="*50)
        print("开始加载数据文件")
        print("="*50)
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            print(f"❌ 错误: 数据目录不存在: {self.data_dir}")
            return {}
            
        print(f"✅ 数据目录存在: {self.data_dir}")
        
        # 列出数据目录中的所有文件
        print("数据目录中的文件:")
        for file in self.data_dir.iterdir():
            if file.is_file():
                print(f"  - {file.name}")
        print()
        
        # 支持多种文件格式，按优先级顺序尝试
        # 完全跳过公司数据加载
        data_file_mappings = {
            'investment_events': ['investment_events.csv'],
            'investment_structures': ['investment_structure.csv'],
            'investors': ['investors.json']
            # 注释掉公司数据，完全跳过加载
            # 'companies': ['company_data.md']
            # 投资方数据现在从单独的JSON文件加载
        }
        
        loaded_data: Dict[str, List[Dict]] = {}
        
        for data_type, filename_options in data_file_mappings.items():
            print(f"正在加载 {data_type} 数据...")
            data_loaded = False
            
            for filename in filename_options:
                file_path = self.data_dir / filename
                print(f"  尝试加载: {file_path}")
                
                if not file_path.exists():
                    print(f"  ❌ 文件不存在: {file_path}")
                    continue
                
                try:
                    # 根据文件扩展名选择合适的解析方法
                    if filename.endswith('.csv'):
                        print(f"  ✅ 检测到CSV文件: {filename}")
                        data = self._parse_csv_file(file_path)
                    elif filename.endswith('.md'):
                        print(f"  ✅ 检测到Markdown文件: {filename}")
                        data = self._parse_md_csv_file(file_path)
                    elif filename.endswith('.json'):
                        print(f"  ✅ 检测到JSON文件: {filename}")
                        data = self._parse_json_file(file_path)
                    else:
                        print(f"  ❌ 不支持的文件格式: {filename}")
                        continue
                    
                    loaded_data[data_type] = data
                    print(f"  ✅ 成功加载 {data_type}: {len(data)} 条记录")
                    data_loaded = True
                    break
                    
                except Exception as e:
                    print(f"  ❌ 加载数据文件失败 {file_path}: {e}")
                    continue
            
            if not data_loaded:
                print(f"  ❌ 无法加载 {data_type} 数据，所有候选文件都不存在或加载失败")
                loaded_data[data_type] = []
        
        print("\n数据加载总结:")
        for data_type, data in loaded_data.items():
            print(f"  {data_type}: {len(data)} 条记录")
        print("="*50 + "\n")
        
        return loaded_data
    
    def _parse_csv_file(self, file_path: Path) -> List[Dict]:
        """直接解析CSV文件 - 支持包含NUL字符的文件"""
        try:
            logger.info(f"开始解析CSV文件: {file_path}")
            
            # 首先尝试UTF-8编码
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"文件 {file_path.name} 使用UTF-8编码，大小: {len(content)} 字符")
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试其他编码
                logger.warning(f"UTF-8解码失败，尝试其他编码: {file_path.name}")
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                    logger.info(f"文件 {file_path.name} 使用GBK编码")
                except UnicodeDecodeError:
                    # 最后使用错误忽略模式
                    logger.warning(f"尝试忽略编码错误: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
            
            # 使用改进的CSV解析器
            result = self.parser.parse_csv_data(content)
            logger.info(f"CSV文件 {file_path.name} 解析完成: {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"CSV文件解析失败 {file_path}: {e}")
            logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
            return []
    
    def _parse_md_csv_file(self, file_path: Path) -> List[Dict]:
        """解析Markdown文件中的CSV数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找CSV数据部分（假设在```csv代码块中）
        lines = content.split('\n')
        csv_lines = []
        in_csv_block = False
        
        for line in lines:
            if line.strip() == '```csv':
                in_csv_block = True
                continue
            elif line.strip() == '```' and in_csv_block:
                in_csv_block = False
                continue
            elif in_csv_block:
                csv_lines.append(line)
        
        if not csv_lines:
            # 如果没有找到CSV代码块，尝试直接解析表格
            csv_lines = self._extract_table_from_md(content)
        
        if not csv_lines:
            return []
        
        # 解析CSV数据
        csv_content = '\n'.join(csv_lines)
        return self.parser.parse_csv_data(csv_content)
    
    def _extract_table_from_md(self, content: str) -> List[str]:
        """从Markdown中提取表格数据"""
        lines = content.split('\n')
        csv_lines = []
        
        # 查找表格开始（包含|分隔符的行）
        table_started = False
        for line in lines:
            if '|' in line and not table_started:
                table_started = True
                csv_lines.append(line)
            elif table_started and '|' in line:
                csv_lines.append(line)
            elif table_started and '|' not in line:
                break
        
        return csv_lines
    
    def _parse_json_file(self, file_path: Path) -> List[Dict]:
        """解析JSON文件"""
        try:
            logger.info(f"开始解析JSON文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 确保返回的是列表
            if isinstance(data, dict):
                data = [data]
            
            logger.info(f"JSON文件 {file_path.name} 解析完成: {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"JSON文件解析失败 {file_path}: {e}")
            logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
            return []
    
    def run_data_parsing_stage(self, raw_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """运行数据解析阶段"""
        logger.info("开始数据解析阶段")
        stage_start = datetime.now()
        
        parsed_data = {}
        
        # 解析公司数据
        if 'companies' in raw_data:
            parsed_data['companies'] = self.parser.parse_companies(raw_data['companies'])
            logger.info(f"公司数据解析完成: {len(parsed_data['companies'])} 条")
        
        # 解析投资事件数据
        if 'investment_events' in raw_data:
            parsed_data['investment_events'] = self.parser.parse_investment_events(raw_data['investment_events'])
            logger.info(f"投资事件数据解析完成: {len(parsed_data['investment_events'])} 条")
        
        # 解析投资方数据
        if 'investors' in raw_data:
            # 投资方数据已经是JSON格式，不需要再进行解析
            parsed_data['investors'] = raw_data['investors']
            logger.info(f"投资方数据加载完成: {len(parsed_data['investors'])} 条")
        
        # 解析投资结构数据
        if 'investment_structures' in raw_data:
            parsed_data['investment_structures'] = self.parser.parse_investment_structure(raw_data['investment_structures'])
            logger.info(f"投资结构数据解析完成: {len(parsed_data['investment_structures'])} 条")
        
        stage_time = (datetime.now() - stage_start).total_seconds()
        self.pipeline_stats['stage_stats']['data_parsing'] = {
            'duration': stage_time,
            'records_processed': sum(len(data) for data in parsed_data.values())
        }
        
        return parsed_data
    
    async def run_entity_building_stage(self, parsed_data: Dict[str, List[Dict]]) -> Dict:
        """运行实体构建阶段"""
        print("\n" + "="*50)
        print("开始实体构建阶段")
        print("="*50)
        stage_start = datetime.now()

        # 步骤1：数据验证
        print("\n步骤1：数据验证...")
        companies_raw = parsed_data.get('companies', [])
        investment_events_raw = parsed_data.get('investment_events', [])
        investors_raw = parsed_data.get('investors', [])
        investment_structures_raw = parsed_data.get('investment_structures', [])

        # 检查是否有公司数据，如果没有则跳过相关验证和构建
        has_companies = len(companies_raw) > 0
        
        if has_companies:
            company_validation = self.validator.validate_company_data(companies_raw)
            print(f"  公司数据验证：{company_validation['valid_records']}/{company_validation['total_records']} 有效")
        else:
            company_validation = {'valid_records': 0, 'total_records': 0, 'valid': True}
            print("  跳过公司数据验证（无公司数据）")

        event_validation = self.validator.validate_investment_event_data(investment_events_raw)
        print(f"  投资事件验证：{event_validation['valid_records']}/{event_validation['total_records']} 有效")

        investor_validation = self.validator.validate_investor_data(investors_raw)
        print(f"  投资方数据验证：{investor_validation['valid_records']}/{investor_validation['total_records']} 有效")
        
        structure_validation = self.validator.validate_investment_structure_data(investment_structures_raw)
        print(f"  投资结构数据验证：{structure_validation['valid_records']}/{structure_validation['total_records']} 有效")

        # 步骤2：构建实体
        print("\n步骤2：构建实体...")
        companies = self.builder.build_company_entities(companies_raw) if has_companies else {}
        print(f"  公司实体构建完成: {len(companies)} 个")
        
        investors = self.builder.build_investor_entities(investors_raw)
        print(f"  投资方实体构建完成: {len(investors)} 个")
        
        # 对投资事件数据进行解析和字段映射
        print("  解析投资事件数据...")
        investment_events = self.parser.parse_investment_events(investment_events_raw)
        
        # 构建投资关系
        print("  构建投资关系...")
        self.builder.build_investment_relationships(investment_events)
        relationships = self.builder.knowledge_graph['relationships']
        print(f"  投资关系构建完成: {len(relationships)} 个")
        
        # 构建投资结构关系
        if investment_structures_raw:
            print("  构建投资结构关系...")
            self.builder.build_investment_structure_relationships(investment_structures_raw)
            # 合并投资结构关系到现有关系
            structure_relationships = self.builder.knowledge_graph.get('structure_relationships', [])
            relationships.extend(structure_relationships)
            print(f"  投资结构关系构建完成: {len(structure_relationships)} 个")

        # 步骤3：LLM增强优化
        print("\n步骤3：LLM增强优化...")
        # 设置知识图谱到optimizer
        self.optimizer.set_knowledge_graph(self.builder.knowledge_graph)
        
        # 实体描述优化 - 完成后自动保存
        print("  进行实体描述优化...")
        enhanced_companies = await self.optimizer.optimize_entity_descriptions(companies, 'company') if has_companies else {}
        enhanced_investors = await self.optimizer.optimize_entity_descriptions(investors, 'investor')
        print(f"  实体描述优化完成: {len(enhanced_companies)} 公司, {len(enhanced_investors)} 投资方")
        
        # 自动保存实体描述优化结果
        await self._auto_save_llm_results({
            'stage': 'entity_description_enhancement',
            'enhanced_companies': enhanced_companies,
            'enhanced_investors': enhanced_investors,
            'companies': companies,
            'investors': investors,
            'relationships': relationships
        })
        
        # 行业分类优化 - 完成后自动保存（仅在存在公司数据时执行）
        industry_classifications = {}
        if has_companies and enhanced_companies:
            print("  进行行业分类优化...")
            industry_classifications = await self.optimizer.optimize_industry_classification(enhanced_companies)
            print(f"  行业分类优化完成: {len(industry_classifications)} 公司")
            
            # 自动保存行业分类结果
            await self._auto_save_llm_results({
                'stage': 'industry_classification',
                'industry_classifications': industry_classifications,
                'enhanced_companies': enhanced_companies,
                'enhanced_investors': enhanced_investors,
                'relationships': relationships
            })
        else:
            print("  跳过行业分类优化（无公司数据）")
        
        # 投资方名称标准化 - 完成后自动保存
        print("  进行投资方名称标准化...")
        investor_names = {i.get('name', '') for i in investors_raw if i.get('name')}
        standardized_names = await self.optimizer.optimize_investor_name_standardization(investor_names)
        print(f"  投资方名称标准化完成: {len(standardized_names)} 个名称")
        
        # 自动保存投资方名称标准化结果
        await self._auto_save_llm_results({
            'stage': 'investor_name_standardization',
            'standardized_names': standardized_names,
            'enhanced_companies': enhanced_companies,
            'enhanced_investors': enhanced_investors,
            'relationships': relationships
        })
        
        # 处理所有待处理的增强任务
        print("  处理所有待处理的增强任务...")
        enhancement_results = await self.optimizer.process_all_pending_enhancements()
        
        # 自动保存完整的增强结果
        await self._auto_save_llm_results({
            'stage': 'complete_enhancement',
            'enhancement_results': enhancement_results,
            'enhanced_companies': enhanced_companies,
            'enhanced_investors': enhanced_investors,
            'industry_classifications': industry_classifications,
            'standardized_names': standardized_names,
            'relationships': relationships
        })

        # 计算总的处理次数
        total_processed = (enhancement_results['entity_descriptions']['processed'] + 
                          enhancement_results['industry_classifications']['processed'] + 
                          enhancement_results['investor_standardizations']['processed'])
        print(f"  LLM调用次数：{total_processed} 次")

        # 步骤4：知识图谱验证
        print("\n步骤4：知识图谱验证...")
        kg_data = {
            'companies': enhanced_companies,
            'investors': enhanced_investors,
            'relationships': relationships
        }
        kg_validation = self.validator.validate_knowledge_graph(kg_data)
        print(f"  知识图谱验证完成")

        # 更新构建器内部状态
        self.builder.companies = enhanced_companies
        self.builder.investors = enhanced_investors
        self.builder.relationships = relationships

        stage_time = (datetime.now() - stage_start).total_seconds()
        self.pipeline_stats['stage_stats']['entity_building'] = {
            'duration': stage_time,
            'companies_created': len(enhanced_companies),
            'investors_created': len(enhanced_investors),
            'relationships_created': len(relationships)
        }

        print("\n" + "="*50)
        print("实体构建阶段完成")
        print("="*50)

        return {
            'companies': enhanced_companies,
            'investors': enhanced_investors,
            'relationships': relationships,
            'validations': {
                'companies': company_validation,
                'events': event_validation,
                'investors': investor_validation,
                'knowledge_graph': kg_validation
            },
            'enhancements': {
                'enhanced_companies': enhanced_companies,
                'enhanced_investors': enhanced_investors,
                'industry_classifications': industry_classifications,
                'standardized_names': standardized_names,
                'enhancement_results': enhancement_results
            }
        }
    
    def run_quality_check_stage(self, kg_data: Dict) -> Dict:
        """运行质量检查阶段"""
        logger.info("开始质量检查阶段")
        stage_start = datetime.now()
        
        quality_report = self._perform_quality_checks(kg_data)
        
        stage_time = (datetime.now() - stage_start).total_seconds()
        self.pipeline_stats['stage_stats']['quality_check'] = {
            'duration': stage_time,
            'checks_performed': len(quality_report)
        }
        
        return quality_report
    
    def _perform_quality_checks(self, kg_data: Dict) -> Dict:
        """执行质量检查"""
        quality_report: Dict[str, Any] = {
            'entity_coverage': {},
            'data_completeness': {},
            'relationship_validation': {},
            'recommendations': []
        }
        companies = kg_data.get('companies', {})
        investors = kg_data.get('investors', {})
        relationships = kg_data.get('relationships', [])
        
        # 实体覆盖率检查
        quality_report['entity_coverage'] = {
            'total_companies': len(companies),
            'total_investors': len(investors),
            'companies_with_description': sum(1 for c in companies.values() if c.get('description')),
            'investors_with_description': sum(1 for i in investors.values() if i.get('description'))
        }
        
        # 数据完整性检查
        quality_report['data_completeness'] = {
            'companies_with_contact_info': sum(1 for c in companies.values() 
                                             if c.get('contact_info', {}).get('address') or 
                                                c.get('contact_info', {}).get('website')),
            'investors_with_focus_info': sum(1 for i in investors.values() 
                                           if i.get('investment_focus', {}).get('industries'))
        }
        
        # 关系验证
        valid_relationships = 0
        for rel in relationships:
            if rel.get('investor_id') and rel.get('company_id'):
                valid_relationships += 1
        
        quality_report['relationship_validation'] = {
            'total_relationships': len(relationships),
            'valid_relationships': valid_relationships,
            'invalid_relationships': len(relationships) - valid_relationships
        }
        
        # 生成建议
        if quality_report['entity_coverage']['companies_with_description'] < len(companies) * 0.5:
            quality_report['recommendations'].append("建议增强公司描述信息")
        
        if quality_report['entity_coverage']['investors_with_description'] < len(investors) * 0.3:
            quality_report['recommendations'].append("建议增强投资方描述信息")
        
        return quality_report
    
    def save_intermediate_results(self, data: Dict, stage_name: str):
        """保存中间结果"""
        output_file = self.output_dir / f"{stage_name}_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"中间结果已保存: {output_file}")
    
    async def _auto_save_llm_results(self, llm_data: Dict):
        """自动保存LLM处理结果 - 增强版本"""
        try:
            stage = llm_data.get('stage', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建自动保存目录
            auto_save_dir = self.output_dir / "auto_save"
            auto_save_dir.mkdir(exist_ok=True)
            
            # 构建详细的保存数据
            save_data = {
                'stage': stage,
                'timestamp': timestamp,
                'data': llm_data,
                'pipeline_stats': self.pipeline_stats.copy(),
                'build_stats': self.builder.get_build_statistics(),
                'llm_enhancement_queue': await self._get_llm_queue_safe(),
                'entity_matcher_stats': self.entity_matcher.get_stats() if hasattr(self, 'entity_matcher') else {}
            }
            
            # 保存LLM处理结果
            save_file = auto_save_dir / f"llm_{stage}_{timestamp}.json"
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"LLM处理结果自动保存: {save_file}")
            
            # 同时保存增量知识图谱数据
            await self._save_incremental_knowledge_graph(llm_data, stage, timestamp)
            
            # 保存处理进度快照
            await self._save_progress_snapshot(stage, timestamp)
            
        except Exception as e:
            logger.error(f"自动保存LLM结果时出错: {e}")
    
    async def _get_llm_queue_safe(self) -> List:
        """安全获取LLM增强队列"""
        try:
            llm_queue = self.builder.get_llm_enhancement_batch()
            # 如果是协程对象，需要await
            if asyncio.iscoroutine(llm_queue):
                llm_queue = await llm_queue
            return llm_queue if llm_queue else []
        except Exception as e:
            logger.error(f"获取LLM队列时出错: {e}")
            return []
    
    async def _save_progress_snapshot(self, stage: str, timestamp: str):
        """保存处理进度快照"""
        try:
            snapshot_dir = self.output_dir / "progress_snapshots"
            snapshot_dir.mkdir(exist_ok=True)
            
            # 安全获取知识图谱数据，避免协程对象问题
            kg = self.builder.knowledge_graph
            # 如果knowledge_graph是协程对象，需要await
            if asyncio.iscoroutine(kg):
                logger.warning("检测到knowledge_graph是协程对象，正在等待其完成...")
                kg = await kg
            
            # 构建进度快照
            snapshot = {
                'stage': stage,
                'timestamp': timestamp,
                'pipeline_stats': self.pipeline_stats.copy(),
                'build_stats': self.builder.get_build_statistics(),
                'entity_count': len(kg.get('companies', {})) + len(kg.get('investors', {})),
                'relationship_count': len(kg.get('relationships', [])),
                'llm_queue_size': len(await self._get_llm_queue_safe())
            }
            
            # 保存快照
            snapshot_file = snapshot_dir / f"progress_{stage}_{timestamp}.json"
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            
            logger.info(f"进度快照已保存: {snapshot_file}")
            
        except Exception as e:
            logger.error(f"保存进度快照时出错: {e}")
    
    async def _save_incremental_knowledge_graph(self, llm_data: Dict, stage: str, timestamp: str):
        """保存增量知识图谱数据"""
        try:
            # 创建增量保存目录
            incremental_dir = self.output_dir / "incremental_kg"
            incremental_dir.mkdir(exist_ok=True)
            
            # 构建增量知识图谱数据
            incremental_kg = {
                'stage': stage,
                'timestamp': timestamp,
                'enhanced_companies': llm_data.get('enhanced_companies', {}),
                'enhanced_investors': llm_data.get('enhanced_investors', {}),
                'industry_classifications': llm_data.get('industry_classifications', {}),
                'standardized_names': llm_data.get('standardized_names', {}),
                'relationships': llm_data.get('relationships', []),
                'enhancement_results': llm_data.get('enhancement_results', {})
            }
            
            # 保存增量知识图谱
            kg_file = incremental_dir / f"kg_{stage}_{timestamp}.json"
            with open(kg_file, 'w', encoding='utf-8') as f:
                json.dump(incremental_kg, f, ensure_ascii=False, indent=2)
            
            logger.info(f"增量知识图谱已保存: {kg_file}")
            
        except Exception as e:
            logger.error(f"保存增量知识图谱时出错: {e}")
    
    async def save_final_results(self, kg_data: Dict, quality_report: Dict):
        """保存最终结果"""
        logger.info("步骤8：保存结果...")
        
        # 保存知识图谱
        kg_file = self.output_dir / "knowledge_graph.json"
        self.builder.save_knowledge_graph(str(kg_file))
        
        # 保存质量报告
        quality_file = self.output_dir / "quality_report.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        # 保存验证结果
        validation_file = self.output_dir / "validation_results.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(kg_data.get('validations', {}), f, ensure_ascii=False, indent=2)
        
        # 保存增强结果
        enhancement_file = self.output_dir / "enhancement_results.json"
        with open(enhancement_file, 'w', encoding='utf-8') as f:
            json.dump(kg_data.get('enhancements', {}), f, ensure_ascii=False, indent=2)
        
        # 保存LLM增强队列
        try:
            llm_queue = self.builder.get_llm_enhancement_batch()
            # 如果是协程对象，需要await
            if asyncio.iscoroutine(llm_queue):
                logger.warning("检测到llm_queue是协程对象，正在等待其完成...")
                llm_queue = await llm_queue
                
            if llm_queue:
                llm_file = self.output_dir / "llm_enhancement_queue.json"
                with open(llm_file, 'w', encoding='utf-8') as f:
                    json.dump(llm_queue, f, ensure_ascii=False, indent=2)
                logger.info(f"LLM增强队列已保存: {len(llm_queue)} 个项目")
        except Exception as e:
            logger.error(f"保存LLM增强队列时出错: {e}")
        
        # 保存统计信息
        stats = {
            'pipeline_stats': self.pipeline_stats,
            'build_stats': self.builder.get_build_statistics(),
            'completion_time': datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / "pipeline_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到：{self.output_dir}")
        logger.info(f"知识图谱文件：{kg_file}")
        logger.info(f"质量报告文件：{quality_file}")
        logger.info(f"验证结果文件：{validation_file}")
        logger.info(f"增强结果文件：{enhancement_file}")
        logger.info(f"统计文件：{stats_file}")

    def export_to_neo4j(self, kg_data: Dict) -> Dict:
        """导出知识图谱到Neo4j"""
        if not self.neo4j_manager:
            return {"error": "Neo4j管理器未初始化"}
        
        try:
            # 导出知识图谱数据
            export_result = self.neo4j_manager.integrate_with_neo4j(kg_data)
            
            # 获取统计信息
            stats = self.neo4j_manager.get_integration_status()
            
            return {
                "success": export_result.get("success", False),
                "export_result": export_result,
                "statistics": stats,
                "message": "知识图谱已成功导出到Neo4j" if export_result.get("success") else "知识图谱导出到Neo4j失败"
            }
        except Exception as e:
            logger.error(f"Neo4j导出失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "知识图谱导出到Neo4j失败"
            }
    
    async def run_full_pipeline(self, save_intermediate: bool = True) -> Dict[str, Any]:
        """运行完整的知识图谱构建流水线"""
        logger.info("开始运行完整的知识图谱构建流水线")
        self.pipeline_stats['start_time'] = datetime.now().isoformat()
        
        # 初始化neo4j_results
        neo4j_results: Optional[Dict[str, Any]] = None
        
        try:
            # 阶段1: 数据加载
            logger.info("=== 阶段1: 数据加载 ===")
            raw_data: Dict[str, List[Dict]] = self.load_data_files()
            
            if save_intermediate:
                self.save_intermediate_results(raw_data, "raw_data")
            
            # 阶段2: 数据解析
            logger.info("=== 阶段2: 数据解析 ===")
            parsed_data: Dict[str, List[Dict]] = self.run_data_parsing_stage(raw_data)
            
            if save_intermediate:
                self.save_intermediate_results(parsed_data, "parsed_data")
            
            # 阶段3: 实体构建
            logger.info("=== 阶段3: 实体构建 ===")
            kg_data: Dict[str, Any] = await self.run_entity_building_stage(parsed_data)
            
            if save_intermediate:
                self.save_intermediate_results(kg_data, "knowledge_graph")
            
            # 阶段4: 质量检查
            logger.info("=== 阶段4: 质量检查 ===")
            quality_report: Dict[str, Any] = self.run_quality_check_stage(kg_data)
            
            # 保存最终结果
            await self.save_final_results(kg_data, quality_report)
            
            # 更新流水线统计
            self.pipeline_stats['end_time'] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(self.pipeline_stats['start_time'])
            end_time = datetime.fromisoformat(self.pipeline_stats['end_time'])
            total_time: float = (end_time - start_time).total_seconds()
            self.pipeline_stats['total_processing_time'] = total_time
            
            logger.info(f"知识图谱构建完成！总耗时: {total_time:.2f}秒")
            
            # 步骤9: Neo4j知识图谱导出
            if self.enable_neo4j and self.neo4j_manager:
                logger.info("步骤9: Neo4j知识图谱导出")
                neo4j_results = self.export_to_neo4j(kg_data)
                logger.info(f"Neo4j导出完成: {neo4j_results}")
            
            # 获取LLM增强队列，确保不是协程对象
            try:
                llm_enhancement_batch = self.builder.get_llm_enhancement_batch()
                # 如果是协程对象，需要await
                if asyncio.iscoroutine(llm_enhancement_batch):
                    logger.warning("检测到llm_enhancement_batch是协程对象，正在等待其完成...")
                    llm_enhancement_batch = await llm_enhancement_batch
                llm_enhancement_count = len(llm_enhancement_batch) if llm_enhancement_batch else 0
            except Exception as e:
                logger.error(f"获取LLM增强队列时出错: {e}")
                llm_enhancement_count = 0
            
            return {
                'success': True,
                'knowledge_graph': kg_data,
                'quality_report': quality_report,
                'statistics': self.pipeline_stats,
                'llm_enhancement_required': llm_enhancement_count,
                'neo4j_results': neo4j_results
            }
            
        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}")
            self.pipeline_stats['end_time'] = datetime.now().isoformat()
            
            # 只有当start_time不为None时才计算总处理时间
            if self.pipeline_stats.get('start_time') is not None:
                start_time = datetime.fromisoformat(self.pipeline_stats['start_time'])
                end_time = datetime.fromisoformat(self.pipeline_stats['end_time'])
                total_time: float = (end_time - start_time).total_seconds()
                self.pipeline_stats['total_processing_time'] = total_time
            
            return {
                'success': False,
                'error': str(e),
                'statistics': self.pipeline_stats,
                'neo4j_results': neo4j_results
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='金融知识图谱构建工具')
    parser.add_argument("--verbose", action="store_true", 
                       help="详细日志输出")
    
    args = parser.parse_args()
    
    print("="*60)
    print("金融知识图谱构建工具")
    print("="*60)
    
    # 显示配置信息
    config_manager = get_config_manager()
    print("配置信息:")
    config_status = config_manager.get_configuration_status()
    print(f"  配置加载状态: {'✅ 已加载' if config_status['is_loaded'] else '❌ 未加载'}")
    print(f"  提供商数量: {config_status['providers_count']}")
    print(f"  有效配置: {'✅ 是' if config_status['has_valid_config'] else '❌ 否'}")
    if config_status['provider_models']:
        print(f"  可用模型: {', '.join(config_status['provider_models'])}")
    if config_status['load_errors']:
        print(f"  加载错误: {config_status['load_errors']}")
    print()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建流水线
    pipeline = Pipeline()
    
    # 运行完整的流水线（使用asyncio运行）
    async def run_pipeline():
        return await pipeline.run_full_pipeline(save_intermediate=True)
    
    result = asyncio.run(run_pipeline())
    
    # 输出结果摘要
    if result['success']:
        print("\n" + "="*60)
        print("知识图谱构建成功！")
        print("="*60)
        
        kg_data = result['knowledge_graph']
        stats = result['statistics']
        
        print(f"处理时间: {stats['total_processing_time']:.2f}秒")
        print(f"公司实体: {len(kg_data.get('companies', {}))} 个")
        print(f"投资方实体: {len(kg_data.get('investors', {}))} 个")
        print(f"投资关系: {len(kg_data.get('relationships', []))} 个")
        
        if result['llm_enhancement_required'] > 0:
            print(f"需要LLM增强: {result['llm_enhancement_required']} 个项目")
        
        print(f"输出目录: {pipeline.output_dir}")
        print("="*60)
        
    else:
        print(f"\n构建失败: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()