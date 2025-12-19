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
    log_file='output/pipeline.log',
    console_output=True,
    file_output=True
)
logger = get_logger('financial_kg')

# 设置日志级别为WARNING，减少INFO级别的日志输出
logger.setLevel(logging.WARNING)


class Pipeline:
    """构建流水线"""
    
    def __init__(self, data_dir: str = "dataset", output_dir: str = "output", 
                 enable_neo4j: bool = False, neo4j_config: Optional[Dict[str, Any]] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.parser = DataParser()
        self.builder = KGBuilder()
        self.validator = DataValidator()
        self.optimizer = OptimizedBatchProcessor()
        
        # Neo4j集成
        self.enable_neo4j = enable_neo4j and NEO4J_INTEGRATION_AVAILABLE
        self.neo4j_manager: Optional[IntegrationManager] = None
        if self.enable_neo4j:
            neo4j_config_obj = Config(**(neo4j_config or {}))
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
        logger.info("开始加载数据文件")
        
        # 支持多种文件格式，按优先级顺序尝试
        data_file_mappings = {
            'companies': ['company_data.csv', 'company_data.md'],
            'investment_events': ['investment_events.csv', 'investment_events.md'],
            'investors': ['investment_structure.csv', 'investment_structure.md'],
            'investment_structures': ['investment_structure.csv', 'investment_structure.md']
        }
        
        loaded_data: Dict[str, List[Dict]] = {}
        
        for data_type, filename_options in data_file_mappings.items():
            data_loaded = False
            
            for filename in filename_options:
                file_path = self.data_dir / filename
                
                if not file_path.exists():
                    continue
                
                try:
                    # 根据文件扩展名选择合适的解析方法
                    if filename.endswith('.csv'):
                        logger.info(f"检测到CSV文件: {filename}")
                        data = self._parse_csv_file(file_path)
                    elif filename.endswith('.md'):
                        logger.info(f"检测到Markdown文件: {filename}")
                        data = self._parse_md_csv_file(file_path)
                    else:
                        logger.warning(f"不支持的文件格式: {filename}")
                        continue
                    
                    loaded_data[data_type] = data
                    logger.info(f"加载 {data_type}: {len(data)} 条记录")
                    data_loaded = True
                    break
                    
                except Exception as e:
                    logger.error(f"加载数据文件失败 {file_path}: {e}")
                    continue
            
            if not data_loaded:
                logger.warning(f"无法加载 {data_type} 数据，所有候选文件都不存在或加载失败")
                loaded_data[data_type] = []
        
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
            parsed_data['investors'] = self.parser.parse_investment_institutions(raw_data['investors'])
            logger.info(f"投资方数据解析完成: {len(parsed_data['investors'])} 条")
        
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
        logger.info("开始实体构建阶段")
        stage_start = datetime.now()

        # 步骤1：数据验证
        logger.info("步骤1：数据验证...")
        companies_raw = parsed_data.get('companies', [])
        investment_events_raw = parsed_data.get('investment_events', [])
        investors_raw = parsed_data.get('investors', [])
        investment_structures_raw = parsed_data.get('investment_structures', [])

        company_validation = self.validator.validate_company_data(companies_raw)
        logger.info(f"公司数据验证：{company_validation['valid_records']}/{company_validation['total_records']} 有效")

        event_validation = self.validator.validate_investment_event_data(investment_events_raw)
        logger.info(f"投资事件验证：{event_validation['valid_records']}/{event_validation['total_records']} 有效")

        investor_validation = self.validator.validate_investor_data(investors_raw)
        logger.info(f"投资方数据验证：{investor_validation['valid_records']}/{investor_validation['total_records']} 有效")
        
        structure_validation = self.validator.validate_investment_structure_data(investment_structures_raw)
        logger.info(f"投资结构数据验证：{structure_validation['valid_records']}/{structure_validation['total_records']} 有效")

        # 步骤2：构建实体
        logger.info("步骤2：构建实体...")
        companies = self.builder.build_company_entities(companies_raw)
        investors = self.builder.build_investor_entities(investors_raw)
        
        # 对投资事件数据进行解析和字段映射
        logger.info("解析投资事件数据...")
        investment_events = self.parser.parse_investment_events(investment_events_raw)
        
        # 构建投资关系
        self.builder.build_investment_relationships(investment_events)
        relationships = self.builder.knowledge_graph['relationships']
        
        # 构建投资结构关系
        if investment_structures_raw:
            self.builder.build_investment_structure_relationships(investment_structures_raw)
            # 合并投资结构关系到现有关系
            structure_relationships = self.builder.knowledge_graph.get('structure_relationships', [])
            relationships.extend(structure_relationships)

        # 步骤3：LLM增强优化
        logger.info("步骤3：LLM增强优化...")
        # 设置知识图谱到optimizer
        self.optimizer.set_knowledge_graph(self.builder.knowledge_graph)
        enhanced_companies = await self.optimizer.optimize_entity_descriptions(companies, 'company')
        enhanced_investors = await self.optimizer.optimize_entity_descriptions(investors, 'investor')
        industry_classifications = await self.optimizer.optimize_industry_classification(enhanced_companies)
        investor_names = {i.get('name', '') for i in investors_raw if i.get('name')}
        standardized_names = await self.optimizer.optimize_investor_name_standardization(investor_names)
        enhancement_results = await self.optimizer.process_all_pending_enhancements()

        logger.info(f"LLM增强优化完成：")
        logger.info(f"  - 增强实体描述：{len(enhanced_companies)} 公司, {len(enhanced_investors)} 投资方")
        logger.info(f"  - 行业分类优化：{len(industry_classifications)} 公司")
        logger.info(f"  - 投资方名称标准化：{len(standardized_names)} 个名称")
        logger.info(f"  - LLM调用次数：{enhancement_results['processed']} 次")

        # 步骤4：知识图谱验证
        logger.info("步骤4：知识图谱验证...")
        kg_data = {
            'companies': enhanced_companies,
            'investors': enhanced_investors,
            'relationships': relationships
        }
        kg_validation = self.validator.validate_knowledge_graph(kg_data)

        logger.info(f"知识图谱验证完成：")
        logger.info(f"  - 实体一致性：{kg_validation['entity_consistency']['valid']}/{kg_validation['entity_consistency']['total']} 有效")
        logger.info(f"  - 关系完整性：{kg_validation['relationship_completeness']['valid']}/{kg_validation['relationship_completeness']['total']} 有效")
        logger.info(f"  - 总体得分：{kg_validation['overall_score']:.2f}")

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
    parser.add_argument('--data-dir', type=str, default='src/dataset', help='数据文件目录')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--enable-neo4j', action='store_true', help='启用Neo4j集成')
    parser.add_argument('--neo4j-uri', type=str, help='Neo4j URI')
    parser.add_argument('--neo4j-user', type=str, help='Neo4j用户名')
    parser.add_argument('--neo4j-password', type=str, help='Neo4j密码')
    parser.add_argument('--env-file', type=str, help='自定义.env文件路径')
    parser.add_argument('--show-config', action='store_true', help='显示配置信息')
    parser.add_argument("--no-intermediate", action="store_true", 
                       help="不保存中间结果")
    parser.add_argument("--verbose", action="store_true", 
                       help="详细日志输出")
    
    args = parser.parse_args()
    
    print("="*60)
    print("金融知识图谱构建工具")
    print("="*60)
    
    # 加载配置（如果指定了自定义env文件）
    if args.env_file:
        config_loaded = load_configuration(args.env_file)
        if not config_loaded:
            print(f"⚠️  加载自定义配置文件失败: {args.env_file}")
            sys.exit(1)
    
    # 显示配置信息
    config_manager = get_config_manager()
    if args.show_config:
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
    
    # 构建Neo4j配置
    neo4j_config = None
    if args.enable_neo4j:
        neo4j_config = {
            'uri': args.neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'username': args.neo4j_user or os.getenv('NEO4J_USERNAME', 'neo4j'),
            'password': args.neo4j_password or os.getenv('NEO4J_PASSWORD', 'password')
        }
    
    # 创建流水线
    pipeline = Pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        enable_neo4j=args.enable_neo4j,
        neo4j_config=neo4j_config
    )
    
    # 运行完整的流水线（使用asyncio运行）
    async def run_pipeline():
        return await pipeline.run_full_pipeline(save_intermediate=not args.no_intermediate)
    
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
        
        print(f"输出目录: {args.output_dir}")
        print("="*60)
        
    else:
        print(f"\n构建失败: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()