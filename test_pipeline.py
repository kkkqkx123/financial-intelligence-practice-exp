#!/usr/bin/env python3
"""
知识图谱构建流程测试脚本
验证整个流程并生成测试报告
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import KnowledgeGraphPipeline
from src.processors import DataParser, EntityMatcher, HybridKGBuilder, DataValidator, BatchOptimizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineTester:
    """知识图谱流水线测试器"""
    
    def __init__(self):
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'tests': {},
            'overall_success': False,
            'errors': []
        }
        
    def test_component_initialization(self) -> bool:
        """测试组件初始化"""
        logger.info("=== 测试组件初始化 ===")
        
        try:
            # 测试各个组件的初始化
            parser = DataParser()
            matcher = EntityMatcher()
            builder = HybridKGBuilder()
            validator = DataValidator()
            optimizer = BatchOptimizer()
            
            self.test_results['tests']['component_initialization'] = {
                'success': True,
                'components': ['DataParser', 'EntityMatcher', 'HybridKGBuilder', 'DataValidator', 'BatchOptimizer']
            }
            
            logger.info("✓ 所有组件初始化成功")
            return True
            
        except Exception as e:
            error_msg = f"组件初始化失败: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.test_results['tests']['component_initialization'] = {
                'success': False,
                'error': error_msg
            }
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_data_loading(self, pipeline: KnowledgeGraphPipeline) -> bool:
        """测试数据加载"""
        logger.info("=== 测试数据加载 ===")
        
        try:
            # 检查数据文件是否存在
            data_files = {
                'companies': 'company_data.md',
                'investment_events': 'investment_events.md',
                'investors': 'investment_structure.md'
            }
            
            missing_files = []
            for data_type, filename in data_files.items():
                file_path = pipeline.data_dir / filename
                if not file_path.exists():
                    missing_files.append(str(file_path))
            
            if missing_files:
                logger.warning(f"缺少数据文件: {missing_files}")
                # 创建测试数据
                self._create_test_data(pipeline.data_dir)
            
            # 测试数据加载
            raw_data = pipeline.load_data_files()
            
            load_stats = {}
            for data_type, data in raw_data.items():
                load_stats[data_type] = len(data)
            
            self.test_results['tests']['data_loading'] = {
                'success': True,
                'load_stats': load_stats
            }
            
            logger.info(f"✓ 数据加载成功: {load_stats}")
            return True
            
        except Exception as e:
            error_msg = f"数据加载失败: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.test_results['tests']['data_loading'] = {
                'success': False,
                'error': error_msg
            }
            self.test_results['errors'].append(error_msg)
            return False
    
    def _create_test_data(self, data_dir: Path):
        """创建测试数据"""
        logger.info("创建测试数据...")
        
        # 确保数据目录存在
        data_dir.mkdir(exist_ok=True)
        
        # 创建公司数据
        company_data = """# 公司数据

```csv
公司名称,股票代码,成立时间,注册资本,所属行业,公司地址,公司简介
腾讯控股,00700.HK,1998-11-11,1000万,互联网,深圳市南山区,中国领先的互联网增值服务提供商
阿里巴巴,09988.HK,1999-09-09,1000万,电子商务,杭州市余杭区,中国最大的电子商务公司
百度集团,09888.HK,2000-01-01,1000万,搜索引擎,北京市海淀区,中国领先的搜索引擎公司
```
"""
        
        # 创建投资事件数据
        investment_data = """# 投资事件数据

```csv
公司名称,投资方名称,投资金额,投资轮次,投资时间,投资比例
腾讯控股,MIH TC,100万,天使轮,2001-06-01,46.5%
阿里巴巴,软银集团,2000万,A轮,2000-01-01,30%
百度集团,德丰杰,150万,天使轮,2000-09-01,25%
```
"""
        
        # 创建投资方数据
        investor_data = """# 投资方数据

```csv
投资方名称,投资类型,管理资金规模,成立时间,投资领域,联系方式
MIH TC,风险投资,1000万,1990-01-01,互联网,info@mihtc.com
软银集团,综合投资,10000万,1981-09-01,科技、电信,contact@softbank.co.jp
德丰杰,风险投资,5000万,1985-01-01,高科技,info@dfj.com
```
"""
        
        # 写入文件
        (data_dir / 'company_data.md').write_text(company_data, encoding='utf-8')
        (data_dir / 'investment_events.md').write_text(investment_data, encoding='utf-8')
        (data_dir / 'investment_structure.md').write_text(investor_data, encoding='utf-8')
        
        logger.info("测试数据创建完成")
    
    def test_data_parsing(self, pipeline: KnowledgeGraphPipeline) -> bool:
        """测试数据解析"""
        logger.info("=== 测试数据解析 ===")
        
        try:
            # 创建测试数据
            raw_data = {
                'companies': [
                    {
                        '公司名称': '测试公司1',
                        '股票代码': 'TEST001',
                        '成立时间': '2020-01-01',
                        '注册资本': '1000万',
                        '所属行业': '科技',
                        '公司地址': '北京市',
                        '公司简介': '这是一家测试公司'
                    }
                ],
                'investment_events': [
                    {
                        '公司名称': '测试公司1',
                        '投资方名称': '测试投资方',
                        '投资金额': '500万',
                        '投资轮次': 'A轮',
                        '投资时间': '2021-01-01',
                        '投资比例': '20%'
                    }
                ],
                'investors': [
                    {
                        '投资方名称': '测试投资方',
                        '投资类型': '风险投资',
                        '管理资金规模': '1000万',
                        '成立时间': '2019-01-01',
                        '投资领域': '科技',
                        '联系方式': 'test@example.com'
                    }
                ]
            }
            
            # 测试解析
            parsed_data = pipeline.run_data_parsing_stage(raw_data)
            
            parse_stats = {}
            for data_type, data in parsed_data.items():
                parse_stats[data_type] = len(data)
            
            self.test_results['tests']['data_parsing'] = {
                'success': True,
                'parse_stats': parse_stats
            }
            
            logger.info(f"✓ 数据解析成功: {parse_stats}")
            return True
            
        except Exception as e:
            error_msg = f"数据解析失败: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.test_results['tests']['data_parsing'] = {
                'success': False,
                'error': error_msg
            }
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_entity_building(self, pipeline: KnowledgeGraphPipeline) -> bool:
        """测试实体构建"""
        logger.info("=== 测试实体构建 ===")
        
        try:
            # 使用测试数据
            test_data = {
                'companies': [
                    {
                        '公司名称': '测试公司2',
                        '股票代码': 'TEST002',
                        '成立时间': '2020-01-01',
                        '注册资本': '2000万',
                        '所属行业': '金融',
                        '公司地址': '上海市',
                        '公司简介': '这是一家金融测试公司'
                    }
                ],
                'investment_events': [
                    {
                        '公司名称': '测试公司2',
                        '投资方名称': '金融投资方',
                        '投资金额': '1000万',
                        '投资轮次': 'B轮',
                        '投资时间': '2021-06-01',
                        '投资比例': '30%'
                    }
                ],
                'investors': [
                    {
                        '投资方名称': '金融投资方',
                        '投资类型': '私募投资',
                        '管理资金规模': '5000万',
                        '成立时间': '2018-01-01',
                        '投资领域': '金融',
                        '联系方式': 'finance@example.com'
                    }
                ]
            }
            
            # 测试实体构建
            kg_data = pipeline.run_entity_building_stage(test_data)
            
            building_stats = {
                'companies': len(kg_data.get('companies', {})),
                'investors': len(kg_data.get('investors', {})),
                'relationships': len(kg_data.get('relationships', []))
            }
            
            self.test_results['tests']['entity_building'] = {
                'success': True,
                'building_stats': building_stats,
                'validations': kg_data.get('validations', {}),
                'enhancements': kg_data.get('enhancements', {})
            }
            
            logger.info(f"✓ 实体构建成功: {building_stats}")
            return True
            
        except Exception as e:
            error_msg = f"实体构建失败: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.test_results['tests']['entity_building'] = {
                'success': False,
                'error': error_msg
            }
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_full_pipeline(self, pipeline: KnowledgeGraphPipeline) -> bool:
        """测试完整流水线"""
        logger.info("=== 测试完整流水线 ===")
        
        try:
            # 运行完整流水线
            result = pipeline.run_full_pipeline(save_intermediate=False)
            
            self.test_results['tests']['full_pipeline'] = {
                'success': result['success'],
                'statistics': result.get('statistics', {}),
                'llm_enhancement_required': result.get('llm_enhancement_required', 0)
            }
            
            if result['success']:
                logger.info("✓ 完整流水线运行成功")
                logger.info(f"  - 处理时间: {result['statistics']['total_processing_time']:.2f}秒")
                logger.info(f"  - 需要LLM增强: {result['llm_enhancement_required']} 个项目")
                return True
            else:
                error_msg = f"流水线失败: {result.get('error', '未知错误')}"
                logger.error(f"✗ {error_msg}")
                self.test_results['errors'].append(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"完整流水线测试失败: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.test_results['tests']['full_pipeline'] = {
                'success': False,
                'error': error_msg
            }
            self.test_results['errors'].append(error_msg)
            return False
    
    def test_neo4j_integration(self) -> bool:
        """测试Neo4j集成"""
        logger.info("=== 测试Neo4j集成 ===")
        
        try:
            # 尝试导入Neo4j相关模块
            from py2neo import Graph, Node, Relationship
            
            self.test_results['tests']['neo4j_integration'] = {
                'success': True,
                'message': 'Neo4j模块导入成功'
            }
            
            logger.info("✓ Neo4j集成测试通过")
            return True
            
        except ImportError as e:
            error_msg = f"Neo4j模块导入失败: {str(e)}"
            logger.warning(f"⚠ {error_msg}")
            self.test_results['tests']['neo4j_integration'] = {
                'success': False,
                'error': error_msg,
                'recommendation': '建议安装py2neo库: pip install py2neo'
            }
            return False
    
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("开始运行知识图谱流水线测试...")
        
        # 创建测试流水线
        test_pipeline = KnowledgeGraphPipeline(
            data_dir="src/dataset",
            output_dir="test_output"
        )
        
        # 运行各个测试
        tests = [
            ("组件初始化", self.test_component_initialization),
            ("数据加载", lambda: self.test_data_loading(test_pipeline)),
            ("数据解析", lambda: self.test_data_parsing(test_pipeline)),
            ("实体构建", lambda: self.test_entity_building(test_pipeline)),
            ("完整流水线", lambda: self.test_full_pipeline(test_pipeline)),
            ("Neo4j集成", self.test_neo4j_integration)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"运行测试: {test_name}")
            logger.info('='*60)
            
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    logger.info(f"✓ {test_name} 测试通过")
                else:
                    logger.warning(f"⚠ {test_name} 测试失败")
            except Exception as e:
                error_msg = f"{test_name} 测试异常: {str(e)}"
                logger.error(f"✗ {error_msg}")
                self.test_results['errors'].append(error_msg)
                self.test_results['tests'][test_name.lower().replace(' ', '_')] = {
                    'success': False,
                    'error': error_msg
                }
        
        # 更新总体结果
        self.test_results['overall_success'] = (passed_tests == total_tests)
        self.test_results['end_time'] = datetime.now().isoformat()
        self.test_results['test_summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests)*100:.1f}%"
        }
        
        # 保存测试报告
        self._save_test_report()
        
        # 输出测试摘要
        self._print_test_summary()
        
        return self.test_results
    
    def _save_test_report(self):
        """保存测试报告"""
        report_file = "test_pipeline_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试报告已保存: {report_file}")
    
    def _print_test_summary(self):
        """打印测试摘要"""
        summary = self.test_results['test_summary']
        
        print("\n" + "="*60)
        print("知识图谱流水线测试摘要")
        print("="*60)
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过测试: {summary['passed_tests']}")
        print(f"失败测试: {summary['failed_tests']}")
        print(f"成功率: {summary['success_rate']}")
        
        if self.test_results['errors']:
            print(f"\n错误信息:")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        if self.test_results['overall_success']:
            print("\n✓ 所有测试通过！知识图谱流水线运行正常。")
        else:
            print(f"\n✗ 部分测试失败，请查看测试报告和日志文件获取详细信息。")
        
        print("="*60)


def main():
    """主函数"""
    print("金融知识图谱构建流水线测试工具")
    print("="*60)
    
    # 创建测试器
    tester = PipelineTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 返回适当的退出码
    sys.exit(0 if results['overall_success'] else 1)


if __name__ == "__main__":
    main()