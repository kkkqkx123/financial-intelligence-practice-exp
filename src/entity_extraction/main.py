"""
实体抽取主程序 - 使用适配器模式集成现有的实体抽取器
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entity_extraction.financial_entity_adapter import FinancialEntityAdapter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityExtractionPipeline:
    """实体抽取管道类"""
    
    def __init__(self, output_dir: str = "src/extraction_results/entities"):
        """
        初始化管道
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.adapter = FinancialEntityAdapter()
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集配置
        self.dataset_configs = {
            'investment_structure': {
                'input_file': 'src/dataset/investment_structure.csv',
                'output_file': self.output_dir / 'investment_structure_entities.csv',
                'description': '投资结构数据'
            },
            'investment_events': {
                'input_file': 'src/dataset/investment_events.csv',
                'output_file': self.output_dir / 'investment_events_entities.csv',
                'description': '投资事件数据'
            },
            'company_data': {
                'input_file': 'src/dataset/company_data.csv',
                'output_file': self.output_dir / 'company_data_entities.csv',
                'description': '公司数据'
            }
        }
    
    def process_dataset(self, dataset_name: str, custom_input_file: Optional[str] = None) -> Dict:
        """
        处理单个数据集
        
        Args:
            dataset_name: 数据集名称
            custom_input_file: 自定义输入文件路径（可选）
            
        Returns:
            处理结果统计信息
        """
        logger.info(f"开始处理 {dataset_name} 数据集...")
        
        config = self.dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        input_file = custom_input_file or config['input_file']
        output_file = config['output_file']
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return {
                'dataset_name': dataset_name,
                'status': 'failed',
                'error': f'输入文件不存在: {input_file}',
                'entity_count': 0
            }
        
        try:
            # 使用适配器处理数据
            result_df = self.adapter.process_data(input_file, str(output_file))
            
            # 统计信息
            entity_count = len(result_df)
            entity_types = result_df['entity_type'].value_counts().to_dict() if not result_df.empty else {}
            
            logger.info(f"{dataset_name} 处理完成，抽取了 {entity_count} 个实体")
            
            return {
                'dataset_name': dataset_name,
                'status': 'success',
                'entity_count': entity_count,
                'entity_types': entity_types,
                'output_file': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"处理 {dataset_name} 时发生错误: {str(e)}")
            return {
                'dataset_name': dataset_name,
                'status': 'failed',
                'error': str(e),
                'entity_count': 0
            }
    
    def process_all_datasets(self) -> List[Dict]:
        """
        处理所有数据集
        
        Returns:
            所有数据集的处理结果列表
        """
        logger.info("开始处理所有数据集...")
        results = []
        
        for dataset_name in self.dataset_configs.keys():
            result = self.process_dataset(dataset_name)
            results.append(result)
        
        # 生成汇总报告
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: List[Dict]):
        """生成处理汇总报告"""
        total_entities = sum(result['entity_count'] for result in results)
        successful_count = sum(1 for result in results if result['status'] == 'success')
        
        report = f"""
实体抽取处理汇总报告
==================

处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
总数据集数量: {len(results)}
成功处理: {successful_count}
失败处理: {len(results) - successful_count}
总实体数量: {total_entities}

详细结果:
"""
        
        for result in results:
            report += f"\n{result['dataset_name']}:"
            report += f"\n  状态: {result['status']}"
            report += f"\n  实体数量: {result['entity_count']}"
            
            if result['status'] == 'success' and result['entity_types']:
                report += f"\n  实体类型分布:"
                for entity_type, count in result['entity_types'].items():
                    report += f"\n    {entity_type}: {count}"
            elif result['status'] == 'failed':
                report += f"\n  错误信息: {result.get('error', '未知错误')}"
            
            report += "\n"
        
        # 保存报告
        report_file = self.output_dir / 'extraction_summary_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"汇总报告已保存到: {report_file}")
        print(report)
    
    def validate_environment(self) -> bool:
        """验证运行环境"""
        logger.info("验证运行环境...")
        
        # 检查数据集文件
        missing_files = []
        for config in self.dataset_configs.values():
            input_file = config['input_file']
            if not os.path.exists(input_file):
                missing_files.append(input_file)
        
        if missing_files:
            logger.warning(f"以下数据集文件不存在: {missing_files}")
            return False
        
        logger.info("环境验证通过")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='金融数据实体抽取工具')
    parser.add_argument(
        '--dataset',
        choices=['investment_structure', 'investment_events', 'company_data', 'all'],
        default='all',
        help='要处理的数据集名称，默认为处理所有数据集'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='自定义输入文件路径（仅当指定单个数据集时有效）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/extraction_results/entities',
        help='输出目录路径，默认为 src/extraction_results/entities'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证环境，不执行处理'
    )
    
    args = parser.parse_args()
    
    # 创建管道实例
    pipeline = EntityExtractionPipeline(output_dir=args.output_dir)
    
    # 验证环境
    if not pipeline.validate_environment():
        if not args.validate_only:
            logger.error("环境验证失败，请检查数据集文件是否存在")
            return 1
    
    if args.validate_only:
        logger.info("环境验证完成")
        return 0
    
    # 处理数据
    try:
        if args.dataset == 'all':
            results = pipeline.process_all_datasets()
        else:
            result = pipeline.process_dataset(args.dataset, args.input_file)
            results = [result]
        
        # 输出结果统计
        print("\n处理结果统计:")
        print("=" * 50)
        for result in results:
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            print(f"{status_symbol} {result['dataset_name']}: {result['entity_count']} 个实体")
        
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())