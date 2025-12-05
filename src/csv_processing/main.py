"""
CSV数据实体抽取主程序
统一处理所有数据集的实体抽取任务
"""

import os
import sys
import argparse
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from investment_structure_extractor import InvestmentStructureExtractor
from investment_events_extractor import InvestmentEventsExtractor
from company_data_extractor import CompanyDataExtractor


class EntityExtractionPipeline:
    """实体抽取管道"""
    
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化抽取器
        self.extractors = {
            'investment_structure': InvestmentStructureExtractor(),
            'investment_events': InvestmentEventsExtractor(),
            'company_data': CompanyDataExtractor()
        }
        
        # 数据集文件映射
        self.dataset_files = {
            'investment_structure': 'investment_structure.csv',
            'investment_events': 'investment_events.csv',
            'company_data': 'company_data.csv'
        }
    
    def process_all_datasets(self):
        """处理所有数据集"""
        print("=== 开始批量实体抽取 ===")
        start_time = datetime.now()
        
        results = {}
        
        for dataset_name, extractor in self.extractors.items():
            print(f"\n--- 处理数据集: {dataset_name} ---")
            
            # 构建文件路径
            csv_filename = self.dataset_files.get(dataset_name)
            if not csv_filename:
                print(f"警告: 未找到数据集 {dataset_name} 对应的文件名")
                continue
            
            csv_file = os.path.join(self.dataset_dir, csv_filename)
            
            # 检查文件是否存在
            if not os.path.exists(csv_file):
                print(f"警告: 文件 {csv_file} 不存在，跳过处理")
                continue
            
            try:
                # 处理数据集
                result = extractor.process_dataset(csv_file, self.output_dir)
                results[dataset_name] = result
                
                print(f"数据集 {dataset_name} 处理完成")
                
            except Exception as e:
                print(f"处理数据集 {dataset_name} 时出错: {e}")
                results[dataset_name] = {'error': str(e)}
        
        # 生成汇总报告
        self._generate_summary_report(results, start_time)
        
        print(f"\n=== 所有数据集处理完成 ===")
        print(f"总耗时: {datetime.now() - start_time}")
        
        return results
    
    def process_single_dataset(self, dataset_name: str):
        """处理单个数据集"""
        print(f"=== 开始处理数据集: {dataset_name} ===")
        start_time = datetime.now()
        
        extractor = self.extractors.get(dataset_name)
        if not extractor:
            print(f"错误: 未找到数据集 {dataset_name} 对应的抽取器")
            return None
        
        csv_filename = self.dataset_files.get(dataset_name)
        if not csv_filename:
            print(f"错误: 未找到数据集 {dataset_name} 对应的文件名")
            return None
        
        csv_file = os.path.join(self.dataset_dir, csv_filename)
        
        if not os.path.exists(csv_file):
            print(f"错误: 文件 {csv_file} 不存在")
            return None
        
        try:
            result = extractor.process_dataset(csv_file, self.output_dir)
            print(f"数据集 {dataset_name} 处理完成，耗时: {datetime.now() - start_time}")
            return result
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            return {'error': str(e)}
    
    def _generate_summary_report(self, results: dict, start_time: datetime):
        """生成处理汇总报告"""
        report_file = os.path.join(self.output_dir, 'extraction_summary_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 实体抽取处理汇总报告\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {datetime.now() - start_time}\n\n")
            
            f.write("## 处理结果概览\n\n")
            
            total_records = 0
            total_entities = 0
            
            for dataset_name, result in results.items():
                f.write(f"### {dataset_name}\n")
                
                if 'error' in result:
                    f.write(f"- **状态**: 处理失败\n")
                    f.write(f"- **错误**: {result['error']}\n\n")
                    continue
                
                metadata = result.get('metadata', {})
                statistics = result.get('statistics', {})
                
                f.write(f"- **状态**: 处理成功\n")
                f.write(f"- **记录数**: {metadata.get('total_records', 0)}\n")
                f.write(f"- **实体数**: {metadata.get('total_entities', 0)}\n")
                f.write(f"- **唯一实体**: {statistics.get('unique_entity_count', 0)}\n")
                f.write(f"- **处理耗时**: {metadata.get('processing_duration', 'N/A')}\n")
                
                # 实体类型分布
                entity_counts = statistics.get('entity_type_counts', {})
                if entity_counts:
                    f.write("- **实体类型分布**:\n")
                    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  - {entity_type}: {count}\n")
                
                f.write("\n")
                
                total_records += metadata.get('total_records', 0)
                total_entities += metadata.get('total_entities', 0)
            
            f.write("## 总体统计\n\n")
            f.write(f"- **总记录数**: {total_records}\n")
            f.write(f"- **总实体数**: {total_entities}\n")
            f.write(f"- **输出目录**: {self.output_dir}\n")
            
            f.write("\n## 输出文件\n\n")
            for dataset_name in results.keys():
                csv_filename = f"{dataset_name}_entities.csv"
                f.write(f"- `{csv_filename}`\n")
        
        print(f"汇总报告已生成: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSV数据实体抽取工具')
    parser.add_argument('--dataset-dir', type=str, default='src/dataset',
                       help='数据集目录路径')
    parser.add_argument('--output-dir', type=str, default='src/extraction_results/entities',
                       help='输出目录路径')
    parser.add_argument('--dataset', type=str, choices=['investment_structure', 'investment_events', 'company_data', 'all'],
                       default='all', help='要处理的数据集名称')
    
    args = parser.parse_args()
    
    # 创建管道实例
    pipeline = EntityExtractionPipeline(args.dataset_dir, args.output_dir)
    
    if args.dataset == 'all':
        # 处理所有数据集
        pipeline.process_all_datasets()
    else:
        # 处理单个数据集
        pipeline.process_single_dataset(args.dataset)


if __name__ == "__main__":
    main()