"""
实体抽取适配器 - 将FinancialEntityExtractor适配到新的CSV处理框架
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Any
from abc import ABC, abstractmethod

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csv_processing.base_extractor import BaseEntityExtractor
from entity_extraction.financial_entity_extractor import FinancialEntityExtractor


class FinancialEntityAdapter(BaseEntityExtractor):
    """金融实体抽取适配器 - 适配现有的FinancialEntityExtractor到新的CSV处理框架"""
    
    def __init__(self):
        super().__init__()
        self.financial_extractor = FinancialEntityExtractor()
        self._entity_types = [
            '投资机构', '被投企业', '投资轮次', '投资规模', '投资时间', '地区', '行业'
        ]
    
    def extract_entities(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        使用现有的FinancialEntityExtractor抽取实体
        
        Args:
            data_item: 单行数据字典
            
        Returns:
            实体字典，键为实体类型，值为实体列表
        """
        # 根据数据内容判断使用哪个抽取方法
        if '机构名称' in data_item:
            # 投资结构数据
            return self.financial_extractor.extract_from_investment_structure(data_item)
        elif '投资方' in data_item and '融资方' in data_item:
            # 投资事件数据
            return self.financial_extractor.extract_from_investment_events(data_item)
        elif '公司名称' in data_item:
            # 公司数据
            return self.financial_extractor.extract_from_company_data(data_item)
        else:
            # 无法识别的数据类型，返回空结果
            return {}
    
    def process_dataset(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """
        处理数据文件并抽取实体，返回完整的结果字典
        
        Args:
            input_file: 输入CSV文件路径
            output_dir: 输出目录路径
            
        Returns:
            完整的结果字典，包含metadata、entities和statistics
        """
        from datetime import datetime
        import pickle
        
        print(f"=== 开始处理数据集: {os.path.basename(input_file)} ===")
        start_time = datetime.now()
        
        try:
            # 读取数据
            df = pd.read_csv(input_file, encoding='utf-8')
            print(f"CSV文件包含 {len(df)} 条记录，列名: {list(df.columns)}")
            
            entity_dict = {}  # 实体名称 -> 实体类别列表
            
            # 处理每一行数据
            for index, row in df.iterrows():
                data_item = row.to_dict()
                entities = self.extract_entities(data_item)
                
                # 将实体转换为名称-类别映射
                for entity_type, entity_list in entities.items():
                    for entity_name in entity_list:
                        if entity_name and entity_name != 'nan':
                            entity_name = entity_name.strip()
                            if entity_name not in entity_dict:
                                entity_dict[entity_name] = []
                            if entity_type not in entity_dict[entity_name]:
                                entity_dict[entity_name].append(entity_type)
            
            # 计算统计信息
            statistics = self._calculate_statistics(entity_dict)
            
            # 构建完整结果
            results = {
                'metadata': {
                    'processing_time': datetime.now().isoformat(),
                    'total_records': len(df),
                    'total_entities': len(entity_dict),
                    'processing_duration': str(datetime.now() - start_time),
                    'data_source': input_file
                },
                'entities': entity_dict,
                'statistics': statistics
            }
            
            # 保存为pickle格式 - 与基类保持一致，只保存实体字典
            if output_dir and entity_dict:
                self._save_entities_pickle(entity_dict, os.path.join(output_dir, self.get_output_filename()))
                print(f"实体抽取完成，共抽取 {len(entity_dict)} 个唯一实体")
            elif entity_dict:
                print(f"实体抽取完成，共抽取 {len(entity_dict)} 个唯一实体")
            else:
                print("未抽取到任何实体")
            
            # 打印摘要
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            print(f"处理文件 {input_file} 时发生错误: {str(e)}")
            return {}
    
    def _calculate_statistics(self, entity_dict: Dict[str, List[str]]) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {
            'total_entities': len(entity_dict),
            'entity_type_counts': {},
            'unique_values': set(entity_dict.keys())
        }
        
        for entity_name, entity_types in entity_dict.items():
            for entity_type in entity_types:
                if entity_type not in stats['entity_type_counts']:
                    stats['entity_type_counts'][entity_type] = 0
                stats['entity_type_counts'][entity_type] += 1
        
        stats['unique_entity_count'] = len(stats['unique_values'])
        del stats['unique_values']  # 集合无法JSON序列化
        
        return stats
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印结果摘要"""
        print(f"\n=== 处理结果摘要 ===")
        print(f"处理记录数: {results['metadata']['total_records']}")
        print(f"抽取实体数: {results['metadata']['total_entities']}")
        print(f"唯一实体值: {results['statistics']['unique_entity_count']}")
        
        print(f"\n实体类型分布:")
        for entity_type, count in results['statistics']['entity_type_counts'].items():
            print(f"  {entity_type}: {count}")
    
    def get_entity_types(self) -> List[str]:
        """获取支持的实体类型列表"""
        return self._entity_types
    
    def get_output_filename(self) -> str:
        """获取输出文件名"""
        return "financial_entities.pkl"
    
    def validate_data_format(self, data_item: Dict[str, Any]) -> bool:
        """验证数据格式是否支持"""
        # 检查是否包含任何支持的字段
        supported_fields = [
            '机构名称', '投资方', '融资方', '公司名称'
        ]
        
        return any(field in data_item for field in supported_fields)