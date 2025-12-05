"""
CSV数据抽取基类
提供通用的CSV数据读取和实体抽取框架
"""

import pandas as pd
import re
import os
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime


class BaseEntityExtractor(ABC):
    """实体抽取基类"""
    
    def __init__(self):
        self.extracted_entities = []
        self.entity_types = set()
    
    @abstractmethod
    def get_entity_types(self) -> List[str]:
        """获取该数据集支持的实体类型"""
        pass
    
    @abstractmethod
    def extract_entities(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从数据项中提取实体"""
        pass
    
    @abstractmethod
    def get_output_filename(self) -> str:
        """获取输出文件名（应该是.pkl后缀）"""
        pass
    
    def read_csv_data(self, csv_file: str) -> List[Dict[str, Any]]:
        """读取CSV数据"""
        print(f"正在读取CSV文件: {csv_file}")
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"CSV文件包含 {len(df)} 条记录，列名: {list(df.columns)}")
            
            data_list = []
            for idx, row in df.iterrows():
                # 将NaN值转换为空字符串
                row_dict = {}
                for col in df.columns:
                    value = row.get(col, '')
                    if pd.isna(value):
                        value = ''
                    else:
                        value = str(value).strip()
                    row_dict[col] = value
                row_dict['id'] = idx
                data_list.append(row_dict)
            
            print(f"成功读取 {len(data_list)} 条记录")
            return data_list
            
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return []
    
    def process_dataset(self, csv_file: str, output_dir: str) -> Dict[str, Any]:
        """处理整个数据集"""
        print(f"=== 开始处理数据集: {os.path.basename(csv_file)} ===")
        start_time = datetime.now()
        
        # 1. 读取数据
        data_list = self.read_csv_data(csv_file)
        if not data_list:
            return {}
        
        # 2. 抽取实体
        entity_dict = {}  # 实体名称 -> 实体类别列表
        for idx, data_item in enumerate(data_list):
            if idx % 100 == 0:
                print(f"处理第 {idx+1}/{len(data_list)} 条记录")
            
            entities = self.extract_entities(data_item)
            
            # 将实体转换为名称-类别映射格式
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    if entity_name and entity_name != 'nan':
                        entity_name = entity_name.strip()
                        if entity_name not in entity_dict:
                            entity_dict[entity_name] = []
                        if entity_type not in entity_dict[entity_name]:
                            entity_dict[entity_name].append(entity_type)
        
        # 3. 统计信息
        statistics = self._calculate_statistics(entity_dict)
        
        # 4. 构建结果
        results = {
            'metadata': {
                'processing_time': datetime.now().isoformat(),
                'total_records': len(data_list),
                'total_entities': len(entity_dict),
                'processing_duration': str(datetime.now() - start_time),
                'data_source': csv_file
            },
            'entities': entity_dict,
            'statistics': statistics
        }
        
        # 5. 保存结果
        self._save_entities_pickle(entity_dict, os.path.join(output_dir, self.get_output_filename()))
        
        print("=== 处理完成 ===")
        self._print_summary(results)
        
        return results
    
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
    
    def _save_entities_pickle(self, entity_dict: Dict[str, List[str]], output_file: str):
        """保存实体到pickle文件"""
        import pickle
        
        # 确保输出目录存在，但不再自动创建子目录
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(entity_dict, f)
        
        print(f"实体数据已保存到: {output_file}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印结果摘要"""
        print(f"\n=== 处理结果摘要 ===")
        print(f"处理记录数: {results['metadata']['total_records']}")
        print(f"抽取实体数: {results['metadata']['total_entities']}")
        print(f"唯一实体值: {results['statistics']['unique_entity_count']}")
        
        print(f"\n实体类型分布:")
        for entity_type, count in results['statistics']['entity_type_counts'].items():
            print(f"  {entity_type}: {count}")


class TextProcessor:
    """文本处理工具类"""
    
    @staticmethod
    def extract_years(text: str) -> List[str]:
        """从文本中提取年份"""
        if not text:
            return []
        year_pattern = r'(\d{4})年?'
        years = re.findall(year_pattern, text)
        return list(set(years))
    
    @staticmethod
    def extract_money_amounts(text: str) -> List[str]:
        """从文本中提取金额"""
        if not text:
            return []
        money_pattern = r'(\d+(?:\.\d+)?(?:万|亿|千万|百万)?(?:人民币|美元|元)?)'
        amounts = re.findall(money_pattern, text)
        return list(set(amounts))
    
    @staticmethod
    def extract_locations(text: str) -> List[str]:
        """从文本中提取地点信息"""
        if not text:
            return []
        
        # 常见城市和省名
        location_patterns = [
            r'位于([\u4e00-\u9fa5]+(?:市|省|区|县|国))',
            r'([\u4e00-\u9fa5]+(?:市|省|区|县|国))',
            r'(北京|上海|广州|深圳|杭州|南京|成都|重庆|天津|武汉|西安|苏州|青岛|大连|宁波|厦门|无锡|佛山|温州|绍兴|嘉兴|金华|台州|湖州|舟山|丽水|衢州)'
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.extend(matches)
        
        return list(set(locations))
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        return text