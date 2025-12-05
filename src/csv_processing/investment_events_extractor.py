"""
投资事件数据实体抽取器
处理投资事件数据集，提取投资方、融资方、投资轮次、金额、时间等实体
"""

import re
from typing import Dict, List, Any
from .base_extractor import BaseEntityExtractor, TextProcessor


class InvestmentEventsExtractor(BaseEntityExtractor):
    """投资事件数据实体抽取器"""
    
    def __init__(self):
        super().__init__()
        self.entity_types = ['投资方', '融资方', '投资轮次', '投资金额', '投资时间', '事件资讯']
    
    def get_entity_types(self) -> List[str]:
        return self.entity_types
    
    def get_output_filename(self) -> str:
        return 'investment_events_entities.pkl'
    
    def extract_entities(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从投资事件数据项中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 提取投资方
        investor = data_item.get('投资方', '')
        if investor:
            investors = self._parse_investors(investor)
            entities['投资方'] = investors
        
        # 提取融资方
        investee = data_item.get('融资方', '')
        if investee:
            entities['融资方'] = [investee]
        
        # 提取投资轮次
        round_name = data_item.get('轮次', '')
        if round_name:
            standardized_round = self._standardize_round(round_name)
            entities['投资轮次'] = [standardized_round]
        
        # 提取投资金额
        amount = data_item.get('金额', '')
        if amount:
            parsed_amounts = self._parse_amount(amount)
            entities['投资金额'] = parsed_amounts
        
        # 提取投资时间
        time_str = data_item.get('融资时间', '')
        if time_str:
            parsed_times = self._parse_time(time_str)
            entities['投资时间'] = parsed_times
        
        # 提取事件资讯
        event_info = data_item.get('事件资讯', '')
        if event_info:
            entities['事件资讯'] = [event_info]
        
        return entities
    
    def _parse_investors(self, investor_str: str) -> List[str]:
        """解析投资方字符串"""
        if not investor_str:
            return []
        
        # 分割多个投资方（按顿号、逗号、空格、分号等）
        investor_parts = re.split(r'[、,，；;\s]+', investor_str.strip())
        
        investors = []
        for part in investor_parts:
            part = part.strip()
            if part and len(part) > 1:  # 过滤掉单个字符
                investors.append(part)
        
        return list(set(investors))  # 去重
    
    def _parse_amount(self, amount_str: str) -> List[str]:
        """解析投资金额字符串"""
        if not amount_str:
            return []
        
        amounts = []
        
        # 提取具体金额（包含数字和单位）
        amount_patterns = [
            r'(\d+(?:\.\d+)?(?:万|亿|千万|百万)?(?:人民币|美元|美金|元)?)',  # 数字+单位
            r'(\d+(?:\.\d+)?(?:万|亿)?元)',  # 数字+万元/亿元
            r'([\u4e00-\u9fa5]*\d+(?:\.\d+)?[\u4e00-\u9fa5]*)',  # 包含中文的金额
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, amount_str)
            amounts.extend(matches)
        
        # 如果没有找到具体金额，保留原始描述
        if not amounts and amount_str:
            amounts.append(amount_str.strip())
        
        return list(set(amounts))
    
    def _parse_time(self, time_str: str) -> List[str]:
        """解析时间字符串"""
        if not time_str:
            return []
        
        times = []
        
        # 提取年份
        years = TextProcessor.extract_years(time_str)
        if years:
            times.extend(years)
        
        # 提取月份
        month_pattern = r'(\d{1,2})月'
        months = re.findall(month_pattern, time_str)
        if months:
            times.extend([f"{month}月" for month in months])
        
        # 提取完整日期
        date_pattern = r'(\d{4})[年\-./](\d{1,2})[月\-./](\d{1,2})[日号]?'
        dates = re.findall(date_pattern, time_str)
        if dates:
            for date in dates:
                times.append(f"{date[0]}年{date[1]}月{date[2]}日")
        
        # 如果没有找到具体时间，保留原始描述
        if not times and time_str:
            times.append(time_str.strip())
        
        return list(set(times))
    
    def _standardize_round(self, round_name: str) -> str:
        """标准化轮次名称"""
        round_mapping = {
            '天使轮': ['天使轮', '天使', '种子轮', '天使投资'],
            'Pre-A轮': ['Pre-A轮', 'Pre-A', 'PreA'],
            'A轮': ['A轮', 'A+轮', 'A++轮', 'A+'],
            'B轮': ['B轮', 'B+轮', 'B++轮', 'B+'],
            'C轮': ['C轮', 'C+轮', 'C++轮', 'C+'],
            'D轮': ['D轮', 'D+轮', 'D+'],
            'E轮': ['E轮'],
            'F轮': ['F轮'],
            'Pre-IPO': ['Pre-IPO', 'PreIPO', '上市前'],
            '战略融资': ['战略融资', '战略投资'],
            '收购并购': ['收购并购', '并购', '收购'],
            '上市': ['上市', 'IPO'],
            '新三板': ['新三板'],
            '未知': ['未知', '未披露', ' undisclosed']
        }
        
        round_name_lower = round_name.lower()
        for standard_name, keywords in round_mapping.items():
            if any(keyword.lower() in round_name_lower for keyword in keywords):
                return standard_name
        return round_name  # 如果没有匹配，返回原始值