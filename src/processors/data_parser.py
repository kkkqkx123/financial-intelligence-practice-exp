"""
硬编码数据解析器 - 基于实际数据集结构的优化实现
"""

import re
from typing import Dict, List, Optional, Union
from datetime import datetime
from .config import (
    ROUND_MAPPING, AMOUNT_RULES, CAPITAL_RULES, 
    DATE_PATTERNS, PERFORMANCE_CONFIG
)


class DataParser:
    """硬编码优先的数据解析器"""
    
    def __init__(self):
        self.stats = {
            'companies_parsed': 0,
            'events_parsed': 0,
            'institutions_parsed': 0,
            'parse_errors': 0
        }
    
    # ==================== 公司数据解析 ====================
    
    def parse_companies(self, text: str) -> List[Dict]:
        """解析公司数据"""
        companies = []
        lines = text.strip().split('\n')
        
        # 跳过统计行和表头
        data_started = False
        for line in lines:
            if line.startswith('名称,公司名称,公司介绍'):
                data_started = True
                continue
            if not data_started or not line.strip():
                continue
            
            try:
                fields = self._smart_split(line)
                if len(fields) >= 11:
                    company = {
                        'short_name': fields[0].strip(),
                        'full_name': fields[1].strip(),
                        'description': fields[2].strip(),
                        'registration_name': fields[3].strip(),
                        'address': fields[4].strip() if fields[4].strip() else None,
                        'registration_id': fields[5].strip(),
                        'establish_date': self._parse_date(fields[6]),
                        'legal_representative': fields[7].strip(),
                        'registered_capital': self._normalize_capital(fields[8]),
                        'credit_code': fields[9].strip(),
                        'website': self._normalize_website(fields[10]),
                        'parsed_by': 'hardcoded',
                        'parse_confidence': 1.0
                    }
                    companies.append(company)
                    self.stats['companies_parsed'] += 1
            except Exception as e:
                self.stats['parse_errors'] += 1
                print(f"解析公司数据失败: {line[:50]}... 错误: {e}")
        
        return companies
    
    # ==================== 投资事件解析 ====================
    
    def parse_investment_events(self, text: str) -> List[Dict]:
        """解析投资事件数据"""
        events = []
        lines = text.strip().split('\n')
        
        data_started = False
        for line in lines:
            if line.startswith('事件资讯,投资方,融资方'):
                data_started = True
                continue
            if not data_started or not line.strip():
                continue
            
            try:
                fields = self._smart_split(line)
                if len(fields) >= 6:
                    event = {
                        'description': fields[0].strip(),
                        'investors': self._parse_investors(fields[1]),
                        'investee': fields[2].strip(),
                        'investment_date': self._parse_date(fields[3]),
                        'round': self._normalize_round(fields[4]),
                        'amount': self._normalize_amount(fields[5]),
                        'parsed_by': 'hardcoded',
                        'parse_confidence': 1.0
                    }
                    events.append(event)
                    self.stats['events_parsed'] += 1
            except Exception as e:
                self.stats['parse_errors'] += 1
                print(f"解析投资事件失败: {line[:50]}... 错误: {e}")
        
        return events
    
    # ==================== 投资机构解析 ====================
    
    def parse_investment_institutions(self, text: str) -> List[Dict]:
        """解析投资机构数据"""
        institutions = []
        lines = text.strip().split('\n')
        
        # 跳过两行统计信息
        data_started = False
        skip_count = 0
        for line in lines:
            if line.startswith('机构名称,介绍,行业,规模,轮次'):
                data_started = True
                continue
            if not data_started or not line.strip():
                continue
            if skip_count < 1:  # 跳过第一行统计
                skip_count += 1
                continue
            
            try:
                fields = self._smart_split(line)
                if len(fields) >= 5:
                    institution = {
                        'name': fields[0].strip(),
                        'description': fields[1].strip() if fields[1].strip() else None,
                        'industries': self._parse_industries(fields[2]),
                        'scale': self._normalize_scale(fields[3]),
                        'preferred_rounds': self._parse_preferred_rounds(fields[4]),
                        'parsed_by': 'hardcoded',
                        'parse_confidence': 1.0
                    }
                    institutions.append(institution)
                    self.stats['institutions_parsed'] += 1
            except Exception as e:
                self.stats['parse_errors'] += 1
                print(f"解析投资机构失败: {line[:50]}... 错误: {e}")
        
        return institutions
    
    # ==================== 核心解析函数 ====================
    
    def _smart_split(self, line: str) -> List[str]:
        """智能分割 - 处理引号和逗号"""
        fields = []
        current = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                fields.append(current)
                current = ""
            else:
                current += char
        
        fields.append(current)
        return fields
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """日期格式标准化"""
        if not date_str or date_str.strip() == '':
            return None
        
        date_str = date_str.strip()
        
        # 预定义模式匹配
        for pattern, converter in DATE_PATTERNS:
            match = re.search(pattern, date_str)
            if match:
                try:
                    # 验证日期有效性
                    date_obj = datetime.strptime(converter(match.group()), '%Y-%m-%d')
                    if date_obj <= datetime.now():
                        return converter(match.group())
                except ValueError:
                    continue
        
        return None  # 无法解析的日期
    
    def _normalize_capital(self, capital_str: str) -> Optional[float]:
        """注册资本标准化"""
        if not capital_str or capital_str.strip() == '':
            return None
        
        capital_str = capital_str.strip()
        
        # 预定义转换规则
        for pattern, converter in CAPITAL_RULES:
            match = re.search(pattern, capital_str)
            if match:
                try:
                    return converter(match.group(1))
                except ValueError:
                    continue
        
        # 兜底处理：提取纯数字（默认万元）
        numbers = re.findall(r'\d+(?:\.\d+)?', capital_str)
        if numbers:
            return float(numbers[0]) * 10000
        
        return None
    
    def _normalize_amount(self, amount_str: str) -> Optional[float]:
        """投资金额标准化"""
        if not amount_str or amount_str.strip() == '' or amount_str == '未披露':
            return None
        
        amount_str = amount_str.strip()
        
        # 处理模糊金额
        if '数千万' in amount_str:
            base_amount = 50000000  # 默认5000万
        elif '数百万' in amount_str:
            base_amount = 5000000   # 默认500万
        elif '数十万' in amount_str:
            base_amount = 500000    # 默认50万
        else:
            base_amount = None
        
        # 预定义转换规则
        for pattern, converter in AMOUNT_RULES:
            match = re.search(pattern, amount_str)
            if match:
                try:
                    return converter(match.group(1))
                except ValueError:
                    continue
        
        # 如果找到模糊金额基准
        if base_amount:
            return base_amount
        
        # 最后尝试提取纯数字
        numbers = re.findall(r'\d+(?:\.\d+)?', amount_str)
        if numbers:
            return float(numbers[0]) * 10000  # 默认万元
        
        return None
    
    def _normalize_round(self, round_str: str) -> Optional[str]:
        """投资轮次标准化"""
        if not round_str:
            return None
        
        round_str = round_str.strip()
        return ROUND_MAPPING.get(round_str, round_str)
    
    def _normalize_website(self, website: str) -> Optional[str]:
        """网站地址标准化"""
        if not website or website.strip() == '':
            return None
        
        website = website.strip()
        
        # 添加协议前缀
        if not website.startswith(('http://', 'https://')):
            website = 'http://' + website
        
        # 基本格式验证
        if re.match(r'^https?://[\w\-]+(\.[\w\-]+)+[/#?]?.*$', website):
            return website
        
        return None
    
    def _parse_investors(self, investors_str: str) -> List[str]:
        """解析投资方列表"""
        if not investors_str or investors_str.strip() == '':
            return []
        
        # 支持多种分隔符
        separators = [' ', '、', ',', '，', ';', '；']
        investors = [inv.strip() for inv in re.split('|'.join(map(re.escape, separators)), investors_str) if inv.strip()]
        
        return investors
    
    def _parse_industries(self, industries_str: str) -> List[str]:
        """解析投资行业列表"""
        if not industries_str or industries_str.strip() == '':
            return []
        
        # 支持多种分隔符
        separators = [' ', '、', ',', '，', ';', '；']
        industries = [ind.strip() for ind in re.split('|'.join(map(re.escape, separators)), industries_str) if ind.strip()]
        
        return industries
    
    def _parse_preferred_rounds(self, rounds_str: str) -> List[str]:
        """解析偏好轮次列表"""
        if not rounds_str or rounds_str.strip() == '':
            return []
        
        # 支持多种分隔符
        separators = [' ', '、', ',', '，', ';', '；']
        rounds = []
        
        for round_item in re.split('|'.join(map(re.escape, separators)), rounds_str):
            round_item = round_item.strip()
            if round_item:
                normalized = self._normalize_round(round_item)
                if normalized:
                    rounds.append(normalized)
        
        return rounds
    
    def _normalize_scale(self, scale_str: str) -> Optional[str]:
        """管理规模标准化"""
        if not scale_str or scale_str.strip() == '':
            return None
        
        scale_str = scale_str.strip()
        
        # 提取数字和单位
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)', scale_str)
        if match:
            amount, unit = match.groups()
            return f"{amount}{unit}"
        
        return scale_str
    
    def get_stats(self) -> Dict:
        """获取解析统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'companies_parsed': 0,
            'events_parsed': 0,
            'institutions_parsed': 0,
            'parse_errors': 0
        }