"""
基于financial_ie_scheme.py的金融实体抽取器
复用现有的实体抽取逻辑，适配新的CSV处理框架
"""

import re
import pandas as pd
from typing import Dict, List, Any
import sys
import os

# 添加父目录到路径，以便导入financial_ie_scheme
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有的常量定义
from financial_ie_scheme import (
    FINANCIAL_ENTITY_TYPES,
    FINANCIAL_RELATION_TYPES,
    ROUND_MAPPING,
    INDUSTRY_MAPPING
)


class FinancialEntityExtractor:
    """基于现有代码的金融实体抽取器"""
    
    def __init__(self):
        self.entity_types = FINANCIAL_ENTITY_TYPES['chinese']
    
    def extract_from_investment_structure(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从投资结构数据中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 投资机构
        institution = data_item.get('机构名称', '')
        if institution and institution != 'nan':
            entities['投资机构'] = [institution]
        
        # 行业提取与标准化
        raw_industries = data_item.get('行业', '')
        if raw_industries and raw_industries != 'nan':
            industries = self._parse_industries(raw_industries)
            entities['行业'] = industries
        
        # 投资轮次提取与标准化
        raw_rounds = data_item.get('轮次', '')
        if raw_rounds and raw_rounds != 'nan':
            rounds = self._parse_rounds(raw_rounds)
            entities['投资轮次'] = rounds
        
        # 投资规模提取
        raw_scale = data_item.get('规模', '')
        if raw_scale and raw_scale != 'nan':
            scales = self._parse_scale(raw_scale)
            entities['投资规模'] = scales
        
        # 从描述中提取时间
        description = data_item.get('介绍', '')
        if description and description != '暂无信息':
            years = self._extract_years(description)
            if years:
                entities['投资时间'] = years
        
        # 从描述中提取地区
        if description:
            locations = self._extract_locations(description)
            if locations:
                entities['地区'] = locations
        
        return entities
    
    def extract_from_investment_events(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从投资事件数据中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 投资方（转换为投资机构）
        investor = data_item.get('投资方', '')
        if investor and investor != 'nan':
            investors = self._parse_investors(investor)
            entities['投资机构'] = investors
        
        # 融资方（转换为被投企业）
        investee = data_item.get('融资方', '')
        if investee and investee != 'nan':
            entities['被投企业'] = [investee]
        
        # 投资轮次
        round_name = data_item.get('轮次', '')
        if round_name and round_name != 'nan':
            standardized_round = self._standardize_round(round_name)
            entities['投资轮次'] = [standardized_round]
        
        # 投资金额（转换为投资规模）
        amount = data_item.get('金额', '')
        if amount and amount != 'nan':
            parsed_amounts = self._parse_amount(amount)
            entities['投资规模'] = parsed_amounts
        
        # 投资时间
        time_str = data_item.get('融资时间', '')
        if time_str and time_str != 'nan':
            parsed_times = self._parse_time(time_str)
            entities['投资时间'] = parsed_times
        
        return entities
    
    def extract_from_company_data(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从公司数据中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 公司名称（转换为被投企业）
        company_name = data_item.get('公司名称', '')
        if company_name and company_name != 'nan':
            entities['被投企业'] = [company_name]
        
        # 从公司介绍中提取行业
        company_intro = data_item.get('公司介绍', '')
        if company_intro and company_intro != 'nan':
            industries = self._extract_industries_from_intro(company_intro)
            if industries:
                entities['行业'] = industries
        
        # 从地址中提取地区
        address = data_item.get('地址', '')
        if address and address != 'nan':
            locations = self._extract_locations(address)
            if locations:
                entities['地区'] = locations
        
        # 成立时间
        establish_time = data_item.get('成立时间', '')
        if establish_time and establish_time != 'nan':
            times = self._parse_time(establish_time)
            entities['投资时间'] = times
        
        return entities
    
    # 以下方法复用自 financial_ie_scheme.py
    def _parse_industries(self, industries_str: str) -> List[str]:
        """解析行业字符串"""
        if not industries_str or industries_str == 'nan':
            return []
        
        industries = []
        industry_parts = re.split(r'[\s、]+', industries_str.strip())
        
        for part in industry_parts:
            part = part.strip()
            if not part:
                continue
            
            match = re.match(r'(\D+)(\d+)', part)
            if match:
                industry_name = match.group(1).strip()
            else:
                industry_name = part
            
            standardized_name = self._standardize_industry(industry_name)
            if standardized_name:
                industries.append(standardized_name)
        
        return list(set(industries))
    
    def _parse_rounds(self, rounds_str: str) -> List[str]:
        """解析投资轮次字符串"""
        if not rounds_str or rounds_str == 'nan':
            return []
        
        rounds = []
        round_parts = re.split(r'[、,\s]+', rounds_str.strip())
        
        for part in round_parts:
            part = part.strip()
            if not part:
                continue
            
            standardized_round = self._standardize_round(part)
            if standardized_round:
                rounds.append(standardized_round)
        
        return list(set(rounds))
    
    def _parse_scale(self, scale_str: str) -> List[str]:
        """解析投资规模字符串"""
        if not scale_str or scale_str == 'nan':
            return []
        
        scales = []
        scale_matches = re.findall(r'(\d+(?:\.\d+)?(?:万|亿|千万|百万)?(?:人民币|美元|元)?)', scale_str)
        
        for match in scale_matches:
            if match:
                scales.append(match)
        
        if not scales and scale_str:
            scales.append(scale_str.strip())
        
        return scales
    
    def _extract_years(self, text: str) -> List[str]:
        """从文本中提取年份"""
        if not text:
            return []
        year_pattern = r'(\d{4})年?'
        years = re.findall(year_pattern, text)
        return list(set(years))
    
    def _extract_locations(self, text: str) -> List[str]:
        """从文本中提取地点信息"""
        if not text:
            return []
        
        location_patterns = [
            r'位于([\u4e00-\u9fa5]+(?:市|省|区|县|国))',
            r'([\u4e00-\u9fa5]+(?:市|省|区|县|国))',
            r'([北京|上海|广州|深圳|杭州|南京|成都|重庆|天津|武汉|西安|苏州|青岛|大连|宁波|厦门|无锡|佛山|温州|绍兴|嘉兴|金华|台州|湖州|舟山|丽水|衢州]+)'
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.extend(matches)
        
        return list(set(locations))
    
    def _standardize_industry(self, industry_name: str) -> str:
        """标准化行业名称"""
        for standard_name, keywords in INDUSTRY_MAPPING.items():
            if any(keyword in industry_name for keyword in keywords):
                return standard_name
        return "其他"
    
    def _standardize_round(self, round_name: str) -> str:
        """标准化轮次名称"""
        for standard_name, keywords in ROUND_MAPPING.items():
            if any(keyword in round_name for keyword in keywords):
                return standard_name
        return "未知"
    
    # 新增的方法
    def _parse_investors(self, investor_str: str) -> List[str]:
        """解析投资方字符串"""
        if not investor_str:
            return []
        
        investor_parts = re.split(r'[、,，；;\s]+', investor_str.strip())
        
        investors = []
        for part in investor_parts:
            part = part.strip()
            if part and len(part) > 1:
                investors.append(part)
        
        return list(set(investors))
    
    def _parse_amount(self, amount_str: str) -> List[str]:
        """解析投资金额字符串"""
        if not amount_str:
            return []
        
        amounts = []
        
        # 提取具体金额
        amount_patterns = [
            r'(\d+(?:\.\d+)?(?:万|亿|千万|百万)?(?:人民币|美元|美金|元)?)',
            r'(\d+(?:\.\d+)?(?:万|亿)?元)',
            r'([\u4e00-\u9fa5]*\d+(?:\.\d+)?[\u4e00-\u9fa5]*)',
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, amount_str)
            amounts.extend(matches)
        
        if not amounts and amount_str:
            amounts.append(amount_str.strip())
        
        return list(set(amounts))
    
    def _parse_time(self, time_str: str) -> List[str]:
        """解析时间字符串"""
        if not time_str:
            return []
        
        times = []
        
        # 提取年份
        years = self._extract_years(time_str)
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
        
        if not times and time_str:
            times.append(time_str.strip())
        
        return list(set(times))
    
    def _extract_industries_from_intro(self, intro: str) -> List[str]:
        """从公司介绍中提取行业信息"""
        if not intro:
            return []
        
        industry_keywords = {
            '科技': ['科技', '技术', '智能', '软件', '硬件', '互联网', 'IT', '人工智能', 'AI', '大数据', '云计算'],
            '金融': ['金融', '银行', '保险', '证券', '投资', '基金', '支付', '信贷', '理财'],
            '医疗': ['医疗', '医药', '健康', '医院', '药品', '医疗器械', '生物科技', '基因', '诊断'],
            '教育': ['教育', '培训', '学校', '大学', '学院', '课程', '学习', '知识'],
            '电商': ['电商', '电子商务', '零售', '销售', '购物', '贸易', '商业'],
            '制造': ['制造', '生产', '工厂', '工业', '加工', '制造', '装备'],
            '房地产': ['房地产', '房产', '地产', '建筑', '物业', '开发', '建设'],
            '汽车': ['汽车', '车辆', '交通', '出行', '运输', '物流', '驾驶'],
            '文娱': ['文化', '娱乐', '影视', '音乐', '游戏', '体育', '传媒', '出版'],
            '餐饮': ['餐饮', '食品', '饮料', '餐厅', '酒店', '旅游', '休闲']
        }
        
        intro_lower = intro.lower()
        found_industries = []
        
        for industry, keywords in industry_keywords.items():
            for keyword in keywords:
                if keyword.lower() in intro_lower:
                    found_industries.append(industry)
                    break
        
        return list(set(found_industries)) if found_industries else ['其他']