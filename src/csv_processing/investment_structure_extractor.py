"""
投资结构数据实体抽取器
处理投资结构数据集，提取投资机构、行业、投资轮次、投资规模等实体
"""

import re
from typing import Dict, List, Any
from .base_extractor import BaseEntityExtractor, TextProcessor


class InvestmentStructureExtractor(BaseEntityExtractor):
    """投资结构数据实体抽取器"""
    
    def __init__(self):
        super().__init__()
        self.entity_types = ['投资机构', '行业', '投资轮次', '投资规模', '投资时间', '地区']
    
    def get_entity_types(self) -> List[str]:
        return self.entity_types
    
    def get_output_filename(self) -> str:
        return 'investment_structure_entities.pkl'
    
    def extract_entities(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从投资结构数据项中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 提取投资机构
        institution = data_item.get('机构名称', '')
        if institution:
            entities['投资机构'] = [institution]
        
        # 提取行业
        industries = data_item.get('行业', '')
        if industries:
            parsed_industries = self._parse_industries(industries)
            entities['行业'] = parsed_industries
        
        # 提取投资轮次
        rounds = data_item.get('轮次', '')
        if rounds:
            parsed_rounds = self._parse_rounds(rounds)
            entities['投资轮次'] = parsed_rounds
        
        # 提取投资规模
        scale = data_item.get('规模', '')
        if scale:
            parsed_scales = self._parse_scale(scale)
            entities['投资规模'] = parsed_scales
        
        # 从介绍中提取时间和地区
        description = data_item.get('介绍', '')
        if description:
            # 提取时间
            years = TextProcessor.extract_years(description)
            if years:
                entities['投资时间'] = years
            
            # 提取地区
            locations = TextProcessor.extract_locations(description)
            if locations:
                entities['地区'] = locations
        
        return entities
    
    def _parse_industries(self, industries_str: str) -> List[str]:
        """解析行业字符串"""
        if not industries_str:
            return []
        
        industries = []
        # 分割行业（按空格或顿号）
        industry_parts = re.split(r'[\s、]+', industries_str.strip())
        
        for part in industry_parts:
            part = part.strip()
            if not part:
                continue
            
            # 提取数字和文字
            match = re.match(r'(\D+)(\d+)', part)
            if match:
                industry_name = match.group(1).strip()
            else:
                industry_name = part
            
            # 标准化行业名称
            standardized_name = self._standardize_industry(industry_name)
            if standardized_name:
                industries.append(standardized_name)
        
        return list(set(industries))  # 去重
    
    def _parse_rounds(self, rounds_str: str) -> List[str]:
        """解析投资轮次字符串"""
        if not rounds_str:
            return []
        
        rounds = []
        # 分割轮次（按顿号、逗号或空格）
        round_parts = re.split(r'[、,\s]+', rounds_str.strip())
        
        for part in round_parts:
            part = part.strip()
            if not part:
                continue
            
            # 标准化轮次名称
            standardized_round = self._standardize_round(part)
            if standardized_round:
                rounds.append(standardized_round)
        
        return list(set(rounds))  # 去重
    
    def _parse_scale(self, scale_str: str) -> List[str]:
        """解析投资规模字符串"""
        if not scale_str:
            return []
        
        scales = []
        # 提取金额信息
        scale_matches = re.findall(r'(\d+(?:\.\d+)?(?:万|亿|千万|百万)?(?:人民币|美元|元)?)', scale_str)
        
        for match in scale_matches:
            if match:
                scales.append(match)
        
        # 如果没有找到具体金额，保留原始描述
        if not scales and scale_str:
            scales.append(scale_str.strip())
        
        return scales
    
    def _standardize_industry(self, industry_name: str) -> str:
        """标准化行业名称"""
        industry_mapping = {
            '企业服务': ['企业服务', 'SaaS', '软件服务', '工具软件与服务'],
            '汽车交通': ['汽车交通', '汽车', '交通', '出行', '无人机'],
            '硬件': ['硬件', '智能硬件', '设备'],
            '医疗': ['医疗', '医疗健康', '生物医药'],
            '金融': ['金融', '金融科技', '支付'],
            '文娱内容游戏': ['文娱', '内容', '游戏', '娱乐', '文娱内容游戏'],
            '电商消费': ['电商', '消费', '零售', '电商消费'],
            '生活服务': ['生活服务', '本地生活', '服务'],
            '高科技': ['高科技', '技术', '科技', '人工智能', '高科技'],
            '智能制造': ['智能制造', '制造', '工业'],
            '物流仓储': ['物流仓储', '物流'],
            '房产': ['房产', '房地产'],
            '教育': ['教育'],
            '旅游': ['旅游'],
            '体育健身': ['体育健身', '体育'],
            '社区社交': ['社区社交', '社区'],
            '农业': ['农业'],
            '其他': ['其他']
        }
        
        for standard_name, keywords in industry_mapping.items():
            if any(keyword in industry_name for keyword in keywords):
                return standard_name
        return "其他"
    
    def _standardize_round(self, round_name: str) -> str:
        """标准化轮次名称"""
        round_mapping = {
            '天使轮': ['天使轮', '天使', '种子轮'],
            'Pre-A轮': ['Pre-A轮', 'Pre-A'],
            'A轮': ['A轮', 'A+轮', 'A++轮'],
            'B轮': ['B轮', 'B+轮', 'B++轮'], 
            'C轮': ['C轮', 'C+轮', 'C++轮'],
            'D轮': ['D轮', 'D+轮'],
            'E轮': ['E轮'],
            'F轮': ['F轮'],
            '战略融资': ['战略融资', '战略投资'],
            '收购并购': ['收购并购', '并购', '收购'],
            '上市': ['上市', 'IPO'],
            '未知': ['未知']
        }
        
        for standard_name, keywords in round_mapping.items():
            if any(keyword in round_name for keyword in keywords):
                return standard_name
        return "未知"