"""
公司数据实体抽取器
处理公司数据集，提取公司名称、行业、地区、法人代表、注册资金等实体
"""

import re
from typing import Dict, List, Any
from .base_extractor import BaseEntityExtractor, TextProcessor


class CompanyDataExtractor(BaseEntityExtractor):
    """公司数据实体抽取器"""
    
    def __init__(self):
        super().__init__()
        self.entity_types = ['公司名称', '行业', '地区', '法人代表', '注册资金', '成立时间', '公司介绍']
    
    def get_entity_types(self) -> List[str]:
        return self.entity_types
    
    def get_output_filename(self) -> str:
        return 'company_data_entities.pkl'
    
    def extract_entities(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """从公司数据项中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # 提取公司名称
        company_name = data_item.get('公司名称', '')
        if company_name:
            entities['公司名称'] = [company_name]
        
        # 提取行业（从公司介绍中分析）
        company_intro = data_item.get('公司介绍', '')
        if company_intro:
            industries = self._extract_industries_from_intro(company_intro)
            if industries:
                entities['行业'] = industries
        
        # 提取地区（从地址中分析）
        address = data_item.get('地址', '')
        if address:
            locations = self._extract_locations_from_address(address)
            if locations:
                entities['地区'] = locations
        
        # 提取法人代表
        legal_rep = data_item.get('法人代表', '')
        if legal_rep:
            entities['法人代表'] = [legal_rep]
        
        # 提取注册资金
        registered_capital = data_item.get('注册资金', '')
        if registered_capital:
            capitals = self._parse_registered_capital(registered_capital)
            entities['注册资金'] = capitals
        
        # 提取成立时间
        establish_time = data_item.get('成立时间', '')
        if establish_time:
            times = self._parse_establish_time(establish_time)
            entities['成立时间'] = times
        
        # 提取公司介绍
        intro = data_item.get('公司介绍', '')
        if intro:
            entities['公司介绍'] = [intro]
        
        return entities
    
    def _extract_industries_from_intro(self, intro: str) -> List[str]:
        """从公司介绍中提取行业信息"""
        if not intro:
            return []
        
        # 行业关键词映射
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
    
    def _extract_locations_from_address(self, address: str) -> List[str]:
        """从地址中提取地区信息"""
        if not address:
            return []
        
        # 中国省市关键词
        location_keywords = [
            '北京', '上海', '天津', '重庆', '河北', '山西', '辽宁', '吉林', '黑龙江', '江苏',
            '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '海南',
            '四川', '贵州', '云南', '陕西', '甘肃', '青海', '台湾', '内蒙古', '广西', '西藏',
            '宁夏', '新疆', '香港', '澳门',
            '广州', '深圳', '杭州', '南京', '成都', '武汉', '西安', '青岛', '大连', '宁波',
            '厦门', '苏州', '无锡', '佛山', '温州', '绍兴', '嘉兴', '金华', '台州', '湖州',
            '舟山', '丽水', '衢州', '常州', '徐州', '南通', '连云港', '淮安', '盐城', '扬州',
            '镇江', '泰州', '宿迁'
        ]
        
        found_locations = []
        for location in location_keywords:
            if location in address:
                found_locations.append(location)
        
        # 如果没有找到具体地点，使用通用文本提取
        if not found_locations:
            locations = TextProcessor.extract_locations(address)
            found_locations.extend(locations)
        
        return list(set(found_locations)) if found_locations else ['未知']
    
    def _parse_registered_capital(self, capital_str: str) -> List[str]:
        """解析注册资金字符串"""
        if not capital_str:
            return []
        
        capitals = []
        
        # 提取金额（包含数字和单位）
        capital_patterns = [
            r'(\d+(?:\.\d+)?(?:万|亿|千万|百万)?(?:人民币|美元|元)?)',  # 数字+单位
            r'(\d+(?:\.\d+)?(?:万|亿)?元)',  # 数字+万元/亿元
            r'([\u4e00-\u9fa5]*\d+(?:\.\d+)?[\u4e00-\u9fa5]*)',  # 包含中文的金额
        ]
        
        for pattern in capital_patterns:
            matches = re.findall(pattern, capital_str)
            capitals.extend(matches)
        
        # 如果没有找到具体金额，保留原始描述
        if not capitals and capital_str:
            capitals.append(capital_str.strip())
        
        return list(set(capitals))
    
    def _parse_establish_time(self, time_str: str) -> List[str]:
        """解析成立时间字符串"""
        if not time_str:
            return []
        
        times = []
        
        # 提取年份
        years = TextProcessor.extract_years(time_str)
        if years:
            times.extend(years)
        
        # 提取完整日期
        date_pattern = r'(\d{4})[年\-./](\d{1,2})[月\-./](\d{1,2})[日号]?'
        dates = re.findall(date_pattern, time_str)
        if dates:
            for date in dates:
                times.append(f"{date[0]}年{date[1]}月{date[2]}日")
        
        # 提取月份
        month_pattern = r'(\d{1,2})月'
        months = re.findall(month_pattern, time_str)
        if months:
            times.extend([f"{month}月" for month in months])
        
        # 如果没有找到具体时间，保留原始描述
        if not times and time_str:
            times.append(time_str.strip())
        
        return list(set(times))