"""
硬编码数据解析器 - 基于实际数据集结构的优化实现
"""

import re
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from .config import (
    ROUND_MAPPING, AMOUNT_RULES, CAPITAL_RULES, 
    DATE_PATTERNS, PERFORMANCE_CONFIG
)
from .field_mapping import (
    COMPANY_FIELD_MAPPING, 
    INVESTMENT_EVENT_FIELD_MAPPING, 
    INVESTOR_FIELD_MAPPING,
    apply_field_mapping
)

# 配置日志
logger = logging.getLogger(__name__)


class DataParser:
    """硬编码优先的数据解析器"""
    
    def __init__(self):
        self.stats = {
            'total_records': 0,
            'companies_processed': 0,
            'events_processed': 0,
            'investors_processed': 0,
            'errors': 0
        }
    
    # ==================== 公司数据解析 ====================
    
    def parse_companies(self, data: Union[str, List[Dict]]) -> List[Dict]:
        """解析公司数据 - 增强错误处理和日志"""
        companies = []
        logger.info(f"开始解析公司数据，输入类型: {type(data)}")
        
        # 如果是列表，说明是已经解析好的字典数据
        if isinstance(data, list):
            logger.info(f"处理字典列表数据，共 {len(data)} 条记录")
            for i, item in enumerate(data):
                try:
                    if isinstance(item, dict):
                        company = {
                            'short_name': item.get('short_name', item.get('name', '')).strip(),
                            'full_name': item.get('full_name', item.get('name', '')).strip(),
                            'description': item.get('description', '').strip(),
                            'registration_name': item.get('registration_name', item.get('name', '')).strip(),
                            'address': item.get('address') or item.get('contact_info', {}).get('address'),
                            'registration_id': item.get('registration_id', '').strip(),
                            'establish_date': item.get('establish_date') or item.get('registration_date'),
                            'legal_representative': item.get('legal_representative', '').strip(),
                            'registered_capital': item.get('registered_capital'),
                            'credit_code': item.get('credit_code', '').strip(),
                            'website': item.get('website'),
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        companies.append(company)
                        self.stats['companies_processed'] += 1
                        self.stats['total_records'] += 1
                    else:
                        logger.warning(f"跳过非字典项 {i}: {type(item)}")
                        self.stats['errors'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析公司数据失败 (索引 {i}): {str(item)[:100]}... 错误: {e}")
            logger.info(f"字典列表处理完成，成功: {len(companies)} 条，失败: {self.stats['errors']} 条")
        
        # 应用字段映射，转换为构建器期望的格式
        logger.info("应用公司数据字段映射")
        mapped_companies = apply_field_mapping(companies, COMPANY_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_companies)} 条记录")
        
        return mapped_companies
        
        # 如果是字符串，使用原有的解析逻辑
        logger.info("处理字符串格式数据")
        text = str(data)
        lines = text.strip().split('\n')
        logger.info(f"字符串数据共 {len(lines)} 行")
        
        # 跳过统计行和表头
        data_started = False
        line_count = 0
        for line_num, line in enumerate(lines, 1):
            if line.startswith('公司简称,公司全称,公司描述,注册名称'):
                data_started = True
                logger.info(f"在第 {line_num} 行找到表头")
                continue
            if not data_started or not line.strip():
                continue
            
            line_count += 1
            try:
                fields = self._smart_split(line)
                logger.debug(f"第 {line_num} 行分割为 {len(fields)} 个字段")
                
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
                    self.stats['companies_processed'] += 1
                    self.stats['total_records'] += 1
                else:
                    logger.warning(f"第 {line_num} 行字段数不足: {len(fields)} < 11")
                    self.stats['errors'] += 1
                    
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"解析公司数据失败 (第 {line_num} 行): {line[:100]}... 错误: {e}")
        
        logger.info(f"字符串格式处理完成，成功: {len(companies)} 条，失败: {self.stats['errors']} 条，处理行数: {line_count}")
        
        # 应用字段映射，转换为构建器期望的格式
        logger.info("应用公司数据字段映射")
        mapped_companies = apply_field_mapping(companies, COMPANY_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_companies)} 条记录")
        
        return mapped_companies
    
    # ==================== 投资事件解析 ====================
    
    def parse_investment_events(self, data: Union[str, List[Dict]]) -> List[Dict]:
        """解析投资事件数据 - 增强错误处理和日志"""
        events = []
        logger.info(f"开始解析投资事件数据，输入类型: {type(data)}")
        
        # 如果是列表，说明是已经解析好的字典数据
        if isinstance(data, list):
            for item in data:
                try:
                    if isinstance(item, dict):
                        event = {
                            'description': item.get('description', item.get('event', '')).strip(),
                            'investors': item.get('investors', item.get('investment_partners', [])),
                            'investee': item.get('investee', item.get('company', '')).strip(),
                            'investment_date': item.get('investment_date') or item.get('date'),
                            'round': item.get('round', item.get('funding_round', '')),
                            'amount': item.get('amount', item.get('funding_amount')),
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        events.append(event)
                        self.stats['events_processed'] += 1
                        self.stats['total_records'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析投资事件数据失败: {item}... 错误: {e}")
            # 应用字段映射，转换为构建器期望的格式
        logger.info("应用投资事件字段映射")
        mapped_events = apply_field_mapping(events, INVESTMENT_EVENT_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_events)} 条记录")
        
        return mapped_events
        
        # 如果是字符串，使用原有的解析逻辑
        text = str(data)
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
                    self.stats['events_processed'] += 1
                    self.stats['total_records'] += 1
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"解析投资事件失败: {line[:50]}... 错误: {e}")
        
        # 应用字段映射，转换为构建器期望的格式
        logger.info("应用投资事件字段映射")
        mapped_events = apply_field_mapping(events, INVESTMENT_EVENT_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_events)} 条记录")
        
        return mapped_events
    
    # ==================== 投资机构解析 ====================
    
    def parse_investment_institutions(self, data: Union[str, List[Dict]]) -> List[Dict]:
        """解析投资机构数据"""
        institutions = []
        
        # 如果是列表，说明是已经解析好的字典数据
        if isinstance(data, list):
            for item in data:
                try:
                    if isinstance(item, dict):
                        institution = {
                            'name': item.get('name', item.get('institution_name', '')).strip(),
                            'description': item.get('description', item.get('introduction', '')).strip(),
                            'industries': item.get('industries', item.get('sectors', [])),
                            'scale': item.get('scale', item.get('fund_size')),
                            'preferred_rounds': item.get('preferred_rounds', item.get('investment_stages', [])),
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        institutions.append(institution)
                        self.stats['investors_processed'] += 1
                        self.stats['total_records'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析投资机构数据失败: {item}... 错误: {e}")
            # 应用字段映射，转换为构建器期望的格式
        logger.info("应用投资机构字段映射")
        mapped_institutions = apply_field_mapping(institutions, INVESTOR_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_institutions)} 条记录")
        
        return mapped_institutions
        
        # 如果是字符串，使用原有的解析逻辑
        text = str(data)
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
                    self.stats['investors_processed'] += 1
                    self.stats['total_records'] += 1
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"解析投资机构失败: {line[:50]}... 错误: {e}")
        
        # 应用字段映射，转换为构建器期望的格式
        logger.info("应用投资机构字段映射")
        mapped_institutions = apply_field_mapping(institutions, INVESTOR_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_institutions)} 条记录")
        
        return mapped_institutions
    
    # ==================== 核心解析函数 ====================
    
    def _smart_split(self, line: str) -> List[str]:
        """智能分割 - 处理引号和逗号"""
        fields: List[str] = []
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
                    date_str = converter(match.group())
                    assert isinstance(date_str, str), f"Converter should return str, got {type(date_str)}"
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    if date_obj <= datetime.now():
                        return date_str
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
    
    def parse_csv_data(self, csv_content: str) -> List[Dict]:
        """解析CSV数据内容 - 支持包含NUL字符的文件"""
        import csv
        from io import StringIO
        
        result = []
        try:
            # 清理NUL字符和其他不可见字符
            cleaned_content = csv_content.replace('\x00', '').replace('\ufeff', '')
            
            # 使用StringIO创建文件对象
            string_buffer = StringIO(cleaned_content)
            
            # 逐行读取，跳过包含NUL字符的行
            lines = []
            for i, line in enumerate(string_buffer):
                if '\x00' not in line:
                    lines.append(line)
                else:
                    logger.warning(f"跳过包含NUL字符的行 {i+1}: {line[:50]}...")
            
            if not lines:
                logger.warning("CSV内容为空或所有行都包含NUL字符")
                return []
            
            # 重新构建CSV内容
            cleaned_csv = ''.join(lines)
            string_buffer = StringIO(cleaned_csv)
            
            # 解析CSV
            csv_reader = csv.DictReader(string_buffer)
            for row_num, row in enumerate(csv_reader, 1):
                try:
                    # 清理和标准化数据
                    cleaned_row = {}
                    for key, value in row.items():
                        if key:  # 确保键不为空
                            if isinstance(value, str):
                                cleaned_row[key.strip()] = value.strip()
                            else:
                                cleaned_row[key.strip()] = value
                    
                    # 确保有有效数据
                    if any(cleaned_row.values()):
                        result.append(cleaned_row)
                    else:
                        logger.warning(f"跳过空行 {row_num}")
                        
                except Exception as e:
                    logger.warning(f"解析行 {row_num} 失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"CSV解析失败: {e}")
            return []
        
        logger.info(f"CSV解析完成: {len(result)} 条有效记录")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取解析统计数据"""
        return {
            'total_records': self.stats['total_records'],
            'companies_processed': self.stats['companies_processed'],
            'events_processed': self.stats['events_processed'],
            'investors_processed': self.stats['investors_processed'],
            'errors': self.stats['errors']
        }
    
    def reset_stats(self):
        """重置解析统计数据"""
        self.stats = {
            'total_records': 0,
            'companies_processed': 0,
            'events_processed': 0,
            'investors_processed': 0,
            'errors': 0
        }