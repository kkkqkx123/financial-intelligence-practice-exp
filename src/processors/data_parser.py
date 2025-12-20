"""
硬编码数据解析器 - 基于实际数据集结构的优化实现
"""

import csv
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

# 设置日志级别为ERROR，减少日志输出
logger.setLevel(logging.ERROR)


class DataParser:
    """硬编码优先的数据解析器"""
    
    def __init__(self):
        self.stats = {
            'total_records': 0,
            'companies_processed': 0,
            'events_processed': 0,
            'investors_processed': 0,
            'structures_processed': 0,
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
                        # 处理可能为列表的字段
                        short_name = item.get('名称', item.get('short_name', item.get('name', '')))
                        if isinstance(short_name, list):
                            short_name = str(short_name[0]) if short_name else ''
                        else:
                            short_name = str(short_name) if short_name else ''
                        
                        full_name = item.get('公司名称', item.get('full_name', item.get('name', '')))
                        if isinstance(full_name, list):
                            full_name = str(full_name[0]) if full_name else ''
                        else:
                            full_name = str(full_name) if full_name else ''
                        
                        description = item.get('公司介绍', item.get('description', ''))
                        if isinstance(description, list):
                            description = str(description[0]) if description else ''
                        else:
                            description = str(description) if description else ''
                        
                        registration_name = item.get('工商', item.get('registration_name', item.get('name', '')))
                        if isinstance(registration_name, list):
                            registration_name = str(registration_name[0]) if registration_name else ''
                        else:
                            registration_name = str(registration_name) if registration_name else ''
                        
                        address = item.get('地址') or item.get('contact_info', {}).get('address')
                        
                        registration_id = item.get('工商注册id', item.get('registration_id', ''))
                        if isinstance(registration_id, list):
                            registration_id = str(registration_id[0]) if registration_id else ''
                        else:
                            registration_id = str(registration_id) if registration_id else ''
                        
                        establish_date = item.get('成立时间', item.get('establish_date') or item.get('registration_date'))
                        
                        legal_representative = item.get('法人代表', item.get('legal_representative', ''))
                        if isinstance(legal_representative, list):
                            legal_representative = str(legal_representative[0]) if legal_representative else ''
                        else:
                            legal_representative = str(legal_representative) if legal_representative else ''
                        
                        registered_capital = item.get('注册资金', item.get('registered_capital'))
                        
                        credit_code = item.get('统一信用代码', item.get('credit_code', ''))
                        if isinstance(credit_code, list):
                            credit_code = str(credit_code[0]) if credit_code else ''
                        else:
                            credit_code = str(credit_code) if credit_code else ''
                        
                        website = item.get('网址', item.get('website'))
                        
                        company = {
                            'short_name': short_name.strip(),
                            'full_name': full_name.strip(),
                            'description': description.strip(),
                            'registration_name': registration_name.strip(),
                            'address': address,
                            'registration_id': registration_id.strip(),
                            'establish_date': establish_date,
                            'legal_representative': legal_representative.strip(),
                            'registered_capital': registered_capital,
                            'credit_code': credit_code.strip(),
                            'website': website,
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
        
        # 如果是字符串，使用原有的解析逻辑
        if isinstance(data, str):
            logger.info("处理字符串格式数据")
            text = str(data)
            lines = text.strip().split('\n')
            logger.info(f"字符串数据共 {len(lines)} 行")
            
            # 跳过统计行和表头
            data_started = False
            line_count = 0
            for line_num, line in enumerate(lines, 1):
                if line.startswith('名称,公司名称,公司介绍,工商'):
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
            logger.info(f"处理字典列表数据，共 {len(data)} 条记录")
            for i, item in enumerate(data):
                try:
                    if isinstance(item, dict):
                        # 处理可能为列表的字段
                        description = item.get('事件资讯', item.get('description', item.get('event', '')))
                        if isinstance(description, list):
                            description = str(description[0]) if description else ''
                        else:
                            description = str(description) if description else ''
                        
                        # 处理投资方字段，确保返回列表
                        investors = item.get('投资方', item.get('investors', item.get('investment_partners', [])))
                        if investors is None:
                            investors = []
                        elif isinstance(investors, str):
                            # 如果是字符串，尝试解析为列表
                            investors = self._parse_investors(investors)
                        elif isinstance(investors, list):
                            # 如果是列表，确保每个元素都是字符串
                            investors = [str(inv) if inv is not None else '' for inv in investors]
                        else:
                            # 其他类型转换为字符串再解析
                            investors = self._parse_investors(str(investors))
                        
                        # 处理融资方字段，确保返回字符串
                        investee = item.get('融资方', item.get('investee', item.get('company', '')))
                        if investee is None:
                            investee = ''
                        elif isinstance(investee, list):
                            investee = str(investee[0]) if investee else ''
                        else:
                            investee = str(investee)
                        investee = investee.strip()
                        
                        investment_date = item.get('融资时间', item.get('investment_date') or item.get('date'))
                        if isinstance(investment_date, list):
                            investment_date = str(investment_date[0]) if investment_date else ''
                        
                        round_field = item.get('轮次', item.get('round', item.get('funding_round', '')))
                        if isinstance(round_field, list):
                            round_field = str(round_field[0]) if round_field else ''
                        else:
                            round_field = str(round_field) if round_field else ''
                        
                        amount = item.get('金额', item.get('amount', item.get('funding_amount')))
                        if isinstance(amount, list):
                            amount = str(amount[0]) if amount else ''
                        else:
                            amount = str(amount) if amount else ''
                        
                        event = {
                            'description': description.strip(),
                            'investors': investors,
                            'investee': investee,
                            'investment_date': investment_date,
                            'round': round_field.strip(),
                            'amount': amount.strip(),
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        events.append(event)
                        self.stats['events_processed'] += 1
                        self.stats['total_records'] += 1
                    else:
                        logger.warning(f"跳过非字典项 {i}: {type(item)}")
                        self.stats['errors'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析投资事件失败 (索引 {i}): {str(item)[:100]}... 错误: {e}")
            logger.info(f"字典列表处理完成，成功: {len(events)} 条，失败: {self.stats['errors']} 条")
            # 不应用字段映射，因为已经使用了构建器期望的英文键名
            return events
        
        # 如果是字符串，使用原有的解析逻辑
        text = str(data)
        lines = text.strip().split('\n')
        print(f"投资事件文本行数: {len(lines)}")
        print(f"第一行内容: {lines[0] if lines else '无'}")
        
        data_started = False
        for i, line in enumerate(lines):
            if line.startswith('事件资讯,投资方,融资方,融资时间,轮次,金额'):
                data_started = True
                print("找到投资事件表头")
                continue
            if not data_started or not line.strip():
                continue
            
            print(f"处理第 {i} 行: {line[:50]}...")
            try:
                fields = self._smart_split(line)
                print(f"分割后字段数: {len(fields)}")
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
                    print(f"成功解析投资事件: {event['description'][:30]}...")
                else:
                    print(f"字段数不足: {len(fields)} < 6")
            except Exception as e:
                self.stats['errors'] += 1
                print(f"解析投资事件失败: {line[:50]}... 错误: {e}")
        
        # 应用字段映射，转换为构建器期望的格式
        logger.info("应用投资事件字段映射")
        mapped_events = apply_field_mapping(events, INVESTMENT_EVENT_FIELD_MAPPING)
        logger.info(f"字段映射完成，共 {len(mapped_events)} 条记录")
        
        return mapped_events
    
    # ==================== 投资机构解析 ====================
    
    def parse_investment_institutions(self, data: Union[str, List[Dict]]) -> List[Dict]:
        """解析投资机构数据"""
        institutions = []
        logger.info(f"开始解析投资机构数据，输入类型: {type(data)}")
        
        # 如果是列表，说明是已经解析好的字典数据
        if isinstance(data, list):
            logger.info(f"处理字典列表数据，共 {len(data)} 条记录")
            for i, item in enumerate(data):
                try:
                    if isinstance(item, dict):
                        # 处理可能为列表的字段
                        name = item.get('机构名称', item.get('name', item.get('institution_name', '')))
                        if isinstance(name, list):
                            name = str(name[0]) if name else ''
                        else:
                            name = str(name) if name else ''
                        
                        description = item.get('介绍', item.get('description', item.get('introduction', '')))
                        if isinstance(description, list):
                            description = str(description[0]) if description else ''
                        else:
                            description = str(description) if description else ''
                        
                        industries = item.get('行业', item.get('industries', item.get('sectors', [])))
                        scale = item.get('规模', item.get('scale', item.get('fund_size')))
                        preferred_rounds = item.get('轮次', item.get('preferred_rounds', item.get('investment_stages', [])))
                        
                        institution = {
                            'name': name.strip(),
                            'description': description.strip(),
                            'industries': industries,
                            'scale': scale,
                            'preferred_rounds': preferred_rounds,
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        institutions.append(institution)
                        self.stats['investors_processed'] += 1
                        self.stats['total_records'] += 1
                    else:
                        logger.warning(f"跳过非字典项 {i}: {type(item)}")
                        self.stats['errors'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析投资机构数据失败 (索引 {i}): {str(item)[:100]}... 错误: {e}")
            logger.info(f"字典列表处理完成，成功: {len(institutions)} 条，失败: {self.stats['errors']} 条")
            
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
    
    # ==================== 投资结构解析 ====================
    
    def parse_investment_structure(self, data: Union[str, List[Dict]]) -> List[Dict]:
        """解析投资结构数据"""
        structures = []
        logger.info(f"开始解析投资结构数据，输入类型: {type(data)}")
        
        # 如果是列表，说明是已经解析好的字典数据
        if isinstance(data, list):
            logger.info(f"处理字典列表数据，共 {len(data)} 条记录")
            for i, item in enumerate(data):
                try:
                    if isinstance(item, dict):
                        # 处理可能为列表的字段
                        name = item.get('机构名称', item.get('name', ''))
                        if isinstance(name, list):
                            name = str(name[0]) if name else ''
                        else:
                            name = str(name) if name else ''
                        
                        description = item.get('介绍', item.get('description', ''))
                        if isinstance(description, list):
                            description = str(description[0]) if description else ''
                        else:
                            description = str(description) if description else ''
                        
                        industries = item.get('行业', item.get('industries', ''))
                        if isinstance(industries, list):
                            industries = ', '.join(str(ind) for ind in industries if ind) if industries else ''
                        else:
                            industries = str(industries) if industries else ''
                        
                        scale = item.get('规模', item.get('scale', ''))
                        if isinstance(scale, list):
                            scale = str(scale[0]) if scale else ''
                        else:
                            scale = str(scale) if scale else ''
                        
                        rounds = item.get('轮次', item.get('rounds', ''))
                        if isinstance(rounds, list):
                            rounds = ', '.join(str(r) for r in rounds if r) if rounds else ''
                        else:
                            rounds = str(rounds) if rounds else ''
                        
                        structure = {
                            'name': name.strip(),
                            'description': description.strip(),
                            'industries': industries.strip(),
                            'scale': scale.strip(),
                            'rounds': rounds.strip(),
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        structures.append(structure)
                        self.stats['structures_processed'] = self.stats.get('structures_processed', 0) + 1
                        self.stats['total_records'] += 1
                    else:
                        logger.warning(f"跳过非字典项 {i}: {type(item)}")
                        self.stats['errors'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析投资结构数据失败 (索引 {i}): {str(item)[:100]}... 错误: {e}")
            logger.info(f"字典列表处理完成，成功: {len(structures)} 条，失败: {self.stats['errors']} 条")
        
        # 如果是字符串，使用原有的解析逻辑
        if isinstance(data, str):
            logger.info("处理字符串格式数据")
            text = str(data)
            lines = text.strip().split('\n')
            logger.info(f"字符串数据共 {len(lines)} 行")
            
            # 跳过统计行和表头
            data_started = False
            line_count = 0
            for line_num, line in enumerate(lines, 1):
                if line.startswith('机构名称,介绍,行业,规模,轮次'):
                    data_started = True
                    logger.info(f"在第 {line_num} 行找到表头")
                    continue
                if not data_started or not line.strip():
                    continue
                
                line_count += 1
                try:
                    fields = self._smart_split(line)
                    logger.debug(f"第 {line_num} 行分割为 {len(fields)} 个字段")
                    
                    if len(fields) >= 5:
                        structure = {
                            'name': fields[0].strip(),
                            'description': fields[1].strip() if fields[1].strip() else None,
                            'industries': fields[2].strip() if fields[2].strip() else None,
                            'scale': fields[3].strip() if fields[3].strip() else None,
                            'rounds': fields[4].strip() if fields[4].strip() else None,
                            'parsed_by': 'hardcoded',
                            'parse_confidence': 1.0
                        }
                        structures.append(structure)
                        self.stats['structures_processed'] = self.stats.get('structures_processed', 0) + 1
                        self.stats['total_records'] += 1
                    else:
                        logger.warning(f"第 {line_num} 行字段数不足: {len(fields)} < 5")
                        self.stats['errors'] += 1
                        
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"解析投资结构数据失败 (第 {line_num} 行): {line[:100]}... 错误: {e}")
            
            logger.info(f"字符串格式处理完成，成功: {len(structures)} 条，失败: {self.stats['errors']} 条，处理行数: {line_count}")
        
        # 应用字段映射，转换为构建器期望的格式
        logger.info("应用投资结构字段映射")
        # 注意：这里可能需要创建一个新的字段映射
        # mapped_structures = apply_field_mapping(structures, INVESTMENT_STRUCTURE_FIELD_MAPPING)
        mapped_structures = structures  # 暂时使用原始数据
        logger.info(f"字段映射完成，共 {len(mapped_structures)} 条记录")
        
        return mapped_structures

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
    
    def _parse_date(self, date_str: Union[str, List[str]]) -> Optional[str]:
        """日期格式标准化"""
        if not date_str:
            return None
        
        # 处理列表类型输入
        if isinstance(date_str, list):
            if not date_str:
                return None
            # 只取第一个非空日期
            for date in date_str:
                if date and str(date).strip():
                    date_str = str(date)
                    break
            else:
                return None
        else:
            date_str = str(date_str)
        
        if date_str.strip() == '':
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
    
    def _normalize_capital(self, capital_str: Union[str, List[str]]) -> Optional[float]:
        """注册资本标准化"""
        if not capital_str:
            return None
        
        # 处理列表类型输入
        if isinstance(capital_str, list):
            if not capital_str:
                return None
            # 只取第一个非空值
            for capital in capital_str:
                if capital and str(capital).strip():
                    capital_str = str(capital)
                    break
            else:
                return None
        else:
            capital_str = str(capital_str)
        
        if capital_str.strip() == '':
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
    
    def _normalize_amount(self, amount_str: Union[str, List[str]]) -> Optional[float]:
        """投资金额标准化 - 增强处理，支持更多样化的金额表达方式"""
        if not amount_str:
            return None
        
        # 处理列表类型输入
        if isinstance(amount_str, list):
            if not amount_str:
                return None
            # 只取第一个非空值
            for amount in amount_str:
                if amount and str(amount).strip():
                    amount_str = str(amount)
                    break
            else:
                return None
        else:
            amount_str = str(amount_str)
        
        if amount_str.strip() == '' or amount_str == '未披露':
            return None
        
        amount_str = amount_str.strip()
        
        # 处理相对金额表达，如"过千万"、"数千万"、"近亿"等
        if '过千万' in amount_str or '近千万' in amount_str:
            return 1000 * 10000  # 1000万，作为"过千万"的保守估计
        elif '过亿' in amount_str or '近亿' in amount_str:
            return 1 * 10000 * 10000  # 1亿，作为"近亿"的保守估计
        
        # 处理模糊金额
        if '数千万' in amount_str:
            base_amount = 3000 * 10000  # 3000万，作为"数千万"的中等估计
        elif '数百万' in amount_str:
            base_amount = 5000000   # 默认500万
        elif '数十万' in amount_str:
            base_amount = 500000    # 默认50万
        else:
            base_amount = None
        
        # 处理"约"、"左右"、"大概"等模糊表达
        if '约' in amount_str or '左右' in amount_str or '大概' in amount_str:
            # 移除这些模糊词汇，然后进行正常解析
            amount_str = re.sub(r'约|左右|大概', '', amount_str).strip()
        
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
    
    def _normalize_round(self, round_str: Union[str, List[str]]) -> Optional[str]:
        """投资轮次标准化"""
        if not round_str:
            return None
        
        # 处理列表类型输入
        if isinstance(round_str, list):
            if not round_str:
                return None
            # 只取第一个非空值
            for round_val in round_str:
                if round_val and str(round_val).strip():
                    round_str = str(round_val)
                    break
            else:
                return None
        else:
            round_str = str(round_str)
        
        round_str = round_str.strip()
        return ROUND_MAPPING.get(round_str, round_str)
    
    def _normalize_website(self, website: Union[str, List[str]]) -> Optional[str]:
        """网站地址标准化"""
        if not website:
            return None
        
        # 处理列表类型输入
        if isinstance(website, list):
            if not website:
                return None
            # 只取第一个非空值
            for site in website:
                if site and str(site).strip():
                    website = str(site)
                    break
            else:
                return None
        else:
            website = str(website)
        
        if website.strip() == '':
            return None
        
        website = website.strip()
        
        # 添加协议前缀
        if not website.startswith(('http://', 'https://')):
            website = 'http://' + website
        
        # 基本格式验证
        if re.match(r'^https?://[\w\-]+(\.[\w\-]+)+[/#?]?.*$', website):
            return website
        
        return None
    
    def _parse_investors(self, investors_str: Union[str, List[str]]) -> List[str]:
        """解析投资方列表 - 增强处理，支持多种分隔符和格式"""
        if not investors_str:
            return []
        
        # 处理列表类型输入
        if isinstance(investors_str, list):
            if not investors_str:
                return []
            # 将列表转换为字符串
            investors_list = []
            for inv in investors_str:
                if inv and str(inv).strip():
                    investors_list.append(str(inv).strip())
            return investors_list
        
        investors_str = str(investors_str)
        
        if investors_str.strip() == '' or investors_str.strip() == '未披露机构':
            return []
        
        investors_str = investors_str.strip()
        
        # 特殊处理：如果包含"等"字，可能是多个机构的合并表示
        if '等' in investors_str and '、' not in investors_str and ' ' not in investors_str:
            # 例如"红杉资本等" -> ["红杉资本"]
            investors_str = investors_str.replace('等', '')
        
        # 支持多种分隔符：空格、顿号、逗号、分号
        # 注意：空格分隔符需要特别处理，因为机构名称中可能包含空格
        # 优先使用更明确的分隔符
        primary_separators = ['、', ',', '，', ';', '；']
        
        # 首先尝试使用主要分隔符分割
        for sep in primary_separators:
            if sep in investors_str:
                investors = [inv.strip() for inv in investors_str.split(sep) if inv.strip()]
                return investors
        
        # 如果没有明确分隔符，但包含空格，则尝试智能分割
        if ' ' in investors_str:
            # 尝试基于常见投资机构名称模式进行分割
            # 例如："红杉资本 经纬中国" -> ["红杉资本", "经纬中国"]
            # 但要注意避免错误分割如"中国平安"这样的名称
            
            # 常见投资机构后缀列表
            common_suffixes = ['资本', '创投', '投资', '基金', '资产', '集团', '控股', '科技', '产业']
            
            # 尝试基于后缀分割
            investors = []
            current_investor = ""
            
            for word in investors_str.split():
                if current_investor:
                    current_investor += " " + word
                else:
                    current_investor = word
                
                # 检查当前词是否是常见后缀，如果是且当前投资者不为空，则分割
                if any(suffix in word for suffix in common_suffixes):
                    investors.append(current_investor)
                    current_investor = ""
            
            # 添加最后一个投资者（如果有）
            if current_investor:
                investors.append(current_investor)
            
            # 如果分割结果合理（至少2个机构），则使用分割结果
            if len(investors) >= 2:
                return [inv.strip() for inv in investors if inv.strip()]
        
        # 如果以上方法都不适用，则返回整个字符串作为单个投资机构
        return [investors_str] if investors_str else []
    
    def _parse_industries(self, industries_str: Union[str, List[str]]) -> List[str]:
        """解析行业列表 - 增强处理，支持'企业服务36家 文娱内容游戏9家 金融9家'等格式"""
        if not industries_str:
            return []
        
        # 处理列表类型输入
        if isinstance(industries_str, list):
            if not industries_str:
                return []
            # 将列表转换为字符串列表
            industries_list = []
            for ind in industries_str:
                if ind and str(ind).strip():
                    industries_list.append(str(ind).strip())
            return industries_list
        
        industries_str = str(industries_str)
        
        if industries_str.strip() == '':
            return []
        
        industries_str = industries_str.strip()
        
        # 处理类似"企业服务36家 文娱内容游戏9家 金融9家"的格式
        # 使用正则表达式匹配行业名称和数量
        industry_matches = re.findall(r'([^0-9]+)(\d+)家', industries_str)
        
        if industry_matches:
            # 提取行业名称
            industries = [match[0].strip() for match in industry_matches]
            return industries
        
        # 处理其他分隔符情况
        separators = [' ', '、', ',', '，', ';', '；', '|']
        
        # 尝试各种分隔符
        for sep in separators:
            if sep in industries_str:
                industries = [ind.strip() for ind in industries_str.split(sep) if ind.strip()]
                return industries
        
        # 如果没有分隔符，返回整个字符串作为单个行业
        return [industries_str]
    
    def _parse_preferred_rounds(self, rounds_str: Union[str, List[str]]) -> List[str]:
        """解析偏好轮次列表 - 增强处理，支持'A轮56家 B轮12家'等格式"""
        if not rounds_str:
            return []
        
        # 处理列表类型输入
        if isinstance(rounds_str, list):
            if not rounds_str:
                return []
            # 将列表转换为字符串列表
            rounds_list = []
            for r in rounds_str:
                if r and str(r).strip():
                    rounds_list.append(str(r).strip())
            return rounds_list
        
        rounds_str = str(rounds_str)
        
        if rounds_str.strip() == '':
            return []
        
        rounds_str = rounds_str.strip()
        
        # 处理类似"A轮56家、B轮47家、C轮36家"的格式
        # 使用正则表达式匹配轮次名称和数量
        round_matches = re.findall(r'([^0-9]+)(\d+)家', rounds_str)
        
        if round_matches:
            # 提取轮次名称
            rounds = [match[0].strip() for match in round_matches]
            return rounds
        
        # 处理其他分隔符情况
        separators = [' ', '、', ',', '，', ';', '；', '|']
        
        # 尝试各种分隔符
        for sep in separators:
            if sep in rounds_str:
                rounds = [r.strip() for r in rounds_str.split(sep) if r.strip()]
                return rounds
        
        # 如果没有分隔符，返回整个字符串作为单个轮次
        return [rounds_str]
    
    def _normalize_scale(self, scale_str: Union[str, List[str]]) -> Optional[str]:
        """管理规模标准化"""
        if not scale_str:
            return None
        
        # 处理列表类型输入
        if isinstance(scale_str, list):
            if not scale_str:
                return None
            # 只取第一个非空值
            for s in scale_str:
                if s and str(s).strip():
                    scale_str = str(s)
                    break
            else:
                return None
        else:
            scale_str = str(scale_str)
        
        if scale_str.strip() == '':
            return None
        
        scale_str = scale_str.strip()
        
        # 提取数字和单位
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)', scale_str)
        if match:
            amount, unit = match.groups()
            return f"{amount}{unit}"
        
        return scale_str
    
    def parse_csv_data(self, csv_content: str) -> List[Dict]:
        """解析CSV数据内容 - 支持包含NUL字符的文件，增强错误处理"""
        import csv
        from io import StringIO
        
        result = []
        error_rows = []
        empty_rows = 0
        
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
                    # 检查行是否为空或只有空值
                    if not row or all(not value or not str(value).strip() for value in row.values()):
                        empty_rows += 1
                        logger.debug(f"跳过空行 {row_num}")
                        continue
                    
                    # 清理和标准化数据
                    cleaned_row = {}
                    has_valid_data = False
                    
                    for key, value in row.items():
                        if key:  # 确保键不为空
                            clean_key = key.strip()
                            if isinstance(value, str):
                                clean_value = value.strip()
                                # 检查是否有有效内容
                                if clean_value:
                                    has_valid_data = True
                                cleaned_row[clean_key] = clean_value
                            else:
                                if value is not None:
                                    has_valid_data = True
                                cleaned_row[clean_key] = value
                    
                    # 确保有有效数据
                    if has_valid_data:
                        result.append(cleaned_row)
                    else:
                        empty_rows += 1
                        logger.debug(f"跳过无有效数据的行 {row_num}")
                        
                except Exception as e:
                    error_rows.append((row_num, str(e)))
                    logger.warning(f"解析行 {row_num} 失败，已跳过: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"CSV解析失败: {e}")
            return []
        
        # 记录解析统计信息
        total_lines = len(result) + len(error_rows) + empty_rows
        logger.info(f"CSV解析完成: 总行数={total_lines}, 有效记录={len(result)}, 空行={empty_rows}, 错误行={len(error_rows)}")
        
        # 记录错误行的详细信息（仅在调试模式下）
        if error_rows and logger.isEnabledFor(logging.DEBUG):
            logger.debug("错误行详情:")
            for row_num, error in error_rows[:5]:  # 只显示前5个错误
                logger.debug(f"  行 {row_num}: {error}")
            if len(error_rows) > 5:
                logger.debug(f"  ... 还有 {len(error_rows) - 5} 个错误")
        
        # 如果错误行过多，发出警告
        if len(error_rows) > 0 and len(result) > 0:
            error_rate = len(error_rows) / (len(result) + len(error_rows))
            if error_rate > 0.1:  # 错误率超过10%
                logger.warning(f"CSV解析错误率较高: {error_rate:.2%} ({len(error_rows)}/{len(result) + len(error_rows)})")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取解析统计数据"""
        return {
            'total_records': self.stats['total_records'],
            'companies_processed': self.stats['companies_processed'],
            'events_processed': self.stats['events_processed'],
            'investors_processed': self.stats['investors_processed'],
            'structures_processed': self.stats.get('structures_processed', 0),
            'errors': self.stats['errors']
        }
    
    def reset_stats(self):
        """重置解析统计数据"""
        self.stats = {
            'total_records': 0,
            'companies_processed': 0,
            'events_processed': 0,
            'investors_processed': 0,
            'structures_processed': 0,
            'errors': 0
        }