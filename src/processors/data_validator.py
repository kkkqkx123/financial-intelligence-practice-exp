"""
数据验证和质量检查系统
提供全面的数据质量验证功能
"""

import re
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter

from .config import CONFIDENCE_THRESHOLDS, VALIDATION_RULES

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.validation_rules = VALIDATION_RULES
        self.validation_stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings': 0,
            'errors': []
        }
    
    def validate_company_data(self, companies: List[Dict]) -> Dict:
        """验证公司数据"""
        logger.info(f"开始验证公司数据，共 {len(companies)} 条记录")
        
        validation_results: Dict[str, Any] = {
            'total_records': len(companies),
            'valid_records': 0,  # type: ignore[assignment]
            'invalid_records': 0,  # type: ignore[assignment]
            'field_statistics': {},
            'validation_errors': [],  # type: ignore[assignment]
            'data_quality_score': 0.0
        }
        
        field_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'filled': 0, 'empty': 0, 'valid': 0, 'invalid': 0})
        
        for idx, company in enumerate(companies):
            record_valid = True
            record_errors = []
            
            # 验证公司名称
            name = company.get('公司名称', '')
            if not name or len(name.strip()) < 2:
                record_valid = False
                record_errors.append(f"记录 {idx+1}: 公司名称无效")
            field_stats['公司名称']['filled' if name else 'empty'] += 1
            
            # 验证工商注册ID
            reg_id = company.get('工商注册id', '')
            if reg_id and not self._is_valid_registration_id(reg_id):
                record_errors.append(f"记录 {idx+1}: 工商注册ID格式无效")
            field_stats['工商注册id']['filled' if reg_id else 'empty'] += 1
            field_stats['工商注册id']['valid' if self._is_valid_registration_id(reg_id) else 'invalid'] += 1
            
            # 验证统一信用代码
            credit_code = company.get('统一信用代码', '')
            if credit_code and not self._is_valid_credit_code(credit_code):
                record_errors.append(f"记录 {idx+1}: 统一信用代码格式无效")
            field_stats['统一信用代码']['filled' if credit_code else 'empty'] += 1
            field_stats['统一信用代码']['valid' if self._is_valid_credit_code(credit_code) else 'invalid'] += 1
            
            # 验证注册资金
            capital = company.get('注册资金', '')
            if capital and not self._is_valid_capital(capital):
                record_errors.append(f"记录 {idx+1}: 注册资金格式无效")
            field_stats['注册资金']['filled' if capital else 'empty'] += 1
            
            # 验证成立时间
            establish_date = company.get('成立时间', '')
            if establish_date and not self._is_valid_date(establish_date):
                record_errors.append(f"记录 {idx+1}: 成立时间格式无效")
            field_stats['成立时间']['filled' if establish_date else 'empty'] += 1
            
            # 验证联系方式
            website = company.get('网址', '')
            if website and not self._is_valid_url(website):
                record_errors.append(f"记录 {idx+1}: 网址格式无效")
            field_stats['网址']['filled' if website else 'empty'] += 1
            
            # 更新统计
            if record_valid:
                validation_results['valid_records'] += 1
            else:
                validation_results['invalid_records'] += 1
            
            validation_results['validation_errors'].extend(record_errors)
        
        validation_results['field_statistics'] = dict(field_stats)
        validation_results['data_quality_score'] = self._calculate_quality_score(validation_results)
        
        logger.info(f"公司数据验证完成: {validation_results['valid_records']}/{validation_results['total_records']} 有效")
        return validation_results
    
    def validate_investment_event_data(self, events: List[Dict]) -> Dict:
        """验证投资事件数据"""
        logger.info(f"开始验证投资事件数据，共 {len(events)} 条记录")
        
        validation_results: Dict[str, Any] = {
            'total_records': len(events),
            'valid_records': 0,  # type: ignore[assignment]
            'invalid_records': 0,  # type: ignore[assignment]
            'field_statistics': {},
            'validation_errors': [],  # type: ignore[assignment]
            'data_quality_score': 0.0,
            'amount_analysis': {},
            'round_analysis': {}
        }
        
        field_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'filled': 0, 'empty': 0, 'valid': 0, 'invalid': 0})
        amount_patterns: Counter[str] = Counter()
        round_patterns: Counter[str] = Counter()
        
        for idx, event in enumerate(events):
            record_valid = True
            record_errors = []
            
            # 验证投资方
            investor = event.get('投资方', '')
            if not investor or len(investor.strip()) < 2:
                record_errors.append(f"记录 {idx+1}: 投资方信息无效")
            field_stats['投资方']['filled' if investor else 'empty'] += 1
            
            # 验证融资方
            company = event.get('融资方', '')
            if not company or len(company.strip()) < 2:
                record_valid = False
                record_errors.append(f"记录 {idx+1}: 融资方信息无效")
            field_stats['融资方']['filled' if company else 'empty'] += 1
            
            # 验证融资时间
            date = event.get('融资时间', '')
            if date and not self._is_valid_date(date):
                record_errors.append(f"记录 {idx+1}: 融资时间格式无效")
            field_stats['融资时间']['filled' if date else 'empty'] += 1
            
            # 验证轮次
            round_info = event.get('轮次', '')
            if round_info:
                round_patterns[round_info] += 1
                if not self._is_valid_round(round_info):
                    record_errors.append(f"记录 {idx+1}: 轮次信息无效")
            field_stats['轮次']['filled' if round_info else 'empty'] += 1
            
            # 验证金额
            amount = event.get('金额', '')
            if amount:
                amount_patterns[amount] += 1
                if not self._is_valid_amount(amount):
                    record_errors.append(f"记录 {idx+1}: 金额格式无效")
            field_stats['金额']['filled' if amount else 'empty'] += 1
            
            # 更新统计
            if record_valid:
                validation_results['valid_records'] += 1
            else:
                validation_results['invalid_records'] += 1
            
            validation_results['validation_errors'].extend(record_errors)
        
        validation_results['field_statistics'] = dict(field_stats)
        validation_results['amount_analysis'] = dict(amount_patterns.most_common(10))
        validation_results['round_analysis'] = dict(round_patterns.most_common(10))
        validation_results['data_quality_score'] = self._calculate_quality_score(validation_results)
        
        logger.info(f"投资事件数据验证完成: {validation_results['valid_records']}/{validation_results['total_records']} 有效")
        return validation_results
    
    def validate_investor_data(self, investors: List[Dict]) -> Dict:
        """验证投资方数据"""
        logger.info(f"开始验证投资方数据，共 {len(investors)} 条记录")
        
        validation_results: Dict[str, Any] = {
            'total_records': len(investors),
            'valid_records': 0,  # type: ignore[assignment]
            'invalid_records': 0,  # type: ignore[assignment]
            'field_statistics': {},
            'validation_errors': [],  # type: ignore[assignment]
            'data_quality_score': 0.0
        }
        
        field_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'filled': 0, 'empty': 0, 'valid': 0, 'invalid': 0})
        
        for idx, investor in enumerate(investors):
            record_valid = True
            record_errors = []
            
            # 验证机构名称
            name = investor.get('机构名称', '')
            if not name or len(name.strip()) < 2:
                record_valid = False
                record_errors.append(f"记录 {idx+1}: 机构名称无效")
            field_stats['机构名称']['filled' if name else 'empty'] += 1
            
            # 验证介绍
            description = investor.get('介绍', '')
            if description and len(description.strip()) < 10:
                record_errors.append(f"记录 {idx+1}: 介绍信息过短")
            field_stats['介绍']['filled' if description else 'empty'] += 1
            
            # 验证行业
            industry = investor.get('行业', '')
            field_stats['行业']['filled' if industry else 'empty'] += 1
            
            # 验证规模
            scale = investor.get('规模', '')
            if scale and not self._is_valid_scale(scale):
                record_errors.append(f"记录 {idx+1}: 规模信息无效")
            field_stats['规模']['filled' if scale else 'empty'] += 1
            
            # 验证轮次
            rounds = investor.get('轮次', '')
            field_stats['轮次']['filled' if rounds else 'empty'] += 1
            
            # 更新统计
            if record_valid:
                validation_results['valid_records'] += 1
            else:
                validation_results['invalid_records'] += 1
            
            validation_results['validation_errors'].extend(record_errors)
        
        validation_results['field_statistics'] = dict(field_stats)
        validation_results['data_quality_score'] = self._calculate_quality_score(validation_results)
        
        logger.info(f"投资方数据验证完成: {validation_results['valid_records']}/{validation_results['total_records']} 有效")
        return validation_results
    
    def validate_knowledge_graph(self, kg_data: Dict) -> Dict:
        """验证知识图谱"""
        logger.info("开始验证知识图谱")
        
        companies = kg_data.get('entities', {}).get('companies', {})
        investors = kg_data.get('entities', {}).get('investors', {})
        relationships = kg_data.get('relationships', [])
        
        validation_results: Dict[str, Any] = {
            'entity_validation': {},
            'relationship_validation': {},
            'consistency_checks': {},
            'data_integrity': {},
            'recommendations': []  # type: ignore[arg-type]
        }
        
        # 实体验证
        validation_results['entity_validation'] = {
            'total_companies': len(companies),
            'total_investors': len(investors),
            'companies_with_ids': sum(1 for c in companies.values() if c.get('id')),
            'investors_with_ids': sum(1 for i in investors.values() if i.get('id')),
            'companies_with_names': sum(1 for c in companies.values() if c.get('name')),
            'investors_with_names': sum(1 for i in investors.values() if i.get('name'))
        }
        
        # 关系验证
        valid_relationships = 0
        orphaned_relationships = 0
        relationship_types: Counter[str] = Counter()
        
        for rel in relationships:
            if rel.get('investor_id') and rel.get('company_id'):
                # 检查ID是否存在
                investor_exists = rel['investor_id'] in investors
                company_exists = rel['company_id'] in companies
                
                if investor_exists and company_exists:
                    valid_relationships += 1
                else:
                    orphaned_relationships += 1
            
            rel_type = rel.get('type', 'unknown')
            relationship_types[rel_type] += 1
        
        validation_results['relationship_validation'] = {
            'total_relationships': len(relationships),
            'valid_relationships': valid_relationships,
            'orphaned_relationships': orphaned_relationships,
            'relationship_types': dict(relationship_types)
        }
        
        # 一致性检查
        validation_results['consistency_checks'] = {
            'id_consistency': self._check_id_consistency(companies, investors, relationships),
            'name_consistency': self._check_name_consistency(companies, investors),
            'date_consistency': self._check_date_consistency(relationships)
        }
        
        # 数据完整性
        validation_results['data_integrity'] = {
            'missing_descriptions': {
                'companies': sum(1 for c in companies.values() if not c.get('description')),
                'investors': sum(1 for i in investors.values() if not i.get('description'))
            },
            'low_confidence_entities': {
                'companies': sum(1 for c in companies.values() 
                               if c.get('metadata', {}).get('confidence', 0) < CONFIDENCE_THRESHOLDS['entity_linking']),
                'investors': sum(1 for i in investors.values() 
                               if i.get('metadata', {}).get('confidence', 0) < CONFIDENCE_THRESHOLDS['entity_linking'])
            }
        }
        
        # 生成建议
        if validation_results['data_integrity']['missing_descriptions']['companies'] > len(companies) * 0.5:
            validation_results['recommendations'].append("大量公司缺少描述信息，建议增强")
        
        if validation_results['data_integrity']['missing_descriptions']['investors'] > len(investors) * 0.5:
            validation_results['recommendations'].append("大量投资方缺少描述信息，建议增强")
        
        if orphaned_relationships > 0:
            validation_results['recommendations'].append(f"发现 {orphaned_relationships} 个孤立关系，需要清理")
        
        logger.info("知识图谱验证完成")
        return validation_results
    
    def _is_valid_registration_id(self, reg_id: str) -> bool:
        """验证工商注册ID格式"""
        if not reg_id:
            return True  # 空值视为有效（可选字段）
        
        # 工商注册号通常是15位数字
        return bool(re.match(r'^\d{15}$', reg_id))
    
    def _is_valid_credit_code(self, credit_code: str) -> bool:
        """验证统一信用代码格式"""
        if not credit_code:
            return True  # 空值视为有效（可选字段）
        
        # 统一信用代码格式：18位，第一位为数字或字母，其余为数字和字母
        return bool(re.match(r'^[0-9A-HJ-NPQRTUWXY]{2}\d{6}[0-9A-HJ-NPQRTUWXY]{10}$', credit_code))
    
    def _is_valid_capital(self, capital: str) -> bool:
        """验证注册资金格式"""
        if not capital:
            return True  # 空值视为有效（可选字段）
        
        # 支持格式：100万人民币，100万元，100万，100万美元等
        return bool(re.match(r'^\d+(\.\d+)?\s*(万|亿)?\s*(人民币|美元|元)?$', capital))
    
    def _is_valid_date(self, date_str: str) -> bool:
        """验证日期格式"""
        if not date_str:
            return True  # 空值视为有效（可选字段）
        
        # 支持多种日期格式
        date_patterns = [
            r'^\d{4}-\d{1,2}-\d{1,2}$',  # 2023-01-01
            r'^\d{4}/\d{1,2}/\d{1,2}$',  # 2023/01/01
            r'^\d{4}年\d{1,2}月\d{1,2}日$',  # 2023年1月1日
            r'^\d{4}\.\d{1,2}\.\d{1,2}$',  # 2023.01.01
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, date_str):
                return True
        
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """验证URL格式"""
        if not url:
            return True  # 空值视为有效（可选字段）
        
        # 简单的URL验证
        return bool(re.match(r'^https?://[\w\-._~:/?#\[\]@!$&\'()*+,;=%]+$', url))
    
    def _is_valid_amount(self, amount: str) -> bool:
        """验证投资金额格式"""
        if not amount or amount == '未披露':
            return True
        
        # 支持格式：100万人民币，100万美元，数千万人民币等
        return bool(re.match(r'^(\d+(\.\d+)?|数\d+)(万|亿)?\s*(人民币|美元)?$|^未披露$', amount))
    
    def _is_valid_round(self, round_str: str) -> bool:
        """验证投资轮次格式"""
        if not round_str:
            return True  # 空值视为有效（可选字段）
        
        valid_rounds = ['种子轮', '天使轮', 'Pre-A', 'A轮', 'A+轮', 'B轮', 'B+轮', 
                       'C轮', 'D轮', 'E轮', 'F轮', 'Pre-IPO', 'IPO', '战略融资', 
                       '并购', '股权投资', '债权融资', '新三板', '其他']
        
        return any(round_name in round_str for round_name in valid_rounds)
    
    def _is_valid_scale(self, scale: str) -> bool:
        """验证规模格式"""
        if not scale:
            return True  # 空值视为有效（可选字段）
        
        # 支持格式：10亿人民币，10亿美元等
        return bool(re.match(r'^\d+(\.\d+)?\s*(万|亿)?\s*(人民币|美元)$', scale))
    
    def _check_id_consistency(self, companies: Dict, investors: Dict, relationships: List) -> Dict:
        """检查ID一致性"""
        company_ids = set(companies.keys())
        investor_ids = set(investors.keys())
        
        missing_investor_ids = set()
        missing_company_ids = set()
        
        for rel in relationships:
            inv_id = rel.get('investor_id')
            comp_id = rel.get('company_id')
            
            if inv_id and inv_id not in investor_ids:
                missing_investor_ids.add(inv_id)
            
            if comp_id and comp_id not in company_ids:
                missing_company_ids.add(comp_id)
        
        return {
            'missing_investor_ids': list(missing_investor_ids),
            'missing_company_ids': list(missing_company_ids),
            'total_missing': len(missing_investor_ids) + len(missing_company_ids)
        }
    
    def _check_name_consistency(self, companies: Dict, investors: Dict) -> Dict:
        """检查名称一致性"""
        company_names = set()
        duplicate_company_names = set()
        
        for company in companies.values():
            name = company.get('name')
            if name in company_names:
                duplicate_company_names.add(name)
            else:
                company_names.add(name)
        
        investor_names = set()
        duplicate_investor_names = set()
        
        for investor in investors.values():
            name = investor.get('name')
            if name in investor_names:
                duplicate_investor_names.add(name)
            else:
                investor_names.add(name)
        
        return {
            'duplicate_company_names': list(duplicate_company_names),
            'duplicate_investor_names': list(duplicate_investor_names),
            'total_duplicates': len(duplicate_company_names) + len(duplicate_investor_names)
        }
    
    def _check_date_consistency(self, relationships: List) -> Dict:
        """检查日期一致性"""
        invalid_dates: List[str] = []
        future_dates: List[str] = []
        
        for rel in relationships:
            date_str = rel.get('date', '')
            if date_str:
                try:
                    # 尝试解析日期
                    if self._is_valid_date(date_str):
                        # 这里可以添加更复杂的日期验证逻辑
                        pass
                except Exception as e:
                    invalid_dates.append(date_str)
        
        return {
            'invalid_dates': invalid_dates,
            'future_dates': future_dates,
            'total_date_issues': len(invalid_dates) + len(future_dates)
        }
    
    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """计算数据质量分数"""
        total_records: int = validation_results.get('total_records', 0)
        valid_records: int = validation_results.get('valid_records', 0)
        
        if total_records == 0:
            return 0.0
        
        # 基础分数：有效记录比例
        base_score: float = valid_records / total_records
        
        # 字段完整性加分
        field_stats: Dict = validation_results.get('field_statistics', {})
        completeness_bonus: float = 0.0
        
        for field, stats in field_stats.items():
            if stats['filled'] > 0:
                completeness = stats['filled'] / (stats['filled'] + stats['empty'])
                completeness_bonus += completeness * 0.1
        
        # 限制加分上限
        completeness_bonus = min(completeness_bonus, 0.3)
        
        final_score: float = base_score + completeness_bonus
        return min(final_score, 1.0)
    
    def get_validation_summary(self) -> Dict:
        """获取验证统计摘要"""
        return {
            'total_checks': self.validation_stats['total_checks'],
            'passed_checks': self.validation_stats['passed_checks'],
            'failed_checks': self.validation_stats['failed_checks'],
            'warnings': self.validation_stats['warnings'],
            'success_rate': (self.validation_stats['passed_checks'] / 
                           max(self.validation_stats['total_checks'], 1))
        }
    
    def generate_validation_report(self, validation_results: Dict, data_type: str) -> str:
        """生成验证报告"""
        report = f"""
# {data_type} 数据验证报告

## 概览
- 总记录数: {validation_results.get('total_records', 0)}
- 有效记录数: {validation_results.get('valid_records', 0)}
- 无效记录数: {validation_results.get('invalid_records', 0)}
- 数据质量分数: {validation_results.get('data_quality_score', 0):.2f}

## 字段统计
"""
        
        field_stats = validation_results.get('field_statistics', {})
        for field, stats in field_stats.items():
            total = stats['filled'] + stats['empty']
            if total > 0:
                fill_rate = stats['filled'] / total * 100
                report += f"- {field}: 填充率 {fill_rate:.1f}% ({stats['filled']}/{total})\n"
        
        # 错误详情
        errors = validation_results.get('validation_errors', [])
        if errors:
            report += f"\n## 验证错误 (前10个)\n"
            for error in errors[:10]:
                report += f"- {error}\n"
            if len(errors) > 10:
                report += f"- ... 还有 {len(errors) - 10} 个错误\n"
        
        # 特殊分析
        if 'amount_analysis' in validation_results:
            report += f"\n## 金额分析 (前10种)\n"
            for amount, count in validation_results['amount_analysis'].items():
                report += f"- {amount}: {count} 次\n"
        
        if 'round_analysis' in validation_results:
            report += f"\n## 轮次分析 (前10种)\n"
            for round_name, count in validation_results['round_analysis'].items():
                report += f"- {round_name}: {count} 次\n"
        
        return report