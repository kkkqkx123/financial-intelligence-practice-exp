"""
混合式知识图谱构建实现 - 硬编码优先，LLM增强
演示如何最小化LLM使用，同时保持高质量输出
"""

import re
import json
import asyncio
import httpx
import os
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time

# 完全禁用httpx和httpcore的日志记录，避免污染过程记录
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

async def call_llm(prompt: str) -> Dict:
    """调用LLM API的函数"""
    try:
        # 获取API配置
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            print("警告: 未设置LLM API密钥，使用模拟响应")
            return {"error": "API密钥未设置"}
        
        # 构建API请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            "messages": [
                {"role": "system", "content": "你是一个金融领域的专家，专门处理公司、投资机构和投资关系相关的信息。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        # 发送请求
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 尝试解析JSON响应
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 如果不是JSON格式，返回原始内容
                return {"response": content}
                
    except Exception as e:
        print(f"LLM API调用失败: {str(e)}")
        return {"error": str(e)}

class OptimizedKGBuilder:
    """优化的知识图谱构建器 - 硬编码优先"""
    
    def __init__(self):
        # 预构建的索引和规则库
        self.company_index = {}
        self.institution_index = {}
        self.alias_mapping = {}
        
        # 统计信息
        self.stats = {
            'hardcoded_success': 0,
            'llm_fallback': 0,
            'total_processed': 0,
            'processing_time': 0
        }
    
    # ==================== 硬编码数据解析 ====================
    
    def parse_datasets_hardcoded(self, company_text: str, event_text: str, institution_text: str) -> Dict:
        """硬编码数据解析主函数"""
        start_time = time.time()
        
        companies = self._parse_companies_hardcoded(company_text)
        events = self._parse_events_hardcoded(event_text)
        institutions = self._parse_institutions_hardcoded(institution_text)
        
        # 构建索引
        self._build_indexes(companies, institutions)
        
        self.stats['processing_time'] = time.time() - start_time
        return {
            'companies': companies,
            'events': events,
            'institutions': institutions
        }
    
    def _parse_companies_hardcoded(self, text: str) -> List[Dict]:
        """硬编码解析公司数据"""
        companies = []
        lines = text.strip().split('\n')
        
        # 跳过非数据行
        data_started = False
        for line in lines:
            if line.startswith('名称,公司名称,公司介绍'):
                data_started = True
                continue
            if not data_started or not line.strip():
                continue
            
            # 使用优化的分割算法
            fields = self._smart_split(line)
            if len(fields) >= 11:
                company = {
                    'short_name': fields[0].strip(),
                    'full_name': fields[1].strip(),
                    'description': fields[2].strip(),
                    'registration_name': fields[3].strip(),
                    'address': fields[4].strip() if fields[4].strip() else None,
                    'registration_id': fields[5].strip(),
                    'establish_date': self._parse_date_hardcoded(fields[6]),
                    'legal_representative': fields[7].strip(),
                    'registered_capital': self._normalize_capital_hardcoded(fields[8]),
                    'credit_code': fields[9].strip(),
                    'website': fields[10].strip() if fields[10].strip() else None,
                    'parsed_by': 'hardcoded'
                }
                companies.append(company)
        
        return companies
    
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
    
    def _parse_date_hardcoded(self, date_str: str) -> Optional[str]:
        """硬编码日期解析"""
        if not date_str or date_str.strip() == '':
            return None
        
        date_str = date_str.strip()
        
        # 预定义的模式
        patterns = [
            (r'\d{4}-\d{1,2}-\d{1,2}', lambda x: x),  # YYYY-MM-DD
            (r'\d{4}/\d{1,2}/\d{1,2}', lambda x: x.replace('/', '-')),  # YYYY/MM/DD
            (r'\d{4}年\d{1,2}月\d{1,2}日', lambda x: x.replace('年', '-').replace('月', '-').replace('日', '')),
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, date_str)
            if match:
                return converter(match.group())
        
        return date_str  # 返回原始值
    
    def _normalize_capital_hardcoded(self, capital_str: str) -> Optional[float]:
        """硬编码注册资本标准化"""
        if not capital_str or capital_str.strip() == '':
            return None
        
        capital_str = capital_str.strip()
        
        # 预定义转换规则
        rules = [
            (r'(\d+(?:\.\d+)?)万元人民币', lambda x: float(x) * 10000),
            (r'(\d+(?:\.\d+)?)万元', lambda x: float(x) * 10000),
            (r'(\d+(?:\.\d+)?)万美元', lambda x: float(x) * 10000 * 7.2),
            (r'(\d+(?:\.\d+)?)百万美元', lambda x: float(x) * 1000000 * 7.2),
            (r'(\d+(?:\.\d+)?)亿元人民币', lambda x: float(x) * 100000000),
        ]
        
        for pattern, converter in rules:
            match = re.search(pattern, capital_str)
            if match:
                try:
                    return converter(match.group(1))
                except ValueError:
                    continue
        
        # 最后尝试提取纯数字
        numbers = re.findall(r'\d+(?:\.\d+)?', capital_str)
        if numbers:
            return float(numbers[0]) * 10000  # 默认万元
        
        return None
    
    def _parse_events_hardcoded(self, text: str) -> List[Dict]:
        """硬编码解析投资事件"""
        events = []
        lines = text.strip().split('\n')
        
        data_started = False
        for line in lines:
            if line.startswith('事件资讯,投资方,融资方'):
                data_started = True
                continue
            if not data_started or not line.strip():
                continue
            
            fields = self._smart_split(line)
            if len(fields) >= 6:
                event = {
                    'description': fields[0].strip(),
                    'investors': fields[1].strip().split() if fields[1].strip() else [],
                    'investee': fields[2].strip(),
                    'investment_date': self._parse_date_hardcoded(fields[3]),
                    'round': self._normalize_round_hardcoded(fields[4]),
                    'amount': self._normalize_amount_hardcoded(fields[5]),
                    'parsed_by': 'hardcoded'
                }
                events.append(event)
        
        return events
    
    def _normalize_round_hardcoded(self, round_str: str) -> Optional[str]:
        """硬编码轮次标准化"""
        if not round_str:
            return None
        
        round_str = round_str.strip()
        
        # 直接映射表
        round_mapping = {
            '天使轮': 'Angel', '种子轮': 'Seed', 'Pre-A轮': 'Pre-A',
            'A轮': 'A', 'A+轮': 'A+', 'Pre-B轮': 'Pre-B',
            'B轮': 'B', 'B+轮': 'B+', 'C轮': 'C',
            'D轮': 'D', 'E轮': 'E', 'Pre-IPO': 'Pre-IPO',
            'IPO': 'IPO', '战略投资': 'Strategic', '并购': 'M&A'
        }
        
        return round_mapping.get(round_str, round_str)
    
    def _normalize_amount_hardcoded(self, amount_str: str) -> Optional[float]:
        """硬编码投资金额标准化"""
        if not amount_str or amount_str.strip() in ['未披露', '', '0']:
            return None
        
        amount_str = amount_str.strip()
        
        # 金额转换规则
        rules = [
            (r'(\d+(?:\.\d+)?)万人民币', lambda x: float(x) * 10000),
            (r'(\d+(?:\.\d+)?)亿元', lambda x: float(x) * 100000000),
            (r'(\d+(?:\.\d+)?)百万美元', lambda x: float(x) * 1000000 * 7.2),
            (r'(\d+(?:\.\d+)?)万美元', lambda x: float(x) * 10000 * 7.2),
        ]
        
        for pattern, converter in rules:
            match = re.search(pattern, amount_str)
            if match:
                try:
                    return converter(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _parse_institutions_hardcoded(self, text: str) -> List[Dict]:
        """硬编码解析投资机构"""
        institutions = []
        lines = text.strip().split('\n')
        
        # 跳过前两行统计信息
        data_lines = lines[2:] if len(lines) > 2 else lines
        
        for line in data_lines:
            if not line.strip():
                continue
            
            fields = self._smart_split(line)
            if len(fields) >= 5:
                institution = {
                    'name': fields[0].strip(),
                    'description': fields[1].strip() if fields[1].strip() else None,
                    'industries': fields[2].strip().split() if fields[2].strip() else [],
                    'scale': self._normalize_scale_hardcoded(fields[3]),
                    'preferred_rounds': fields[4].strip().split() if fields[4].strip() else [],
                    'parsed_by': 'hardcoded'
                }
                institutions.append(institution)
        
        return institutions
    
    def _normalize_scale_hardcoded(self, scale_str: str) -> Optional[str]:
        """硬编码规模标准化"""
        if not scale_str or scale_str.strip() == '':
            return None
        
        scale_str = scale_str.strip()
        
        # 简单的关键词匹配
        if '亿' in scale_str:
            numbers = re.findall(r'\d+(?:\.\d+)?', scale_str)
            if numbers:
                return f"{numbers[0]}亿元"
        
        return scale_str
    
    # ==================== 硬编码实体链接 ====================
    
    def _build_indexes(self, companies: List[Dict], institutions: List[Dict]):
        """构建快速查找索引"""
        # 公司索引
        for company in companies:
            # 主名称索引
            self.company_index[company['full_name']] = company
            
            # 别名索引（硬编码生成）
            aliases = self._generate_company_aliases_hardcoded(company)
            for alias in aliases:
                if alias not in self.company_index:
                    self.company_index[alias] = company
        
        # 机构索引
        for institution in institutions:
            self.institution_index[institution['name']] = institution
            
            # 机构别名
            aliases = self._generate_institution_aliases_hardcoded(institution)
            for alias in aliases:
                if alias not in self.institution_index:
                    self.institution_index[alias] = institution
    
    def _generate_company_aliases_hardcoded(self, company: Dict) -> List[str]:
        """硬编码生成公司别名"""
        aliases = []
        full_name = company['full_name']
        short_name = company['short_name']
        
        # 规则1: 简称
        if short_name and short_name != full_name:
            aliases.append(short_name)
        
        # 规则2: 去除公司后缀
        suffixes = ['有限公司', '股份有限公司', '有限责任公司', '公司', '集团']
        for suffix in suffixes:
            if full_name.endswith(suffix):
                aliases.append(full_name[:-len(suffix)])
        
        # 规则3: 去除地域前缀
        regions = ['北京', '上海', '深圳', '杭州', '广州', '成都', '南京', '武汉']
        for region in regions:
            if full_name.startswith(region):
                aliases.append(full_name[len(region):])
        
        # 规则4: 提取英文部分
        english_parts = re.findall(r'[A-Za-z]+', full_name)
        aliases.extend(english_parts)
        
        return list(set(aliases))
    
    def _generate_institution_aliases_hardcoded(self, institution: Dict) -> List[str]:
        """硬编码生成机构别名"""
        aliases = []
        name = institution['name']
        
        # 预定义的机构别名映射
        institution_aliases = {
            '红杉资本': ['红杉', 'Sequoia', '红杉中国'],
            'IDG资本': ['IDG', 'IDG Capital'],
            '真格基金': ['真格', 'ZhenFund'],
            '经纬创投': ['经纬', 'Matrix Partners'],
            '高瓴资本': ['高瓴', 'Hillhouse'],
            '腾讯投资': ['腾讯', 'Tencent'],
            '阿里巴巴': ['阿里', 'Alibaba'],
            '百度投资': ['百度', 'Baidu'],
            '京东集团': ['京东', 'JD'],
            '小米科技': ['小米', 'Xiaomi']
        }
        
        # 查找预定义别名
        for standard_name, alias_list in institution_aliases.items():
            if name == standard_name:
                aliases.extend(alias_list)
                break
        
        # 通用别名生成规则
        if '资本' in name:
            aliases.append(name.replace('资本', ''))
        if '基金' in name:
            aliases.append(name.replace('基金', ''))
        if '创投' in name:
            aliases.append(name.replace('创投', ''))
        
        return list(set(aliases))
    
    def link_entities_hardcoded(self, companies: List[Dict], events: List[Dict], institutions: List[Dict]) -> Dict:
        """硬编码实体链接主函数"""
        linked_data = {
            'company_links': [],
            'institution_links': [],
            'failed_links': []
        }
        
        # 公司实体链接
        for event in events:
            investee = event['investee']
            
            # 多级硬编码匹配
            link_result = self._match_company_hardcoded(investee)
            
            if link_result['success']:
                linked_data['company_links'].append({
                    'event': event,
                    'company': link_result['company'],
                    'match_type': link_result['type'],
                    'confidence': link_result['confidence']
                })
                self.stats['hardcoded_success'] += 1
            else:
                # 记录失败，后续用LLM处理
                linked_data['failed_links'].append({
                    'type': 'company',
                    'event': event,
                    'reason': link_result['reason']
                })
                self.stats['llm_fallback'] += 1
            
            # 投资机构链接
            for investor in event['investors']:
                inst_result = self._match_institution_hardcoded(investor)
                
                if inst_result['success']:
                    linked_data['institution_links'].append({
                        'investor_name': investor,
                        'institution': inst_result['institution'],
                        'match_type': inst_result['type'],
                        'confidence': inst_result['confidence']
                    })
                    self.stats['hardcoded_success'] += 1
                else:
                    linked_data['failed_links'].append({
                        'type': 'institution',
                        'investor_name': investor,
                        'reason': inst_result['reason']
                    })
                    self.stats['llm_fallback'] += 1
        
        self.stats['total_processed'] += len(events) + sum(len(e['investors']) for e in events)
        return linked_data
    
    def _match_company_hardcoded(self, investee_name: str) -> Dict:
        """硬编码公司匹配"""
        investee_name = investee_name.strip()
        
        # 级别1: 精确匹配
        if investee_name in self.company_index:
            return {
                'success': True,
                'company': self.company_index[investee_name],
                'type': 'exact_match',
                'confidence': 1.0
            }
        
        # 级别2: 子字符串匹配（双向）
        for company_name, company_data in self.company_index.items():
            if (len(investee_name) > 3 and len(company_name) > 3 and
                (investee_name in company_name or company_name in investee_name)):
                return {
                    'success': True,
                    'company': company_data,
                    'type': 'substring_match',
                    'confidence': 0.8
                }
        
        # 级别3: 编辑距离匹配（简单实现）
        best_match = None
        best_similarity = 0
        
        for company_name, company_data in self.company_index.items():
            similarity = self._calculate_similarity(investee_name, company_name)
            if similarity > 0.7 and similarity > best_similarity:  # 阈值可调
                best_similarity = similarity
                best_match = company_data
        
        if best_match:
            return {
                'success': True,
                'company': best_match,
                'type': 'similarity_match',
                'confidence': best_similarity
            }
        
        return {
            'success': False,
            'reason': 'no_hardcoded_match_found'
        }
    
    def _match_institution_hardcoded(self, investor_name: str) -> Dict:
        """硬编码机构匹配"""
        investor_name = investor_name.strip()
        
        # 级别1: 精确匹配
        if investor_name in self.institution_index:
            return {
                'success': True,
                'institution': self.institution_index[investor_name],
                'type': 'exact_match',
                'confidence': 1.0
            }
        
        # 级别2: 预定义别名匹配
        for standard_name, aliases in self._get_institution_aliases().items():
            if investor_name in aliases and standard_name in self.institution_index:
                return {
                    'success': True,
                    'institution': self.institution_index[standard_name],
                    'type': 'alias_match',
                    'confidence': 0.95
                }
        
        # 级别3: 子字符串匹配
        for inst_name, inst_data in self.institution_index.items():
            if (len(investor_name) > 2 and len(inst_name) > 2 and
                (investor_name in inst_name or inst_name in investor_name)):
                return {
                    'success': True,
                    'institution': inst_data,
                    'type': 'substring_match',
                    'confidence': 0.7
                }
        
        return {
            'success': False,
            'reason': 'no_hardcoded_match_found'
        }
    
    def _get_institution_aliases(self) -> Dict[str, List[str]]:
        """获取机构别名映射 - 硬编码"""
        return {
            '红杉资本': ['红杉', 'Sequoia', '红杉中国'],
            'IDG资本': ['IDG', 'IDG Capital'],
            '真格基金': ['真格', 'ZhenFund'],
            '经纬创投': ['经纬', 'Matrix Partners'],
            '高瓴资本': ['高瓴', 'Hillhouse'],
            '腾讯投资': ['腾讯', 'Tencent'],
            '阿里巴巴': ['阿里', 'Alibaba'],
            '百度投资': ['百度', 'Baidu'],
            '京东集团': ['京东', 'JD'],
            '小米科技': ['小米', 'Xiaomi']
        }
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """简单的相似度计算 - 硬编码"""
        # 基于字符的相似度
        if s1 == s2:
            return 1.0
        
        # 包含关系
        if s1 in s2 or s2 in s1:
            return 0.9
        
        # 计算公共字符比例
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # 长度惩罚
        length_diff = abs(len(s1) - len(s2)) / max(len(s1), len(s2))
        
        return jaccard * (1 - length_diff * 0.3)
    
    # ==================== LLM 增强阶段 ====================
    
    async def enhance_with_llm(self, failed_links: List[Dict]) -> List[Dict]:
        """LLM增强处理 - 仅处理硬编码失败的案例"""
        enhanced_results = []
        
        for failed in failed_links:
            if failed['type'] == 'company':
                result = await self._llm_enhance_company_matching(failed)
            elif failed['type'] == 'institution':
                result = await self._llm_enhance_institution_matching(failed)
            else:
                continue
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    async def _llm_enhance_company_matching(self, failed: Dict) -> Dict:
        """LLM增强公司匹配"""
        event = failed['event']
        investee_name = event['investee']
        
        # 获取候选公司列表
        candidates = list(self.company_index.keys())[:20]  # 限制候选数量
        
        # 构建LLM提示
        prompt = f"""
        任务：在投资事件中匹配正确的公司实体
        
        投资事件描述：{event['description'][:200]}
        融资方名称：{investee_name}
        
        候选公司列表（前20个）：
        {chr(10).join(f"- {c}" for c in candidates)}
        
        要求：
        1. 分析融资方名称与候选公司的相似性
        2. 考虑投资事件的行业背景
        3. 评估匹配的置信度（0-1）
        
        输出格式：
        {{
            "matched_company": "匹配的公司名称或null",
            "confidence": 0.85,
            "reasoning": "匹配理由说明"
        }}
        """
        
        # 调用LLM API
        result = await call_llm(prompt)
        
        # 检查是否有错误
        if "error" in result:
            print(f"LLM调用出错: {result['error']}")
            # 使用模拟响应作为后备
            result = self._simulate_llm_response(investee_name, candidates)
        
        if result.get('matched_company') and result['matched_company'] in self.company_index:
            return {
                'success': True,
                'type': 'llm_enhanced',
                'company': self.company_index[result['matched_company']],
                'confidence': result['confidence'],
                'reasoning': result['reasoning']
            }
        
        return {
            'success': False,
            'type': 'llm_failed',
            'reason': 'no_confident_match'
        }
    
    async def _llm_enhance_institution_matching(self, failed: Dict) -> Dict:
        """LLM增强机构匹配"""
        investor_name = failed['investor_name']
        
        # 获取候选机构
        candidates = list(self.institution_index.keys())[:15]
        
        prompt = f"""
        任务：识别投资机构的规范名称
        
        投资方名称：{investor_name}
        
        候选投资机构：
        {chr(10).join(f"- {c}" for c in candidates)}
        
        要求：
        1. 识别是否为已知机构的简称、别名或分支机构
        2. 考虑中文投资机构的命名习惯
        3. 评估匹配置信度
        
        输出格式：
        {{
            "matched_institution": "匹配机构名称或null", 
            "confidence": 0.9,
            "match_type": "简称/别名/分支机构",
            "reasoning": "匹配理由"
        }}
        """
        
        # 调用LLM API
        result = await call_llm(prompt)
        
        # 检查是否有错误
        if "error" in result:
            print(f"LLM调用出错: {result['error']}")
            # 使用模拟响应作为后备
            result = self._simulate_llm_institution_response(investor_name, candidates)
        
        if result.get('matched_institution') and result['matched_institution'] in self.institution_index:
            return {
                'success': True,
                'type': 'llm_enhanced',
                'institution': self.institution_index[result['matched_institution']],
                'confidence': result['confidence'],
                'match_type': result['match_type'],
                'reasoning': result['reasoning']
            }
        
        return {
            'success': False,
            'type': 'llm_failed',
            'reason': 'no_confident_match'
        }
    
    def _simulate_llm_response(self, investee_name: str, candidates: List[str]) -> Dict:
        """模拟LLM响应 - 实际应用中替换为真实LLM调用"""
        # 简单的相似度匹配模拟
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = self._calculate_similarity(investee_name, candidate)
            if score > 0.6 and score > best_score:
                best_score = score
                best_match = candidate
        
        if best_match:
            return {
                "matched_company": best_match,
                "confidence": best_score,
                "reasoning": f"基于名称相似度匹配：{investee_name} ≈ {best_match}"
            }
        
        return {
            "matched_company": None,
            "confidence": 0.0,
            "reasoning": "未找到高置信度匹配"
        }
    
    def _simulate_llm_institution_response(self, investor_name: str, candidates: List[str]) -> Dict:
        """模拟LLM机构匹配 - 实际应用中替换为真实LLM调用"""
        # 检查是否在别名映射中
        for standard_name, aliases in self._get_institution_aliases().items():
            if investor_name in aliases and standard_name in self.institution_index:
                return {
                    "matched_institution": standard_name,
                    "confidence": 0.9,
                    "match_type": "别名识别",
                    "reasoning": f"识别为{standard_name}的常用简称"
                }
        
        # 相似度匹配
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = self._calculate_similarity(investor_name, candidate)
            if score > 0.5 and score > best_score:
                best_score = score
                best_match = candidate
        
        if best_match:
            return {
                "matched_institution": best_match,
                "confidence": best_score,
                "match_type": "相似度匹配",
                "reasoning": f"基于名称相似度：{investor_name} ≈ {best_match}"
            }
        
        return {
            "matched_institution": None,
            "confidence": 0.0,
            "match_type": "无匹配",
            "reasoning": "未找到合适的机构匹配"
        }
    
    # ==================== 性能统计 ====================
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        total = self.stats['total_processed']
        hardcoded_success = self.stats['hardcoded_success']
        llm_fallback = self.stats['llm_fallback']
        
        return {
            'total_processed': total,
            'hardcoded_success': hardcoded_success,
            'llm_fallback': llm_fallback,
            'hardcoded_ratio': hardcoded_success / total if total > 0 else 0,
            'llm_fallback_ratio': llm_fallback / total if total > 0 else 0,
            'processing_time_seconds': self.stats['processing_time'],
            'efficiency_improvement': f"{hardcoded_success}/{total} = {hardcoded_success/total*100:.1f}%"
        }

# ==================== 使用示例 ====================

async def demo_optimized_workflow():
    """演示优化的工作流程"""
    
    # 模拟数据
    company_text = """总计:10000条记录
名称,公司名称,公司介绍,工商,地址,工商注册id,成立时间,法人代表,注册资金,统一信用代码,网址
字节跳动,北京字节跳动科技有限公司,"信息科技公司",北京字节跳动科技有限公司,北京市海淀区,91110108MA1234567,2012-03-09,张一鸣,10000万元人民币,91110108MA1234567,www.bytedance.com
腾讯,深圳市腾讯计算机系统有限公司,"互联网综合服务",深圳市腾讯计算机系统有限公司,深圳市南山区,9144030019236096F,1998-11-11,马化腾,6500万元人民币,9144030019236096F,www.tencent.com"""
    
    event_text = """总计:2000条记录
事件资讯,投资方,融资方,融资时间,轮次,金额
"字节跳动完成B轮融资","红杉资本 新浪微博","北京字节跳动科技有限公司","2013-05-01","B轮","1000万美元"
"腾讯收购Supercell","腾讯","Supercell Oy","2016-06-21","并购","86亿美元"""
    
    institution_text = """总计:1000条记录
机构名称,介绍,行业,规模,轮次
红杉资本,"全球知名投资机构","科技 医疗","100+亿人民币","天使轮 A轮 B轮"
腾讯投资,"腾讯旗下投资部门","游戏 社交","1000+亿人民币","战略投资 并购"""
    
    # 构建优化的知识图谱
    builder = OptimizedKGBuilder()
    
    # 步骤1: 硬编码解析
    print("步骤1: 硬编码数据解析...")
    parsed_data = builder.parse_datasets_hardcoded(company_text, event_text, institution_text)
    
    print(f"解析完成:")
    print(f"  - 公司数量: {len(parsed_data['companies'])}")
    print(f"  - 事件数量: {len(parsed_data['events'])}")
    print(f"  - 机构数量: {len(parsed_data['institutions'])}")
    
    # 步骤2: 硬编码实体链接
    print("\n步骤2: 硬编码实体链接...")
    linked_data = builder.link_entities_hardcoded(
        parsed_data['companies'], 
        parsed_data['events'], 
        parsed_data['institutions']
    )
    
    print(f"链接结果:")
    print(f"  - 公司链接成功: {len(linked_data['company_links'])}")
    print(f"  - 机构链接成功: {len(linked_data['institution_links'])}")
    print(f"  - 需要LLM处理: {len(linked_data['failed_links'])}")
    
    # 步骤3: LLM增强（仅处理失败案例）
    if linked_data['failed_links']:
        print(f"\n步骤3: LLM增强处理 {len(linked_data['failed_links'])} 个失败案例...")
        enhanced_results = await builder.enhance_with_llm(linked_data['failed_links'])
        
        successful_enhancements = sum(1 for r in enhanced_results if r.get('success'))
        print(f"LLM增强成功: {successful_enhancements}")
    
    # 性能统计
    stats = builder.get_performance_stats()
    print(f"\n性能统计:")
    print(f"  - 总处理量: {stats['total_processed']}")
    print(f"  - 硬编码成功率: {stats['hardcoded_ratio']:.1%}")
    print(f"  - LLM回退率: {stats['llm_fallback_ratio']:.1%}")
    print(f"  - 处理时间: {stats['processing_time_seconds']:.2f}秒")
    print(f"  - 效率提升: {stats['efficiency_improvement']}")

if __name__ == "__main__":
    asyncio.run(demo_optimized_workflow())