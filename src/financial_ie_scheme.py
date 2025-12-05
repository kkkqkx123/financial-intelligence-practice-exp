"""
金融投资领域知识图谱实体关系抽取完整实现
基于ChatIE两阶段提示框架，专门适配SmoothNLP投资结构数据集
"""

import pandas as pd
import json
import re
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# 金融领域实体类型定义（基于数据集特点）
FINANCIAL_ENTITY_TYPES = {
    'chinese': ['投资机构', '被投企业', '行业', '投资轮次', '投资规模', '投资时间', '地区']
}

# 金融领域关系类型定义
FINANCIAL_RELATION_TYPES = {
    'chinese': {
        '投资': ['投资机构', '被投企业'],
        '属于': ['投资机构', '行业'], 
        '参与': ['投资机构', '投资轮次'],
        '管理': ['投资机构', '投资规模'],
        '位于': ['投资机构', '地区'],
        '领投': ['投资机构', '被投企业'],
        '跟投': ['投资机构', '被投企业'],
        '天使投资': ['投资机构', '被投企业'],
        '战略投资': ['投资机构', '被投企业'],
        '收购': ['投资机构', '被投企业']
    }
}

# 投资轮次标准化映射
ROUND_MAPPING = {
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

# 行业标准化映射（基于数据集中的行业分类）
INDUSTRY_MAPPING = {
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

class FinancialInvestmentExtractor:
    """金融投资信息抽取器"""
    
    def __init__(self):
        self.extracted_entities = {}
        self.extracted_relations = []
        self.knowledge_triples = []
        
    def parse_investment_data(self, csv_file: str) -> List[Dict[str, Any]]:
        """解析投资结构CSV数据"""
        print(f"正在解析CSV文件: {csv_file}")
        df = pd.read_csv(csv_file)
        investment_data = []
        
        print(f"CSV文件包含 {len(df)} 条记录，列名: {list(df.columns)}")
        
        for idx, row in df.iterrows():
            institution = str(row.get('机构名称', '')).strip()
            description = str(row.get('介绍', '')).strip()
            industries = str(row.get('行业', '')).strip()
            scale = str(row.get('规模', '')).strip()
            rounds = str(row.get('轮次', '')).strip()
            
            # 跳过空记录
            if not institution or institution == 'nan':
                continue
                
            # 构建富文本描述
            text_parts = []
            text_parts.append(f"{institution}")
            
            if description and description != '暂无信息' and description != 'nan':
                text_parts.append(f"{description}")
                
            if industries and industries != 'nan':
                text_parts.append(f"主要投资领域包括{industries}")
                
            if scale and scale != 'nan':
                text_parts.append(f"管理资金规模为{scale}")
                
            if rounds and rounds != 'nan':
                text_parts.append(f"参与的投资轮次包括：{rounds}")
            
            full_text = "。".join(text_parts) + "。" if text_parts else ""
            
            investment_data.append({
                'id': idx,
                'institution': institution,
                'text': full_text,
                'raw_industries': industries,
                'raw_scale': scale,
                'raw_rounds': rounds,
                'description': description
            })
        
        print(f"成功解析 {len(investment_data)} 条有效投资记录")
        return investment_data
    
    def extract_entities_rule_based(self, data_item: Dict[str, Any]) -> Dict[str, List[str]]:
        """基于规则提取实体"""
        entities = {entity_type: [] for entity_type in FINANCIAL_ENTITY_TYPES['chinese']}
        
        institution = data_item['institution']
        raw_industries = data_item['raw_industries']
        raw_rounds = data_item['raw_rounds']
        raw_scale = data_item['raw_scale']
        description = data_item['description']
        
        # 投资机构
        if institution and institution != 'nan':
            entities['投资机构'] = [institution]
        
        # 行业提取与标准化
        if raw_industries and raw_industries != 'nan':
            industries = self._parse_industries(raw_industries)
            entities['行业'] = industries
        
        # 投资轮次提取与标准化
        if raw_rounds and raw_rounds != 'nan':
            rounds = self._parse_rounds(raw_rounds)
            entities['投资轮次'] = rounds
        
        # 投资规模提取
        if raw_scale and raw_scale != 'nan':
            scales = self._parse_scale(raw_scale)
            entities['投资规模'] = scales
        
        # 从描述中提取时间
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
    
    def _parse_industries(self, industries_str: str) -> List[str]:
        """解析行业字符串"""
        if not industries_str or industries_str == 'nan':
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
                count = match.group(2)
            else:
                industry_name = part
                count = "1"
            
            # 标准化行业名称
            standardized_name = self._standardize_industry(industry_name)
            if standardized_name:
                industries.append(standardized_name)
        
        return list(set(industries))  # 去重
    
    def _parse_rounds(self, rounds_str: str) -> List[str]:
        """解析投资轮次字符串"""
        if not rounds_str or rounds_str == 'nan':
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
        if not scale_str or scale_str == 'nan':
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
    
    def _extract_years(self, text: str) -> List[str]:
        """从文本中提取年份"""
        year_pattern = r'(\d{4})年?'
        years = re.findall(year_pattern, text)
        return list(set(years))
    
    def _extract_locations(self, text: str) -> List[str]:
        """从文本中提取地点信息"""
        # 简单的地点提取规则
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
    
    def create_knowledge_triples(self, data_item: Dict[str, Any], entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """创建知识图谱三元组"""
        triples = []
        institution = data_item['institution']
        
        # 基于规则构建关系三元组
        
        # 1. 投资机构-属于-行业
        if institution and entities.get('行业'):
            for industry in entities['行业']:
                triples.append({
                    'subject': institution,
                    'predicate': '属于',
                    'object': industry,
                    'confidence': 0.9,
                    'source': 'rule_based',
                    'text_snippet': data_item['text'][:100] + "..."
                })
        
        # 2. 投资机构-参与-投资轮次
        if institution and entities.get('投资轮次'):
            for round_name in entities['投资轮次']:
                triples.append({
                    'subject': institution,
                    'predicate': '参与',
                    'object': round_name,
                    'confidence': 0.9,
                    'source': 'rule_based',
                    'text_snippet': data_item['text'][:100] + "..."
                })
        
        # 3. 投资机构-管理-投资规模
        if institution and entities.get('投资规模'):
            for scale in entities['投资规模']:
                triples.append({
                    'subject': institution,
                    'predicate': '管理',
                    'object': scale,
                    'confidence': 0.8,
                    'source': 'rule_based',
                    'text_snippet': data_item['text'][:100] + "..."
                })
        
        # 4. 投资机构-位于-地区
        if institution and entities.get('地区'):
            for location in entities['地区']:
                triples.append({
                    'subject': institution,
                    'predicate': '位于',
                    'object': location,
                    'confidence': 0.7,
                    'source': 'rule_based',
                    'text_snippet': data_item['text'][:100] + "..."
                })
        
        return triples
    
    def run_extraction_pipeline(self, csv_file: str, output_dir: str = None) -> Dict[str, Any]:
        """运行完整的抽取流程"""
        print("=== 开始金融投资信息抽取流程 ===")
        start_time = datetime.now()
        
        # 1. 数据预处理
        print("1. 解析投资数据...")
        investment_data = self.parse_investment_data(csv_file)
        
        if not investment_data:
            print("错误：未找到有效的投资数据")
            return {}
        
        # 2. 实体和关系抽取
        print("2. 执行信息抽取...")
        all_entities = {}
        all_relations = []
        all_triples = []
        
        for idx, data_item in enumerate(investment_data):
            print(f"处理第 {idx+1}/{len(investment_data)} 条记录: {data_item['institution']}")
            
            # 基于规则的实体抽取
            entities = self.extract_entities_rule_based(data_item)
            
            # 创建知识图谱三元组
            triples = self.create_knowledge_triples(data_item, entities)
            
            # 汇总结果
            all_entities[data_item['institution']] = {
                'entities': entities,
                'text': data_item['text'],
                'id': data_item['id']
            }
            all_triples.extend(triples)
        
        # 3. 结果统计
        print("3. 统计抽取结果...")
        statistics = self._calculate_statistics(all_entities, all_triples)
        
        # 4. 构建最终结果
        results = {
            'metadata': {
                'extraction_time': datetime.now().isoformat(),
                'total_records': len(investment_data),
                'extraction_duration': str(datetime.now() - start_time),
                'data_source': csv_file
            },
            'entities': all_entities,
            'knowledge_triples': all_triples,
            'statistics': statistics
        }
        
        # 5. 保存结果
        if output_dir:
            self._save_results(results, output_dir)
        
        print("=== 抽取完成 ===")
        self._print_summary(results)
        
        return results
    
    def _calculate_statistics(self, entities: Dict, triples: List) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {
            'total_institutions': len(entities),
            'total_triples': len(triples),
            'entity_type_counts': {},
            'relation_type_counts': {},
            'top_industries': {},
            'top_rounds': {},
            'scale_distribution': {}
        }
        
        # 统计实体类型
        for institution, data in entities.items():
            entity_dict = data['entities']
            for entity_type, entity_list in entity_dict.items():
                if entity_list:
                    if entity_type not in stats['entity_type_counts']:
                        stats['entity_type_counts'][entity_type] = 0
                    stats['entity_type_counts'][entity_type] += len(entity_list)
        
        # 统计关系类型
        relation_counts = {}
        for triple in triples:
            predicate = triple['predicate']
            if predicate not in relation_counts:
                relation_counts[predicate] = 0
            relation_counts[predicate] += 1
        stats['relation_type_counts'] = relation_counts
        
        # 统计行业分布
        industry_counts = {}
        for institution, data in entities.items():
            industries = data['entities'].get('行业', [])
            for industry in industries:
                if industry not in industry_counts:
                    industry_counts[industry] = 0
                industry_counts[industry] += 1
        stats['top_industries'] = dict(sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # 统计轮次分布
        round_counts = {}
        for institution, data in entities.items():
            rounds = data['entities'].get('投资轮次', [])
            for round_name in rounds:
                if round_name not in round_counts:
                    round_counts[round_name] = 0
                round_counts[round_name] += 1
        stats['top_rounds'] = round_counts
        
        return stats
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """保存抽取结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果
        full_result_file = os.path.join(output_dir, 'financial_extraction_full.json')
        with open(full_result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存知识图谱三元组（CSV格式，便于Neo4j导入）
        triples_file = os.path.join(output_dir, 'knowledge_triples.csv')
        with open(triples_file, 'w', encoding='utf-8') as f:
            f.write("subject,predicate,object,confidence,source\n")
            for triple in results['knowledge_triples']:
                f.write(f"\"{triple['subject']}\",\"{triple['predicate']}\",\"{triple['object']}\",{triple['confidence']},\"{triple['source']}\"\n")
        
        # 保存实体信息（CSV格式）
        entities_file = os.path.join(output_dir, 'entities.csv')
        with open(entities_file, 'w', encoding='utf-8') as f:
            f.write("institution,entity_type,entity_name\n")
            for institution, data in results['entities'].items():
                for entity_type, entity_list in data['entities'].items():
                    for entity_name in entity_list:
                        f.write(f"\"{institution}\",\"{entity_type}\",\"{entity_name}\"\n")
        
        print(f"结果已保存到目录: {output_dir}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """打印结果摘要"""
        print("\n=== 抽取结果摘要 ===")
        print(f"处理记录数: {results['metadata']['total_records']}")
        print(f"知识图谱三元组数: {results['statistics']['total_triples']}")
        print(f"投资机构数: {results['statistics']['total_institutions']}")
        
        print("\n实体类型分布:")
        for entity_type, count in results['statistics']['entity_type_counts'].items():
            print(f"  {entity_type}: {count}")
        
        print("\n关系类型分布:")
        for relation_type, count in results['statistics']['relation_type_counts'].items():
            print(f"  {relation_type}: {count}")
        
        print("\n热门行业 (Top 10):")
        for industry, count in results['statistics']['top_industries'].items():
            print(f"  {industry}: {count}")
        
        print("\n投资轮次分布:")
        for round_name, count in results['statistics']['top_rounds'].items():
            print(f"  {round_name}: {count}")

# 使用示例和测试函数
def test_extractor():
    """测试抽取器"""
    print("开始测试金融投资信息抽取器...")
    
    # 创建抽取器实例
    extractor = FinancialInvestmentExtractor()
    
    # 测试数据路径
    csv_file = "d:/Source/torch/financial-intellgience/大作业/知识图谱/SmoothNLP投资结构数据集-1k.csv"
    output_dir = "d:/Source/torch/financial-intellgience/大作业/知识图谱/extraction_results"
    
    # 运行抽取流程
    results = extractor.run_extraction_pipeline(csv_file, output_dir)
    
    # 打印部分结果示例
    if results and results['knowledge_triples']:
        print("\n=== 知识图谱三元组示例 ===")
        for i, triple in enumerate(results['knowledge_triples'][:10]):
            print(f"{i+1}. {triple['subject']} --[{triple['predicate']}]--> {triple['object']} (置信度: {triple['confidence']})")
    
    return results

if __name__ == "__main__":
    # 运行测试
    results = test_extractor()
    
    if results:
        print("\n金融投资信息抽取测试完成！")
    else:
        print("\n抽取过程出现错误！")