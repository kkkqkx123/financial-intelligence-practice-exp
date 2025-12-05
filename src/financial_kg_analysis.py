"""
金融投资领域知识图谱实体关系抽取 - 分析报告
基于SmoothNLP投资结构数据集的分析结果
"""

import json
import pandas as pd
from datetime import datetime
from collections import Counter
import os

class FinancialKGAnalysis:
    """金融知识图谱分析报告生成器"""
    
    def __init__(self, extraction_results_file: str):
        """初始化分析器"""
        self.extraction_results_file = extraction_results_file
        self.results = None
        self.load_results()
    
    def load_results(self):
        """加载抽取结果"""
        try:
            with open(self.extraction_results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"成功加载 {len(self.results.get('entities', {}))} 个投资机构的抽取结果")
        except Exception as e:
            print(f"加载结果文件失败: {e}")
            self.results = None
    
    def generate_analysis_report(self, output_file: str = None):
        """生成完整的分析报告"""
        if not self.results:
            print("没有可用的抽取结果")
            return
        
        report = []
        report.append("# 金融投资领域知识图谱实体关系抽取分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. 基本信息统计
        report.append("## 1. 基本信息统计")
        report.append(self._generate_basic_stats())
        report.append("")
        
        # 2. 实体类型分析
        report.append("## 2. 实体类型分析")
        report.append(self._generate_entity_analysis())
        report.append("")
        
        # 3. 关系类型分析
        report.append("## 3. 关系类型分析")
        report.append(self._generate_relation_analysis())
        report.append("")
        
        # 4. 行业分布分析
        report.append("## 4. 行业分布分析")
        report.append(self._generate_industry_analysis())
        report.append("")
        
        # 5. 投资轮次分析
        report.append("## 5. 投资轮次分析")
        report.append(self._generate_round_analysis())
        report.append("")
        
        # 6. 知识图谱质量评估
        report.append("## 6. 知识图谱质量评估")
        report.append(self._generate_quality_assessment())
        report.append("")
        
        # 7. 抽取方法分析
        report.append("## 7. 抽取方法分析")
        report.append(self._generate_method_analysis())
        report.append("")
        
        # 8. 结论与建议
        report.append("## 8. 结论与建议")
        report.append(self._generate_conclusions())
        
        # 合并报告内容
        full_report = "\n".join(report)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"分析报告已保存到: {output_file}")
        
        return full_report
    
    def _generate_basic_stats(self) -> str:
        """生成基本统计信息"""
        metadata = self.results.get('metadata', {})
        stats = self.results.get('statistics', {})
        
        lines = []
        lines.append(f"- **数据源**: {metadata.get('data_source', '未知')}")
        lines.append(f"- **处理记录数**: {metadata.get('total_records', 0)}")
        lines.append(f"- **抽取投资机构数**: {stats.get('total_institutions', 0)}")
        lines.append(f"- **知识图谱三元组数**: {stats.get('total_triples', 0)}")
        lines.append(f"- **抽取耗时**: {metadata.get('extraction_duration', '未知')}")
        lines.append(f"- **抽取完成时间**: {metadata.get('extraction_time', '未知')}")
        
        return "\n".join(lines)
    
    def _generate_entity_analysis(self) -> str:
        """生成实体类型分析"""
        entity_counts = self.results.get('statistics', {}).get('entity_type_counts', {})
        
        lines = []
        lines.append("### 实体类型分布")
        
        if entity_counts:
            lines.append("| 实体类型 | 数量 | 占比 |")
            lines.append("|----------|------|------|")
            
            total_entities = sum(entity_counts.values())
            for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_entities * 100) if total_entities > 0 else 0
                lines.append(f"| {entity_type} | {count} | {percentage:.1f}% |")
        
        # 添加实体抽取示例
        lines.append("")
        lines.append("### 实体抽取示例")
        entities = self.results.get('entities', {})
        sample_institutions = list(entities.keys())[:3]
        
        for inst in sample_institutions:
            entity_data = entities[inst]
            lines.append(f"**{inst}**:")
            for entity_type, entity_list in entity_data['entities'].items():
                if entity_list:
                    lines.append(f"  - {entity_type}: {', '.join(entity_list[:3])}{'...' if len(entity_list) > 3 else ''}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_relation_analysis(self) -> str:
        """生成关系类型分析"""
        relation_counts = self.results.get('statistics', {}).get('relation_type_counts', {})
        
        lines = []
        lines.append("### 关系类型分布")
        
        if relation_counts:
            lines.append("| 关系类型 | 数量 | 占比 |")
            lines.append("|----------|------|------|")
            
            total_relations = sum(relation_counts.values())
            for relation_type, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_relations * 100) if total_relations > 0 else 0
                lines.append(f"| {relation_type} | {count} | {percentage:.1f}% |")
        
        # 添加关系三元组示例
        lines.append("")
        lines.append("### 关系三元组示例")
        triples = self.results.get('knowledge_triples', [])
        sample_triples = triples[:10]
        
        for i, triple in enumerate(sample_triples, 1):
            lines.append(f"{i}. **{triple['subject']}** --[{triple['predicate']}]--> **{triple['object']}** (置信度: {triple['confidence']})")
        
        return "\n".join(lines)
    
    def _generate_industry_analysis(self) -> str:
        """生成行业分布分析"""
        top_industries = self.results.get('statistics', {}).get('top_industries', {})
        
        lines = []
        lines.append("### 热门行业分布 (Top 10)")
        
        if top_industries:
            lines.append("| 行业 | 投资机构数量 |")
            lines.append("|------|--------------|")
            
            for industry, count in list(top_industries.items())[:10]:
                lines.append(f"| {industry} | {count} |")
        
        # 分析行业覆盖度
        total_institutions = self.results.get('statistics', {}).get('total_institutions', 0)
        institutions_with_industry = sum(1 for _, data in self.results.get('entities', {}).items() 
                                       if data['entities'].get('行业', []))
        
        lines.append("")
        lines.append("### 行业覆盖度分析")
        lines.append(f"- **有行业信息的投资机构数**: {institutions_with_industry}")
        lines.append(f"- **行业信息覆盖率**: {(institutions_with_industry/total_institutions*100):.1f}%")
        lines.append(f"- **涉及的行业类别数**: {len(top_industries)}")
        
        return "\n".join(lines)
    
    def _generate_round_analysis(self) -> str:
        """生成投资轮次分析"""
        top_rounds = self.results.get('statistics', {}).get('top_rounds', {})
        
        lines = []
        lines.append("### 投资轮次分布")
        
        if top_rounds:
            lines.append("| 投资轮次 | 投资机构数量 |")
            lines.append("|----------|--------------|")
            
            for round_name, count in sorted(top_rounds.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {round_name} | {count} |")
        
        # 分析轮次覆盖度
        total_institutions = self.results.get('statistics', {}).get('total_institutions', 0)
        institutions_with_rounds = sum(1 for _, data in self.results.get('entities', {}).items() 
                                     if data['entities'].get('投资轮次', []))
        
        lines.append("")
        lines.append("### 投资轮次覆盖度分析")
        lines.append(f"- **有轮次信息的投资机构数**: {institutions_with_rounds}")
        lines.append(f"- **轮次信息覆盖率**: {(institutions_with_rounds/total_institutions*100):.1f}%")
        lines.append(f"- **涉及的投资轮次数**: {len(top_rounds)}")
        
        return "\n".join(lines)
    
    def _generate_quality_assessment(self) -> str:
        """生成质量评估"""
        triples = self.results.get('knowledge_triples', [])
        
        lines = []
        lines.append("### 数据质量评估")
        
        if triples:
            # 置信度分析
            confidence_scores = [triple.get('confidence', 0) for triple in triples]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            high_confidence_count = sum(1 for score in confidence_scores if score >= 0.8)
            
            lines.append(f"- **平均置信度**: {avg_confidence:.3f}")
            lines.append(f"- **高置信度三元组比例**: {(high_confidence_count/len(triples)*100):.1f}%")
            lines.append(f"- **数据来源**: 基于规则抽取")
            
            # 数据一致性检查
            unique_subjects = len(set(triple['subject'] for triple in triples))
            unique_objects = len(set(triple['object'] for triple in triples))
            
            lines.append(f"- **唯一主体数**: {unique_subjects}")
            lines.append(f"- **唯一客体数**: {unique_objects}")
            lines.append(f"- **三元组重复率**: {((len(triples) - len(set((t['subject'], t['predicate'], t['object']) for t in triples)))/len(triples)*100):.1f}%")
        
        return "\n".join(lines)
    
    def _generate_method_analysis(self) -> str:
        """生成方法分析"""
        lines = []
        lines.append("### 抽取方法概述")
        lines.append("本系统采用基于规则的信息抽取方法，主要特点包括：")
        lines.append("")
        lines.append("#### 1. 实体抽取策略")
        lines.append("- **结构化数据解析**: 直接从CSV文件的预定义字段提取信息")
        lines.append("- **行业标准化**: 使用预定义的行业映射表进行标准化")
        lines.append("- **轮次标准化**: 使用预定义的轮次映射表进行标准化")
        lines.append("- **文本模式匹配**: 从描述文本中提取时间和地理信息")
        lines.append("")
        lines.append("#### 2. 关系构建策略")
        lines.append("- **基于规则的关系推断**: 根据实体类型自动构建关系三元组")
        lines.append("- **置信度评分**: 为每个三元组分配置信度分数")
        lines.append("- **关系类型定义**: 预定义了10种金融领域关系类型")
        lines.append("")
        lines.append("#### 3. 数据标准化")
        lines.append("- **行业分类标准化**: 将原始行业描述映射到标准分类")
        lines.append("- **投资轮次标准化**: 将原始轮次描述映射到标准轮次")
        lines.append("- **数据清洗**: 处理缺失值和异常数据")
        
        return "\n".join(lines)
    
    def _generate_conclusions(self) -> str:
        """生成结论与建议"""
        stats = self.results.get('statistics', {})
        
        lines = []
        lines.append("### 主要发现")
        
        # 分析主要发现
        top_industries = stats.get('top_industries', {})
        top_rounds = stats.get('top_rounds', {})
        
        lines.append("1. **行业分布特征**")
        if top_industries:
            top_3_industries = list(top_industries.items())[:3]
            lines.append(f"   - 最活跃的投资领域: {', '.join([industry for industry, _ in top_3_industries])}")
            lines.append(f"   - 行业集中度: 前3大行业覆盖 {sum(count for _, count in top_3_industries)} 家投资机构")
        
        lines.append("")
        lines.append("2. **投资轮次特征**")
        if top_rounds:
            most_common_round = max(top_rounds.items(), key=lambda x: x[1])
            lines.append(f"   - 最常见投资轮次: {most_common_round[0]} ({most_common_round[1]} 家机构)")
            lines.append(f"   - 轮次多样性: 共涉及 {len(top_rounds)} 种不同的投资轮次")
        
        lines.append("")
        lines.append("3. **数据质量评估**")
        total_institutions = stats.get('total_institutions', 0)
        institutions_with_industry = sum(1 for _, data in self.results.get('entities', {}).items() 
                                         if data['entities'].get('行业', []))
        institutions_with_rounds = sum(1 for _, data in self.results.get('entities', {}).items() 
                                       if data['entities'].get('投资轮次', []))
        
        lines.append(f"   - 行业信息完整性: {(institutions_with_industry/total_institutions*100):.1f}%")
        lines.append(f"   - 轮次信息完整性: {(institutions_with_rounds/total_institutions*100):.1f}%")
        lines.append(f"   - 知识图谱三元组数: {stats.get('total_triples', 0)}")
        
        lines.append("")
        lines.append("### 改进建议")
        lines.append("1. **增强实体抽取能力**")
        lines.append("   - 集成大语言模型进行实体识别")
        lines.append("   - 增加被投企业实体抽取")
        lines.append("   - 改进地理位置信息提取")
        lines.append("")
        lines.append("2. **优化关系抽取**")
        lines.append("   - 实现基于上下文的动态关系抽取")
        lines.append("   - 增加投资金额、时间等具体关系信息")
        lines.append("   - 引入关系强度评分机制")
        lines.append("")
        lines.append("3. **提升数据质量**")
        lines.append("   - 增加数据验证和清洗规则")
        lines.append("   - 建立实体消歧和合并机制")
        lines.append("   - 引入人工校验和质量评估流程")
        lines.append("")
        lines.append("4. **扩展应用场景**")
        lines.append("   - 构建可视化知识图谱")
        lines.append("   - 实现智能查询和推荐功能")
        lines.append("   - 支持动态更新和维护")
        
        return "\n".join(lines)

# 生成分析报告的函数
def generate_financial_kg_report():
    """生成金融知识图谱分析报告"""
    print("开始生成金融知识图谱分析报告...")
    
    # 分析器配置
    extraction_results_file = "d:/Source/torch/financial-intellgience/大作业/知识图谱/extraction_results/financial_extraction_full.json"
    output_report_file = "d:/Source/torch/financial-intellgience/大作业/知识图谱/extraction_results/financial_kg_analysis_report.md"
    
    # 创建分析器
    analyzer = FinancialKGAnalysis(extraction_results_file)
    
    # 生成报告
    report = analyzer.generate_analysis_report(output_report_file)
    
    # 打印报告摘要
    print("\n=== 分析报告摘要 ===")
    print(report[:2000] + "..." if len(report) > 2000 else report)
    
    return report

if __name__ == "__main__":
    # 运行报告生成
    report = generate_financial_kg_report()
    print("\n金融知识图谱分析报告生成完成！")