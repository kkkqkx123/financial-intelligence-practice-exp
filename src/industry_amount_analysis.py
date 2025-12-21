"""
行业金额分析脚本
分析投资方在不同行业的投资金额分布
"""

import pandas as pd
import json
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

class IndustryAmountAnalyzer:
    def __init__(self, investment_events_file, investment_structure_file):
        self.investment_events_file = investment_events_file
        self.investment_structure_file = investment_structure_file
        self.investment_events_df = None
        self.investment_structure_df = None
        self.industry_amount_data = None
        
    def load_data(self):
        """加载投资事件和投资结构数据"""
        try:
            # 加载投资事件数据
            self.investment_events_df = pd.read_csv(self.investment_events_file)
            print(f"成功加载投资事件数据: {len(self.investment_events_df)} 条记录")
            
            # 加载投资结构数据
            self.investment_structure_df = pd.read_csv(self.investment_structure_file)
            print(f"成功加载投资结构数据: {len(self.investment_structure_df)} 条记录")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
        return True
    
    def preprocess_amount(self, amount_str):
        """预处理金额字符串，转换为数值"""
        if pd.isna(amount_str) or amount_str == '未披露' or amount_str == '无投资':
            return None
        
        # 处理金额字符串
        amount_str = str(amount_str).strip()
        
        # 提取数字和单位
        match = re.search(r'([\d\.]+)([亿万]*)', amount_str)
        if not match:
            return None
            
        number = float(match.group(1))
        unit = match.group(2)
        
        # 单位转换
        if unit == '亿':
            return number * 100000000
        elif unit == '万':
            return number * 10000
        else:
            return number
    
    def extract_industries(self, industry_str):
        """从行业字符串中提取行业列表"""
        if pd.isna(industry_str):
            return []
        
        industries = []
        # 分割行业字符串，处理多种格式
        industry_str = str(industry_str).replace('、', ' ').replace('，', ' ')
        parts = re.findall(r'[\u4e00-\u9fff]+[\u4e00-\u9fff\s]*[\u4e00-\u9fff]+', industry_str)
        
        for part in parts:
            # 清理行业名称
            industry = part.strip()
            if industry and '家' not in industry:
                industries.append(industry)
        
        return list(set(industries))  # 去重
    
    def analyze_industry_amount_relationships(self):
        """分析行业与金额的关系 - 逐个行业详细分析"""
        if self.investment_events_df is None or self.investment_structure_df is None:
            print("请先加载数据")
            return None
        
        # 创建行业-投资方映射
        investor_industry_map = {}
        for _, row in self.investment_structure_df.iterrows():
            investor = row['机构名称']
            industries = self.extract_industries(row['行业'])
            investor_industry_map[investor] = industries
        
        # 获取所有行业列表
        all_industries = set()
        for industries in investor_industry_map.values():
            all_industries.update(industries)
        
        print(f"发现 {len(all_industries)} 个行业，开始逐个分析...")
        
        # 逐个行业分析
        industry_detailed_stats = {}
        investor_industry_detailed_stats = {}
        
        valid_records = 0
        total_amount = 0
        
        for industry in sorted(all_industries):
            print(f"正在分析行业: {industry}")
            
            # 分析该行业的投资事件
            industry_amounts = []
            industry_investors = defaultdict(list)
            
            for _, event in self.investment_events_df.iterrows():
                investors = str(event['投资方']).split(' ')
                amount = self.preprocess_amount(event['金额'])
                
                if amount is None:
                    continue
                    
                # 检查是否有投资方属于该行业
                relevant_investors = []
                for investor in investors:
                    if investor in investor_industry_map and industry in investor_industry_map[investor]:
                        relevant_investors.append(investor)
                
                if relevant_investors:
                    valid_records += 1
                    total_amount += amount
                    industry_amounts.append(amount)
                    
                    # 记录每个投资方的投资
                    for investor in relevant_investors:
                        industry_investors[investor].append(amount)
            
            # 计算该行业的详细统计
            if industry_amounts:
                # 投资方统计
                investor_stats = {}
                for investor, amounts in industry_investors.items():
                    investor_stats[investor] = {
                        '投资次数': len(amounts),
                        '总金额': sum(amounts),
                        '平均金额': sum(amounts) / len(amounts),
                        '最大金额': max(amounts),
                        '最小金额': min(amounts),
                        '投资占比': sum(amounts) / sum(industry_amounts) if sum(industry_amounts) > 0 else 0
                    }
                
                # 行业总体统计
                industry_detailed_stats[industry] = {
                    '投资次数': len(industry_amounts),
                    '总金额': sum(industry_amounts),
                    '平均金额': sum(industry_amounts) / len(industry_amounts),
                    '最大金额': max(industry_amounts),
                    '最小金额': min(industry_amounts),
                    '投资方数量': len(investor_stats),
                    '主要投资方': sorted(investor_stats.items(), key=lambda x: x[1]['总金额'], reverse=True)[:5],
                    '投资方统计': investor_stats,
                    '金额分布': {
                        '小于100万': len([a for a in industry_amounts if a < 1000000]),
                        '100万-1000万': len([a for a in industry_amounts if 1000000 <= a < 10000000]),
                        '1000万-1亿': len([a for a in industry_amounts if 10000000 <= a < 100000000]),
                        '1亿以上': len([a for a in industry_amounts if a >= 100000000])
                    }
                }
                
                # 投资方-行业统计
                for investor, stats in investor_stats.items():
                    if investor not in investor_industry_detailed_stats:
                        investor_industry_detailed_stats[investor] = {}
                    investor_industry_detailed_stats[investor][industry] = stats
        
        # 总体统计
        overall_stats = {
            '总记录数': len(self.investment_events_df),
            '有效金额记录数': valid_records,
            '总投资金额': total_amount,
            '平均投资金额': total_amount / valid_records if valid_records > 0 else 0,
            '行业数量': len(industry_detailed_stats),
            '有投资记录的行业': len([industry for industry, stats in industry_detailed_stats.items() if stats['投资次数'] > 0])
        }
        
        self.industry_amount_data = {
            'overall_stats': overall_stats,
            'industry_detailed_stats': industry_detailed_stats,
            'investor_industry_detailed_stats': investor_industry_detailed_stats,
            'all_industries': list(all_industries)
        }
        
        print(f"行业分析完成！共分析 {len(industry_detailed_stats)} 个有投资记录的行业")
        return self.industry_amount_data
    
    def generate_visualizations(self, output_dir='analysis_results'):
        """生成可视化图表"""
        if self.industry_amount_data is None:
            print("请先进行分析")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        industry_detailed_stats = self.industry_amount_data['industry_detailed_stats']
        
        # 1. 行业投资金额分布图（前20名）
        industries_sorted = sorted(industry_detailed_stats.items(), 
                                 key=lambda x: x[1]['总金额'], reverse=True)[:20]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 总金额分布
        industries_names = [item[0] for item in industries_sorted]
        total_amounts = [item[1]['总金额'] / 100000000 for item in industries_sorted]  # 转换为亿元
        
        ax1.barh(industries_names, total_amounts, color='skyblue')
        ax1.set_xlabel('总投资金额（亿元）')
        ax1.set_title('各行业总投资金额分布（前20名）')
        ax1.grid(axis='x', alpha=0.3)
        
        # 平均金额分布
        avg_amounts = [item[1]['平均金额'] / 100000000 for item in industries_sorted]  # 转换为亿元
        
        ax2.barh(industries_names, avg_amounts, color='lightcoral')
        ax2.set_xlabel('平均投资金额（亿元）')
        ax2.set_title('各行业平均投资金额分布（前20名）')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/industry_amount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 行业投资次数分布图
        industries_count_sorted = sorted(industry_detailed_stats.items(), 
                                       key=lambda x: x[1]['投资次数'], reverse=True)[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        industries_names_count = [item[0] for item in industries_count_sorted]
        investment_counts = [item[1]['投资次数'] for item in industries_count_sorted]
        
        bars = ax.bar(industries_names_count, investment_counts, color='lightgreen')
        ax.set_xlabel('行业')
        ax.set_ylabel('投资次数')
        ax.set_title('各行业投资次数分布（前15名）')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱状图上显示数值
        for bar, count in zip(bars, investment_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/industry_investment_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 行业投资方数量分布图
        industries_investor_sorted = sorted(industry_detailed_stats.items(), 
                                          key=lambda x: x[1]['投资方数量'], reverse=True)[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        industries_names_investor = [item[0] for item in industries_investor_sorted]
        investor_counts = [item[1]['投资方数量'] for item in industries_investor_sorted]
        
        bars = ax.bar(industries_names_investor, investor_counts, color='orange')
        ax.set_xlabel('行业')
        ax.set_ylabel('投资方数量')
        ax.set_title('各行业投资方数量分布（前15名）')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱状图上显示数值
        for bar, count in zip(bars, investor_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/industry_investor_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到 {output_dir}")
    
    def generate_report(self, output_dir='analysis_results'):
        """生成详细的逐个行业分析报告"""
        if self.industry_amount_data is None:
            print("请先进行分析")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        overall_stats = self.industry_amount_data['overall_stats']
        industry_detailed_stats = self.industry_amount_data['industry_detailed_stats']
        all_industries = self.industry_amount_data['all_industries']
        
        # 生成Markdown报告
        report_content = f"""# 行业金额关系分析报告（逐个行业详细分析）

## 分析概况
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总记录数**: {overall_stats['总记录数']:,} 条
- **有效金额记录数**: {overall_stats['有效金额记录数']:,} 条
- **总投资金额**: {overall_stats['总投资金额']/100000000:,.2f} 亿元
- **平均投资金额**: {overall_stats['平均投资金额']/1000000:,.2f} 百万元
- **发现行业总数**: {len(all_industries)} 个
- **有投资记录的行业**: {overall_stats['有投资记录的行业']} 个

## 行业投资金额排名（前20名）

| 排名 | 行业 | 投资次数 | 总金额（亿元） | 平均金额（百万元） | 最大金额（亿元） | 投资方数量 | 主要投资方 |
|------|------|----------|----------------|-------------------|------------------|------------|------------|
"""
        
        # 添加行业排名表格
        industries_sorted = sorted(industry_detailed_stats.items(), 
                                 key=lambda x: x[1]['总金额'], reverse=True)[:20]
        
        for i, (industry, stats) in enumerate(industries_sorted, 1):
            main_investors = ", ".join([inv[0] for inv in stats['主要投资方'][:3]])
            report_content += f"| {i} | {industry} | {stats['投资次数']} | {stats['总金额']/100000000:,.2f} | {stats['平均金额']/1000000:,.2f} | {stats['最大金额']/100000000:,.2f} | {stats['投资方数量']} | {main_investors} |\n"
        
        # 添加投资次数排名
        report_content += "\n## 行业投资次数排名（前15名）\n\n"
        industries_count_sorted = sorted(industry_detailed_stats.items(), 
                                       key=lambda x: x[1]['投资次数'], reverse=True)[:15]
        
        for i, (industry, stats) in enumerate(industries_count_sorted, 1):
            report_content += f"{i}. **{industry}**: {stats['投资次数']} 次投资，总金额 {stats['总金额']/100000000:,.2f} 亿元，平均金额 {stats['平均金额']/1000000:,.2f} 百万元\n"
        
        # 添加关键发现
        report_content += "\n## 关键发现\n\n"
        
        # 找出投资金额最大的行业
        top_industry_by_amount = max(industry_detailed_stats.items(), key=lambda x: x[1]['总金额'])
        # 找出投资次数最多的行业
        top_industry_by_count = max(industry_detailed_stats.items(), key=lambda x: x[1]['投资次数'])
        # 找出平均投资金额最大的行业
        top_industry_by_avg = max(industry_detailed_stats.items(), key=lambda x: x[1]['平均金额'])
        # 找出投资方最多的行业
        top_industry_by_investors = max(industry_detailed_stats.items(), key=lambda x: x[1]['投资方数量'])
        
        report_content += f"""1. **投资金额最大的行业**: {top_industry_by_amount[0]}，总投资金额 {top_industry_by_amount[1]['总金额']/100000000:,.2f} 亿元
2. **投资次数最多的行业**: {top_industry_by_count[0]}，共 {top_industry_by_count[1]['投资次数']} 次投资
3. **平均投资金额最大的行业**: {top_industry_by_avg[0]}，平均投资金额 {top_industry_by_avg[1]['平均金额']/1000000:,.2f} 百万元
4. **投资方最多的行业**: {top_industry_by_investors[0]}，共有 {top_industry_by_investors[1]['投资方数量']} 个投资方

## 逐个行业详细分析

"""
        
        # 添加逐个行业详细分析
        for i, (industry, stats) in enumerate(industries_sorted, 1):
            report_content += f"\n### {i}. {industry}\n\n"
            report_content += f"**总体统计**:\n"
            report_content += f"- 投资次数: {stats['投资次数']} 次\n"
            report_content += f"- 总投资金额: {stats['总金额']/100000000:,.2f} 亿元\n"
            report_content += f"- 平均投资金额: {stats['平均金额']/1000000:,.2f} 百万元\n"
            report_content += f"- 最大单笔投资: {stats['最大金额']/100000000:,.2f} 亿元\n"
            report_content += f"- 最小单笔投资: {stats['最小金额']/10000:,.1f} 万元\n"
            report_content += f"- 投资方数量: {stats['投资方数量']} 个\n\n"
            
            report_content += f"**金额分布**:\n"
            report_content += f"- 小于100万: {stats['金额分布']['小于100万']} 次\n"
            report_content += f"- 100万-1000万: {stats['金额分布']['100万-1000万']} 次\n"
            report_content += f"- 1000万-1亿: {stats['金额分布']['1000万-1亿']} 次\n"
            report_content += f"- 1亿以上: {stats['金额分布']['1亿以上']} 次\n\n"
            
            report_content += f"**主要投资方（按投资金额排序）**:\n"
            for j, (investor, inv_stats) in enumerate(stats['主要投资方'][:5], 1):
                report_content += f"{j}. **{investor}**: {inv_stats['投资次数']} 次投资，总金额 {inv_stats['总金额']/100000000:,.2f} 亿元，占比 {inv_stats['投资占比']*100:.1f}%\n"
            
            report_content += "\n---\n"
        
        # 添加数据说明
        report_content += """
## 可视化图表

分析生成了以下可视化图表：
- `industry_amount_distribution.png`: 行业投资金额分布图
- `industry_investment_count.png`: 行业投资次数分布图

这些图表直观展示了不同行业的投资活跃度和资金规模。

## 数据说明

- 数据来源: investment_events.csv 和 investment_structure.csv
- 金额单位: 原始数据中的金额已统一转换为人民币元
- 行业分类: 基于投资结构数据中的行业信息进行关联分析
- 分析方法: 逐个行业详细分析，包含投资方统计和金额分布
"""
        
        # 保存报告
        report_file = f'{output_dir}/industry_amount_analysis_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存JSON数据
        json_file = f'{output_dir}/industry_amount_analysis_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.industry_amount_data, f, ensure_ascii=False, indent=2)
        
        print(f"分析报告已保存到: {report_file}")
        print(f"JSON数据已保存到: {json_file}")
        
        return report_file, json_file

def main():
    """主函数"""
    print("开始行业金额关系分析...")
    
    # 文件路径
    investment_events_file = "dataset/investment_events.csv"
    investment_structure_file = "dataset/investment_structure.csv"
    
    # 创建分析器
    analyzer = IndustryAmountAnalyzer(investment_events_file, investment_structure_file)
    
    # 加载数据
    if not analyzer.load_data():
        print("数据加载失败，请检查文件路径")
        return
    
    # 进行分析
    print("正在进行行业金额关系分析...")
    analysis_results = analyzer.analyze_industry_amount_relationships()
    
    if analysis_results:
        print("分析完成，生成可视化图表和报告...")
        
        # 生成可视化图表
        analyzer.generate_visualizations()
        
        # 生成报告
        analyzer.generate_report()
        
        # 输出关键统计信息
        overall_stats = analysis_results['overall_stats']
        print(f"\n=== 分析结果摘要 ===")
        print(f"总记录数: {overall_stats['总记录数']:,}")
        print(f"有效金额记录数: {overall_stats['有效金额记录数']:,}")
        print(f"总投资金额: {overall_stats['总投资金额']/100000000:,.2f} 亿元")
        print(f"平均投资金额: {overall_stats['平均投资金额']/1000000:,.2f} 百万元")
        print(f"涉及行业数量: {overall_stats['行业数量']}")
        
        print("\n行业金额关系分析完成！")
    else:
        print("分析失败")

if __name__ == "__main__":
    main()