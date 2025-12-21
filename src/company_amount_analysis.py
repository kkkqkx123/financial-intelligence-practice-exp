"""
融资方（公司）与金额的关系分析
分析各融资方获得的投资金额分布
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CompanyAmountAnalyzer:
    def __init__(self):
        self.investment_events_df = None
        self.company_stats = {}
        self.overall_stats = {}
        
    def load_data(self):
        """加载投资事件数据"""
        try:
            # 读取投资事件数据
            self.investment_events_df = pd.read_csv('dataset/investment_events.csv', encoding='utf-8')
            print(f"成功加载 {len(self.investment_events_df)} 条投资事件记录")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def preprocess_amount(self, amount_str):
        """预处理金额字段"""
        if pd.isna(amount_str) or amount_str == '未披露':
            return None
        
        try:
            # 提取数字部分
            import re
            match = re.search(r'(\d+(?:\.\d+)?)(?:亿|万)?', str(amount_str))
            if match:
                num = float(match.group(1))
                if '亿' in str(amount_str):
                    num *= 100000000  # 转换为元
                elif '万' in str(amount_str):
                    num *= 10000  # 转换为元
                return num
        except:
            pass
        
        return None
    
    def analyze_company_amounts(self):
        """分析各融资方的投资金额"""
        if self.investment_events_df is None:
            print("请先加载数据")
            return
        
        # 预处理金额字段
        self.investment_events_df['金额数值'] = self.investment_events_df['金额'].apply(self.preprocess_amount)
        
        # 过滤有效金额记录
        valid_amount_df = self.investment_events_df[self.investment_events_df['金额数值'].notna()]
        
        # 按融资方分组统计
        company_groups = valid_amount_df.groupby('融资方')
        
        # 统计各融资方的投资情况
        for company, group in company_groups:
            amounts = group['金额数值'].tolist()
            
            self.company_stats[company] = {
                '投资次数': len(group),
                '总金额': sum(amounts),
                '平均金额': sum(amounts) / len(amounts) if amounts else 0,
                '最大金额': max(amounts) if amounts else 0,
                '最小金额': min(amounts) if amounts else 0,
                '金额列表': amounts
            }
        
        # 计算总体统计
        all_amounts = valid_amount_df['金额数值'].tolist()
        
        self.overall_stats = {
            '总记录数': len(self.investment_events_df),
            '有效金额记录数': len(valid_amount_df),
            '融资方数量': len(self.company_stats),
            '总投资金额': sum(all_amounts) if all_amounts else 0,
            '平均投资金额': sum(all_amounts) / len(all_amounts) if all_amounts else 0,
            '最大投资金额': max(all_amounts) if all_amounts else 0,
            '最小投资金额': min(all_amounts) if all_amounts else 0
        }
        
        print(f"分析完成，共统计 {len(self.company_stats)} 个融资方")
        print(f"总投资金额: {self.overall_stats['总投资金额']:,.0f} 元")
        print(f"平均投资金额: {self.overall_stats['平均投资金额']:,.0f} 元")
    
    def generate_visualizations(self, output_dir='analysis_results'):
        """生成可视化图表"""
        if not self.company_stats:
            print("请先进行分析")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 融资方投资金额排名（前20）
        top_companies = sorted(self.company_stats.items(), 
                              key=lambda x: x[1]['总金额'], reverse=True)[:20]
        
        plt.figure(figsize=(12, 8))
        companies = [c[0] for c in top_companies]
        amounts = [c[1]['总金额'] / 100000000 for c in top_companies]  # 转换为亿元
        
        plt.barh(range(len(companies)), amounts)
        plt.yticks(range(len(companies)), companies)
        plt.xlabel('投资金额 (亿元)')
        plt.title('融资方投资金额排名 (前20)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/company_amount_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 融资方投资次数排名（前20）
        top_companies_count = sorted(self.company_stats.items(), 
                                   key=lambda x: x[1]['投资次数'], reverse=True)[:20]
        
        plt.figure(figsize=(12, 8))
        companies = [c[0] for c in top_companies_count]
        counts = [c[1]['投资次数'] for c in top_companies_count]
        
        plt.barh(range(len(companies)), counts)
        plt.yticks(range(len(companies)), companies)
        plt.xlabel('投资次数')
        plt.title('融资方投资次数排名 (前20)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/company_investment_count_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 投资金额分布直方图
        plt.figure(figsize=(10, 6))
        amounts = [stats['总金额'] / 100000000 for stats in self.company_stats.values()]  # 转换为亿元
        plt.hist(amounts, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('投资金额 (亿元)')
        plt.ylabel('融资方数量')
        plt.title('融资方投资金额分布')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/company_amount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化图表生成完成")
    
    def generate_report(self, output_dir='analysis_results'):
        """生成分析报告"""
        if not self.company_stats:
            print("请先进行分析")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告内容
        report_content = f"""# 融资方投资金额分析报告

## 总体统计
- **总记录数**: {self.overall_stats['总记录数']:,} 条
- **有效金额记录数**: {self.overall_stats['有效金额记录数']:,} 条
- **融资方数量**: {self.overall_stats['融资方数量']:,} 个
- **总投资金额**: {self.overall_stats['总投资金额']:,.0f} 元 ({self.overall_stats['总投资金额']/100000000:,.2f} 亿元)
- **平均投资金额**: {self.overall_stats['平均投资金额']:,.0f} 元 ({self.overall_stats['平均投资金额']/100000000:,.2f} 亿元)
- **最大投资金额**: {self.overall_stats['最大投资金额']:,.0f} 元 ({self.overall_stats['最大投资金额']/100000000:,.2f} 亿元)
- **最小投资金额**: {self.overall_stats['最小投资金额']:,.0f} 元

## 融资方投资金额排名 (前10)

| 排名 | 融资方 | 投资次数 | 总投资金额 (亿元) | 平均投资金额 (亿元) |
|------|--------|----------|-------------------|---------------------|
"""
        
        # 添加前10名融资方
        top_companies = sorted(self.company_stats.items(), 
                              key=lambda x: x[1]['总金额'], reverse=True)[:10]
        
        for i, (company, stats) in enumerate(top_companies, 1):
            report_content += f"| {i} | {company} | {stats['投资次数']} | {stats['总金额']/100000000:,.2f} | {stats['平均金额']/100000000:,.2f} |\n"
        
        report_content += """
## 融资方投资次数排名 (前10)

| 排名 | 融资方 | 投资次数 | 总投资金额 (亿元) | 平均投资金额 (亿元) |
|------|--------|----------|-------------------|---------------------|
"""
        
        # 添加投资次数前10名融资方
        top_companies_count = sorted(self.company_stats.items(), 
                                   key=lambda x: x[1]['投资次数'], reverse=True)[:10]
        
        for i, (company, stats) in enumerate(top_companies_count, 1):
            report_content += f"| {i} | {company} | {stats['投资次数']} | {stats['总金额']/100000000:,.2f} | {stats['平均金额']/100000000:,.2f} |\n"
        
        report_content += """
## 关键发现

### 1. 投资金额集中度分析
- 前10大融资方占据了总投资金额的 {top10_percentage:.1f}%
- 平均每个融资方获得 {avg_amount_per_company:,.0f} 元投资

### 2. 投资频次分析
- 平均每个融资方获得 {avg_investments_per_company:.1f} 次投资
- 投资次数最多的融资方获得了 {max_investments} 次投资

### 3. 金额分布特征
- 投资金额主要集中在 {amount_range} 范围内
- 存在明显的融资方分层现象

## 可视化图表

生成的图表包括：
1. **融资方投资金额排名图** - 显示前20名融资方的总投资金额
2. **融资方投资次数排名图** - 显示前20名融资方的投资次数
3. **融资方投资金额分布图** - 显示投资金额的分布情况

报告生成时间: {timestamp}
""".format(
            top10_percentage=sum([stats['总金额'] for _, stats in top_companies[:10]]) / self.overall_stats['总投资金额'] * 100,
            avg_amount_per_company=self.overall_stats['总投资金额'] / len(self.company_stats),
            avg_investments_per_company=self.overall_stats['有效金额记录数'] / len(self.company_stats),
            max_investments=max([stats['投资次数'] for stats in self.company_stats.values()]),
            amount_range="小额到中额",
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 保存报告
        report_path = f'{output_dir}/company_amount_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告已保存到: {report_path}")
        
        # 保存JSON结果
        result_data = {
            'overall_stats': self.overall_stats,
            'company_stats': self.company_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        json_path = f'{output_dir}/company_amount_analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"分析结果已保存到: {json_path}")

def main():
    """主函数"""
    print("开始融资方投资金额分析...")
    
    analyzer = CompanyAmountAnalyzer()
    
    # 加载数据
    if not analyzer.load_data():
        return
    
    # 进行分析
    analyzer.analyze_company_amounts()
    
    # 生成可视化图表
    analyzer.generate_visualizations()
    
    # 生成报告
    analyzer.generate_report()
    
    print("融资方投资金额分析完成！")

if __name__ == "__main__":
    main()