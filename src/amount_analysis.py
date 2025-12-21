"""
金额数据高级统计分析脚本
分析investment_events.csv中的金额数据，并生成统计分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmountAnalyzer:
    def __init__(self, csv_path: str):
        """初始化分析器"""
        self.csv_path = csv_path
        self.df = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载并预处理数据"""
        logger.info(f"加载数据文件: {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            logger.info(f"成功读取 {len(self.df)} 条记录")
            
            # 数据预处理
            self._preprocess_data()
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def _preprocess_data(self):
        """数据预处理"""
        # 处理空值
        self.df = self.df.replace({np.nan: None})
        
        # 标准化金额格式
        self.df['金额数值'] = self.df['金额'].apply(self._standardize_amount)
        
        # 标记有效金额数据
        self.df['金额有效'] = self.df['金额数值'].notna()
        
        # 提取年份
        self.df['年份'] = self.df['融资时间'].apply(self._extract_year)
        
        logger.info(f"金额数据有效性: {self.df['金额有效'].sum()}/{len(self.df)} 条记录")
    
    def _standardize_amount(self, amount: str) -> float:
        """标准化金额格式"""
        if pd.isna(amount) or amount == '未披露':
            return None
        
        try:
            # 提取数字部分
            import re
            match = re.search(r'(\d+(?:\.\d+)?)(?:亿|万)?', str(amount))
            if match:
                num = float(match.group(1))
                if '亿' in str(amount):
                    num *= 100000000  # 转换为元
                elif '万' in str(amount):
                    num *= 10000  # 转换为元
                return num
        except:
            pass
        
        return None
    
    def _extract_year(self, date_str: str) -> int:
        """提取年份"""
        if pd.isna(date_str):
            return None
        
        try:
            return int(date_str.split('-')[0])
        except:
            return None
    
    def analyze_amount_distribution(self):
        """分析金额分布"""
        logger.info("开始分析金额分布")
        
        valid_amounts = self.df[self.df['金额有效']]['金额数值']
        
        if len(valid_amounts) == 0:
            logger.warning("没有有效的金额数据可供分析")
            return
        
        # 基础统计
        stats_dict = {
            '总记录数': len(self.df),
            '有效金额记录数': len(valid_amounts),
            '金额缺失率': 1 - len(valid_amounts) / len(self.df),
            '总金额': valid_amounts.sum(),
            '平均金额': valid_amounts.mean(),
            '中位数金额': valid_amounts.median(),
            '最大金额': valid_amounts.max(),
            '最小金额': valid_amounts.min(),
            '金额标准差': valid_amounts.std(),
            '金额偏度': stats.skew(valid_amounts.dropna()),
            '金额峰度': stats.kurtosis(valid_amounts.dropna())
        }
        
        # 金额区间分布
        amount_ranges = [
            (0, 1000000),      # 100万以下
            (1000000, 10000000),   # 100万-1000万
            (10000000, 50000000),  # 1000万-5000万
            (50000000, 100000000), # 5000万-1亿
            (100000000, float('inf'))  # 1亿以上
        ]
        
        range_labels = ['100万以下', '100万-1000万', '1000万-5000万', '5000万-1亿', '1亿以上']
        range_counts = []
        
        for min_val, max_val in amount_ranges:
            if max_val == float('inf'):
                count = len(valid_amounts[valid_amounts >= min_val])
            else:
                count = len(valid_amounts[(valid_amounts >= min_val) & (valid_amounts < max_val)])
            range_counts.append(count)
        
        stats_dict['金额区间分布'] = dict(zip(range_labels, range_counts))
        
        self.analysis_results['金额分布'] = stats_dict
        logger.info("金额分布分析完成")
    
    def analyze_amount_by_round(self):
        """按轮次分析金额"""
        logger.info("开始按轮次分析金额")
        
        valid_data = self.df[self.df['金额有效']]
        
        if len(valid_data) == 0:
            return
        
        round_analysis = {}
        
        for round_name in valid_data['轮次'].unique():
            if pd.isna(round_name):
                continue
                
            round_data = valid_data[valid_data['轮次'] == round_name]
            amounts = round_data['金额数值']
            
            round_stats = {
                '记录数': len(round_data),
                '总金额': amounts.sum(),
                '平均金额': amounts.mean(),
                '中位数金额': amounts.median(),
                '最大金额': amounts.max(),
                '最小金额': amounts.min(),
                '标准差': amounts.std()
            }
            
            round_analysis[round_name] = round_stats
        
        # 按平均金额排序
        sorted_rounds = sorted(round_analysis.items(), key=lambda x: x[1]['平均金额'], reverse=True)
        round_analysis = dict(sorted_rounds)
        
        self.analysis_results['轮次金额分析'] = round_analysis
        logger.info("轮次金额分析完成")
    
    def analyze_amount_by_time(self):
        """按时间分析金额趋势"""
        logger.info("开始按时间分析金额趋势")
        
        valid_data = self.df[self.df['金额有效'] & self.df['年份'].notna()]
        
        if len(valid_data) == 0:
            return
        
        time_analysis = {}
        
        for year in sorted(valid_data['年份'].unique()):
            year_data = valid_data[valid_data['年份'] == year]
            amounts = year_data['金额数值']
            
            year_stats = {
                '投资次数': len(year_data),
                '总金额': amounts.sum(),
                '平均金额': amounts.mean(),
                '中位数金额': amounts.median(),
                '最大单笔投资': amounts.max()
            }
            
            time_analysis[int(year)] = year_stats
        
        self.analysis_results['时间趋势分析'] = time_analysis
        logger.info("时间趋势分析完成")
    
    def analyze_amount_by_investor(self, top_n: int = 20):
        """按投资方分析金额"""
        logger.info("开始按投资方分析金额")
        
        valid_data = self.df[self.df['金额有效']]
        
        if len(valid_data) == 0:
            return
        
        # 展开投资方列表
        investor_amounts = {}
        
        for _, row in valid_data.iterrows():
            if pd.notna(row['投资方']):
                investors = str(row['投资方']).split()
                amount_per_investor = row['金额数值'] / len(investors) if len(investors) > 0 else row['金额数值']
                
                for investor in investors:
                    if investor not in investor_amounts:
                        investor_amounts[investor] = []
                    investor_amounts[investor].append(amount_per_investor)
        
        # 计算每个投资方的统计
        investor_analysis = {}
        
        for investor, amounts in investor_amounts.items():
            amounts_series = pd.Series(amounts)
            investor_analysis[investor] = {
                '投资次数': len(amounts),
                '总投资额': amounts_series.sum(),
                '平均投资额': amounts_series.mean(),
                '中位数投资额': amounts_series.median(),
                '最大单笔投资': amounts_series.max()
            }
        
        # 按总投资额排序，取前top_n
        sorted_investors = sorted(investor_analysis.items(), key=lambda x: x[1]['总投资额'], reverse=True)[:top_n]
        investor_analysis = dict(sorted_investors)
        
        self.analysis_results['投资方金额分析'] = investor_analysis
        logger.info("投资方金额分析完成")
    
    def create_visualizations(self, output_dir: str):
        """创建可视化图表"""
        logger.info("开始创建可视化图表")
        
        plt.style.use('seaborn-v0_8')
        
        # 1. 金额分布直方图
        valid_amounts = self.df[self.df['金额有效']]['金额数值']
        if len(valid_amounts) > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(valid_amounts, bins=50, alpha=0.7, edgecolor='black')
            plt.title('投资金额分布')
            plt.xlabel('金额（元）')
            plt.ylabel('频次')
            plt.yscale('log')  # 对数尺度更好地显示分布
            
            # 2. 轮次平均金额柱状图
            if '轮次金额分析' in self.analysis_results:
                plt.subplot(2, 2, 2)
                rounds = list(self.analysis_results['轮次金额分析'].keys())[:10]  # 前10个轮次
                avg_amounts = [self.analysis_results['轮次金额分析'][r]['平均金额'] for r in rounds]
                
                plt.bar(rounds, avg_amounts, alpha=0.7)
                plt.title('各轮次平均投资金额')
                plt.xlabel('轮次')
                plt.ylabel('平均金额（元）')
                plt.xticks(rotation=45)
            
            # 3. 时间趋势图
            if '时间趋势分析' in self.analysis_results:
                plt.subplot(2, 2, 3)
                years = list(self.analysis_results['时间趋势分析'].keys())
                avg_amounts = [self.analysis_results['时间趋势分析'][y]['平均金额'] for y in years]
                
                plt.plot(years, avg_amounts, marker='o')
                plt.title('年度平均投资金额趋势')
                plt.xlabel('年份')
                plt.ylabel('平均金额（元）')
            
            # 4. 金额区间分布饼图
            if '金额分布' in self.analysis_results and '金额区间分布' in self.analysis_results['金额分布']:
                plt.subplot(2, 2, 4)
                range_data = self.analysis_results['金额分布']['金额区间分布']
                labels = list(range_data.keys())
                sizes = list(range_data.values())
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.title('投资金额区间分布')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'amount_analysis_charts.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("可视化图表创建完成")
    
    def save_analysis_results(self, output_dir: str):
        """保存分析结果"""
        logger.info("保存分析结果")
        
        # 保存JSON结果
        output_file = os.path.join(output_dir, 'amount_analysis_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
        
        # 生成分析报告
        self._generate_report(output_dir)
        
        logger.info(f"分析结果已保存到: {output_dir}")
    
    def _generate_report(self, output_dir: str):
        """生成分析报告"""
        report_file = os.path.join(output_dir, 'amount_analysis_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 投资金额数据分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**数据源**: {self.csv_path}\n")
            f.write(f"**总记录数**: {len(self.df)}\n\n")
            
            if '金额分布' in self.analysis_results:
                stats = self.analysis_results['金额分布']
                f.write("## 1. 金额分布概况\n\n")
                f.write(f"- **有效金额记录**: {stats['有效金额记录数']}/{stats['总记录数']} ({stats['金额缺失率']*100:.1f}%缺失率)\n")
                f.write(f"- **总投资金额**: {stats['总金额']:,.0f} 元\n")
                f.write(f"- **平均投资金额**: {stats['平均金额']:,.0f} 元\n")
                f.write(f"- **中位数投资金额**: {stats['中位数金额']:,.0f} 元\n")
                f.write(f"- **金额标准差**: {stats['金额标准差']:,.0f} 元\n")
                f.write(f"- **金额偏度**: {stats['金额偏度']:.2f}\n")
                f.write(f"- **金额峰度**: {stats['金额峰度']:.2f}\n\n")
            
            if '轮次金额分析' in self.analysis_results:
                f.write("## 2. 轮次金额分析\n\n")
                f.write("| 轮次 | 投资次数 | 平均金额（元） | 中位数金额（元） | 最大金额（元） |\n")
                f.write("|------|----------|---------------|-----------------|---------------|\n")
                
                for round_name, stats in self.analysis_results['轮次金额分析'].items():
                    f.write(f"| {round_name} | {stats['记录数']} | {stats['平均金额']:,.0f} | {stats['中位数金额']:,.0f} | {stats['最大金额']:,.0f} |\n")
                f.write("\n")
            
            if '时间趋势分析' in self.analysis_results:
                f.write("## 3. 时间趋势分析\n\n")
                f.write("| 年份 | 投资次数 | 总金额（元） | 平均金额（元） |\n")
                f.write("|------|----------|-------------|---------------|\n")
                
                for year, stats in sorted(self.analysis_results['时间趋势分析'].items()):
                    f.write(f"| {year} | {stats['投资次数']} | {stats['总金额']:,.0f} | {stats['平均金额']:,.0f} |\n")
                f.write("\n")
            
            if '投资方金额分析' in self.analysis_results:
                f.write("## 4. 投资方金额分析（前20名）\n\n")
                f.write("| 投资方 | 投资次数 | 总投资额（元） | 平均投资额（元） |\n")
                f.write("|--------|----------|---------------|-----------------|\n")
                
                for investor, stats in self.analysis_results['投资方金额分析'].items():
                    f.write(f"| {investor} | {stats['投资次数']} | {stats['总投资额']:,.0f} | {stats['平均投资额']:,.0f} |\n")
    
    def run_complete_analysis(self, output_dir: str = None):
        """运行完整分析流程"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.csv_path), '..', 'analysis_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.load_data()
        self.analyze_amount_distribution()
        self.analyze_amount_by_round()
        self.analyze_amount_by_time()
        self.analyze_amount_by_investor()
        self.create_visualizations(output_dir)
        self.save_analysis_results(output_dir)
        
        logger.info("完整分析流程完成")
        return self.analysis_results

def main():
    """主函数"""
    # 文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dataset", "investment_events.csv")
    output_dir = os.path.join(base_dir, "analysis_results")
    
    # 运行分析
    analyzer = AmountAnalyzer(csv_path)
    results = analyzer.run_complete_analysis(output_dir)
    
    print("分析完成！结果已保存到:", output_dir)

if __name__ == "__main__":
    main()