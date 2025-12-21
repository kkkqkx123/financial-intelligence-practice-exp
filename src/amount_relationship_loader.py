"""
金额关系数据加载器
将金额数据与轮次、行业、投资方等属性的关系存入Neo4j数据库
"""

import pandas as pd
import numpy as np
from py2neo import Graph
import os
import logging
from typing import Dict, List, Any
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmountRelationshipLoader:
    def __init__(self, uri: str, user: str, password: str):
        """初始化Neo4j连接"""
        self.graph = Graph(uri, auth=(user, password))
        
    def close(self):
        """关闭连接"""
        pass
    
    def load_amount_relationships(self, csv_path: str, analysis_results_path: str):
        """加载金额关系数据"""
        logger.info("开始加载金额关系数据")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"成功读取 {len(df)} 条投资事件记录")
            
            # 读取分析结果
            with open(analysis_results_path, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
            
            # 数据预处理
            df = self._preprocess_data(df)
            
            # 创建金额统计节点
            self._create_amount_statistics_nodes(analysis_results)
            
            # 创建轮次金额关系
            self._create_round_amount_relationships(analysis_results)
            
            # 创建时间金额关系
            self._create_time_amount_relationships(analysis_results)
            
            # 创建投资方金额关系
            self._create_investor_amount_relationships(analysis_results)
            
            # 创建金额分布关系
            self._create_amount_distribution_relationships(analysis_results)
            
            logger.info("金额关系数据加载完成")
            
        except Exception as e:
            logger.error(f"加载金额关系数据失败: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 处理空值
        df = df.replace({np.nan: None})
        
        # 标准化金额格式
        df['金额数值'] = df['金额'].apply(self._standardize_amount)
        
        # 标记有效金额数据
        df['金额有效'] = df['金额数值'].notna()
        
        # 提取年份
        df['年份'] = df['融资时间'].apply(self._extract_year)
        
        return df
    
    def _standardize_amount(self, amount: str) -> float:
        """标准化金额格式"""
        if pd.isna(amount) or amount == '未披露':
            return None
        
        try:
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
    
    def _create_amount_statistics_nodes(self, analysis_results: Dict):
        """创建金额统计节点"""
        logger.info("创建金额统计节点")
        
        if '金额分布' in analysis_results:
            stats = analysis_results['金额分布']
            
            query = """
            MERGE (s:金额统计 {统计类型: '总体统计'})
            SET s.总记录数 = $total_records,
                s.有效金额记录数 = $valid_records,
                s.金额缺失率 = $missing_rate,
                s.总投资金额 = $total_amount,
                s.平均投资金额 = $avg_amount,
                s.中位数投资金额 = $median_amount,
                s.最大投资金额 = $max_amount,
                s.最小投资金额 = $min_amount,
                s.金额标准差 = $std_amount,
                s.金额偏度 = $skewness,
                s.金额峰度 = $kurtosis,
                s.数据来源 = '金额分析脚本',
                s.创建时间 = datetime()
            RETURN s
            """
            
            self.graph.run(query,
                   total_records=stats['总记录数'],
                   valid_records=stats['有效金额记录数'],
                   missing_rate=stats['金额缺失率'],
                   total_amount=stats['总金额'],
                   avg_amount=stats['平均金额'],
                   median_amount=stats['中位数金额'],
                   max_amount=stats['最大金额'],
                   min_amount=stats['最小金额'],
                   std_amount=stats['金额标准差'],
                   skewness=stats['金额偏度'],
                   kurtosis=stats['金额峰度'])
    
    def _create_round_amount_relationships(self, analysis_results: Dict):
        """创建轮次金额关系"""
        logger.info("创建轮次金额关系")
        
        if '轮次金额分析' in analysis_results:
            for round_name, stats in analysis_results['轮次金额分析'].items():
                # 创建轮次统计节点
                query = """
                MERGE (r:轮次统计 {轮次名称: $round_name})
                SET r.投资次数 = $investment_count,
                    r.总金额 = $total_amount,
                    r.平均金额 = $avg_amount,
                    r.中位数金额 = $median_amount,
                    r.最大金额 = $max_amount,
                    r.最小金额 = $min_amount,
                    r.标准差 = $std_amount,
                    r.数据来源 = '金额分析脚本',
                    r.创建时间 = datetime()
                RETURN r
                """
                
                self.graph.run(query,
                       round_name=round_name.strip(),
                       investment_count=stats['记录数'],
                       total_amount=stats['总金额'],
                       avg_amount=stats['平均金额'],
                       median_amount=stats['中位数金额'],
                       max_amount=stats['最大金额'],
                       min_amount=stats['最小金额'],
                       std_amount=stats['标准差'])
                
                # 连接到轮次节点
                query = """
                MATCH (rnd:轮次 {轮次名称: $round_name})
                MATCH (rs:轮次统计 {轮次名称: $round_name})
                MERGE (rnd)-[rel:金额统计]->(rs)
                SET rel.统计类型 = '轮次金额分析',
                    rel.数据来源 = '金额分析脚本',
                    rel.创建时间 = datetime()
                RETURN rel
                """
                
                self.graph.run(query, round_name=round_name.strip())
    
    def _create_time_amount_relationships(self, analysis_results: Dict):
        """创建时间金额关系"""
        logger.info("创建时间金额关系")
        
        if '时间趋势分析' in analysis_results:
            for year, stats in analysis_results['时间趋势分析'].items():
                # 创建年度统计节点
                query = """
                MERGE (y:年度统计 {年份: $year})
                SET y.投资次数 = $investment_count,
                    y.总金额 = $total_amount,
                    y.平均金额 = $avg_amount,
                    y.中位数金额 = $median_amount,
                    y.最大单笔投资 = $max_investment,
                    y.数据来源 = '金额分析脚本',
                    y.创建时间 = datetime()
                RETURN y
                """
                
                self.graph.run(query,
                       year=int(year),
                       investment_count=stats['投资次数'],
                       total_amount=stats['总金额'],
                       avg_amount=stats['平均金额'],
                       median_amount=stats['中位数金额'],
                       max_investment=stats['最大单笔投资'])
    
    def _create_investor_amount_relationships(self, analysis_results: Dict):
        """创建投资方金额关系"""
        logger.info("创建投资方金额关系")
        
        if '投资方金额分析' in analysis_results:
            for investor, stats in analysis_results['投资方金额分析'].items():
                # 创建投资方统计节点
                query = """
                MERGE (is:投资方统计 {投资方名称: $investor_name})
                SET is.投资次数 = $investment_count,
                    is.总投资额 = $total_amount,
                    is.平均投资额 = $avg_amount,
                    is.中位数投资额 = $median_amount,
                    is.最大单笔投资 = $max_investment,
                    is.数据来源 = '金额分析脚本',
                    is.创建时间 = datetime()
                RETURN is
                """
                
                self.graph.run(query,
                       investor_name=investor,
                       investment_count=stats['投资次数'],
                       total_amount=stats['总投资额'],
                       avg_amount=stats['平均投资额'],
                       median_amount=stats['中位数投资额'],
                       max_investment=stats['最大单笔投资'])
                
                # 连接到投资方节点
                query = """
                MATCH (i:投资方 {投资方名称: $investor_name})
                MATCH (is:投资方统计 {投资方名称: $investor_name})
                MERGE (i)-[rel:金额统计]->(is)
                SET rel.统计类型 = '投资方金额分析',
                    rel.数据来源 = '金额分析脚本',
                    rel.创建时间 = datetime()
                RETURN rel
                """
                
                self.graph.run(query, investor_name=investor)
    
    def _create_amount_distribution_relationships(self, analysis_results: Dict):
        """创建金额分布关系"""
        logger.info("创建金额分布关系")
        
        if '金额分布' in analysis_results and '金额区间分布' in analysis_results['金额分布']:
            range_data = analysis_results['金额分布']['金额区间分布']
            
            for range_name, count in range_data.items():
                # 创建金额区间节点
                query = """
                MERGE (ar:金额区间 {区间名称: $range_name})
                SET ar.投资次数 = $investment_count,
                    ar.数据来源 = '金额分析脚本',
                    ar.创建时间 = datetime()
                RETURN ar
                """
                
                self.graph.run(query, range_name=range_name, investment_count=count)
                
                # 连接到总体统计节点
                query = """
                MATCH (s:金额统计 {统计类型: '总体统计'})
                MATCH (ar:金额区间 {区间名称: $range_name})
                MERGE (s)-[rel:包含区间]->(ar)
                SET rel.统计类型 = '金额分布',
                    rel.数据来源 = '金额分析脚本',
                    rel.创建时间 = datetime()
                RETURN rel
                """
                
                self.graph.run(query, range_name=range_name)
    
    def create_advanced_queries(self):
        """创建高级查询示例"""
        logger.info("创建高级查询示例")
        
        # 查询不同轮次的金额分布
        query1 = """
        MATCH (rs:轮次统计)
        RETURN rs.轮次名称 AS 轮次, 
               rs.投资次数 AS 投资次数,
               rs.平均金额 AS 平均金额
        ORDER BY rs.平均金额 DESC
        """
        
        # 查询投资方投资金额排名
        query2 = """
        MATCH (is:投资方统计)
        RETURN is.投资方名称 AS 投资方,
               is.总投资额 AS 总投资额,
               is.投资次数 AS 投资次数
        ORDER BY is.总投资额 DESC
        LIMIT 20
        """
        
        # 查询年度投资趋势
        query3 = """
        MATCH (y:年度统计)
        RETURN y.年份 AS 年份,
               y.总金额 AS 总金额,
               y.投资次数 AS 投资次数
        ORDER BY y.年份
        """
        
        # 保存查询到文件
        queries = {
            "轮次金额分布查询": query1,
            "投资方金额排名查询": query2,
            "年度投资趋势查询": query3
        }
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "advanced_queries.json"), 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        
        logger.info("高级查询示例已保存")

def main():
    """主函数"""
    # Neo4j连接配置
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "1234567kk"
    
    # 文件路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "dataset", "investment_events.csv")
    ANALYSIS_RESULTS_PATH = os.path.join(BASE_DIR, "analysis_results", "amount_analysis_results.json")
    
    # 创建加载器
    loader = AmountRelationshipLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # 加载金额关系数据
        loader.load_amount_relationships(CSV_PATH, ANALYSIS_RESULTS_PATH)
        
        # 创建高级查询示例
        loader.create_advanced_queries()
        
        logger.info("金额关系数据加载完成！")
        
    except Exception as e:
        logger.error(f"金额关系数据加载过程中发生错误: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    main()