"""
行业关系数据加载器
将行业金额关系数据加载到Neo4j数据库
"""

import json
import os
from datetime import datetime
from py2neo import Graph
from typing import Dict, List, Any
import logging

class IndustryRelationshipLoader:
    def __init__(self, uri, user, password):
        self.graph = Graph(uri, auth=(user, password))
        
    def close(self):
        # py2neo的Graph对象不需要显式关闭
        pass
    
    def load_industry_data_to_neo4j(self, json_file_path):
        """将行业金额关系数据加载到Neo4j"""
        
        # 读取JSON数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            industry_data = json.load(f)
        
        overall_stats = industry_data['overall_stats']
        industry_stats = industry_data['industry_detailed_stats']
        
        # 1. 创建行业统计节点
        self._create_industry_statistics_node(overall_stats)
        
        # 2. 创建行业节点和行业统计关系
        for industry_name, stats in industry_stats.items():
            self._create_industry_node_and_relationships(industry_name, stats)
        
        # 3. 创建投资方-行业关系
        investor_industry_data = self._extract_investor_industry_data(industry_stats)
        for investor_name, industry_data in investor_industry_data.items():
            for industry_name, stats in industry_data.items():
                self._create_investor_industry_relationship(investor_name, industry_name, stats)
        
        # 4. 创建高级查询示例
        self._create_advanced_queries()
        
        print("行业关系数据加载完成！")
    
    def _create_industry_statistics_node(self, data: Dict[str, Any]):
        """创建行业统计节点"""
        query = """
        MERGE (s:行业统计 {统计类型: '总体统计'})
        SET s.总记录数 = $total_records,
            s.有效金额记录数 = $valid_amount_records,
            s.总投资金额 = $total_investment_amount,
            s.平均投资金额 = $avg_investment_amount,
            s.行业数量 = $industry_count,
            s.数据来源 = '行业金额分析',
            s.创建时间 = datetime()
        RETURN s
        """
        self.graph.run(query, 
                   total_records=data['总记录数'],
                   valid_amount_records=data['有效金额记录数'],
                   total_investment_amount=data['总投资金额'],
                   avg_investment_amount=data['平均投资金额'],
                   industry_count=data['行业数量'])
    
        logging.info("创建行业统计节点: 总体统计")
    
    def _create_industry_node_and_relationships(self, industry_name, stats):
        """创建行业节点和统计关系"""
        # 创建行业节点
        query = """
        MERGE (i:行业 {行业名称: $industry_name})
        SET i.投资次数 = $investment_count,
            i.总金额 = $total_amount,
            i.平均金额 = $avg_amount,
            i.最大金额 = $max_amount,
            i.最小金额 = $min_amount
        """
        self.graph.run(query, 
               industry_name=industry_name,
               investment_count=stats['投资次数'],
               total_amount=stats['总金额'],
               avg_amount=stats['平均金额'],
               max_amount=stats['最大金额'],
               min_amount=stats['最小金额'])
        
        # 创建行业统计关系
        query = """
        MATCH (i:行业 {行业名称: $industry_name})
        MATCH (s:行业统计 {统计类型: '总体统计'})
        MERGE (i)-[r:属于统计]->(s)
        SET r.投资次数 = $investment_count,
            r.总金额 = $total_amount
        """
        self.graph.run(query, 
               industry_name=industry_name,
               investment_count=stats['投资次数'],
               total_amount=stats['总金额'])
    
    def _extract_investor_industry_data(self, industry_stats):
        """从行业统计数据中提取投资方-行业关系数据"""
        investor_industry_data = {}
        
        for industry_name, industry_data in industry_stats.items():
            if '投资方统计' in industry_data:
                for investor_name, investor_stats in industry_data['投资方统计'].items():
                    if investor_name not in investor_industry_data:
                        investor_industry_data[investor_name] = {}
                    investor_industry_data[investor_name][industry_name] = investor_stats
        
        return investor_industry_data
    
    def _create_investor_industry_relationship(self, investor_name, industry_name, stats):
        """创建投资方-行业投资关系"""
        # 首先创建投资方节点
        query = """
        MERGE (inv:投资方 {投资方名称: $investor_name})
        """
        self.graph.run(query, investor_name=investor_name)
        
        # 然后创建投资关系
        query = """
        MATCH (inv:投资方 {投资方名称: $investor_name})
        MATCH (ind:行业 {行业名称: $industry_name})
        MERGE (inv)-[r:投资于行业]->(ind)
        SET r.投资次数 = $investment_count,
            r.总金额 = $total_amount,
            r.平均金额 = $avg_amount
        """
        self.graph.run(query, 
               investor_name=investor_name,
               industry_name=industry_name,
               investment_count=stats['投资次数'],
               total_amount=stats['总金额'],
               avg_amount=stats['平均金额'])
    
    def _create_advanced_queries(self):
        """创建高级查询示例"""
        
        # 创建查询示例节点
        queries = [
            {
                "name": "查询各行业投资金额排名",
                "query": "MATCH (i:行业) RETURN i.行业名称 AS 行业, i.总金额 AS 总投资金额, i.投资次数 AS 投资次数 ORDER BY i.总金额 DESC LIMIT 10"
            },
            {
                "name": "查询投资次数最多的行业",
                "query": "MATCH (i:行业) RETURN i.行业名称 AS 行业, i.投资次数 AS 投资次数, i.总金额 AS 总投资金额 ORDER BY i.投资次数 DESC LIMIT 10"
            },
            {
                "name": "查询特定投资方的行业投资分布",
                "query": "MATCH (inv:投资方)-[r:投资于行业]->(i:行业) WHERE inv.投资方名称 = 'Lightspeed Venture Partners' RETURN i.行业名称 AS 行业, r.投资次数 AS 投资次数, r.总金额 AS 投资金额 ORDER BY r.总金额 DESC"
            },
            {
                "name": "查询平均投资金额最高的行业",
                "query": "MATCH (i:行业) WHERE i.投资次数 > 0 RETURN i.行业名称 AS 行业, i.平均金额 AS 平均投资金额, i.投资次数 AS 投资次数 ORDER BY i.平均金额 DESC LIMIT 10"
            },
            {
                "name": "查询投资方在多个行业的投资情况",
                "query": "MATCH (inv:投资方)-[r:投资于行业]->(i:行业) WITH inv, COUNT(i) AS 投资行业数, SUM(r.投资次数) AS 总投资次数 WHERE 投资行业数 >= 3 RETURN inv.投资方名称 AS 投资方, 投资行业数, 总投资次数 ORDER BY 投资行业数 DESC"
            }
        ]
        
        for i, q in enumerate(queries):
            query_node = """
            MERGE (q:查询示例 {查询名称: $query_name})
            SET q.查询语句 = $query_statement,
                q.示例编号 = $example_id
            """
            self.graph.run(query_node, 
                   query_name=q["name"],
                   query_statement=q["query"],
                   example_id=i+1)

def main():
    """主函数"""
    
    # Neo4j连接配置
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "1234567kk"
    
    # JSON数据文件路径
    json_file_path = "analysis_results/industry_amount_analysis_results.json"
    
    if not os.path.exists(json_file_path):
        print(f"JSON数据文件不存在: {json_file_path}")
        return
    
    print("开始加载行业关系数据到Neo4j...")
    
    try:
        loader = IndustryRelationshipLoader(uri, user, password)
        
        # 测试连接
        result = loader.graph.run("RETURN 'Neo4j连接成功' AS message")
        print(result.data()[0]["message"])
        
        # 加载数据
        loader.load_industry_data_to_neo4j(json_file_path)
        
        print("行业关系数据加载完成！")
        
        # 显示一些查询示例
        print("\n=== 查询示例 ===")
        print("1. 查询各行业投资金额排名:")
        print("   MATCH (i:行业) RETURN i.行业名称 AS 行业, i.总金额 AS 总投资金额 ORDER BY i.总金额 DESC LIMIT 10")
        print("\n2. 查询投资次数最多的行业:")
        print("   MATCH (i:行业) RETURN i.行业名称 AS 行业, i.投资次数 AS 投资次数 ORDER BY i.投资次数 DESC LIMIT 10")
        print("\n3. 查询特定投资方的行业投资分布:")
        print("   MATCH (inv:投资方)-[r:投资于行业]->(i:行业) WHERE inv.投资方名称 = 'Lightspeed Venture Partners' RETURN i.行业名称 AS 行业, r.投资次数 AS 投资次数")
        
    except Exception as e:
        print(f"加载失败: {e}")
    finally:
        if 'loader' in locals():
            loader.close()

if __name__ == "__main__":
    main()