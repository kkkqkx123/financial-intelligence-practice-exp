#!/usr/bin/env python3
"""
测试Neo4j连接和基本数据导入
"""

import json
from datetime import datetime
from pathlib import Path

try:
    from py2neo import Graph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("错误: py2neo库未安装")
    exit(1)

def test_neo4j_connection():
    """测试Neo4j连接"""
    print("=== 测试Neo4j连接 ===")
    
    # Neo4j连接配置
    config = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "1234567kk"
    }
    
    try:
        graph = Graph(config["uri"], auth=(config["username"], config["password"]))
        print("✅ 成功连接到Neo4j数据库")
        
        # 测试查询
        result = graph.run("MATCH (n) RETURN count(n) as node_count").data()
        node_count = result[0]["node_count"] if result else 0
        print(f"当前数据库中有 {node_count} 个节点")
        
        return graph
        
    except Exception as e:
        print(f"❌ 连接Neo4j失败: {e}")
        return None

def test_basic_import(graph):
    """测试基本数据导入"""
    print("\n=== 测试基本数据导入 ===")
    
    try:
        # 清除现有数据
        print("清除现有数据...")
        graph.run("MATCH (n) DETACH DELETE n")
        print("✅ 数据清除完成")
        
        # 创建测试数据
        print("创建测试数据...")
        
        # 创建公司节点
        company_query = """
        CREATE (c:公司 {
            公司ID: 'test_company_001',
            公司名称: '测试公司',
            股票代码: 'TEST001',
            成立时间: '2020-01-01',
            注册资本: '1000万',
            所属行业: '科技',
            数据来源: '测试导入'
        })
        """
        graph.run(company_query)
        
        # 创建投资方节点
        investor_query = """
        CREATE (i:投资方 {
            投资方ID: 'test_investor_001',
            投资方名称: '测试投资方',
            投资方类型: 'VC',
            管理资金规模: '10亿',
            数据来源: '测试导入'
        })
        """
        graph.run(investor_query)
        
        # 创建投资关系
        investment_query = """
        MATCH (c:公司 {公司ID: 'test_company_001'}), (i:投资方 {投资方ID: 'test_investor_001'})
        CREATE (i)-[r:投资 {
            投资金额: '500万',
            融资轮次: 'A轮',
            投资时间: '2023-01-01',
            数据来源: '测试导入'
        }]->(c)
        """
        graph.run(investment_query)
        
        print("✅ 测试数据导入完成")
        
        # 验证数据
        print("验证导入的数据...")
        
        # 查询节点数量
        node_count = graph.run("MATCH (n) RETURN count(n) as count").data()[0]["count"]
        print(f"总节点数: {node_count}")
        
        # 查询关系数量
        rel_count = graph.run("MATCH ()-[r]-() RETURN count(r) as count").data()[0]["count"]
        print(f"总关系数: {rel_count}")
        
        # 查询具体数据
        company_count = graph.run("MATCH (c:公司) RETURN count(c) as count").data()[0]["count"]
        investor_count = graph.run("MATCH (i:投资方) RETURN count(i) as count").data()[0]["count"]
        investment_count = graph.run("MATCH ()-[r:投资]-() RETURN count(r) as count").data()[0]["count"]
        
        print(f"公司节点数: {company_count}")
        print(f"投资方节点数: {investor_count}")
        print(f"投资关系数: {investment_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据导入失败: {e}")
        return False

def run_test_queries(graph):
    """运行测试查询"""
    print("\n=== 运行测试查询 ===")
    
    queries = [
        {
            "name": "所有节点",
            "query": "MATCH (n) RETURN n LIMIT 10"
        },
        {
            "name": "所有关系",
            "query": "MATCH ()-[r]-() RETURN r LIMIT 10"
        },
        {
            "name": "公司节点",
            "query": "MATCH (c:公司) RETURN c LIMIT 5"
        },
        {
            "name": "投资方节点",
            "query": "MATCH (i:投资方) RETURN i LIMIT 5"
        },
        {
            "name": "投资关系",
            "query": "MATCH (i:投资方)-[r:投资]->(c:公司) RETURN i.投资方名称, r.投资金额, c.公司名称"
        }
    ]
    
    for query_info in queries:
        print(f"\n查询: {query_info['name']}")
        try:
            result = graph.run(query_info['query']).data()
            print(f"结果数量: {len(result)}")
            if result:
                for i, record in enumerate(result[:3]):  # 只显示前3条结果
                    print(f"  {i+1}. {record}")
            else:
                print("  无结果")
        except Exception as e:
            print(f"  查询失败: {e}")

def main():
    """主函数"""
    print("开始Neo4j连接和数据导入测试")
    
    # 测试连接
    graph = test_neo4j_connection()
    if not graph:
        print("测试失败: 无法连接到Neo4j")
        return
    
    # 测试数据导入
    if not test_basic_import(graph):
        print("测试失败: 数据导入失败")
        return
    
    # 运行测试查询
    run_test_queries(graph)
    
    print("\n=== 测试完成 ===")
    
    # 最终统计
    try:
        node_count = graph.run("MATCH (n) RETURN count(n) as count").data()[0]["count"]
        rel_count = graph.run("MATCH ()-[r]-() RETURN count(r) as count").data()[0]["count"]
        print(f"最终统计: {node_count} 个节点, {rel_count} 个关系")
    except Exception as e:
        print(f"最终统计失败: {e}")

if __name__ == "__main__":
    main()