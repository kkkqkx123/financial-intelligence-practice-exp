#!/usr/bin/env python3
"""
Neo4j查询脚本
基于投资事件和投资结构数据设计的查询集合
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from py2neo import Graph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("py2neo库未安装，Neo4j功能将不可用")

# 导入配置类
try:
    from integrations.neo4j_exporter import Config
except ImportError:
    # 如果无法导入，定义简单的配置类
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        """连接配置"""
        uri: str = "bolt://localhost:7687"
        username: str = "neo4j"
        password: str = "1234567kk"
        database: str = "neo4j"

# 配置日志
log_file = 'D:/Source/torch/financial-intellgience/src/logs/neo4j_queries.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Neo4jQueryManager:
    """Neo4j查询管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.graph: Optional[Graph] = None
        self._connect()
    
    def _connect(self):
        """连接到Neo4j数据库"""
        if not NEO4J_AVAILABLE:
            raise RuntimeError("py2neo库未安装，无法连接Neo4j")
        
        try:
            self.graph = Graph(
                self.config.uri, 
                auth=(self.config.username, self.config.password)
            )
            logger.info(f"成功连接到Neo4j: {self.config.uri}")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    
    def execute_query(self, query: str, output_file: str = None) -> Dict[str, Any]:
        """
        执行单个查询并保存结果
        
        Args:
            query: Neo4j查询语句
            output_file: 输出文件路径
            
        Returns:
            包含查询结果的字典
        """
        if not self.graph:
            raise RuntimeError("Neo4j连接不可用")
        
        try:
            # 生成默认输出文件名
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"../query/neo4j_query_{timestamp}.json"
            
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 执行查询
            logger.info(f"执行查询: {query[:100]}...")
            start_time = datetime.now()
            
            results = self.graph.run(query).data()
            
            # 计算查询耗时
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 准备结果数据
            result_data = {
                "query": query,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
                "result_count": len(results),
                "results": results,
                "success": True
            }
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"查询完成: {len(results)} 条记录，耗时 {execution_time:.2f}s")
            logger.info(f"结果已保存到: {output_file}")
            
            return result_data
            
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "result_count": 0,
                "results": []
            }
    
    def execute_multiple_queries(self, queries: List[Dict[str, str]], output_dir: str = "../query") -> List[Dict[str, Any]]:
        """
        执行多个查询并将结果保存到文件
        
        Args:
            queries: 查询列表，每个元素是包含"name"和"query"的字典
            output_dir: 输出目录
            
        Returns:
            包含所有查询结果的列表
        """
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for query_info in queries:
            query_name = query_info.get("name", "unnamed_query")
            query = query_info.get("query", "")
            
            if not query:
                logger.warning(f"查询 {query_name} 为空，跳过")
                continue
            
            # 生成输出文件名
            output_file = f"{output_dir}/{query_name}_{timestamp}.json"
            
            # 执行查询
            result = self.execute_query(query, output_file)
            result["query_name"] = query_name
            results.append(result)
        
        return results


def get_basic_statistics_queries() -> List[Dict[str, str]]:
    """获取基础统计查询"""
    return [
        {
            "name": "entity_counts",
            "query": """
            MATCH (n) 
            RETURN labels(n) as entity_type, count(n) as count 
            ORDER BY count DESC
            """
        },
        {
            "name": "relationship_counts",
            "query": """
            MATCH ()-[r]->() 
            RETURN type(r) as relationship_type, count(r) as count 
            ORDER BY count DESC
            """
        },
        {
            "name": "node_properties",
            "query": """
            MATCH (n) 
            WITH labels(n) as labels, keys(n) as properties
            UNWIND properties as property
            RETURN labels, property, count(*) as count
            ORDER BY labels, count DESC
            """
        }
    ]


def get_investment_analysis_queries() -> List[Dict[str, str]]:
    """获取投资分析查询"""
    return [
        {
            "name": "top_investors",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            RETURN i.投资方名称 as investor, count(r) as investment_count
            ORDER BY investment_count DESC
            LIMIT 20
            """
        },
        {
            "name": "top_companies",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            RETURN c.公司名称 as company, count(r) as investment_count
            ORDER BY investment_count DESC
            LIMIT 20
            """
        },
        {
            "name": "investment_rounds_distribution",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.轮次 IS NOT NULL
            RETURN r.轮次 as round, count(r) as count
            ORDER BY count DESC
            """
        },
        {
            "name": "investment_amount_analysis",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.金额 IS NOT NULL AND r.金额 <> '未披露'
            RETURN r.轮次 as round, 
                   count(r) as deal_count,
                   avg(toFloat(replace(replace(r.金额, '亿人民币', ''), '万人民币', ''))) as avg_amount
            ORDER BY avg_amount DESC
            """
        },
        {
            "name": "investor_network_analysis",
            "query": """
            MATCH (i1:投资方)-[:投资]->(c:公司)<-[:投资]-(i2:投资方)
            WHERE id(i1) < id(i2)
            RETURN i1.投资方名称 as investor1, 
                   i2.投资方名称 as investor2, 
                   count(c) as co_investment_count
            ORDER BY co_investment_count DESC
            LIMIT 20
            """
        }
    ]


def get_industry_analysis_queries() -> List[Dict[str, str]]:
    """获取行业分析查询"""
    return [
        {
            "name": "industry_distribution",
            "query": """
            MATCH (c:公司)-[:属于]->(ind:行业)
            RETURN ind.行业名称 as industry, count(c) as company_count
            ORDER BY company_count DESC
            LIMIT 20
            """
        },
        {
            "name": "investment_by_industry",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)-[:属于]->(ind:行业)
            RETURN ind.行业名称 as industry, count(r) as investment_count
            ORDER BY investment_count DESC
            LIMIT 20
            """
        },
        {
            "name": "investor_industry_preferences",
            "query": """
            MATCH (i:投资方)-[:投资]->(c:公司)-[:属于]->(ind:行业)
            WITH i, ind, count(*) as count
            ORDER BY i.投资方名称, count DESC
            WITH i.投资方名称 as investor, collect({industry: ind.行业名称, count: count}) as industries
            RETURN investor, industries
            ORDER BY size(industries) DESC
            LIMIT 20
            """
        },
        {
            "name": "industry_network_analysis",
            "query": """
            MATCH (i:投资方)-[:投资]->(c:公司)-[:属于]->(ind:行业)
            WITH ind, i, count(*) as investment_count
            WITH ind.行业名称 as industry, 
                 collect({investor: i.投资方名称, count: investment_count}) as investors
            RETURN industry, 
                   size(investors) as investor_count,
                   investors
            ORDER BY investor_count DESC
            LIMIT 15
            """
        }
    ]


def get_temporal_analysis_queries() -> List[Dict[str, str]]:
    """获取时序分析查询"""
    return [
        {
            "name": "investment_trend_by_year",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.融资时间 IS NOT NULL
            WITH substring(r.融资时间, 0, 4) as year, count(r) as count
            RETURN year, count
            ORDER BY year
            """
        },
        {
            "name": "investment_trend_by_round",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.融资时间 IS NOT NULL AND r.轮次 IS NOT NULL
            WITH substring(r.融资时间, 0, 4) as year, r.轮次 as round, count(r) as count
            RETURN year, round, count
            ORDER BY year, round
            """
        },
        {
            "name": "investor_activity_trend",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.融资时间 IS NOT NULL
            WITH i.投资方名称 as investor, substring(r.融资时间, 0, 4) as year, count(r) as count
            WITH investor, collect({year: year, count: count}) as activity
            WHERE size(activity) >= 3
            RETURN investor, activity
            ORDER BY size(activity) DESC
            LIMIT 20
            """
        }
    ]


def get_advanced_analysis_queries() -> List[Dict[str, str]]:
    """获取高级分析查询"""
    return [
        {
            "name": "company_investment_path",
            "query": """
            MATCH path = (i1:投资方)-[:投资*1..3]->(c:公司)
            WHERE length(path) >= 2
            RETURN c.公司名称 as company, 
                   length(path) as path_length,
                   [node IN nodes(path) WHERE node:投资方 | node.投资方名称] as investors
            ORDER BY path_length DESC
            LIMIT 20
            """
        },
        {
            "name": "investor_centrality",
            "query": """
            MATCH (i:投资方)-[:投资]->(c:公司)
            WITH i, count(c) as degree
            RETURN i.投资方名称 as investor, degree
            ORDER BY degree DESC
            LIMIT 20
            """
        },
        {
            "name": "community_detection",
            "query": """
            MATCH (i:投资方)-[:投资]->(c:公司)<-[:投资]-(i2:投资方)
            WHERE id(i) < id(i2)
            WITH i, i2, count(c) as shared_companies
            WHERE shared_companies >= 2
            RETURN i.投资方名称 as investor1, 
                   i2.投资方名称 as investor2, 
                   shared_companies
            ORDER BY shared_companies DESC
            LIMIT 30
            """
        },
        {
            "name": "investment_network_density",
            "query": """
            MATCH (i:投资方)
            WITH count(i) as investor_count
            
            MATCH (i1:投资方)-[:投资]->(c:公司)<-[:投资]-(i2:投资方)
            WHERE id(i1) < id(i2)
            WITH investor_count, count(DISTINCT [i1, i2]) as connection_count
            
            RETURN investor_count, 
                   connection_count,
                   toFloat(connection_count) / (investor_count * (investor_count - 1) / 2) as density
            """
        }
    ]


def main():
    """主函数 - 执行所有查询"""
    print("Neo4j查询执行器")
    print("="*50)
    
    # 创建查询管理器
    config = Config(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="1234567kk"
    )
    
    try:
        query_manager = Neo4jQueryManager(config)
        print("✓ Neo4j连接成功")
        
        # 获取所有查询
        all_queries = []
        all_queries.extend(get_basic_statistics_queries())
        all_queries.extend(get_investment_analysis_queries())
        all_queries.extend(get_industry_analysis_queries())
        all_queries.extend(get_temporal_analysis_queries())
        all_queries.extend(get_advanced_analysis_queries())
        
        print(f"准备执行 {len(all_queries)} 个查询")
        
        # 执行查询
        results = query_manager.execute_multiple_queries(all_queries)
        
        # 打印执行摘要
        print("\n查询执行摘要:")
        print("-"*50)
        
        successful_queries = 0
        total_results = 0
        
        for result in results:
            query_name = result.get("query_name", "unknown")
            success = result.get("success", False)
            result_count = result.get("result_count", 0)
            execution_time = result.get("execution_time", 0)
            
            if success:
                successful_queries += 1
                total_results += result_count
                print(f"✓ {query_name}: {result_count} 条记录 ({execution_time:.2f}s)")
            else:
                print(f"✗ {query_name}: 查询失败")
        
        print(f"\n总计: {successful_queries}/{len(all_queries)} 个查询成功执行")
        print(f"总记录数: {total_results}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())