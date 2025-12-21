#!/usr/bin/env python3
"""
Neo4j查询执行器
执行自定义查询并将结果保存到文件中
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

try:
    from py2neo import Graph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("py2neo库未安装，Neo4j功能将不可用")

from integrations.neo4j_exporter import Config

# 配置日志
log_file = 'D:/Source/torch/financial-intellgience/src/logs/neo4j_query_executor.log'
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


class Neo4jQueryExecutor:
    """Neo4j查询执行器"""
    
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
        执行Neo4j查询并将结果保存到文件
        
        Args:
            query: Neo4j查询语句
            output_file: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            包含查询结果和统计信息的字典
        """
        if not self.graph:
            raise RuntimeError("Neo4j连接不可用")
        
        start_time = datetime.now()
        
        # 生成默认输出文件名
        if not output_file:
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            output_file = f"output/neo4j_query_results_{timestamp}.json"
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"执行查询: {query[:100]}...")
            
            # 执行查询
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
                "results": results
            }
            
            # 保存结果到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"查询完成，结果已保存到: {output_file}")
            logger.info(f"返回 {len(results)} 条记录，耗时 {execution_time:.2f} 秒")
            
            return {
                "success": True,
                "output_file": str(output_path),
                "result_count": len(results),
                "execution_time": execution_time,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            error_data = {
                "query": query,
                "timestamp": start_time.isoformat(),
                "error": str(e),
                "success": False
            }
            
            # 保存错误信息到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            return {
                "success": False,
                "output_file": str(output_path),
                "error": str(e)
            }
    
    def execute_multiple_queries(self, queries: List[Dict[str, str]], output_dir: str = "output") -> List[Dict[str, Any]]:
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
            output_file = f"{output_dir}/neo4j_{query_name}_{timestamp}.json"
            
            # 执行查询
            result = self.execute_query(query, output_file)
            result["query_name"] = query_name
            results.append(result)
        
        return results
    
    def close(self):
        """关闭连接"""
        if self.graph:
            logger.info("关闭Neo4j连接")
            # py2neo会自动管理连接，这里不需要特别处理


def get_predefined_queries() -> List[Dict[str, str]]:
    """获取预定义的查询列表"""
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
            "name": "industry_distribution",
            "query": """
            MATCH (c:公司)-[:属于]->(ind:行业)
            RETURN ind.行业名称 as industry, count(c) as company_count
            ORDER BY company_count DESC
            LIMIT 20
            """
        },
        {
            "name": "investment_rounds",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            RETURN r.融资轮次 as round, count(r) as count
            ORDER BY count DESC
            """
        },
        {
            "name": "recent_investments",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.投资时间 IS NOT NULL
            RETURN i.投资方名称 as investor, c.公司名称 as company, r.融资轮次 as round, r.投资时间 as date, r.投资金额 as amount
            ORDER BY r.投资时间 DESC
            LIMIT 50
            """
        },
        {
            "name": "graph_overview",
            "query": """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->()
            RETURN count(DISTINCT n) as total_nodes, count(DISTINCT r) as total_relationships
            """
        }
    ]


def main():
    """主函数 - 执行预定义查询"""
    print("Neo4j查询执行器")
    print("="*50)
    
    # 创建查询执行器
    config = Config(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="1234567kk"
    )
    
    try:
        executor = Neo4jQueryExecutor(config)
        print("✓ Neo4j连接成功")
        
        # 获取预定义查询
        queries = get_predefined_queries()
        print(f"准备执行 {len(queries)} 个预定义查询")
        
        # 执行查询
        results = executor.execute_multiple_queries(queries)
        
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
            output_file = result.get("output_file", "")
            
            if success:
                successful_queries += 1
                total_results += result_count
                print(f"✓ {query_name}: {result_count} 条记录 ({execution_time:.2f}s)")
                print(f"  输出文件: {output_file}")
            else:
                print(f"✗ {query_name}: 查询失败")
                print(f"  错误: {result.get('error', 'Unknown error')}")
        
        print("-"*50)
        print(f"成功执行: {successful_queries}/{len(queries)} 个查询")
        print(f"总记录数: {total_results}")
        
        # 保存执行摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"output/neo4j_query_summary_{timestamp}.json"
        
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "successful_queries": successful_queries,
            "total_results": total_results,
            "results": results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"执行摘要已保存到: {summary_file}")
        
        executor.close()
        
    except Exception as e:
        print(f"✗ 连接或执行失败: {e}")
        return 1
    
    print("\n查询执行完成")
    return 0


if __name__ == "__main__":
    exit(main())