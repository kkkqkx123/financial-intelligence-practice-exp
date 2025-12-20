#!/usr/bin/env python3
"""
高级Neo4j查询执行器
基于项目实现的高级功能设计并执行复杂查询
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AdvancedNeo4jQueryExecutor:
    """高级Neo4j查询执行器"""
    
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
            output_file = f"analysis_results/advanced_query_{timestamp}.json"
        
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
    
    def close(self):
        """关闭连接"""
        if self.graph:
            logger.info("关闭Neo4j连接")
            # py2neo会自动管理连接，这里不需要特别处理


def get_advanced_queries() -> List[Dict[str, str]]:
    """获取基于项目高级功能设计的查询列表"""
    return [
        {
            "name": "llm_enhanced_entity_descriptions",
            "query": """
            MATCH (c:公司)
            WHERE c.实体描述 IS NOT NULL
            RETURN c.公司名称 as company, c.实体描述 as description
            ORDER BY size(c.实体描述) DESC
            LIMIT 20
            """
        },
        {
            "name": "industry_investment_correlation",
            "query": """
            MATCH (c:公司)-[:属于]->(ind:行业), (i:投资方)-[r:投资]->(c)
            RETURN ind.行业名称 as industry, 
                   count(DISTINCT c) as company_count,
                   count(DISTINCT i) as investor_count,
                   avg(toInteger(replace(r.投资金额, '万', ''))) as avg_investment
            ORDER BY company_count DESC
            LIMIT 15
            """
        },
        {
            "name": "investment_stage_evolution",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.投资时间 IS NOT NULL
            WITH r.融资轮次 as round, 
                 date(r.投资时间).year as year,
                 count(*) as count
            RETURN round, year, count
            ORDER BY year, round
            """
        },
        {
            "name": "investor_portfolio_diversity",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)-[:属于]->(ind:行业)
            WITH i, count(DISTINCT ind.行业名称) as industry_diversity, count(r) as investment_count
            RETURN i.投资方名称 as investor, 
                   industry_diversity, 
                   investment_count,
                   industry_diversity * 1.0 / investment_count as diversity_ratio
            ORDER BY industry_diversity DESC
            LIMIT 20
            """
        },
        {
            "name": "cross_industry_investors",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)-[:属于]->(ind:行业)
            WITH i, collect(DISTINCT ind.行业名称) as industries
            WHERE size(industries) > 3
            RETURN i.投资方名称 as investor, industries, size(industries) as industry_count
            ORDER BY industry_count DESC
            LIMIT 15
            """
        },
        {
            "name": "investment_network_centrality",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WITH i, count(r) as investment_count
            ORDER BY investment_count DESC
            WITH collect(i.投资方名称) as top_investors
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE i.投资方名称 IN top_investors[0..10]
            RETURN i.投资方名称 as investor, 
                   count(r) as direct_investments,
                   count(DISTINCT c.公司名称) as unique_companies,
                   collect(DISTINCT c.公司名称)[0..5] as sample_companies
            ORDER BY direct_investments DESC
            """
        },
        {
            "name": "llm_enhanced_industry_analysis",
            "query": """
            MATCH (c:公司)-[:属于]->(ind:行业)
            WHERE c.实体描述 IS NOT NULL AND ind.行业名称 IS NOT NULL
            RETURN ind.行业名称 as industry,
                   count(c) as company_count,
                   collect(c.实体描述)[0..3] as sample_descriptions
            ORDER BY company_count DESC
            LIMIT 10
            """
        },
        {
            "name": "investment_amount_distribution",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE r.投资金额 IS NOT NULL
            WITH 
                CASE 
                    WHEN toInteger(replace(r.投资金额, '万', '')) < 100 THEN '小额投资(<100万)'
                    WHEN toInteger(replace(r.投资金额, '万', '')) < 1000 THEN '中等投资(100-1000万)'
                    WHEN toInteger(replace(r.投资金额, '万', '')) < 5000 THEN '大额投资(1000-5000万)'
                    ELSE '超大额投资(>5000万)'
                END as amount_category,
                count(*) as count
            RETURN amount_category, count
            ORDER BY count DESC
            """
        },
        {
            "name": "standardized_investor_names",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)
            WHERE i.标准化名称 IS NOT NULL
            RETURN i.投资方名称 as original_name, 
                   i.标准化名称 as standardized_name,
                   count(r) as investment_count
            ORDER BY investment_count DESC
            LIMIT 20
            """
        },
        {
            "name": "investment_round_industry_preference",
            "query": """
            MATCH (i:投资方)-[r:投资]->(c:公司)-[:属于]->(ind:行业)
            WITH i.投资方名称 as investor, ind.行业名称 as industry, r.融资轮次 as round, count(*) as count
            ORDER BY investor, count DESC
            RETURN investor, industry, collect(round)[0..3] as preferred_rounds, sum(count) as total_investments
            ORDER BY total_investments DESC
            LIMIT 15
            """
        }
    ]


def main():
    """主函数 - 执行高级查询"""
    print("高级Neo4j查询执行器")
    print("="*50)
    
    # 创建查询执行器
    config = Config(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="1234567kk"
    )
    
    try:
        executor = AdvancedNeo4jQueryExecutor(config)
        print("✓ Neo4j连接成功")
        
        # 获取高级查询
        queries = get_advanced_queries()
        print(f"准备执行 {len(queries)} 个高级查询")
        
        # 执行查询
        results = []
        successful_queries = 0
        total_results = 0
        
        for query_info in queries:
            query_name = query_info.get("name", "unnamed_query")
            query = query_info.get("query", "")
            
            if not query:
                logger.warning(f"查询 {query_name} 为空，跳过")
                continue
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"analysis_results/advanced_{query_name}_{timestamp}.json"
            
            # 执行查询
            result = executor.execute_query(query, output_file)
            result["query_name"] = query_name
            results.append(result)
            
            # 打印执行结果
            success = result.get("success", False)
            result_count = result.get("result_count", 0)
            execution_time = result.get("execution_time", 0)
            
            if success:
                successful_queries += 1
                total_results += result_count
                print(f"✓ {query_name}: {result_count} 条记录 ({execution_time:.2f}s)")
            else:
                print(f"✗ {query_name}: 查询失败")
                print(f"  错误: {result.get('error', 'Unknown error')}")
        
        # 打印执行摘要
        print("\n查询执行摘要:")
        print("-"*50)
        print(f"成功执行: {successful_queries}/{len(queries)} 个查询")
        print(f"总记录数: {total_results}")
        
        # 保存执行摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"analysis_results/advanced_query_summary_{timestamp}.json"
        
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
    
    print("\n高级查询执行完成")
    return 0


if __name__ == "__main__":
    exit(main())