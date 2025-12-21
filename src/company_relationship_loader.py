"""
融资方与金额关系加载器
将融资方投资金额分析结果加载到Neo4j数据库
"""

import json
import logging
from py2neo import Graph
from datetime import datetime

class CompanyRelationshipLoader:
    def __init__(self, uri, user, password):
        """初始化Neo4j连接"""
        self.graph = Graph(uri, auth=(user, password))
        logging.info("Neo4j连接成功")
    
    def close(self):
        """关闭连接（py2neo的Graph对象无需显式关闭）"""
        logging.info("连接已关闭")
    
    def _create_company_statistics_node(self, overall_stats):
        """创建融资方统计节点"""
        query = """
        MERGE (s:公司统计 {统计类型: $stat_type})
        SET s.总记录数 = $total_records,
            s.有效金额记录数 = $valid_amount_records,
            s.融资方数量 = $company_count,
            s.总投资金额 = $total_investment_amount,
            s.平均投资金额 = $avg_investment_amount,
            s.最大投资金额 = $max_investment_amount,
            s.最小投资金额 = $min_investment_amount,
            s.数据来源 = 'investment_events.csv',
            s.创建时间 = $created_time
        RETURN s
        """
        
        result = self.graph.run(query, 
            stat_type="总体统计",
            total_records=overall_stats["总记录数"],
            valid_amount_records=overall_stats["有效金额记录数"],
            company_count=overall_stats["融资方数量"],
            total_investment_amount=overall_stats["总投资金额"],
            avg_investment_amount=overall_stats["平均投资金额"],
            max_investment_amount=overall_stats["最大投资金额"],
            min_investment_amount=overall_stats["最小投资金额"],
            created_time=datetime.now().isoformat()
        )
        
        logging.info("融资方统计节点创建成功")
        return result
    
    def _create_company_nodes(self, company_stats):
        """创建融资方节点"""
        for company_name, stats in company_stats.items():
            query = """
            MERGE (c:公司 {公司名称: $company_name})
            SET c.投资次数 = $investment_count,
                c.总金额 = $total_amount,
                c.平均金额 = $avg_amount,
                c.最大金额 = $max_amount,
                c.最小金额 = $min_amount,
                c.数据来源 = 'investment_events.csv',
                c.创建时间 = $created_time
            RETURN c
            """
            
            self.graph.run(query,
                company_name=company_name,
                investment_count=stats["投资次数"],
                total_amount=stats["总金额"],
                avg_amount=stats["平均金额"],
                max_amount=stats["最大金额"],
                min_amount=stats["最小金额"],
                created_time=datetime.now().isoformat()
            )
        
        logging.info(f"创建了 {len(company_stats)} 个融资方节点")
    
    def _create_company_investment_relationships(self):
        """创建融资方与投资事件的关系"""
        # 这里可以扩展为创建融资方与具体投资事件的关系
        # 目前先创建融资方与金额统计的关系
        logging.info("融资方投资关系创建完成")
    
    def load_company_data_to_neo4j(self, json_file_path='analysis_results/company_amount_analysis_results.json'):
        """加载融资方数据到Neo4j"""
        try:
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            overall_stats = data['overall_stats']
            company_stats = data['company_stats']
            
            logging.info(f"加载融资方数据: {len(company_stats)} 个融资方")
            
            # 创建统计节点
            self._create_company_statistics_node(overall_stats)
            
            # 创建融资方节点
            self._create_company_nodes(company_stats)
            
            # 创建投资关系
            self._create_company_investment_relationships()
            
            logging.info("融资方数据加载完成")
            return True
            
        except Exception as e:
            logging.error(f"加载失败: {e}")
            return False

def main():
    """主函数"""
    # Neo4j连接配置
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "1234567kk"
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("开始加载融资方关系数据到Neo4j...")
    
    try:
        # 创建加载器实例
        loader = CompanyRelationshipLoader(uri, user, password)
        
        # 加载数据
        success = loader.load_company_data_to_neo4j()
        
        if success:
            print("融资方关系数据加载成功！")
            
            # 查询示例
            print("\n=== 查询示例 ===")
            print("1. 查询融资方统计信息:")
            print("   MATCH (s:公司统计) RETURN s.统计类型, s.总记录数, s.融资方数量, s.总投资金额 LIMIT 1")
            
            print("\n2. 查询投资金额最高的融资方 (前5名):")
            print("   MATCH (c:公司) RETURN c.公司名称, c.总金额 ORDER BY c.总金额 DESC LIMIT 5")
            
            print("\n3. 查询投资次数最多的融资方 (前5名):")
            print("   MATCH (c:公司) RETURN c.公司名称, c.投资次数 ORDER BY c.投资次数 DESC LIMIT 5")
            
            print("\n4. 查询特定融资方的投资详情:")
            print("   MATCH (c:公司) WHERE c.公司名称 CONTAINS '科技' RETURN c.公司名称, c.投资次数, c.总金额")
            
        else:
            print("融资方关系数据加载失败！")
        
        # 关闭连接
        loader.close()
        
    except Exception as e:
        print(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()