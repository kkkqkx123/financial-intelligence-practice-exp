'''
企业知识图谱可视化查询脚本
基于 Cypher 查询语言进行图数据探索和可视化
'''

from py2neo import Graph
import pandas as pd

class KnowledgeGraphExplorer:
    def __init__(self, uri, auth):
        """初始化图数据库连接"""
        self.graph = Graph(profile=uri, auth=auth)
        
    def execute_query(self, query, parameters=None):
        """执行Cypher查询并返回结果"""
        try:
            result = self.graph.run(query, parameters).data()
            return result
        except Exception as e:
            print(f"查询执行失败: {str(e)}")
            return []
    
    def get_enterprise_shareholders(self, enterprise_name):
        """
        查看指定企业的持股股东
        
        Args:
            enterprise_name: 企业名称（如"平安银行"）
        """
        query = """
        MATCH p=(holder:股东)-[r:参股]->(enterprise:企业)
        WHERE enterprise.企业名称 = $enterprise_name
        RETURN 
            holder.股东名称 as 股东名称,
            r.持有数量 as 持有数量,
            r.持有比例 as 持有比例,
            r.公告日期 as 公告日期,
            r.报告期 as 报告期
        ORDER BY r.持有比例 DESC
        """
        
        results = self.execute_query(query, {'enterprise_name': enterprise_name})
        
        print(f"\n=== {enterprise_name} 的持股股东 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print(f"\n总计 {len(results)} 个股东")
        else:
            print("未找到相关股东信息")
            
        return results
    
    def get_correlated_enterprises(self, enterprise_name, correlation_threshold=0.8):
        """
        查看与指定企业相关系数超过阈值的企业
        
        Args:
            enterprise_name: 企业名称
            correlation_threshold: 相关系数阈值（默认0.8）
        """
        query = """
        MATCH p=(e1:企业)-[r]->(e2:企业)
        WHERE (e1.企业名称 = $enterprise_name OR e2.企业名称 = $enterprise_name)
          AND abs(r.相关系数) >= $correlation_threshold
        RETURN 
            e1.企业名称 as 企业1,
            e2.企业名称 as 企业2,
            type(r) as 关系类型,
            r.相关系数 as 相关系数
        ORDER BY abs(r.相关系数) DESC
        """
        
        results = self.execute_query(query, {
            'enterprise_name': enterprise_name,
            'correlation_threshold': correlation_threshold
        })
        
        print(f"\n=== 与 {enterprise_name} 相关系数绝对值 ≥ {correlation_threshold} 的企业 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print(f"\n总计 {len(results)} 个相关企业")
        else:
            print("未找到满足条件的企业")
            
        return results
    
    def get_same_industry_enterprises(self, enterprise_name):
        """
        查看与指定企业同行业的所有企业
        
        Args:
            enterprise_name: 企业名称
        """
        query = """
        MATCH p=(e1:企业)-[:行业属于]->(industry:行业)<-[:行业属于]-(e2:企业)
        WHERE e1.企业名称 = $enterprise_name AND e1 <> e2
        RETURN 
            e2.企业名称 as 企业名称,
            e2.股票代码 as 股票代码,
            industry.行业名称 as 所属行业
        ORDER BY e2.企业名称
        """
        
        results = self.execute_query(query, {'enterprise_name': enterprise_name})
        
        print(f"\n=== 与 {enterprise_name} 同行业的企业 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print(f"\n总计 {len(results)} 个同行业企业")
        else:
            print("未找到同行业企业")
            
        return results
    
    def get_enterprise_basic_info(self, enterprise_name):
        """
        获取企业基本信息
        
        Args:
            enterprise_name: 企业名称
        """
        query = """
        MATCH (e:企业)-[:行业属于]->(industry:行业)
        WHERE e.企业名称 = $enterprise_name
        RETURN 
            e.企业名称 as 企业名称,
            e.股票代码 as 股票代码,
            industry.行业名称 as 所属行业
        """
        
        results = self.execute_query(query, {'enterprise_name': enterprise_name})
        
        print(f"\n=== {enterprise_name} 基本信息 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
        else:
            print("未找到企业信息")
            
        return results
    
    def get_concept_enterprises(self, concept_name):
        """
        获取属于特定概念的所有企业
        
        Args:
            concept_name: 概念名称
        """
        query = """
        MATCH (e:企业)-[:概念属于]->(concept:概念)
        WHERE concept.概念名称 = $concept_name
        RETURN 
            e.企业名称 as 企业名称,
            e.股票代码 as 股票代码
        ORDER BY e.企业名称
        """
        
        results = self.execute_query(query, {'concept_name': concept_name})
        
        print(f"\n=== 属于 '{concept_name}' 概念的企业 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print(f"\n总计 {len(results)} 个企业")
        else:
            print("未找到相关企业")
            
        return results
    
    def get_major_shareholders(self, min_hold_ratio=5.0):
        """
        获取持股比例超过阈值的主要股东及其投资企业
        
        Args:
            min_hold_ratio: 最低持股比例（默认5%）
        """
        query = """
        MATCH (holder:股东)-[r:参股]->(enterprise:企业)
        WHERE r.持有比例 >= $min_hold_ratio
        RETURN 
            holder.股东名称 as 股东名称,
            enterprise.企业名称 as 企业名称,
            r.持有比例 as 持股比例,
            r.持有数量 as 持有数量
        ORDER BY r.持有比例 DESC
        """
        
        results = self.execute_query(query, {'min_hold_ratio': min_hold_ratio})
        
        print(f"\n=== 持股比例 ≥ {min_hold_ratio}% 的主要股东 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print(f"\n总计 {len(results)} 条持股记录")
        else:
            print("未找到满足条件的股东")
            
        return results
    
    def get_industry_statistics(self):
        """
        获取各行业企业数量统计
        """
        query = """
        MATCH (e:企业)-[:行业属于]->(industry:行业)
        RETURN 
            industry.行业名称 as 行业名称,
            count(e) as 企业数量
        ORDER BY count(e) DESC
        """
        
        results = self.execute_query(query)
        
        print(f"\n=== 各行业企业数量统计 ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
        else:
            print("未找到行业数据")
            
        return results
    
    def get_correlation_network(self, enterprise_name, depth=2, correlation_threshold=0.6):
        """
        获取企业的相关性网络（多层关联）
        
        Args:
            enterprise_name: 中心企业名称
            depth: 关联深度（默认2层）
            correlation_threshold: 相关系数阈值
        """
        query = f"""
        MATCH path = (center:企业)-[:正相关|负相关*1..{depth}]-(related:企业)
        WHERE center.企业名称 = $enterprise_name
          AND ALL(rel in relationships(path) 
                  WHERE abs(rel.相关系数) >= $correlation_threshold)
        WITH center, related, 
             min(length(path)) as min_length,
             collect(path) as paths
        RETURN 
            center.企业名称 as 中心企业,
            related.企业名称 as 关联企业,
            min_length as 关联层数,
            size(paths) as 路径数量
        ORDER BY min_length, related.企业名称
        """
        
        results = self.execute_query(query, {
            'enterprise_name': enterprise_name,
            'correlation_threshold': correlation_threshold
        })
        
        print(f"\n=== {enterprise_name} 的相关性网络（深度：{depth}，阈值：{correlation_threshold}） ===")
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            print(f"\n总计 {len(results)} 个关联企业")
        else:
            print("未找到相关企业")
            
        return results
    
    def run_demo_queries(self):
        """运行演示查询"""
        print("=" * 60)
        print("企业知识图谱可视化查询演示")
        print("=" * 60)
        
        # 1. 查看平安银行的持股股东
        self.get_enterprise_shareholders("平安银行")
        
        # 2. 查看与平安银行相关系数>0.8的企业
        self.get_correlated_enterprises("平安银行", 0.8)
        
        # 3. 查看与平安银行同行业的企业
        self.get_same_industry_enterprises("平安银行")
        
        # 4. 获取平安银行基本信息
        self.get_enterprise_basic_info("平安银行")
        
        # 5. 查看新能源概念相关企业
        self.get_concept_enterprises("新能源")
        
        # 6. 获取主要股东（持股>5%）
        self.get_major_shareholders(5.0)
        
        # 7. 行业统计
        self.get_industry_statistics()
        
        # 8. 获取相关性网络
        self.get_correlation_network("平安银行", 2, 0.6)
        
        print("\n" + "=" * 60)
        print("演示查询完成！")
        print("=" * 60)


def main():
    """主函数"""
    # Neo4j连接配置
    uri = "http://localhost:7474/"
    auth = ("neo4j", "password")  # 请根据实际配置修改密码
    
    # 创建图数据探索器
    explorer = KnowledgeGraphExplorer(uri, auth)
    
    # 运行演示查询
    explorer.run_demo_queries()


if __name__ == "__main__":
    main()