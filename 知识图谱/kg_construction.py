'''
企业知识图谱构建主脚本
基于 py2neo 和 Neo4j 构建企业、行业、股东、概念知识图谱
'''

from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import numpy as np
import os
from datetime import datetime

class EnterpriseKnowledgeGraph:
    def __init__(self, uri, auth):
        """初始化知识图谱构建器"""
        self.graph = Graph(profile=uri, auth=auth)
        self.node_matcher = NodeMatcher(self.graph)
        self.data_dir = "数据集"
        
    def load_data(self):
        """加载所有数据文件"""
        print("开始加载数据...")
        
        # 企业基础信息
        self.enterprise_basic = pd.read_csv(os.path.join(self.data_dir, "enterprise_basic.csv"))
        print(f"企业基础信息：{len(self.enterprise_basic)} 条记录")
        
        # 企业概念关联
        self.enterprise_concept = pd.read_csv(os.path.join(self.data_dir, "enterprise_concept.csv"))
        print(f"企业概念关联：{len(self.enterprise_concept)} 条记录")
        
        # 企业股东信息
        self.enterprise_holders = pd.read_csv(os.path.join(self.data_dir, "enterprise_holders.csv"))
        print(f"企业股东信息：{len(self.enterprise_holders)} 条记录")
        
        # 股票价格数据
        self.stock_prices = pd.read_csv(os.path.join(self.data_dir, "stock_prices.csv"))
        print(f"股票价格数据：{len(self.stock_prices)} 条记录")
        
        print("数据加载完成！")
        
    def create_enterprise_nodes(self):
        """创建企业实体节点"""
        print("\n开始创建企业实体...")
        
        enterprise_df = self.enterprise_basic[['企业名称', '股票代码']].drop_duplicates()
        created_count = 0
        skipped_count = 0
        
        for _, row in enterprise_df.iterrows():
            enterprise_name = row['企业名称']
            stock_code = row['股票代码']
            
            # 检查是否已存在
            if self.node_matcher.match("企业", 企业名称=enterprise_name, 股票代码=stock_code).first() is None:
                enterprise_node = Node('企业', 
                                     企业名称=enterprise_name, 
                                     股票代码=stock_code)
                self.graph.create(enterprise_node)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"企业实体创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def create_industry_nodes(self):
        """创建行业实体节点"""
        print("\n开始创建行业实体...")
        
        industry_list = self.enterprise_basic['行业'].unique().tolist()
        created_count = 0
        skipped_count = 0
        
        for industry in industry_list:
            if pd.isna(industry):
                continue
                
            if self.node_matcher.match('行业', 行业名称=industry).first() is None:
                industry_node = Node('行业', 行业名称=industry)
                self.graph.create(industry_node)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"行业实体创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def create_holder_nodes(self):
        """创建股东实体节点"""
        print("\n开始创建股东实体...")
        
        holder_list = self.enterprise_holders['股东名称'].unique().tolist()
        created_count = 0
        skipped_count = 0
        
        for holder in holder_list:
            if pd.isna(holder):
                continue
                
            if self.node_matcher.match('股东', 股东名称=holder).first() is None:
                holder_node = Node('股东', 股东名称=holder)
                self.graph.create(holder_node)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"股东实体创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def create_concept_nodes(self):
        """创建概念实体节点"""
        print("\n开始创建概念实体...")
        
        concept_list = self.enterprise_concept['概念名称'].unique().tolist()
        created_count = 0
        skipped_count = 0
        
        for concept in concept_list:
            if pd.isna(concept):
                continue
                
            if self.node_matcher.match('概念', 概念名称=concept).first() is None:
                concept_node = Node('概念', 概念名称=concept)
                self.graph.create(concept_node)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"概念实体创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def compute_corrcoef(self, stock_return: np.ndarray):
        """计算皮尔逊相关系数矩阵"""
        corrcoef = np.corrcoef(stock_return)
        corrcoef[np.isnan(corrcoef)] = 0  # nan位置置0
        return corrcoef
        
    def create_enterprise_relationships(self):
        """创建企业-企业关系（基于股价相关性）"""
        print("\n开始创建企业-企业关系...")
        
        # 准备价格数据
        prices_list = []
        stock_code_list = []
        
        for stock_code, group in self.stock_prices.groupby(by=["股票代码"]):
            stock_code_list.append(stock_code)
            group = group.sort_values(by=["交易日期"], ascending=True)
            prices_list.append(group["收盘价"].tolist())
            
        prices_np = np.array(prices_list)
        return_np = prices_np[:, 1:] / prices_np[:, :-1]  # 计算收益率
        corrcoef = self.compute_corrcoef(return_np)
        
        created_count = 0
        # 创建相关性关系（阈值：0.5）
        for i in range(len(stock_code_list)):
            for j in range(len(stock_code_list)):
                if i != j and abs(corrcoef[i, j]) >= 0.5:
                    enterprise1_node = self.node_matcher.match("企业", 股票代码=stock_code_list[i]).first()
                    enterprise2_node = self.node_matcher.match("企业", 股票代码=stock_code_list[j]).first()
                    
                    if enterprise1_node and enterprise2_node:
                        relation_type = '正相关' if corrcoef[i, j] > 0 else '负相关'
                        relation = Relationship(enterprise1_node, relation_type,
                                              enterprise2_node, 相关系数=float(corrcoef[i, j]))
                        self.graph.create(relation)
                        created_count += 1
                        
        print(f"企业-企业关系创建完成：新建 {created_count} 个关系")
        
    def create_industry_relationships(self):
        """创建企业-行业关系"""
        print("\n开始创建企业-行业关系...")
        
        created_count = 0
        skipped_count = 0
        
        for _, row in self.enterprise_basic.iterrows():
            stock_code = row['股票代码']
            enterprise_name = row['企业名称']
            industry = row['行业']
            
            if pd.isna(industry):
                skipped_count += 1
                continue
                
            enterprise_node = self.node_matcher.match("企业", 企业名称=enterprise_name, 股票代码=stock_code).first()
            industry_node = self.node_matcher.match("行业", 行业名称=industry).first()
            
            if enterprise_node and industry_node:
                relation = Relationship(enterprise_node, '行业属于', industry_node)
                self.graph.create(relation)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"企业-行业关系创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def create_holder_relationships(self):
        """创建企业-股东关系"""
        print("\n开始创建企业-股东关系...")
        
        created_count = 0
        skipped_count = 0
        
        for _, row in self.enterprise_holders.iterrows():
            stock_code = row['股票代码']
            holder_name = row['股东名称']
            hold_amount = row['持有数量（股）']
            hold_ratio = row['持有比例']
            ann_date = row['公告日期']
            end_date = row['报告期']
            
            if pd.isna(holder_name):
                skipped_count += 1
                continue
                
            holder_node = self.node_matcher.match("股东", 股东名称=holder_name).first()
            enterprise_node = self.node_matcher.match("企业", 股票代码=stock_code).first()
            
            if holder_node and enterprise_node:
                relation = Relationship(holder_node, '参股', enterprise_node,
                                    持有数量=hold_amount, 持有比例=hold_ratio,
                                    公告日期=ann_date, 报告期=end_date)
                self.graph.create(relation)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"企业-股东关系创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def create_concept_relationships(self):
        """创建企业-概念关系"""
        print("\n开始创建企业-概念关系...")
        
        created_count = 0
        skipped_count = 0
        
        for _, row in self.enterprise_concept.iterrows():
            concept_name = row['概念名称']
            stock_code = row['股票代码']
            
            if pd.isna(concept_name):
                skipped_count += 1
                continue
                
            enterprise_node = self.node_matcher.match("企业", 股票代码=stock_code).first()
            concept_node = self.node_matcher.match("概念", 概念名称=concept_name).first()
            
            if enterprise_node and concept_node:
                relation = Relationship(enterprise_node, '概念属于', concept_node)
                self.graph.create(relation)
                created_count += 1
            else:
                skipped_count += 1
                
        print(f"企业-概念关系创建完成：新建 {created_count} 个，跳过 {skipped_count} 个")
        
    def build_knowledge_graph(self):
        """构建完整知识图谱"""
        print("=== 开始构建企业知识图谱 ===")
        start_time = datetime.now()
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 创建实体
            self.create_enterprise_nodes()
            self.create_industry_nodes()
            self.create_holder_nodes()
            self.create_concept_nodes()
            
            # 3. 创建关系
            self.create_enterprise_relationships()
            self.create_industry_relationships()
            self.create_holder_relationships()
            self.create_concept_relationships()
            
            end_time = datetime.now()
            print(f"\n=== 知识图谱构建完成！用时：{end_time - start_time} ===")
            
        except Exception as e:
            print(f"构建过程中出现错误：{str(e)}")
            raise
            
    def get_statistics(self):
        """获取知识图谱统计信息"""
        print("\n=== 知识图谱统计信息 ===")
        
        # 实体统计
        enterprise_count = len(list(self.node_matcher.match("企业")))
        industry_count = len(list(self.node_matcher.match("行业")))
        holder_count = len(list(self.node_matcher.match("股东")))
        concept_count = len(list(self.node_matcher.match("概念")))
        
        print(f"企业实体：{enterprise_count} 个")
        print(f"行业实体：{industry_count} 个")
        print(f"股东实体：{holder_count} 个")
        print(f"概念实体：{concept_count} 个")
        
        # 关系统计
        query = """
        MATCH ()-[r]->() 
        RETURN type(r) as relation_type, count(r) as count
        ORDER BY count DESC
        """
        results = self.graph.run(query).data()
        
        print("\n关系类型统计：")
        for result in results:
            print(f"  {result['relation_type']}：{result['count']} 个")
            
        return {
            'enterprise_count': enterprise_count,
            'industry_count': industry_count,
            'holder_count': holder_count,
            'concept_count': concept_count,
            'relationships': results
        }


def main():
    """主函数"""
    # Neo4j连接配置
    uri = "http://localhost:7474/"
    auth = ("neo4j", "password")  # 请根据实际配置修改密码
    
    # 创建知识图谱构建器
    kg_builder = EnterpriseKnowledgeGraph(uri, auth)
    
    # 构建知识图谱
    kg_builder.build_knowledge_graph()
    
    # 获取统计信息
    kg_builder.get_statistics()


if __name__ == "__main__":
    main()