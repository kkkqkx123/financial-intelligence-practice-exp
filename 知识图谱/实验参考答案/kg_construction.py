'''
构建企业知识图谱，参考代码：https://github.com/jm199504/Financial-Knowledge-Graphs
'''
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import numpy as np

# step 1. 数据读取
data_dir = "../数据集/"
enterprise_basic = pd.read_csv(data_dir + "enterprise_basic.csv")
enterprise_concept = pd.read_csv(data_dir + "enterprise_concept.csv")
enterprise_holders = pd.read_csv(data_dir + "enterprise_holders.csv")
stock_notices = pd.read_csv(data_dir + "stock_notices.csv")
stock_prices = pd.read_csv(data_dir + "stock_prices.csv")

# step 2. 安装并连接neo4j，创建NodeMatcher
uri = "bolt://localhost:7687"
auth = ("neo4j", "1234")
graph = Graph(profile="http://localhost:7474/", auth=auth)
# graph.delete_all()   # 清空数据库

node_matcher = NodeMatcher(graph)

# ==========================================================================================================
# step 3. 基于py2neo.Node创建实体，并且补充相关节点属性。请注意实体重复问题！！！
# step 3.1. 创建企业实体
enterprise = enterprise_basic[['企业名称', '股票代码']]
enterprise = enterprise.drop_duplicates(subset=None, keep='first', inplace=False)
for enterprise_name, stock_code in enterprise.values:
   if node_matcher.match("企业", 企业名称=enterprise_name, 股票代码=stock_code).first() is None:
      enterprise_node = Node('企业', 企业名称=enterprise_name, 股票代码=stock_code)
      graph.create(enterprise_node)
print("create enterprise node")

# step 3.2. 创建行业实体
industry_list = enterprise_basic['行业'].unique().tolist()
for industry in industry_list:
   if node_matcher.match('行业', 行业名称=industry).first() is None:
      industry_node = Node('行业', 行业名称=industry)
      graph.create(industry_node)
print("create industry node")

# step 3.3. 创建股东实体
holder_list = enterprise_holders['股东名称'].unique().tolist()
for holder in holder_list:
   if node_matcher.match('股东', 股东名称=holder).first() is None:
      holder_node = Node('股东', 股东名称=holder)
      graph.create(holder_node)
print("create holder node")

# step 3.4. 创建概念实体
concept_list = enterprise_concept['概念名称'].unique().tolist()
for concept in concept_list:
   if node_matcher.match('概念', 概念名称=concept).first() is None:
      concept_node = Node('概念', 概念名称=concept)
      graph.create(concept_node)
print("create concept node")


# ==========================================================================================================
# step 4. 基于py2neo.NodeMatcher查找node，并且基于py2neo.Relationship创建关系，并且补充相关关系属性。
# step 4.1. 基于企业的股票之间的协方差来创建企业-企业关系
def coumpute_corrcoef(stock_price: np.ndarray):
   corrcoef = np.corrcoef(stock_price)
   corrcoef[np.isnan(corrcoef)] = 0  # nan位置赋0
   return corrcoef

prices_list = []
stock_code_list = []
for stock_code, group in stock_prices.groupby(by=["股票代码"]):
   stock_code_list.append(stock_code)
   group = group.sort_values(by=["交易日期"], ascending=True)
   prices_list.append(group["收盘价"].tolist())
prices_np = np.array(prices_list)
return_np = prices_np[:, 1:] / prices_np[:, :-1]
corrcoef = coumpute_corrcoef(return_np)

for i in range(len(stock_code_list)):
   for j in range(len(stock_code_list)):
      if i!=j and abs(corrcoef[i, j])>=0.5:
         enterprise1_node = node_matcher.match("企业", 股票代码=stock_code_list[i]).first()
         enterprise2_node = node_matcher.match("企业", 股票代码=stock_code_list[j]).first()
         relation = Relationship(enterprise1_node, '正相关' if corrcoef[i, j]>0 else '负相关',
                                 enterprise2_node, 相关系数=corrcoef[i, j])
         graph.create(relation)
print("create enterprise-enterprise relation")


# step 4.2. 创建企业-行业关系
for stock_code, enterprise_name, industry, _ in enterprise_basic.values:
   enterprise_node = node_matcher.match("企业", 企业名称=enterprise_name, 股票代码=stock_code).first()
   industry_node = node_matcher.match("行业", 行业名称=industry).first()
   relation = Relationship(enterprise_node, '行业属于', industry_node)
   graph.create(relation)
print("create enterprise-industry relation")

# step 4.3. 创建企业-股东关系
for stock_code, ann_date, end_date, holder_name, hold_amount, hold_ratio in enterprise_holders.values:
   holder_node = node_matcher.match("股东", 股东名称=holder_name).first()
   enterprise_node = node_matcher.match("企业", 股票代码=stock_code).first()
   relation = Relationship(holder_node, '参股', enterprise_node,
                           持有数量=hold_amount, 持有比例=hold_ratio,
                           公告日期=ann_date, 报告期=end_date)
   graph.create(relation)
print("create enterprise-holder relation")

# step 4.4. 创建企业-概念关系
for concept_name, stock_code in enterprise_concept.values:
   enterprise_node = node_matcher.match("企业", 股票代码=stock_code).first()
   concept_node = node_matcher.match("概念", 概念名称=concept_name).first()
   try:
      relation = Relationship(enterprise_node, '概念属于', concept_node)
   except:
      print()
   graph.create(relation)
print("create enterprise-concept relation")

