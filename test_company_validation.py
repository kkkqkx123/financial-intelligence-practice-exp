from src.main import KnowledgeGraphPipeline

# 创建知识图谱流水线
p = KnowledgeGraphPipeline(data_dir='src/dataset')

# 加载数据
data = p.load_data_files()

# 解析公司数据
companies = p.parser.parse_companies(data['companies'][:10])

# 打印公司数据验证结果
print('公司数据验证:')
for i, c in enumerate(companies):
    print(f'  {i+1}. {c.get("公司名称", "未知")}: {c.get("工商注册id", "无")}')