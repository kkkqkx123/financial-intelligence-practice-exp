# 企业知识图谱构建与可视化系统

## 项目概述

本项目基于Neo4j图数据库构建企业知识图谱，整合企业基本信息、股东关系、行业分类、概念标签和股价相关性等多维度数据，实现企业关系的全面分析和可视化展示。

## 项目结构

```
知识图谱/
├── config.py                    # 配置文件
├── kg_construction.py           # 知识图谱构建主程序
├── kg_visualization.py          # 可视化查询程序
├── run_build.py                 # 一键构建脚本
├── run_visualization.py         # 一键查询脚本
├── cypher_queries.cyp           # Cypher查询语句集合
├── README.md                    # 项目说明文档
├── plan/                        # 方案文档目录
│   └── 实现方案.md              # 详细实现方案
└── 数据集/                      # 数据文件目录
    ├── enterprise_basic.csv     # 企业基本信息
    ├── enterprise_concept.csv   # 企业概念标签
    ├── enterprise_holders.csv   # 股东持股信息
    ├── enterprise_notices.csv # 企业公告信息
    └── stock_prices.csv       # 股价数据
```

## 环境要求

- Python 3.7+
- Neo4j 4.0+
- Docker（用于运行Neo4j服务）

## 安装依赖

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install py2neo pandas numpy
```

## 快速开始

### 1. 启动Neo4j服务

确保Docker中的Neo4j服务已启动并运行：
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 2. 修改配置文件

编辑 `config.py` 文件，设置Neo4j连接信息：
```python
NEO4J_CONFIG = {
    'uri': 'bolt://localhost:7687',
    'auth': ('neo4j', 'password')  # 修改为您的密码
}
```

### 3. 构建知识图谱

```bash
# 一键构建
python run_build.py

# 或直接运行
python kg_construction.py
```

### 4. 执行可视化查询

```bash
# 一键查询
python run_visualization.py

# 或直接运行
python kg_visualization.py
```

## 功能特性

### 知识图谱构建

1. **实体创建**
   - 企业实体：包含股票代码、企业名称、行业、交易所
   - 行业实体：行业分类信息
   - 股东实体：股东基本信息
   - 概念实体：概念标签信息

2. **关系建立**
   - 企业-企业：基于股价相关系数的相关性分析
   - 企业-行业：企业所属行业关系
   - 企业-股东：股东持股关系（包含持股数量和比例）
   - 企业-概念：企业所属概念标签关系

3. **数据处理**
   - 自动去重和标准化
   - 相关性计算（皮尔逊相关系数）
   - 批量数据导入优化

### 可视化查询

1. **企业股东查询**：查看特定企业的股东信息
2. **相关企业查询**：查找与目标企业高度相关的企业
3. **同行业企业查询**：查看同一行业的企业
4. **企业基本信息查询**：获取企业详细资料
5. **概念企业查询**：查找特定概念下的企业
6. **主要股东查询**：统计持股企业数量最多的股东
7. **行业统计查询**：各行业企业数量统计
8. **相关性网络查询**：构建企业相关性网络

## 数据模型

### 实体类型

```
企业 {
    股票代码: String,
    企业名称: String,
    行业: String,
    交易所: String
}

行业 {
    行业名称: String
}

股东 {
    股东名称: String
}

概念 {
    概念名称: String
}
```

### 关系类型

```
企业-[:正相关]->企业 {
    相关系数: Float,
    相关性强度: String
}

企业-[:负相关]->企业 {
    相关系数: Float,
    相关性强度: String
}

企业-[:行业属于]->行业

企业-[:参股]->股东 {
    持有数量: Integer,
    持有比例: Float,
    公告日期: String,
    报告期: String
}

企业-[:概念属于]->概念
```

## 使用示例

### 查询平安银行的股东
```cypher
MATCH (holder:股东)-[r:参股]->(enterprise:企业)
WHERE enterprise.企业名称 = "平安银行"
RETURN holder.股东名称, r.持有比例
ORDER BY r.持有比例 DESC
```

### 查找银行行业的企业
```cypher
MATCH (e:企业)-[:行业属于]->(industry:行业)
WHERE industry.行业名称 = "银行"
RETURN e.企业名称, e.股票代码
```

### 分析企业相关性
```cypher
MATCH (e1:企业)-[r:正相关]->(e2:企业)
WHERE e1.企业名称 = "平安银行" AND r.相关系数 > 0.8
RETURN e2.企业名称, r.相关系数
ORDER BY r.相关系数 DESC
```

## 注意事项

1. **数据质量**：确保CSV文件格式正确，编码为UTF-8
2. **性能优化**：大量数据导入时建议使用批量操作
3. **内存管理**：处理大数据集时监控内存使用情况
4. **错误处理**：程序包含完整的错误处理和日志记录
5. **数据备份**：建议定期备份Neo4j数据库

## 故障排除

### 连接Neo4j失败
- 检查Neo4j服务是否启动
- 确认连接URI和认证信息
- 检查防火墙设置

### 数据导入失败
- 验证CSV文件路径和格式
- 检查数据编码
- 确认磁盘空间充足

### 查询性能问题
- 为常用查询字段创建索引
- 优化Cypher查询语句
- 考虑使用Neo4j的查询缓存

## 扩展功能

本项目提供了良好的基础框架，可以进一步扩展：

1. **时间序列分析**：添加时间维度的股价分析
2. **风险模型**：构建企业风险评估模型
3. **预测分析**：基于历史数据预测企业表现
4. **实时监控**：实时更新股价和相关性分析
5. **Web界面**：开发可视化Web界面

## 联系信息

如有问题或建议，请通过以下方式联系：
- 项目维护者：AI助手
- 更新日期：2024年

## 许可证

本项目采用MIT许可证，详见LICENSE文件。