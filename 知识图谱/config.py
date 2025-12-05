"""
知识图谱项目配置文件
"""

# Neo4j数据库连接配置
NEO4J_CONFIG = {
    'uri': 'bolt://localhost:7687',
    'auth': ('neo4j', '1234567kk'),  # 使用默认密码，首次登录需要修改
    'http_uri': 'http://localhost:7474/'
}

# 数据文件路径配置
DATA_CONFIG = {
    'data_dir': '数据集',
    'enterprise_basic': 'enterprise_basic.csv',
    'enterprise_concept': 'enterprise_concept.csv', 
    'enterprise_holders': 'enterprise_holders.csv',
    'stock_prices': 'stock_prices.csv'
}

# 知识图谱构建参数
KG_CONFIG = {
    # 相关性计算参数
    'correlation_threshold': 0.5,  # 企业间相关系数阈值
    'min_correlation_for_network': 0.6,  # 网络分析最小相关系数
    
    # 查询参数
    'default_correlation_threshold': 0.8,  # 默认相关性查询阈值
    'major_shareholder_threshold': 5.0,  # 主要股东持股比例阈值（%）
    'network_depth': 2,  # 网络分析深度
    
    # 性能参数
    'batch_size': 1000,  # 批量处理大小
    'max_query_results': 1000  # 最大查询结果数
}

# 可视化参数
VISUALIZATION_CONFIG = {
    'node_colors': {
        '企业': '#4CAF50',
        '行业': '#2196F3', 
        '股东': '#FF9800',
        '概念': '#9C27B0'
    },
    'relationship_colors': {
        '正相关': '#4CAF50',
        '负相关': '#F44336',
        '行业属于': '#2196F3',
        '参股': '#FF9800',
        '概念属于': '#9C27B0'
    }
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}