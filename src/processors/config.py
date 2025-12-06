# 配置文件模块

# 轮次映射表
ROUND_MAPPING = {
    '天使轮': 'Angel',
    '种子轮': 'Seed', 
    'Pre-A轮': 'Pre-A',
    'A轮': 'A',
    'A+轮': 'A+',
    'Pre-B轮': 'Pre-B',
    'B轮': 'B',
    'B+轮': 'B+',
    'C轮': 'C',
    'D轮': 'D',
    'E轮': 'E',
    'Pre-IPO': 'Pre-IPO',
    'IPO': 'IPO',
    '战略融资': 'Strategic',
    '战略投资': 'Strategic',
    '并购': 'M&A',
    '收购并购': 'M&A',
    '新三板': 'NEEQ',
    '未融资': 'NoFunding'
}

# 金额单位转换规则
AMOUNT_RULES = [
    (r'(\d+(?:\.\d+)?)亿元人民币', lambda x: float(x) * 100000000),
    (r'(\d+(?:\.\d+)?)百万美元', lambda x: float(x) * 1000000 * 7.2),
    (r'(\d+(?:\.\d+)?)万美元', lambda x: float(x) * 10000 * 7.2),
    (r'(\d+(?:\.\d+)?)万人民币', lambda x: float(x) * 10000),
    (r'(\d+(?:\.\d+)?)万元', lambda x: float(x) * 10000),
    (r'(\d+(?:\.\d+)?)人民币', lambda x: float(x)),
    (r'(\d+(?:\.\d+)?)美元', lambda x: float(x) * 7.2),
]

# 注册资本单位转换规则
CAPITAL_RULES = [
    (r'(\d+(?:\.\d+)?)万元人民币', lambda x: float(x) * 10000),
    (r'(\d+(?:\.\d+)?)万元', lambda x: float(x) * 10000),
    (r'(\d+(?:\.\d+)?)百万美元', lambda x: float(x) * 1000000 * 7.2),
    (r'(\d+(?:\.\d+)?)万美元', lambda x: float(x) * 10000 * 7.2),
    (r'(\d+(?:\.\d+)?)亿元人民币', lambda x: float(x) * 100000000),
]

# 日期格式模式
DATE_PATTERNS = [
    (r'\d{4}-\d{1,2}-\d{1,2}', lambda x: x),  # YYYY-MM-DD
    (r'\d{4}/\d{1,2}/\d{1,2}', lambda x: x.replace('/', '-')),  # YYYY/MM/DD
    (r'\d{4}年\d{1,2}月\d{1,2}日', lambda x: x.replace('年', '-').replace('月', '-').replace('日', '')),
]

# 置信度阈值配置
CONFIDENCE_THRESHOLDS = {
    'exact_match': 1.0,
    'alias_match': 0.95,
    'substring_match': 0.8,
    'fuzzy_match': 0.7,
    'llm_match': 0.6
}

# LLM调用配置
LLM_CONFIG = {
    'max_retries': 3,
    'timeout': 30,
    'batch_size': 50,  # 批处理大小
    'confidence_threshold': 0.7,  # 低于此阈值才调用LLM
    'cache_enabled': True,
    'cache_size': 1000
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    'use_multiprocessing': True,
    'max_workers': 4,
    'chunk_size': 1000,
    'enable_progress_bar': True
}

# 批处理配置
BATCH_PROCESSING_CONFIG = {
    'default_batch_size': 50,  # 默认批处理大小
    'max_batch_size': 100,  # 最大批处理大小
    'min_batch_size': 10,  # 最小批处理大小
    'retry_attempts': 3,  # 重试次数
    'retry_delay': 1.0,  # 重试延迟（秒）
    'enable_caching': True,  # 启用缓存
    'cache_ttl': 3600,  # 缓存TTL（秒）
}