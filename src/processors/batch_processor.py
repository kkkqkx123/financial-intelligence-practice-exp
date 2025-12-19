"""
重构的批量处理逻辑

优化LLM查询的批量处理，提高效率并降低成本
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

from .llm_processor import SimplifiedLLMProcessor, get_llm_processor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """批量请求项"""
    id: str
    text: str
    request_type: str  # "entity", "relation", "attribute"
    priority: int = 0  # 优先级，数值越高优先级越高
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0  # 预估token数量
    similarity_hash: str = ""  # 用于相似性分组的哈希值
    
    # 新增字段，用于更细粒度的分组
    event_type: str = ""  # 事件类型，如"投资事件"、"公司信息"等
    text_length_category: str = ""  # 文本长度分类，如"短"、"中"、"长"
    investors: List[str] = field(default_factory=list)  # 投资机构列表
    industries: List[str] = field(default_factory=list)  # 行业列表
    investment_round: str = ""  # 投资轮次
    amount_range: str = ""  # 金额范围，如"千万级"、"亿级"等


@dataclass
class BatchResult:
    """批量处理结果"""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0


class TextSimilarityCalculator:
    """文本相似度计算器"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 中文不使用停用词
            ngram_range=(1, 2)  # 使用1-2gram
        )
        self.is_fitted = False
    
    def calculate_similarity(self, texts: List[str]) -> np.ndarray:
        """计算文本之间的相似度矩阵"""
        if not texts:
            return np.array([])
        
        try:
            # 如果文本数量小于2，直接返回全1矩阵
            if len(texts) < 2:
                return np.ones((len(texts), len(texts)))
            
            # 训练或更新TF-IDF向量化器
            if not self.is_fitted:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.is_fitted = True
            else:
                tfidf_matrix = self.vectorizer.transform(texts)
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"计算文本相似度时出错: {e}")
            # 出错时返回全1矩阵，表示所有文本都相似
            return np.ones((len(texts), len(texts)))
    
    def get_similar_groups(self, texts: List[str], threshold: float = 0.7) -> List[List[int]]:
        """根据相似度阈值获取文本分组"""
        if not texts:
            return []
        
        # 计算相似度矩阵
        similarity_matrix = self.calculate_similarity(texts)
        
        # 使用简单的聚类算法分组
        n = len(texts)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # 创建新组
            group = [i]
            visited[i] = True
            
            # 查找与当前文本相似的文本
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    visited[j] = True
            
            groups.append(group)
        
        return groups


class DynamicBatchSizer:
    """动态批量大小计算器"""
    
    def __init__(self, max_tokens_per_batch: int = 3000, safety_margin: float = 0.8):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.safety_margin = safety_margin
        self.token_estimation_cache = {}
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 使用缓存
        if text in self.token_estimation_cache:
            return self.token_estimation_cache[text]
        
        # 简单的token估算：中文字符*1.5 + 英文单词*1.3 + 标点*0.5
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len(text.split()) - chinese_chars  # 粗略估算
        punctuation = len([c for c in text if not c.isalnum() and not c.isspace()])
        
        estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3 + punctuation * 0.5)
        
        # 缓存结果
        self.token_estimation_cache[text] = estimated_tokens
        
        return estimated_tokens
    
    def calculate_optimal_batch_size(self, requests: List[BatchRequest]) -> int:
        """计算最优批量大小"""
        if not requests:
            return 0
        
        # 计算每个请求的平均token数
        total_tokens = sum(req.estimated_tokens for req in requests)
        avg_tokens = total_tokens / len(requests)
        
        # 计算理想批量大小
        ideal_batch_size = int((self.max_tokens_per_batch * self.safety_margin) / avg_tokens)
        
        # 确保至少处理一个请求
        return max(1, ideal_batch_size)
    
    def create_batches(self, requests: List[BatchRequest]) -> List[List[BatchRequest]]:
        """根据token限制创建批次"""
        if not requests:
            return []
        
        # 按优先级排序
        sorted_requests = sorted(requests, key=lambda x: (-x.priority, x.id))
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for request in sorted_requests:
            # 如果添加当前请求会超过token限制，则创建新批次
            if current_tokens + request.estimated_tokens > self.max_tokens_per_batch * self.safety_margin:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            # 添加请求到当前批次
            current_batch.append(request)
            current_tokens += request.estimated_tokens
        
        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)
        
        return batches


class PriorityManager:
    """优先级管理器"""
    
    def __init__(self):
        self.priority_rules = {
            # 实体类型优先级
            "entity": {
                "company": 10,
                "investor": 9,
                "person": 7,
                "default": 5
            },
            # 关系类型优先级
            "relation": {
                "invests_in": 10,
                "founder_of": 8,
                "works_at": 6,
                "default": 5
            }
        }
    
    def calculate_priority(self, request: BatchRequest) -> int:
        """计算请求的优先级"""
        # 如果请求已有优先级，直接返回
        if request.priority > 0:
            return request.priority
        
        # 根据请求类型和元数据计算优先级
        if request.request_type == "entity":
            entity_type = request.metadata.get("entity_type", "default")
            return self.priority_rules["entity"].get(entity_type, self.priority_rules["entity"]["default"])
        
        elif request.request_type == "relation":
            relation_type = request.metadata.get("relation_type", "default")
            return self.priority_rules["relation"].get(relation_type, self.priority_rules["relation"]["default"])
        
        # 默认优先级
        return 5
    
    def update_priorities(self, requests: List[BatchRequest]) -> None:
        """更新请求列表的优先级"""
        for request in requests:
            request.priority = self.calculate_priority(request)


class ErrorHandlingStrategy:
    """错误处理策略"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_stats = defaultdict(int)
    
    async def process_with_retry(self, processor_func, batch: List[BatchRequest]) -> List[BatchResult]:
        """带重试机制的批处理"""
        results = []
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # 尝试处理批次
                batch_results = await processor_func(batch)
                
                # 检查是否有失败的请求
                failed_requests = []
                for i, result in enumerate(batch_results):
                    if result.success:
                        results.append(result)
                    else:
                        # 记录错误
                        self.error_stats[result.error] += 1
                        failed_requests.append((batch[i], result))
                
                # 如果没有失败的请求，直接返回结果
                if not failed_requests:
                    break
                
                # 如果还有重试机会，准备重试失败的请求
                if retry_count < self.max_retries:
                    logger.warning(f"批次中有 {len(failed_requests)} 个请求失败，准备重试 ({retry_count + 1}/{self.max_retries})")
                    batch = [req for req, _ in failed_requests]
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # 指数退避
                else:
                    # 达到最大重试次数，将失败的请求也加入结果
                    for req, result in failed_requests:
                        results.append(result)
                    logger.error(f"达到最大重试次数，{len(failed_requests)} 个请求仍然失败")
                
                retry_count += 1
                
            except Exception as e:
                logger.error(f"批处理过程中发生异常: {e}")
                
                # 创建失败结果
                batch_results = []
                for req in batch:
                    batch_results.append(BatchResult(
                        request_id=req.id,
                        success=False,
                        error=f"批处理异常: {str(e)}"
                    ))
                
                results.extend(batch_results)
                
                # 如果还有重试机会，等待后重试整个批次
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                    retry_count += 1
                else:
                    logger.error(f"达到最大重试次数，整个批次处理失败")
                    break
        
        return results
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        return dict(self.error_stats)


class OptimizedBatchProcessor:
    """优化的批量处理器"""
    
    def __init__(self, llm_processor: SimplifiedLLMProcessor = None):
        self.llm_processor = llm_processor or get_llm_processor()
        self.similarity_calculator = TextSimilarityCalculator()
        self.batch_sizer = DynamicBatchSizer()
        self.priority_manager = PriorityManager()
        self.error_handler = ErrorHandlingStrategy()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_llm_calls": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "error_count": 0,
            "retry_count": 0
        }
        
        # 初始化knowledge_graph属性
        self.knowledge_graph = None
    
    def set_knowledge_graph(self, knowledge_graph):
        """设置知识图谱实例"""
        self.knowledge_graph = knowledge_graph
    
    def get_companies(self):
        """获取所有公司实体"""
        if hasattr(self.knowledge_graph, 'get_companies'):
            return self.knowledge_graph.get_companies()
        elif hasattr(self.knowledge_graph, 'companies'):
            return self.knowledge_graph.companies
        elif isinstance(self.knowledge_graph, dict) and 'companies' in self.knowledge_graph:
            return self.knowledge_graph['companies']
        else:
            logger.warning("无法获取公司实体，返回空列表")
            return []
    
    def get_investors(self):
        """获取所有投资方实体"""
        if hasattr(self.knowledge_graph, 'get_investors'):
            return self.knowledge_graph.get_investors()
        elif hasattr(self.knowledge_graph, 'investors'):
            return self.knowledge_graph.investors
        elif isinstance(self.knowledge_graph, dict) and 'investors' in self.knowledge_graph:
            return self.knowledge_graph['investors']
        else:
            logger.warning("无法获取投资方实体，返回空列表")
            return []
    
    async def process_requests(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """处理批量请求"""
        if not requests:
            return []
        
        start_time = time.time()
        
        # 更新统计
        self.stats["total_requests"] += len(requests)
        
        # 预处理请求
        processed_requests = self._preprocess_requests(requests)
        
        # 分组相似请求
        similarity_groups = self._group_similar_requests(processed_requests)
        
        # 创建批次
        batches = self._create_batches(similarity_groups)
        
        # 更新统计
        self.stats["total_batches"] += len(batches)
        self.stats["average_batch_size"] = sum(len(batch) for batch in batches) / len(batches) if batches else 0
        
        # 处理批次
        all_results = []
        for batch in batches:
            batch_results = await self._process_batch(batch)
            all_results.extend(batch_results)
            
            # 更新统计
            self.stats["total_llm_calls"] += 1
        
        # 更新处理时间
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        
        return all_results
    
    def _preprocess_requests(self, requests: List[BatchRequest]) -> List[BatchRequest]:
        """预处理请求"""
        processed_requests = []
        
        for request in requests:
            # 估算token数量
            request.estimated_tokens = self.batch_sizer.estimate_tokens(request.text)
            
            # 计算优先级
            self.priority_manager.update_priorities([request])
            
            # 计算相似性哈希（用于快速分组）
            request.similarity_hash = self._calculate_similarity_hash(request.text)
            
            # 提取和设置新字段
            self._extract_request_features(request)
            
            processed_requests.append(request)
        
        return processed_requests
    
    def _extract_request_features(self, request: BatchRequest) -> None:
        """提取请求的特征信息"""
        # 识别事件类型
        request.event_type = self._identify_event_type(request.text)
        
        # 分类文本长度
        request.text_length_category = self._categorize_text_length(request.text)
        
        # 提取投资机构
        request.investors = self._extract_investors(request.text)
        
        # 提取行业
        request.industries = self._extract_industries(request.text)
        
        # 提取投资轮次
        request.investment_round = self._extract_investment_round(request.text)
        
        # 识别金额范围
        request.amount_range = self._identify_amount_range(request.text)
    
    def _identify_event_type(self, text: str) -> str:
        """识别文本的事件类型"""
        text_lower = text.lower()
        
        # 投资事件关键词
        investment_keywords = ["投资", "融资", "轮", "投资方", "融资方", "领投", "跟投", "金额", "估值"]
        
        # 公司信息关键词
        company_keywords = ["公司", "企业", "成立", "总部", "员工", "业务", "产品", "服务"]
        
        # 行业分析关键词
        industry_keywords = ["行业", "市场", "趋势", "领域", "赛道", "细分市场"]
        
        # 计算关键词匹配度
        investment_score = sum(1 for kw in investment_keywords if kw in text_lower)
        company_score = sum(1 for kw in company_keywords if kw in text_lower)
        industry_score = sum(1 for kw in industry_keywords if kw in text_lower)
        
        # 确定事件类型
        if investment_score > company_score and investment_score > industry_score:
            return "投资事件"
        elif company_score > industry_score:
            return "公司信息"
        elif industry_score > 0:
            return "行业分析"
        else:
            return "其他"
    
    def _categorize_text_length(self, text: str) -> str:
        """根据文本长度分类"""
        length = len(text)
        
        if length < 100:
            return "短"
        elif length < 500:
            return "中"
        else:
            return "长"
    
    def _extract_investors(self, text: str) -> List[str]:
        """从文本中提取投资机构"""
        import re
        
        # 常见投资机构后缀
        investor_suffixes = ["资本", "基金", "投资", "创投", "风投", "创投基金", "资产管理", "集团"]
        
        # 查找包含投资机构后缀的词汇
        investors = []
        for suffix in investor_suffixes:
            pattern = f'[^，。；！？\n]*{suffix}[^，。；！？\n]*'
            matches = re.findall(pattern, text)
            investors.extend(matches)
        
        # 去重并清理
        investors = list(set(investors))
        investors = [inv.strip() for inv in investors if inv.strip()]
        
        return investors
    
    def _extract_industries(self, text: str) -> List[str]:
        """从文本中提取行业"""
        import re
        
        # 常见行业关键词
        industry_keywords = [
            "企业服务", "文娱", "内容", "游戏", "金融", "医疗", "健康", "教育", "电商",
            "零售", "物流", "制造", "汽车", "房产", "科技", "互联网", "软件", "硬件",
            "人工智能", "大数据", "云计算", "区块链", "新能源", "环保", "农业", "食品"
        ]
        
        # 查找包含行业关键词的词汇
        industries = []
        for keyword in industry_keywords:
            if keyword in text:
                industries.append(keyword)
        
        # 使用正则表达式提取"XX行业"或"XX领域"
        pattern = r'([^，。；！？\n]*(?:行业|领域|赛道)[^，。；！？\n]*)'
        matches = re.findall(pattern, text)
        industries.extend(matches)
        
        # 去重并清理
        industries = list(set(industries))
        industries = [ind.strip() for ind in industries if ind.strip()]
        
        return industries
    
    def _extract_investment_round(self, text: str) -> str:
        """从文本中提取投资轮次"""
        import re
        
        # 常见轮次
        rounds = ["种子轮", "天使轮", "Pre-A", "A轮", "A+轮", "B轮", "B+轮", "C轮", "C+轮", "D轮", "E轮", "F轮", "Pre-IPO"]
        
        for round_name in rounds:
            if round_name in text:
                return round_name
        
        # 使用正则表达式匹配轮次
        pattern = r'([A-Z]轮|[A-Z]\+轮|Pre-[A-Z]|天使轮|种子轮|Pre-IPO)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        
        return ""
    
    def _identify_amount_range(self, text: str) -> str:
        """识别金额范围"""
        import re
        
        # 金额单位模式
        amount_patterns = [
            (r'([0-9.]+)\s*千万', "千万级"),
            (r'([0-9.]+)\s*亿', "亿级"),
            (r'千万级', "千万级"),
            (r'亿级', "亿级"),
            (r'数千万', "千万级"),
            (r'数亿', "亿级"),
            (r'过千万', "千万级"),
            (r'过亿', "亿级"),
            (r'近千万', "千万级"),
            (r'近亿', "亿级")
        ]
        
        for pattern, range_name in amount_patterns:
            if re.search(pattern, text):
                return range_name
        
        return ""
    
    def _calculate_similarity_hash(self, text: str) -> str:
        """计算文本的相似性哈希值"""
        # 提取前100个字符作为哈希输入
        text_sample = text[:100]
        return hashlib.md5(text_sample.encode()).hexdigest()
    
    def _group_similar_requests(self, requests: List[BatchRequest]) -> List[List[BatchRequest]]:
        """将相似请求分组，使用多维度分组策略"""
        if not requests:
            return []
        
        # 第一步：按事件类型分组
        event_type_groups = defaultdict(list)
        for request in requests:
            event_type_groups[request.event_type].append(request)
        
        # 第二步：在每个事件类型组内，按文本长度分组
        text_length_groups = defaultdict(list)
        for event_type, type_requests in event_type_groups.items():
            for request in type_requests:
                # 创建复合键：事件类型 + 文本长度
                key = f"{event_type}_{request.text_length_category}"
                text_length_groups[key].append(request)
        
        # 第三步：在文本长度组内，按投资机构分组
        investor_groups = defaultdict(list)
        for text_length_key, length_requests in text_length_groups.items():
            for request in length_requests:
                # 如果有投资机构信息，则按投资机构分组
                if request.investors:
                    # 使用第一个投资机构作为分组依据
                    investor = request.investors[0]
                    key = f"{text_length_key}_{investor}"
                else:
                    # 没有投资机构信息，标记为"无投资机构"
                    key = f"{text_length_key}_无投资机构"
                
                investor_groups[key].append(request)
        
        # 第四步：在投资机构组内，按行业分组
        industry_groups = defaultdict(list)
        for investor_key, investor_requests in investor_groups.items():
            for request in investor_requests:
                # 如果有行业信息，则按行业分组
                if request.industries:
                    # 使用第一个行业作为分组依据
                    industry = request.industries[0]
                    key = f"{investor_key}_{industry}"
                else:
                    # 没有行业信息，标记为"无行业"
                    key = f"{investor_key}_无行业"
                
                industry_groups[key].append(request)
        
        # 第五步：在行业组内，按金额范围分组
        amount_groups = defaultdict(list)
        for industry_key, industry_requests in industry_groups.items():
            for request in industry_requests:
                # 如果有金额范围信息，则按金额范围分组
                if request.amount_range:
                    key = f"{industry_key}_{request.amount_range}"
                else:
                    # 没有金额范围信息，标记为"无金额"
                    key = f"{industry_key}_无金额"
                
                amount_groups[key].append(request)
        
        # 第六步：在金额范围组内，按投资轮次分组
        final_groups = defaultdict(list)
        for amount_key, amount_requests in amount_groups.items():
            for request in amount_requests:
                # 如果有投资轮次信息，则按投资轮次分组
                if request.investment_round:
                    key = f"{amount_key}_{request.investment_round}"
                else:
                    # 没有投资轮次信息，标记为"无轮次"
                    key = f"{amount_key}_无轮次"
                
                final_groups[key].append(request)
        
        # 第七步：在每个最终组内，按文本相似度进一步分组
        similarity_groups = []
        for group_key, group_requests in final_groups.items():
            if len(group_requests) <= 1:
                # 如果只有一个请求，直接作为一个组
                similarity_groups.append(group_requests)
                continue
            
            # 提取文本
            texts = [req.text for req in group_requests]
            
            # 计算相似度分组
            groups = self.similarity_calculator.get_similar_groups(texts, threshold=0.7)
            
            # 创建分组
            for group_indices in groups:
                group_requests_similar = [group_requests[i] for i in group_indices]
                similarity_groups.append(group_requests_similar)
        
        return similarity_groups
    
    def _create_batches(self, similarity_groups: List[List[BatchRequest]]) -> List[List[BatchRequest]]:
        """从相似组创建批次"""
        all_batches = []
        
        for group in similarity_groups:
            # 为每个相似组创建批次
            group_batches = self.batch_sizer.create_batches(group)
            all_batches.extend(group_batches)
        
        return all_batches
    
    async def _process_batch(self, batch: List[BatchRequest]) -> List[BatchResult]:
        """处理单个批次"""
        start_time = time.time()
        
        # 准备批处理
        batch_type = batch[0].request_type
        texts = [req.text for req in batch]
        
        # 根据请求类型调用相应的处理函数
        if batch_type == "entity":
            processor_func = self._process_entity_batch
        elif batch_type == "relation":
            processor_func = self._process_relation_batch
        elif batch_type == "attribute":
            processor_func = self._process_attribute_batch
        else:
            # 未知请求类型，返回错误结果
            return [BatchResult(
                request_id=req.id,
                success=False,
                error=f"未知请求类型: {batch_type}"
            ) for req in batch]
        
        # 使用错误处理策略处理批次
        results = await self.error_handler.process_with_retry(processor_func, batch)
        
        # 更新统计
        processing_time = time.time() - start_time
        error_count = sum(1 for r in results if not r.success)
        self.stats["error_count"] += error_count
        
        return results
    
    async def _process_entity_batch(self, batch: List[BatchRequest]) -> List[BatchResult]:
        """处理实体提取批次"""
        texts = [req.text for req in batch]
        request_ids = [req.id for req in batch]
        
        try:
            # 调用LLM处理器提取实体
            entities = await self.llm_processor.extract_entities(texts)
            
            # 创建结果
            results = []
            for i, request_id in enumerate(request_ids):
                # 为每个请求找到对应的实体
                # 这里假设entities的顺序与texts的顺序一致
                request_entities = entities[i] if i < len(entities) else []
                
                results.append(BatchResult(
                    request_id=request_id,
                    success=True,
                    result=request_entities
                ))
            
            return results
        except Exception as e:
            # 创建失败结果
            return [BatchResult(
                request_id=req.id,
                success=False,
                error=f"实体提取失败: {str(e)}"
            ) for req in batch]
    
    async def _process_relation_batch(self, batch: List[BatchRequest]) -> List[BatchResult]:
        """处理关系提取批次"""
        texts = [req.text for req in batch]
        request_ids = [req.id for req in batch]
        
        try:
            # 调用LLM处理器提取关系
            relations = await self.llm_processor.extract_relations(texts)
            
            # 创建结果
            results = []
            for i, request_id in enumerate(request_ids):
                # 为每个请求找到对应的关系
                request_relations = relations[i] if i < len(relations) else []
                
                results.append(BatchResult(
                    request_id=request_id,
                    success=True,
                    result=request_relations
                ))
            
            return results
        except Exception as e:
            # 创建失败结果
            return [BatchResult(
                request_id=req.id,
                success=False,
                error=f"关系提取失败: {str(e)}"
            ) for req in batch]
    
    async def _process_attribute_batch(self, batch: List[BatchRequest]) -> List[BatchResult]:
        """处理属性提取批次"""
        texts = [req.text for req in batch]
        request_ids = [req.id for req in batch]
        
        try:
            # 调用LLM处理器提取属性
            attributes = await self.llm_processor.extract_attributes(texts)
            
            # 创建结果
            results = []
            for i, request_id in enumerate(request_ids):
                # 为每个请求找到对应的属性
                request_attributes = attributes[i] if i < len(attributes) else []
                
                results.append(BatchResult(
                    request_id=request_id,
                    success=True,
                    result=request_attributes
                ))
            
            # 处理所有待处理的增强任务
            try:
                enhancement_results = await self.process_all_pending_enhancements()
                self.stats['enhancements_processed'] = enhancement_results
            except Exception as e:
                logger.error(f"处理增强任务时出错: {str(e)}")
                self.stats['enhancements_processed'] = {
                    'entity_descriptions': {'processed': 0, 'enhanced': 0},
                    'industry_classifications': {'processed': 0, 'enhanced': 0},
                    'investor_standardizations': {'processed': 0, 'enhanced': 0}
                }
            
            return results
        except Exception as e:
            # 创建失败结果
            return [BatchResult(
                request_id=req.id,
                success=False,
                error=f"属性提取失败: {str(e)}"
            ) for req in batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        stats.update(self.error_handler.get_error_stats())
        
        # 计算平均处理时间
        if stats["total_batches"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_batches"]
        else:
            stats["average_processing_time"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_llm_calls": 0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "error_count": 0,
            "retry_count": 0
        }
        self.error_handler.error_stats = defaultdict(int)
    
    async def optimize_entity_descriptions(self, entities: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """优化实体描述"""
        enhanced_entities = {}
        
        for entity_id, entity_data in entities.items():
            # 复制原始数据
            enhanced_entity = entity_data.copy()
            
            # 如果没有描述或描述太短，则使用LLM生成描述
            if not enhanced_entity.get('description') or len(enhanced_entity.get('description', '')) < 10:
                try:
                    # 构建LLM提示
                    if entity_type == 'company':
                        name = enhanced_entity.get('name', '')
                        industry = enhanced_entity.get('industry', '')
                        location = enhanced_entity.get('location', '')
                        
                        prompt = f"""
                        任务：为公司生成简洁、专业的描述
                        
                        公司名称：{name}
                        行业：{industry}
                        地点：{location}
                        
                        要求：
                        1. 生成一段50-100字的简洁公司描述
                        2. 突出公司的主要业务和特点
                        3. 语言要专业、客观
                        
                        输出格式：
                        {{
                            "description": "公司描述内容"
                        }}
                        """
                    elif entity_type == 'investor':
                        name = enhanced_entity.get('name', '')
                        investor_type = enhanced_entity.get('type', '')
                        
                        prompt = f"""
                        任务：为投资机构生成简洁、专业的描述
                        
                        机构名称：{name}
                        机构类型：{investor_type}
                        
                        要求：
                        1. 生成一段50-100字的简洁投资机构描述
                        2. 突出机构的投资领域和特点
                        3. 语言要专业、客观
                        
                        输出格式：
                        {{
                            "description": "投资机构描述内容"
                        }}
                        """
                    else:
                        # 其他类型实体，使用基本描述
                        if entity_type == 'company':
                            name = enhanced_entity.get('name', '')
                            industry = enhanced_entity.get('industry', '')
                            location = enhanced_entity.get('location', '')
                            enhanced_entity['description'] = f"{name}是一家位于{location}的{industry}企业。"
                        elif entity_type == 'investor':
                            name = enhanced_entity.get('name', '')
                            investor_type = enhanced_entity.get('type', '')
                            enhanced_entity['description'] = f"{name}是一家{investor_type}投资机构。"
                        
                        enhanced_entities[entity_id] = enhanced_entity
                        continue
                    
                    # 调用LLM生成描述
                    result = await self.llm_processor.llm_client.generate_completion(prompt)
                    
                    # 检查结果
                    if isinstance(result, dict) and 'description' in result:
                        enhanced_entity['description'] = result['description']
                    elif isinstance(result, str):
                        enhanced_entity['description'] = result
                    else:
                        # 回退到基本描述
                        if entity_type == 'company':
                            name = enhanced_entity.get('name', '')
                            industry = enhanced_entity.get('industry', '')
                            location = enhanced_entity.get('location', '')
                            enhanced_entity['description'] = f"{name}是一家位于{location}的{industry}企业。"
                        elif entity_type == 'investor':
                            name = enhanced_entity.get('name', '')
                            investor_type = enhanced_entity.get('type', '')
                            enhanced_entity['description'] = f"{name}是一家{investor_type}投资机构。"
                
                except Exception as e:
                    print(f"生成实体描述时出错: {str(e)}")
                    # 回退到基本描述
                    if entity_type == 'company':
                        name = enhanced_entity.get('name', '')
                        industry = enhanced_entity.get('industry', '')
                        location = enhanced_entity.get('location', '')
                        enhanced_entity['description'] = f"{name}是一家位于{location}的{industry}企业。"
                    elif entity_type == 'investor':
                        name = enhanced_entity.get('name', '')
                        investor_type = enhanced_entity.get('type', '')
                        enhanced_entity['description'] = f"{name}是一家{investor_type}投资机构。"
            
            enhanced_entities[entity_id] = enhanced_entity
        
        return enhanced_entities
    
    async def optimize_industry_classification(self, companies: Dict[str, Any]) -> Dict[str, Any]:
        """优化行业分类"""
        industry_classifications = {}
        
        for company_id, company_data in companies.items():
            name = company_data.get('name', '')
            industry = company_data.get('industry', '')
            description = company_data.get('description', '')
            
            # 如果已有行业分类且置信度高，直接使用
            if industry and company_data.get('industry_confidence', 0) > 0.6:  # 从0.8降低到0.6
                industry_classifications[company_id] = {
                    'company_id': company_id,
                    'company_name': name,
                    'industry': industry,
                    'confidence': company_data.get('industry_confidence', 0.6),  # 从0.8降低到0.6
                }
                continue
            
            # 如果没有行业分类或置信度低，使用LLM进行分类
            try:
                prompt = f"""
                任务：为公司进行行业分类
                
                公司名称：{name}
                当前行业：{industry if industry else "未分类"}
                公司描述：{description}
                
                要求：
                1. 根据公司名称和描述，确定最合适的行业分类
                2. 从以下行业中选择一个最合适的：科技、金融、医疗健康、教育、零售、制造、房地产、能源、交通、文娱、企业服务、其他
                3. 评估分类的置信度（0-1之间的数值）
                
                输出格式：
                {{
                    "industry": "行业分类",
                    "confidence": 0.9,
                    "reasoning": "分类理由"
                }}
                """
                
                # 调用LLM进行行业分类
                result = await self.llm_processor.llm_client.generate_completion(prompt)
                
                # 检查结果
                if isinstance(result, dict) and 'industry' in result:
                    industry_classifications[company_id] = {
                        'company_id': company_id,
                        'company_name': name,
                        'industry': result['industry'],
                        'confidence': result.get('confidence', 0.6),  # 从0.8降低到0.6
                        'reasoning': result.get('reasoning', '')
                    }
                else:
                    # 回退到简单分类逻辑
                    fallback_industry = self._simple_industry_classification(description)
                    industry_classifications[company_id] = {
                        'company_id': company_id,
                        'company_name': name,
                        'industry': fallback_industry,
                        'confidence': 0.6,
                        'reasoning': '回退到简单分类逻辑'
                    }
            
            except Exception as e:
                print(f"行业分类时出错: {str(e)}")
                # 回退到简单分类逻辑
                fallback_industry = self._simple_industry_classification(description)
                industry_classifications[company_id] = {
                    'company_id': company_id,
                    'company_name': name,
                    'industry': fallback_industry,
                    'confidence': 0.6,
                    'reasoning': '回退到简单分类逻辑'
                }
        
        return industry_classifications
    
    def _simple_industry_classification(self, description: str) -> str:
        """简单的行业分类逻辑（作为回退方案）"""
        if not description:
            return "其他"
        
        # 根据描述推断行业
        if '科技' in description or '技术' in description or '软件' in description or '互联网' in description:
            return '科技'
        elif '金融' in description or '投资' in description or '银行' in description or '保险' in description:
            return '金融'
        elif '医疗' in description or '健康' in description or '生物' in description or '医药' in description:
            return '医疗健康'
        elif '教育' in description or '培训' in description or '学校' in description:
            return '教育'
        elif '零售' in description or '电商' in description or '销售' in description or '商店' in description:
            return '零售'
        elif '制造' in description or '工厂' in description or '生产' in description:
            return '制造'
        elif '房产' in description or '房地产' in description or '地产' in description:
            return '房地产'
        elif '能源' in description or '电力' in description or '石油' in description:
            return '能源'
        elif '交通' in description or '汽车' in description or '运输' in description:
            return '交通'
        elif '文娱' in description or '娱乐' in description or '媒体' in description:
            return '文娱'
        elif '企业服务' in description or '服务' in description:
            return '企业服务'
        else:
            return '其他'
    
    async def optimize_investor_name_standardization(self, investor_names: Set[str]) -> Dict[str, str]:
        """优化投资方名称标准化"""
        standardized_names = {}
        
        for name in investor_names:
            # 如果名称已经是标准格式（包含常见投资机构关键词），直接使用
            if any(keyword in name for keyword in ['资本', '投资', '基金', '创投', 'Venture', 'Capital', 'Investment', 'Fund']):
                standardized_names[name] = name
                continue
            
            # 使用LLM进行名称标准化
            try:
                prompt = f"""
                任务：标准化投资机构名称
                
                原始名称：{name}
                
                要求：
                1. 识别投资机构的标准名称
                2. 保留机构的核心标识（如"资本"、"投资"、"基金"等）
                3. 去除多余的描述性词汇
                4. 如果无法确定标准名称，返回原始名称
                
                输出格式：
                {{
                    "standardized_name": "标准名称",
                    "confidence": 0.9,
                    "reasoning": "标准化理由"
                }}
                """
                
                # 调用LLM进行名称标准化
                result = await self.llm_processor.llm_client.generate_completion(prompt)
                
                # 检查结果
                if isinstance(result, dict) and 'standardized_name' in result:
                    standardized_names[name] = result['standardized_name']
                else:
                    # 回退到简单标准化逻辑
                    fallback_name = self._simple_investor_name_standardization(name)
                    standardized_names[name] = fallback_name
            
            except Exception as e:
                print(f"投资方名称标准化时出错: {str(e)}")
                # 回退到简单标准化逻辑
                fallback_name = self._simple_investor_name_standardization(name)
                standardized_names[name] = fallback_name
        
        return standardized_names
    
    def _simple_investor_name_standardization(self, name: str) -> str:
        """简单的投资方名称标准化逻辑（作为回退方案）"""
        # 去除常见的非核心词汇
        core_name = name
        non_core_words = ['有限公司', '有限责任公司', '股份有限公司', '集团', '控股', '企业']
        
        for word in non_core_words:
            core_name = core_name.replace(word, '')
        
        # 如果去除后名称太短，返回原始名称
        if len(core_name) < 2:
            return name
        
        # 确保包含投资相关关键词
        if not any(keyword in core_name for keyword in ['资本', '投资', '基金', '创投', 'Venture', 'Capital', 'Investment', 'Fund']):
            # 如果没有投资相关关键词，添加"投资"
            core_name += "投资"
        
        return core_name
    
    async def process_all_pending_enhancements(self) -> Dict[str, Any]:
        """处理所有待处理的增强任务"""
        # 模拟处理结果
        results = {
            'entity_descriptions': {'processed': 0, 'enhanced': 0},
            'industry_classifications': {'processed': 0, 'enhanced': 0},
            'investor_standardizations': {'processed': 0, 'enhanced': 0}
        }
        
        # 检查knowledge_graph是否已设置
        if self.knowledge_graph is None:
            print("警告: knowledge_graph未设置，跳过增强处理")
            return results
        
        # 获取待处理的实体
        companies = self.get_companies()
        investors = self.get_investors()
        
        # 处理实体描述优化
        if companies:
            entity_results = await self.optimize_entity_descriptions(companies)
            results['entity_descriptions']['processed'] = len(companies)
            results['entity_descriptions']['enhanced'] = len(entity_results)
        
        # 处理行业分类优化
        if companies:
            industry_results = await self.optimize_industry_classification(companies)
            results['industry_classifications']['processed'] = len(companies)
            results['industry_classifications']['enhanced'] = len(industry_results)
        
        # 处理投资方名称标准化
        if investors:
            investor_names = {inv.get('name', '') for inv in investors if inv.get('name')}
            if investor_names:
                investor_results = await self.optimize_investor_name_standardization(investor_names)
                results['investor_standardizations']['processed'] = len(investor_names)
                results['investor_standardizations']['enhanced'] = len(investor_results)
        
        return results


# 全局批量处理器实例
_global_processor = None


def get_batch_processor() -> OptimizedBatchProcessor:
    """获取全局批量处理器实例"""
    global _global_processor
    if _global_processor is None:
        _global_processor = OptimizedBatchProcessor()
    return _global_processor


async def process_batch_requests(requests: List[BatchRequest]) -> List[BatchResult]:
    """处理批量请求的便捷函数"""
    processor = get_batch_processor()
    return await processor.process_requests(requests)


def create_batch_request(id: str, text: str, request_type: str, 
                        priority: int = 0, metadata: Dict[str, Any] = None) -> BatchRequest:
    """创建批量请求的便捷函数"""
    return BatchRequest(
        id=id,
        text=text,
        request_type=request_type,
        priority=priority,
        metadata=metadata or {}
    )