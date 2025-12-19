"""
测试新实现的功能

包含对简化LLM处理器、优化关系提取流程和批量处理逻辑的测试
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processors.llm_processor import (
    SimplifiedLLMProcessor, 
    Entity, 
    Relation, 
    get_llm_processor
)
from src.processors.relation_extractor import (
    OptimizedRelationExtractor,
    InvestmentEvent,
    get_relation_extractor,
    extract_entities_and_relations
)
from src.processors.batch_processor import (
    OptimizedBatchProcessor,
    BatchRequest,
    BatchResult,
    get_batch_processor,
    create_batch_request
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSuite:
    """测试套件"""
    
    def __init__(self):
        self.test_results = {
            "llm_processor": {},
            "optimized_relation_extractor": {},
            "optimized_batch_processor": {}
        }
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行所有测试...")
        
        # 测试简化LLM处理器
        await self.test_llm_processor()
        
        # 测试优化关系提取流程
        await self.test_optimized_relation_extractor()
        
        # 测试优化批量处理逻辑
        await self.test_optimized_batch_processor()
        
        # 打印测试结果
        self.print_test_results()
    
    async def test_llm_processor(self):
        """测试简化LLM处理器"""
        logger.info("测试简化LLM处理器...")
        
        try:
            # 获取LLM处理器实例
            processor = get_llm_processor()
            
            # 测试文本
            test_texts = [
                "红杉资本投资了字节跳动",
                "腾讯领投了京东的D轮融资",
                "阿里巴巴和蚂蚁金服共同投资了饿了么",
                "IDG资本投资了百度和腾讯",
                "高瓴资本投资了美团点评"
            ]
            
            # 测试实体提取
            start_time = time.time()
            entities = await processor.extract_entities(test_texts)
            entity_time = time.time() - start_time
            
            # 测试关系提取
            start_time = time.time()
            relations = await processor.extract_relations(test_texts)
            relation_time = time.time() - start_time
            
            # 测试属性提取
            start_time = time.time()
            attributes = await processor.extract_attributes(test_texts)
            attribute_time = time.time() - start_time
            
            # 记录结果
            self.test_results["llm_processor"] = {
                "success": True,
                "entity_extraction_time": entity_time,
                "relation_extraction_time": relation_time,
                "attribute_extraction_time": attribute_time,
                "total_entities": len(entities),
                "total_relations": len(relations),
                "total_attributes": len(attributes),
                "sample_entities": [e.name for e in entities[:5]],
                "sample_relations": [f"{r.type}" for r in relations[:5]],
                "sample_attributes": [str(a) for a in attributes[:5]]
            }
            
            logger.info(f"简化LLM处理器测试成功，提取了 {len(entities)} 个实体，{len(relations)} 个关系，{len(attributes)} 个属性")
            
        except Exception as e:
            logger.error(f"简化LLM处理器测试失败: {e}")
            self.test_results["llm_processor"] = {
                "success": False,
                "error": str(e)
            }
    
    async def test_optimized_relation_extractor(self):
        """测试优化关系提取流程"""
        logger.info("测试优化关系提取流程...")
        
        try:
            # 获取关系提取器实例
            extractor = get_relation_extractor()
            
            # 测试文本
            test_texts = [
                "红杉资本投资了字节跳动，投资金额为1亿美元",
                "腾讯领投了京东的D轮融资，金额为10亿美元",
                "阿里巴巴和蚂蚁金服共同投资了饿了么，投资金额为95亿美元",
                "IDG资本投资了百度和腾讯，投资轮次为A轮",
                "高瓴资本投资了美团点评，投资时间为2020年"
            ]
            
            # 测试实体提取
            start_time = time.time()
            entities = await extractor.extract_entities_from_texts(test_texts)
            entity_time = time.time() - start_time
            
            # 测试关系提取
            start_time = time.time()
            relations = await extractor.extract_relations_from_texts(test_texts)
            relation_time = time.time() - start_time
            
            # 测试便捷函数
            start_time = time.time()
            conv_entities, conv_relations = await extract_entities_and_relations(test_texts)
            conv_time = time.time() - start_time
            
            # 获取统计信息
            stats = extractor.get_stats()
            
            # 记录结果
            self.test_results["optimized_relation_extractor"] = {
                "success": True,
                "entity_extraction_time": entity_time,
                "relation_extraction_time": relation_time,
                "convenience_function_time": conv_time,
                "total_entities": len(entities),
                "total_relations": len(relations),
                "total_texts": stats["total_texts"],
                "rule_events": stats["rule_events"],
                "llm_events": stats["llm_events"],
                "high_confidence_events": stats["high_confidence_events"],
                "llm_calls": stats["llm_calls"],
                "sample_entities": [e.name for e in entities[:5]],
                "sample_relations": [f"{r.type}" for r in relations[:5]]
            }
            
            logger.info(f"优化关系提取流程测试成功，提取了 {len(entities)} 个实体，{len(relations)} 个关系")
            
        except Exception as e:
            logger.error(f"优化关系提取流程测试失败: {e}")
            self.test_results["optimized_relation_extractor"] = {
                "success": False,
                "error": str(e)
            }
    
    async def test_optimized_batch_processor(self):
        """测试优化批量处理逻辑"""
        logger.info("测试优化批量处理逻辑...")
        
        try:
            # 获取批量处理器实例
            processor = get_batch_processor()
            
            # 创建测试请求
            test_requests = []
            for i in range(10):
                request = create_batch_request(
                    id=f"test_request_{i}",
                    text=f"红杉资本投资了字节跳动{i}，投资金额为{i}亿美元",
                    request_type="relation",
                    priority=i % 3,
                    metadata={"relation_type": "invests_in"}
                )
                test_requests.append(request)
            
            # 测试批量处理
            start_time = time.time()
            results = await processor.process_requests(test_requests)
            batch_time = time.time() - start_time
            
            # 获取统计信息
            stats = processor.get_stats()
            
            # 计算成功率
            success_count = sum(1 for r in results if r.success)
            success_rate = success_count / len(results) if results else 0
            
            # 记录结果
            self.test_results["optimized_batch_processor"] = {
                "success": True,
                "batch_processing_time": batch_time,
                "total_requests": stats["total_requests"],
                "total_batches": stats["total_batches"],
                "total_llm_calls": stats["total_llm_calls"],
                "average_batch_size": stats["average_batch_size"],
                "error_count": stats["error_count"],
                "success_count": success_count,
                "success_rate": success_rate,
                "average_processing_time": stats.get("average_processing_time", 0)
            }
            
            logger.info(f"优化批量处理逻辑测试成功，处理了 {len(test_requests)} 个请求，成功率: {success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"优化批量处理逻辑测试失败: {e}")
            self.test_results["optimized_batch_processor"] = {
                "success": False,
                "error": str(e)
            }
    
    def print_test_results(self):
        """打印测试结果"""
        logger.info("测试结果:")
        
        for component, results in self.test_results.items():
            logger.info(f"\n{component}:")
            
            if results.get("success", False):
                for key, value in results.items():
                    if key != "success":
                        logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  测试失败: {results.get('error', '未知错误')}")
        
        # 保存测试结果到文件
        self.save_test_results()
    
    def save_test_results(self):
        """保存测试结果到文件"""
        try:
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")


async def main():
    """主函数"""
    test_suite = TestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())