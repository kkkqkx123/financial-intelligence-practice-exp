#!/usr/bin/env python3
"""
金融知识图谱完整测试脚本
运行完整的知识图谱构建流程，包括数据验证、LLM增强、Neo4j导出等
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from main import KnowledgeGraphPipeline
    from integrations.neo4j_exporter import Neo4jConfig
    NEO4J_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neo4j集成不可用: {e}")
    from main import KnowledgeGraphPipeline
    NEO4J_AVAILABLE = False


class CompletePipelineTester:
    """完整流程测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        
    def run_complete_pipeline_test(self, enable_neo4j: bool = False) -> Dict:
        """运行完整的知识图谱构建测试"""
        logger.info("开始完整流程测试")
        self.start_time = time.time()
        
        try:
            # 1. 初始化流水线
            logger.info("1. 初始化知识图谱流水线")
            neo4j_config = None
            if enable_neo4j and NEO4J_AVAILABLE:
                neo4j_config = {
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "database": "neo4j"
                }
            
            pipeline = KnowledgeGraphPipeline(
                data_dir="dataset",
                output_dir="output",
                enable_neo4j=enable_neo4j,
                neo4j_config=neo4j_config
            )
            
            # 2. 运行完整流程
            logger.info("2. 运行完整构建流程")
            results = pipeline.run_full_pipeline(save_intermediate=True)
            
            # 3. 验证结果
            logger.info("3. 验证结果")
            validation_passed = self.validate_results(results)
            
            # 4. 性能分析
            logger.info("4. 性能分析")
            performance_metrics = self.analyze_performance(results)
            
            # 5. 生成测试报告
            logger.info("5. 生成测试报告")
            test_report = self.generate_test_report(results, validation_passed, performance_metrics)
            
            total_time = time.time() - self.start_time
            logger.info(f"完整测试完成，总耗时: {total_time:.2f}秒")
            
            return {
                "success": True,
                "validation_passed": validation_passed,
                "results": results,
                "performance_metrics": performance_metrics,
                "test_report": test_report,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"完整测试失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - self.start_time if self.start_time else 0
            }
    
    def validate_results(self, results: Dict) -> bool:
        """验证测试结果"""
        try:
            # 检查必需字段
            required_keys = ["validation_results", "enhancement_results", "final_knowledge_graph"]
            for key in required_keys:
                if key not in results:
                    logger.error(f"缺少必需字段: {key}")
                    return False
            
            # 验证数据验证结果
            validation_results = results.get("validation_results", {})
            if not validation_results.get("overall_quality", {}).get("is_valid", False):
                logger.warning("数据验证未通过")
            
            # 验证知识图谱数据
            kg_data = results.get("final_knowledge_graph", {})
            if not kg_data.get("entities"):
                logger.error("知识图谱实体为空")
                return False
            
            if not kg_data.get("relations"):
                logger.error("知识图谱关系为空")
                return False
            
            # 验证实体数量
            total_entities = sum(len(entities) for entities in kg_data.get("entities", {}).values())
            if total_entities < 10:
                logger.error(f"实体数量过少: {total_entities}")
                return False
            
            # 验证关系数量
            total_relations = sum(len(relations) for relations in kg_data.get("relations", {}).values())
            if total_relations < 5:
                logger.error(f"关系数量过少: {total_relations}")
                return False
            
            logger.info("结果验证通过")
            return True
            
        except Exception as e:
            logger.error(f"结果验证失败: {str(e)}")
            return False
    
    def analyze_performance(self, results: Dict) -> Dict:
        """分析性能指标"""
        try:
            pipeline_stats = results.get("statistics", {})
            
            metrics = {
                "total_entities": 0,
                "total_relations": 0,
                "processing_time": pipeline_stats.get("total_time", 0),
                "data_quality_score": 0,
                "llm_enhancements": results.get("llm_enhancement_required", 0),
                "neo4j_exported": False
            }
            
            # 统计实体数量
            kg_data = results.get("final_knowledge_graph", {})
            for entity_type, entities in kg_data.get("entities", {}).items():
                metrics["total_entities"] += len(entities)
            
            # 统计关系数量
            for relation_type, relations in kg_data.get("relations", {}).items():
                metrics["total_relations"] += len(relations)
            
            # 数据质量评分
            validation_results = results.get("validation_results", {})
            quality_score = validation_results.get("overall_quality", {}).get("quality_score", 0)
            metrics["data_quality_score"] = quality_score
            
            # Neo4j导出状态
            neo4j_results = results.get("neo4j_results")
            if neo4j_results:
                metrics["neo4j_exported"] = neo4j_results.get("success", False)
            
            # 计算效率指标
            if metrics["processing_time"] > 0:
                metrics["entities_per_second"] = metrics["total_entities"] / metrics["processing_time"]
                metrics["relations_per_second"] = metrics["total_relations"] / metrics["processing_time"]
            
            return metrics
            
        except Exception as e:
            logger.error(f"性能分析失败: {str(e)}")
            return {}
    
    def generate_test_report(self, results: Dict, validation_passed: bool, performance_metrics: Dict) -> Dict:
        """生成测试报告"""
        try:
            report = {
                "test_summary": {
                    "status": "PASSED" if validation_passed else "FAILED",
                    "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_duration": time.time() - self.start_time if self.start_time else 0
                },
                "data_statistics": {
                    "total_entities": performance_metrics.get("total_entities", 0),
                    "total_relations": performance_metrics.get("total_relations", 0),
                    "data_quality_score": performance_metrics.get("data_quality_score", 0),
                    "llm_enhancements": results.get("llm_enhancement_required", 0)
                },
                "performance_metrics": performance_metrics,
                "component_status": {
                    "data_validation": "OK" if results.get("validation_results") else "FAILED",
                    "entity_building": "OK" if results.get("final_knowledge_graph") else "FAILED",
                    "llm_enhancement": "OK" if results.get("enhancement_results") else "FAILED",
                    "neo4j_export": "OK" if results.get("neo4j_results") else "NOT_ENABLED"
                },
                "recommendations": self.generate_recommendations(results, performance_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {str(e)}")
            return {"error": str(e)}
    
    def generate_recommendations(self, results: Dict, performance_metrics: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        try:
            # 数据质量建议
            quality_score = performance_metrics.get("data_quality_score", 0)
            if quality_score < 80:
                recommendations.append(f"数据质量评分较低({quality_score:.1f})，建议检查数据源和清洗规则")
            
            # 性能建议
            entities_per_second = performance_metrics.get("entities_per_second", 0)
            if entities_per_second < 100:
                recommendations.append(f"处理效率较低({entities_per_second:.1f}实体/秒)，建议优化算法或增加并行处理")
            
            # LLM增强建议
            llm_enhancements = performance_metrics.get("llm_enhancements", 0)
            if llm_enhancements > 0:
                recommendations.append(f"有{llm_enhancements}个实体需要LLM增强，建议配置LLM服务")
            
            # Neo4j建议
            neo4j_exported = performance_metrics.get("neo4j_exported", False)
            if not neo4j_exported and NEO4J_AVAILABLE:
                recommendations.append("建议启用Neo4j导出以获得更好的图分析能力")
            
            # 实体数量建议
            total_entities = performance_metrics.get("total_entities", 0)
            if total_entities < 50:
                recommendations.append("实体数量较少，建议增加更多数据源")
            
            if not recommendations:
                recommendations.append("系统运行良好，无需特殊优化")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成建议失败: {str(e)}")
            return ["无法生成改进建议"]


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("金融知识图谱完整测试开始")
    logger.info("=" * 60)
    
    # 检查Neo4j可用性
    enable_neo4j = NEO4J_AVAILABLE
    if enable_neo4j:
        logger.info("Neo4j集成可用，将在测试中启用")
    else:
        logger.info("Neo4j集成不可用，将跳过相关测试")
    
    # 运行测试
    tester = CompletePipelineTester()
    test_results = tester.run_complete_pipeline_test(enable_neo4j=enable_neo4j)
    
    # 保存测试结果
    output_file = "output/complete_test_results.json"
    Path("output").mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"测试结果已保存到: {output_file}")
    
    # 输出测试总结
    if test_results.get("success"):
        logger.info("✅ 测试通过！")
        logger.info(f"实体数量: {test_results['performance_metrics'].get('total_entities', 0)}")
        logger.info(f"关系数量: {test_results['performance_metrics'].get('total_relations', 0)}")
        logger.info(f"数据质量评分: {test_results['performance_metrics'].get('data_quality_score', 0):.1f}")
        logger.info(f"总耗时: {test_results['total_time']:.2f}秒")
        
        # 输出建议
        recommendations = test_results.get("test_report", {}).get("recommendations", [])
        if recommendations:
            logger.info("改进建议:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
    else:
        logger.error("❌ 测试失败！")
        logger.error(f"错误: {test_results.get('error', '未知错误')}")
    
    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()