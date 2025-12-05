#!/usr/bin/env python3
"""
知识图谱构建运行脚本
一键执行知识图谱构建流程
"""

import sys
import os
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kg_construction import EnterpriseKnowledgeGraph
from config import NEO4J_CONFIG, KG_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数：执行知识图谱构建"""
    print("=" * 60)
    print("企业知识图谱构建系统")
    print("=" * 60)
    
    start_time = datetime.now()
    logger.info("开始构建知识图谱...")
    
    try:
        # 创建知识图谱构建器
        kg_builder = EnterpriseKnowledgeGraph(
            uri=NEO4J_CONFIG['uri'],
            auth=NEO4J_CONFIG['auth']
        )
        
        # 构建知识图谱
        kg_builder.build_knowledge_graph()
        
        # 获取统计信息
        stats = kg_builder.get_statistics()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("知识图谱构建完成！")
        print(f"总用时: {duration}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"知识图谱构建失败: {str(e)}")
        print(f"\n错误: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)