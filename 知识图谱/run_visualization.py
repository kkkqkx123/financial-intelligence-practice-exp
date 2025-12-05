#!/usr/bin/env python3
"""
知识图谱可视化查询运行脚本
执行各种图数据探索和可视化查询
"""

import sys
import os
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kg_visualization import KnowledgeGraphExplorer
from config import NEO4J_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数：执行可视化查询"""
    print("=" * 60)
    print("企业知识图谱可视化查询系统")
    print("=" * 60)
    
    logger.info("开始执行可视化查询...")
    
    try:
        # 创建图数据探索器
        explorer = KnowledgeGraphExplorer(
            uri=NEO4J_CONFIG['uri'],
            auth=NEO4J_CONFIG['auth']
        )
        
        # 运行演示查询
        explorer.run_demo_queries()
        
        print("\n可视化查询执行完成！")
        return True
        
    except Exception as e:
        logger.error(f"可视化查询执行失败: {str(e)}")
        print(f"\n错误: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)