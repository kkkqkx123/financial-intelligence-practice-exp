#!/usr/bin/env python3
"""
简单的Neo4j查询执行器
执行单个自定义查询并将结果保存到文件
"""

import sys
import json
from datetime import datetime
from pathlib import Path

try:
    from py2neo import Graph
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("错误: py2neo库未安装，无法执行Neo4j查询")
    sys.exit(1)

from integrations.neo4j_exporter import Config


def execute_query_and_save(query, output_file=None):
    """
    执行Neo4j查询并将结果保存到文件
    
    Args:
        query: Neo4j查询语句
        output_file: 输出文件路径，如果为None则使用默认路径
    """
    # Neo4j连接配置
    config = Config(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="1234567kk"
    )
    
    try:
        # 连接到Neo4j
        graph = Graph(config.uri, auth=(config.username, config.password))
        print("成功连接到Neo4j数据库")
        
        # 生成默认输出文件名
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/neo4j_query_{timestamp}.json"
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行查询
        print(f"执行查询: {query[:100]}...")
        start_time = datetime.now()
        
        results = graph.run(query).data()
        
        # 计算查询耗时
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 准备结果数据
        result_data = {
            "query": query,
            "execution_time": execution_time,
            "timestamp": start_time.isoformat(),
            "result_count": len(results),
            "results": results
        }
        
        # 保存结果到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"查询完成，结果已保存到: {output_file}")
        print(f"返回 {len(results)} 条记录，耗时 {execution_time:.2f} 秒")
        
        return output_file
        
    except Exception as e:
        print(f"查询执行失败: {e}")
        return None


def main():
    """主函数 - 从命令行参数或标准输入读取查询"""
    # 如果有命令行参数，使用参数作为查询
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # 否则从标准输入读取查询
        print("请输入Neo4j查询语句（以空行结束）:")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        query = "\n".join(lines)
    
    if not query.strip():
        print("错误: 没有提供查询语句")
        print("用法:")
        print(f"  python {sys.argv[0]} \"MATCH (n) RETURN count(n)\"")
        print(f"  或者运行 {sys.argv[0]} 然后输入查询语句")
        sys.exit(1)
    
    # 执行查询并保存结果
    output_file = execute_query_and_save(query)
    
    if output_file:
        print(f"查询结果已保存到: {output_file}")
        sys.exit(0)
    else:
        print("查询失败")
        sys.exit(1)


if __name__ == "__main__":
    main()