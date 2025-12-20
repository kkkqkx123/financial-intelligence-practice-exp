#!/usr/bin/env python3
"""
Neo4j查询批处理脚本
从文件中读取查询列表并执行，将结果保存到指定目录
"""

import json
import sys
import argparse
from pathlib import Path

from simple_neo4j_query import execute_query_and_save


def load_queries_from_file(query_file):
    """从文件加载查询列表"""
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式：
        # 1. 简单格式：{"query1": "MATCH ...", "query2": "MATCH ..."}
        # 2. 详细格式：[{"name": "query1", "query": "MATCH ..."}, ...]
        
        if isinstance(data, dict):
            # 简单格式
            queries = []
            for name, query in data.items():
                queries.append({"name": name, "query": query})
            return queries
        elif isinstance(data, list):
            # 详细格式
            return data
        else:
            raise ValueError("不支持的查询文件格式")
            
    except Exception as e:
        print(f"加载查询文件失败: {e}")
        return []


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Neo4j查询批处理工具')
    parser.add_argument('query_file', help='包含查询的JSON文件路径')
    parser.add_argument('-o', '--output-dir', default='output', help='输出目录（默认: output）')
    
    args = parser.parse_args()
    
    # 检查查询文件是否存在
    query_path = Path(args.query_file)
    if not query_path.exists():
        print(f"错误: 查询文件不存在: {args.query_file}")
        sys.exit(1)
    
    # 加载查询
    queries = load_queries_from_file(args.query_file)
    if not queries:
        print("错误: 没有找到有效的查询")
        sys.exit(1)
    
    print(f"加载了 {len(queries)} 个查询")
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行查询
    results = []
    successful_queries = 0
    total_results = 0
    
    for query_info in queries:
        query_name = query_info.get("name", "unnamed_query")
        query = query_info.get("query", "")
        
        if not query:
            print(f"警告: 查询 {query_name} 为空，跳过")
            continue
        
        # 生成输出文件名
        output_file = f"{output_dir}/batch_{query_name}.json"
        
        # 执行查询
        print(f"执行查询: {query_name}")
        output_file_path = f"{output_dir}/batch_{query_name}.json"
        execute_query_and_save(query, output_file_path)
        
        # 检查输出文件是否存在并读取结果数量
        output_path = Path(output_file_path)
        if output_path.exists():
            successful_queries += 1
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                result_count = result_data.get("result_count", 0)
                total_results += result_count
                print(f"  ✓ 成功: {result_count} 条记录")
            except:
                print(f"  ✓ 成功: 无法读取结果数量")
        else:
            print(f"  ✗ 失败")
        
        results.append({
            "query_name": query_name,
            "output_file": output_file_path,
            "success": output_path.exists()
        })
    
    # 保存执行摘要
    summary_file = f"{output_dir}/batch_query_summary.json"
    summary_data = {
        "query_file": args.query_file,
        "total_queries": len(queries),
        "successful_queries": successful_queries,
        "total_results": total_results,
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n批处理完成:")
    print(f"  成功执行: {successful_queries}/{len(queries)} 个查询")
    print(f"  总记录数: {total_results}")
    print(f"  执行摘要: {summary_file}")


if __name__ == "__main__":
    main()