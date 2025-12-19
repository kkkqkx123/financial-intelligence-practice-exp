#!/usr/bin/env python3
"""
测试投资结构数据处理功能
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from processors import DataParser, KGBuilder, DataValidator

def test_investment_structure_processing():
    """测试投资结构数据处理功能"""
    print("开始测试投资结构数据处理功能...")
    
    # 初始化处理器
    parser = DataParser()
    builder = KGBuilder()
    validator = DataValidator()
    
    # 1. 测试投资结构数据解析
    print("\n1. 测试投资结构数据解析...")
    
    # 模拟投资结构数据
    sample_structure_data = [
        {
            "机构名称": "红杉资本",
            "行业": "人工智能,企业服务,医疗健康",
            "轮次": "A轮,B轮,C轮",
            "规模": "1000万-1亿人民币"
        },
        {
            "机构名称": "IDG资本",
            "行业": "金融科技,消费升级",
            "轮次": "种子轮,天使轮",
            "规模": "500万-5000万人民币"
        }
    ]
    
    # 解析投资结构数据
    parsed_structures = parser.parse_investment_structure(sample_structure_data)
    print(f"解析投资结构数据: {len(parsed_structures)} 条记录")
    
    for i, structure in enumerate(parsed_structures):
        print(f"  结构 {i+1}: {structure['name']} - 行业: {structure.get('industries', [])} - 轮次: {structure.get('rounds', [])}")
    
    # 2. 测试投资结构数据验证
    print("\n2. 测试投资结构数据验证...")
    
    validation_result = validator.validate_investment_structure_data(sample_structure_data)
    print(f"验证结果: {validation_result['valid_records']}/{validation_result['total_records']} 有效")
    print(f"数据质量分数: {validation_result['data_quality_score']:.2f}")
    
    # 3. 测试投资结构关系构建
    print("\n3. 测试投资结构关系构建...")
    
    # 创建一些投资方实体
    investors = [
        {"id": "inv_001", "name": "红杉资本", "type": "investor"},
        {"id": "inv_002", "name": "IDG资本", "type": "investor"}
    ]
    
    # 设置投资方实体
    builder.investors = {inv["id"]: inv for inv in investors}
    
    # 构建投资结构关系
    builder.build_investment_structure_relationships(parsed_structures)
    
    # 获取构建的关系
    relationships = builder.knowledge_graph.get('structure_relationships', [])
    print(f"构建投资结构关系: {len(relationships)} 条")
    
    for i, rel in enumerate(relationships):
        print(f"  关系 {i+1}: {rel['source']} -{rel['type']}-> {rel['target']} ({rel.get('properties', {})})")
    
    # 4. 获取构建统计信息
    print("\n4. 构建统计信息...")
    
    stats = builder.get_build_statistics()
    print(f"公司实体: {stats.get('total_companies', 0)}")
    print(f"投资方实体: {stats.get('total_investors', 0)}")
    print(f"关系总数: {stats.get('successful_links', 0)}")
    print(f"LLM增强任务: {stats.get('llm_enhancements', 0)}")
    print(f"投资结构关系: {len(relationships)}")
    
    print("\n投资结构数据处理功能测试完成!")
    return True

if __name__ == "__main__":
    try:
        success = test_investment_structure_processing()
        if success:
            print("\n✅ 所有测试通过!")
        else:
            print("\n❌ 测试失败!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)