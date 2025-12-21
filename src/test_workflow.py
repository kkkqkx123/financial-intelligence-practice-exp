#!/usr/bin/env python3
"""
测试工作流脚本 - 验证Pipeline初始化、数据加载和解析功能
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Pipeline

async def test_workflow():
    """测试工作流"""
    print("=" * 60)
    print("开始测试工作流")
    print("=" * 60)
    
    try:
        # 初始化Pipeline
        print("\n1. 初始化Pipeline...")
        pipeline = Pipeline()
        print("   ✓ Pipeline初始化成功")
        
        # 加载数据
        print("\n2. 加载数据...")
        raw_data = pipeline.load_data_files()
        print(f"   ✓ 数据加载成功: {len(raw_data)} 种数据类型")
        for data_type, records in raw_data.items():
            print(f"     - {data_type}: {len(records)} 条记录")
        
        # 数据解析阶段
        print("\n3. 运行数据解析阶段...")
        parsed_data = pipeline.run_data_parsing_stage(raw_data)
        print(f"   ✓ 数据解析完成: {len(parsed_data)} 种解析数据")
        for data_type, records in parsed_data.items():
            print(f"     - {data_type}: {len(records)} 条记录")
        
        # 实体构建阶段（仅运行到数据验证和实体构建部分）
        print("\n4. 运行实体构建阶段（基础部分）...")
        
        # 由于LLM处理可能耗时较长，我们只测试到实体构建的基础部分
        # 这里我们直接调用builder的方法来构建实体，而不进行LLM增强
        
        # 数据验证
        print("   - 数据验证...")
        companies_raw = parsed_data.get('companies', [])
        investment_events_raw = parsed_data.get('investment_events', [])
        investors_raw = parsed_data.get('investors', [])
        investment_structures_raw = parsed_data.get('investment_structures', [])
        
        has_companies = len(companies_raw) > 0
        
        if has_companies:
            company_validation = pipeline.validator.validate_company_data(companies_raw)
            print(f"     公司数据验证：{company_validation['valid_records']}/{company_validation['total_records']} 有效")
        else:
            print("     跳过公司数据验证（无公司数据）")
        
        event_validation = pipeline.validator.validate_investment_event_data(investment_events_raw)
        print(f"     投资事件验证：{event_validation['valid_records']}/{event_validation['total_records']} 有效")
        
        investor_validation = pipeline.validator.validate_investor_data(investors_raw)
        print(f"     投资方数据验证：{investor_validation['valid_records']}/{investor_validation['total_records']} 有效")
        
        structure_validation = pipeline.validator.validate_investment_structure_data(investment_structures_raw)
        print(f"     投资结构数据验证：{structure_validation['valid_records']}/{structure_validation['total_records']} 有效")
        
        # 构建实体
        print("   - 构建实体...")
        companies = pipeline.builder.build_company_entities(companies_raw) if has_companies else {}
        print(f"     公司实体构建完成: {len(companies)} 个")
        
        investors = pipeline.builder.build_investor_entities(investors_raw)
        print(f"     投资方实体构建完成: {len(investors)} 个")
        
        # 解析投资事件数据
        print("   - 解析投资事件数据...")
        investment_events = pipeline.parser.parse_investment_events(investment_events_raw)
        
        # 构建投资关系
        print("   - 构建投资关系...")
        pipeline.builder.build_investment_relationships(investment_events)
        relationships = pipeline.builder.knowledge_graph['relationships']
        print(f"     投资关系构建完成: {len(relationships)} 个")
        
        # 构建投资结构关系
        if investment_structures_raw:
            print("   - 构建投资结构关系...")
            pipeline.builder.build_investment_structure_relationships(investment_structures_raw)
            structure_relationships = pipeline.builder.knowledge_graph.get('structure_relationships', [])
            relationships.extend(structure_relationships)
            print(f"     投资结构关系构建完成: {len(structure_relationships)} 个")
        
        print("\n" + "=" * 60)
        print("工作流测试完成！")
        print(f"✓ Pipeline初始化成功")
        print(f"✓ 数据加载成功 (共 {sum(len(records) for records in raw_data.values())} 条记录)")
        print(f"✓ 数据解析成功 (共 {sum(len(records) for records in parsed_data.values())} 条记录)")
        print(f"✓ 实体构建成功 (公司: {len(companies)}, 投资方: {len(investors)}, 关系: {len(relationships)})")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(test_workflow())
    
    if success:
        print("\n🎉 所有测试通过！工作流功能正常。")
        sys.exit(0)
    else:
        print("\n💥 测试失败，请检查错误信息。")
        sys.exit(1)