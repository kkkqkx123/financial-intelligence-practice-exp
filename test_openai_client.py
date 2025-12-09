#!/usr/bin/env python3
"""
测试新的OpenAI客户端和轮询池功能
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.processors import get_llm_client, get_llm_processor, get_batch_llm_processor, PollingPool, OpenAICompatibleClient
from src.processors.llm_client import OpenAICompatibleClient, PollingPool

def test_single_client():
    """测试单个OpenAI客户端"""
    print("=== 测试单个OpenAI客户端 ===")
    
    try:
        # 获取LLM客户端
        client = get_llm_client()
        print(f"客户端类型: {type(client).__name__}")
        
        # 测试基本功能
        print("\n1. 测试基本文本生成...")
        test_prompt = "请用一句话介绍人工智能在金融领域的应用。"
        response = client.generate_response(test_prompt)
        print(f"响应: {response}")
        
        # 测试统计信息
        stats = client.get_stats()
        print(f"\n客户端统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"单个客户端测试失败: {e}")
        return False
    
    return True

def test_polling_pool():
    """测试轮询池"""
    print("\n=== 测试轮询池 ===")
    
    try:
        # 创建轮询池
        pool = PollingPool()
        print(f"轮询池中的provider数量: {len(pool.providers)}")
        
        # 测试轮询功能
        print("\n2. 测试轮询池文本生成...")
        test_prompt = "请用一句话介绍区块链技术。"
        
        for i in range(3):
            print(f"\n第{i+1}次调用:")
            response = pool.generate_response(test_prompt)
            print(f"响应: {response}")
            
            # 显示当前使用的provider
            if pool.providers:
                current_provider = pool.providers[pool.current_index - 1] if pool.current_index > 0 else pool.providers[-1]
                print(f"使用的provider: {current_provider.get('name', 'Unknown')}")
            else:
                print(f"使用的provider: 模拟客户端 (无配置provider)")
            
            time.sleep(0.5)  # 短暂延迟
        
        # 测试统计信息
        stats = pool.get_stats()
        print(f"\n轮询池统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"轮询池测试失败: {e}")
        return False
    
    return True

def test_llm_processor():
    """测试LLM处理器"""
    print("\n=== 测试LLM处理器 ===")
    
    try:
        # 获取LLM处理器
        processor = get_llm_processor()
        print(f"处理器类型: {type(processor).__name__}")
        
        # 测试实体描述增强
        print("\n3. 测试实体描述增强...")
        entity_name = "腾讯科技"
        context = {
            "industry": "互联网",
            "founded_year": 1998,
            "location": "深圳"
        }
        
        enhanced_description = processor.enhance_entity_description(entity_name, context)
        print(f"增强后的描述: {enhanced_description}")
        
        # 测试冲突解决
        print("\n4. 测试实体冲突解决...")
        conflict_group = [
            {"name": "阿里巴巴", "description": "电商巨头"},
            {"name": "阿里集团", "description": "电商巨头"}
        ]
        
        resolution = processor.resolve_entity_conflicts(conflict_group)
        print(f"冲突解决结果: {json.dumps(resolution, ensure_ascii=False, indent=2)}")
        
        # 测试统计信息
        stats = processor.get_stats()
        print(f"\n处理器统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"LLM处理器测试失败: {e}")
        return False
    
    return True

def test_batch_processor():
    """测试批量处理器"""
    print("\n=== 测试批量处理器 ===")
    
    try:
        # 获取批量LLM处理器
        batch_processor = get_batch_llm_processor()
        print(f"批量处理器类型: {type(batch_processor).__name__}")
        
        # 测试批量实体描述增强
        print("\n5. 测试批量实体描述增强...")
        entities = [
            {"name": "腾讯科技", "context": {"industry": "互联网"}},
            {"name": "阿里巴巴", "context": {"industry": "电商"}},
            {"name": "百度公司", "context": {"industry": "搜索"}}
        ]
        
        enhanced_descriptions = batch_processor.batch_enhance_entity_descriptions(entities)
        print(f"批量增强结果数量: {len(enhanced_descriptions)}")
        for i, desc in enumerate(enhanced_descriptions):
            print(f"实体 {i+1}: {desc}")
        
        # 测试统计信息
        stats = batch_processor.get_batch_stats()
        print(f"\n批量处理器统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"批量处理器测试失败: {e}")
        return False
    
    return True

def test_environment_config():
    """测试环境变量配置"""
    print("\n=== 测试环境变量配置 ===")
    
    # 检查关键环境变量
    env_vars = [
        'LLM_API_KEY',
        'LLM_BASE_URL', 
        'LLM_MODEL',
        'LLM_MAX_TOKENS',
        'LLM_POLLING_PROVIDERS'
    ]
    
    print("当前环境变量配置:")
    for var in env_vars:
        value = os.getenv(var, '未设置')
        if var == 'LLM_API_KEY' and value != '未设置':
            # 隐藏API密钥的部分内容
            masked_value = value[:8] + '*' * (len(value) - 12) + value[-4:] if len(value) > 12 else '*' * len(value)
            print(f"{var}: {masked_value}")
        else:
            print(f"{var}: {value}")

def main():
    """主测试函数"""
    print("开始测试新的OpenAI客户端和轮询池功能...")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 测试环境变量配置
    test_environment_config()
    
    # 运行各项测试
    tests = [
        ("单个客户端", test_single_client),
        ("轮询池", test_polling_pool),
        ("LLM处理器", test_llm_processor),
        ("批量处理器", test_batch_processor)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            if not success:
                print(f"❌ {test_name} 测试失败")
            else:
                print(f"✅ {test_name} 测试通过")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
        
        print("-" * 50)
    
    # 总结测试结果
    print("\n=== 测试结果总结 ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"总测试数: {total}")
    print(f"通过数: {passed}")
    print(f"失败数: {total - passed}")
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if passed == total:
        print("\n🎉 所有测试通过！新的OpenAI客户端和轮询池功能正常工作。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查配置和实现。")
        return 1

if __name__ == "__main__":
    sys.exit(main())