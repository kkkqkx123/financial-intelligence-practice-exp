#!/usr/bin/env python3
"""
测试HTTP LLM API调用功能
这个脚本演示如何使用OpenAICompatibleClient进行真实的HTTP调用
"""

import os
import sys
from src.processors.llm_client import OpenAICompatibleClient, get_llm_client
from src.processors.llm_processor import OpenAICompatibleProcessor

def test_http_llm_call():
    """测试HTTP LLM调用"""
    print("=== 测试HTTP LLM API调用 ===")
    
    # 检查环境变量配置
    api_key = os.getenv('LLM_API_KEY')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    
    print(f"API密钥配置: {'已配置' if api_key else '未配置'}")
    print(f"基础URL: {base_url}")
    print(f"模型: {model}")
    
    if not api_key:
        print("⚠️  警告: 未配置LLM_API_KEY环境变量")
        print("请设置环境变量: set LLM_API_KEY=your_api_key")
        print("将使用模拟模式进行测试")
        return test_mock_mode()
    
    try:
        # 创建OpenAI兼容客户端
        print("\n1. 创建OpenAICompatibleClient...")
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,
            timeout=30
        )
        print("✅ 客户端创建成功")
        
        # 测试generate_response方法
        print("\n2. 测试generate_response方法...")
        test_prompt = "请用一句话介绍人工智能在金融领域的应用。"
        system_prompt = "你是一个专业的金融分析师，请用中文回答。"
        
        print(f"用户提示词: {test_prompt}")
        print(f"系统提示词: {system_prompt}")
        
        response = client.generate_response(
            prompt=test_prompt,
            system_prompt=system_prompt
        )
        
        print(f"✅ LLM响应: {response}")
        
        # 测试统计信息
        print("\n3. 获取客户端统计信息...")
        stats = client.get_stats()
        print(f"调用统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ HTTP调用失败: {e}")
        return False

def test_mock_mode():
    """测试模拟模式"""
    print("\n=== 使用模拟模式测试 ===")
    
    try:
        # 使用get_llm_client获取客户端（会返回模拟客户端）
        print("1. 获取模拟客户端...")
        client = get_llm_client()
        print(f"客户端类型: {type(client).__name__}")
        
        # 测试generate_response方法
        print("\n2. 测试generate_response方法...")
        test_prompt = "请用一句话介绍人工智能在金融领域的应用。"
        system_prompt = "你是一个专业的金融分析师，请用中文回答。"
        
        response = client.generate_response(
            prompt=test_prompt,
            system_prompt=system_prompt
        )
        
        print(f"✅ 模拟LLM响应: {response}")
        
        # 测试统计信息
        print("\n3. 获取客户端统计信息...")
        stats = client.get_stats()
        print(f"调用统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟模式测试失败: {e}")
        return False

def test_openai_compatible_processor():
    """测试OpenAICompatibleProcessor"""
    print("\n=== 测试OpenAICompatibleProcessor ===")
    
    try:
        # 创建处理器
        print("1. 创建OpenAICompatibleProcessor...")
        processor = OpenAICompatibleProcessor()
        print("✅ 处理器创建成功")
        
        # 测试实体描述增强
        print("\n2. 测试实体描述增强...")
        entity_name = "腾讯控股"
        context = {"industry": "互联网", "location": "深圳"}
        
        description = processor.enhance_entity_description(entity_name, context)
        print(f"实体: {entity_name}")
        print(f"增强描述: {description}")
        
        # 测试统计信息
        print("\n3. 获取处理器统计信息...")
        stats = processor.get_stats()
        print(f"处理器统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理器测试失败: {e}")
        return False

if __name__ == "__main__":
    print("HTTP LLM API调用测试脚本")
    print("=" * 50)
    
    # 测试基本HTTP调用
    success = test_http_llm_call()
    
    if success:
        print("\n✅ HTTP调用测试完成")
    else:
        print("\n❌ HTTP调用测试失败，尝试模拟模式...")
        test_mock_mode()
    
    # 测试处理器
    test_openai_compatible_processor()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n使用说明：")
    print("1. 配置真实API：set LLM_API_KEY=your_openai_api_key")
    print("2. 可选配置：set LLM_BASE_URL=https://api.openai.com/v1")
    print("3. 可选配置：set LLM_MODEL=gpt-3.5-turbo")
    print("4. 未配置API密钥时将自动使用模拟模式")