#!/usr/bin/env python3
"""
配置检查脚本 - 分析配置加载问题
"""

import os
import sys
from pathlib import Path

# 添加src到路径
sys.path.append('src')

def load_env_file():
    """手动加载.env文件"""
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ 未找到.env文件")
        return False
    
    print("✅ 找到.env文件，正在加载...")
    loaded_count = 0
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' not in line:
                print(f"⚠️  第{line_num}行格式错误: {line}")
                continue
            
            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key and value:
                    os.environ[key] = value
                    loaded_count += 1
                    
                    # 安全地显示配置（隐藏API密钥）
                    if 'API_KEY' in key:
                        print(f"  {key}: {value[:10]}...")
                    elif 'PASSWORD' in key:
                        print(f"  {key}: ***")
                    else:
                        print(f"  {key}: {value}")
                        
            except Exception as e:
                print(f"❌ 第{line_num}行加载失败: {e}")
    
    print(f"\n✅ 成功加载 {loaded_count} 个环境变量")
    return True

def check_environment_variables():
    """检查环境变量状态"""
    print("\n=== 环境变量状态 ===")
    
    # 基础配置
    api_key = os.getenv('LLM_API_KEY')
    base_url = os.getenv('LLM_BASE_URL')
    model = os.getenv('LLM_MODEL')
    timeout = os.getenv('LLM_TIMEOUT')
    
    print(f"LLM_API_KEY: {'✅ 已设置' if api_key else '❌ 未设置'}")
    print(f"LLM_BASE_URL: {'✅ 已设置' if base_url else '❌ 未设置'} ({base_url or '无'})")
    print(f"LLM_MODEL: {'✅ 已设置' if model else '❌ 未设置'} ({model or '无'})")
    print(f"LLM_TIMEOUT: {'✅ 已设置' if timeout else '❌ 未设置'} ({timeout or '无'})")
    
    # 多提供商配置
    provider_configs = []
    i = 1
    while True:
        provider_key = os.getenv(f'LLM_PROVIDER_{i}_API_KEY')
        if not provider_key:
            break
        
        provider_url = os.getenv(f'LLM_PROVIDER_{i}_BASE_URL')
        provider_model = os.getenv(f'LLM_PROVIDER_{i}_MODEL')
        
        provider_configs.append({
            'key': provider_key,
            'url': provider_url,
            'model': provider_model,
            'index': i
        })
        i += 1
    
    if provider_configs:
        print(f"\n✅ 发现 {len(provider_configs)} 个多提供商配置:")
        for provider in provider_configs:
            print(f"  提供商{provider['index']}: {provider['model'] or '未知模型'} @ {provider['url'] or '未知URL'}")
    else:
        print("\nℹ️  未发现多提供商配置")
    
    return api_key, base_url, model, provider_configs

def test_llm_client():
    """测试LLM客户端创建"""
    print("\n=== LLM客户端测试 ===")
    
    try:
        from processors.llm_client import get_llm_client
        client = get_llm_client()
        
        client_type = type(client).__name__
        print(f"✅ 客户端创建成功: {client_type}")
        
        if hasattr(client, 'api_key'):
            print(f"API密钥: {'✅ 已设置' if client.api_key else '❌ 未设置'}")
        
        if hasattr(client, 'providers') and client.providers:
            print(f"提供商数量: {len(client.providers)}")
            for i, provider in enumerate(client.providers):
                model = provider.get('model', '未知模型')
                url = provider.get('base_url', '未知URL')
                print(f"  提供商{i+1}: {model} @ {url}")
        
        if hasattr(client, 'model'):
            print(f"模型: {client.model}")
        
        if hasattr(client, 'base_url'):
            print(f"基础URL: {client.base_url}")
            
        return client
        
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        return None

def test_llm_processor():
    """测试LLM处理器"""
    print("\n=== LLM处理器测试 ===")
    
    try:
        from processors.llm_processor import get_llm_processor
        processor = get_llm_processor()
        
        processor_type = type(processor).__name__
        print(f"✅ 处理器创建成功: {processor_type}")
        
        # 测试基本功能
        print("\n测试实体描述增强...")
        result = processor.enhance_entity_description("测试公司", {"industry": "科技"})
        print(f"结果: {result[:50]}...")
        
        return processor
        
    except Exception as e:
        print(f"❌ 处理器创建失败: {e}")
        return None

def main():
    """主函数"""
    print("🔍 配置加载问题分析")
    print("=" * 50)
    
    # 1. 加载.env文件
    env_loaded = load_env_file()
    
    if not env_loaded:
        print("\n❌ 请先创建.env文件并配置API密钥")
        return
    
    # 2. 检查环境变量
    api_key, base_url, model, provider_configs = check_environment_variables()
    
    # 3. 测试LLM客户端
    client = test_llm_client()
    
    # 4. 测试LLM处理器
    processor = test_llm_processor()
    
    # 5. 总结
    print("\n" + "=" * 50)
    print("📊 配置分析总结:")
    
    if api_key and base_url and model:
        print("✅ 基础配置完整")
    elif provider_configs:
        print("✅ 多提供商配置完整")
    else:
        print("❌ 配置不完整")
    
    if client:
        print("✅ LLM客户端工作正常")
    else:
        print("❌ LLM客户端创建失败")
    
    if processor:
        print("✅ LLM处理器工作正常")
    else:
        print("❌ LLM处理器创建失败")

if __name__ == "__main__":
    main()