"""
测试配置管理模块
验证配置加载、验证和集成功能
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.processors.config_manager import (
    ConfigManager, LLMProviderConfig, load_configuration, get_config_manager
)
from src.processors.config_validator import ConfigValidator, validate_configuration


class TestConfigManager(unittest.TestCase):
    """测试配置管理器"""
    
    def setUp(self):
        """测试前置设置"""
        # 保存原始环境变量
        self.original_env = dict(os.environ)
        # 清除相关的环境变量
        for key in list(os.environ.keys()):
            if key.startswith('LLM_'):
                del os.environ[key]
    
    def tearDown(self):
        """测试后置清理"""
        # 恢复原始环境变量
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_llm_provider_config_creation(self):
        """测试LLM提供商配置创建"""
        config = LLMProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            model="test-model",
            max_tokens=1000,
            temperature=0.7,
            timeout=30,
            weight=1
        )
        
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.example.com")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.weight, 1)
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        manager = ConfigManager()
        self.assertFalse(manager.is_config_loaded())
        self.assertEqual(len(manager.get_llm_configs()), 0)
    
    def test_load_from_env_single_provider(self):
        """测试从环境变量加载单个提供商配置"""
        # 设置环境变量
        os.environ['LLM_API_KEY'] = 'test-api-key'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        os.environ['LLM_MAX_TOKENS'] = '2000'
        os.environ['LLM_TEMPERATURE'] = '0.5'
        os.environ['LLM_TIMEOUT'] = '60'
        
        manager = ConfigManager()
        manager._load_from_env()
        
        configs = manager.get_llm_configs()
        self.assertEqual(len(configs), 1)
        
        config = configs[0]
        self.assertEqual(config.api_key, 'test-api-key')
        self.assertEqual(config.base_url, 'https://api.example.com')
        self.assertEqual(config.model, 'test-model')
        self.assertEqual(config.max_tokens, 2000)
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.timeout, 60)
    
    def test_load_from_env_multi_provider(self):
        """测试从环境变量加载多个提供商配置"""
        # 设置多个提供商的环境变量
        os.environ['LLM_PROVIDER_1_API_KEY'] = 'key1'
        os.environ['LLM_PROVIDER_1_BASE_URL'] = 'https://api1.example.com'
        os.environ['LLM_PROVIDER_1_MODEL'] = 'model1'
        os.environ['LLM_PROVIDER_1_WEIGHT'] = '2'
        
        os.environ['LLM_PROVIDER_2_API_KEY'] = 'key2'
        os.environ['LLM_PROVIDER_2_BASE_URL'] = 'https://api2.example.com'
        os.environ['LLM_PROVIDER_2_MODEL'] = 'model2'
        os.environ['LLM_PROVIDER_2_WEIGHT'] = '1'
        
        manager = ConfigManager()
        manager._load_from_env()
        
        configs = manager.get_llm_configs()
        self.assertEqual(len(configs), 2)
        
        # 验证第一个提供商
        self.assertEqual(configs[0].api_key, 'key1')
        self.assertEqual(configs[0].base_url, 'https://api1.example.com')
        self.assertEqual(configs[0].model, 'model1')
        self.assertEqual(configs[0].weight, 2)
        
        # 验证第二个提供商
        self.assertEqual(configs[1].api_key, 'key2')
        self.assertEqual(configs[1].base_url, 'https://api2.example.com')
        self.assertEqual(configs[1].model, 'model2')
        self.assertEqual(configs[1].weight, 1)
    
    def test_load_from_env_fallback(self):
        """测试回退到旧配置格式"""
        os.environ['LLM_API_KEY'] = 'fallback-key'
        os.environ['LLM_BASE_URL'] = 'https://fallback.example.com'
        os.environ['LLM_MODEL'] = 'fallback-model'
        
        manager = ConfigManager()
        manager._load_from_env()
        
        configs = manager.get_llm_configs()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].api_key, 'fallback-key')
    
    def test_load_from_env_file(self):
        """测试从.env文件加载配置"""
        # 创建临时.env文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("LLM_API_KEY=file-api-key\n")
            f.write("LLM_BASE_URL=https://file.example.com\n")
            f.write("LLM_MODEL=file-model\n")
            temp_env_path = f.name
        
        try:
            manager = ConfigManager()
            manager._load_from_env_file(temp_env_path)
            
            configs = manager.get_llm_configs()
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].api_key, 'file-api-key')
            self.assertEqual(configs[0].base_url, 'https://file.example.com')
            self.assertEqual(configs[0].model, 'file-model')
            
        finally:
            os.unlink(temp_env_path)
    
    def test_get_primary_provider(self):
        """测试获取主要提供商"""
        # 设置多个提供商，权重不同
        os.environ['LLM_PROVIDER_1_API_KEY'] = 'key1'
        os.environ['LLM_PROVIDER_1_BASE_URL'] = 'https://api1.example.com'
        os.environ['LLM_PROVIDER_1_MODEL'] = 'model1'
        os.environ['LLM_PROVIDER_1_WEIGHT'] = '1'
        
        os.environ['LLM_PROVIDER_2_API_KEY'] = 'key2'
        os.environ['LLM_PROVIDER_2_BASE_URL'] = 'https://api2.example.com'
        os.environ['LLM_PROVIDER_2_MODEL'] = 'model2'
        os.environ['LLM_PROVIDER_2_WEIGHT'] = '3'  # 权重更高
        
        manager = ConfigManager()
        manager._load_from_env()
        
        primary = manager.get_primary_provider()
        self.assertIsNotNone(primary)
        self.assertEqual(primary.weight, 3)  # 应该返回权重最高的
    
    def test_get_all_providers(self):
        """测试获取所有提供商"""
        os.environ['LLM_API_KEY'] = 'test-key'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        manager = ConfigManager()
        manager._load_from_env()
        
        providers = manager.get_all_providers()
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]['api_key'], 'test-key')
    
    def test_get_stats(self):
        """测试获取统计信息"""
        os.environ['LLM_API_KEY'] = 'test-key'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        manager = ConfigManager()
        manager._load_from_env()
        
        stats = manager.get_stats()
        self.assertEqual(stats['total_providers'], 1)
        self.assertEqual(stats['loaded_from_env'], True)
        self.assertEqual(stats['loaded_from_file'], False)


class TestConfigValidator(unittest.TestCase):
    """测试配置验证器"""
    
    def setUp(self):
        """测试前置设置"""
        self.original_env = dict(os.environ)
        # 清除相关的环境变量
        for key in list(os.environ.keys()):
            if key.startswith('LLM_'):
                del os.environ[key]
    
    def tearDown(self):
        """测试后置清理"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_validate_valid_single_provider(self):
        """测试验证有效的单个提供商配置"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'https://api.openai.com/v1'
        os.environ['LLM_MODEL'] = 'gpt-3.5-turbo'
        
        validator = ConfigValidator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_invalid_api_key(self):
        """测试验证无效的API密钥"""
        os.environ['LLM_API_KEY'] = 'short'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = ConfigValidator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('API密钥过短' in error for error in result['errors']))
    
    def test_validate_missing_required_fields(self):
        """测试验证缺少必需字段"""
        # 只设置API密钥，缺少其他必需字段
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        
        validator = ConfigValidator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('配置不完整' in warning for warning in result['warnings']))
    
    def test_validate_invalid_url(self):
        """测试验证无效的URL"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'not-a-valid-url'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = ConfigValidator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(any('基础URL' in error for error in result['errors']))
    
    def test_validate_multi_provider_config(self):
        """测试验证多提供商配置"""
        os.environ['LLM_PROVIDER_1_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_PROVIDER_1_BASE_URL'] = 'https://api1.example.com'
        os.environ['LLM_PROVIDER_1_MODEL'] = 'model1'
        
        os.environ['LLM_PROVIDER_2_API_KEY'] = 'sk-test223456789012345678901234567890123456789012'
        os.environ['LLM_PROVIDER_2_BASE_URL'] = 'https://api2.example.com'
        os.environ['LLM_PROVIDER_2_MODEL'] = 'model2'
        
        validator = ConfigValidator()
        result = validator.validate_all_configs()
        
        # 多提供商配置应该有效
        self.assertEqual(result['status'], 'valid')
    
    def test_validate_configuration_function(self):
        """测试验证配置便捷函数"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        result = validate_configuration()
        
        self.assertIn('status', result)
        self.assertIn('errors', result)
        self.assertIn('warnings', result)
        self.assertIn('suggestions', result)
        self.assertIn('summary', result)
    
    def test_get_diagnostics_info(self):
        """测试获取诊断信息"""
        validator = ConfigValidator()
        diagnostics = validator.get_diagnostics_info()
        
        self.assertIn('config_manager_available', diagnostics)
        self.assertIn('env_file_exists', diagnostics)
        self.assertIn('common_env_vars', diagnostics)
        self.assertIn('system_info', diagnostics)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前置设置"""
        self.original_env = dict(os.environ)
        # 清除相关的环境变量
        for key in list(os.environ.keys()):
            if key.startswith('LLM_'):
                del os.environ[key]
    
    def tearDown(self):
        """测试后置清理"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_load_configuration_integration(self):
        """测试配置加载集成"""
        # 设置环境变量
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        # 加载配置
        config_loaded = load_configuration()
        self.assertTrue(config_loaded)
        
        # 获取配置管理器
        manager = get_config_manager()
        self.assertIsNotNone(manager)
        self.assertTrue(manager.is_config_loaded())
        
        # 验证配置
        configs = manager.get_llm_configs()
        self.assertEqual(len(configs), 1)
        
        config = configs[0]
        self.assertEqual(config.api_key, 'sk-test123456789012345678901234567890123456789012')
        self.assertEqual(config.base_url, 'https://api.example.com')
        self.assertEqual(config.model, 'test-model')
    
    def test_end_to_end_validation(self):
        """测试端到端验证流程"""
        # 创建临时.env文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("LLM_API_KEY=sk-test123456789012345678901234567890123456789012\n")
            f.write("LLM_BASE_URL=https://api.openai.com/v1\n")
            f.write("LLM_MODEL=gpt-3.5-turbo\n")
            temp_env_path = f.name
        
        try:
            # 从文件加载配置
            manager = ConfigManager()
            manager.load_from_file(temp_env_path)
            
            # 验证配置
            validator = ConfigValidator()
            result = validator.validate_all_configs()
            
            self.assertEqual(result['status'], 'valid')
            self.assertEqual(len(result['errors']), 0)
            
        finally:
            os.unlink(temp_env_path)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)