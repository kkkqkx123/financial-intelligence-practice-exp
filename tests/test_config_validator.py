"""
测试配置验证和错误处理功能
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.processors.config_validator import (
    ConfigValidator, ConfigValidationError, validate_configuration,
    get_configuration_diagnostics
)
from src.processors.config_manager import LLMProviderConfig


class TestConfigValidator(unittest.TestCase):
    """测试配置验证器"""
    
    def setUp(self):
        """测试前置设置"""
        self.original_env = dict(os.environ)
        # 清除相关的环境变量
        for key in list(os.environ.keys()):
            if key.startswith('LLM_'):
                del os.environ[key]
        
        # 使用patch来模拟配置管理器不可用
        self.config_manager_patcher = patch('src.processors.config_manager.get_config_manager')
        self.mock_get_config_manager = self.config_manager_patcher.start()
        self.mock_get_config_manager.side_effect = ImportError("模拟配置管理器不可用")
    
    def tearDown(self):
        """测试后置清理"""
        self.config_manager_patcher.stop()
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def _create_validator(self):
        """创建配置验证器实例（延迟创建以确保补丁生效）"""
        return ConfigValidator()
    
    def test_validate_valid_openai_config(self):
        """测试验证有效的OpenAI配置"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'https://api.openai.com/v1'
        os.environ['LLM_MODEL'] = 'gpt-3.5-turbo'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(len(result['errors']), 0)
        self.assertGreater(len(result['suggestions']), 0)
    
    def test_validate_api_key_too_short(self):
        """测试验证过短的API密钥"""
        os.environ['LLM_API_KEY'] = 'sk-test'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # API密钥长度不足现在生成警告而不是错误
        self.assertTrue(any('API密钥长度' in warning for warning in result['warnings']))
    
    def test_validate_invalid_api_key_empty(self):
        """测试验证空的API密钥"""
        os.environ['LLM_API_KEY'] = ''
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # 空API密钥现在生成警告而不是错误
        self.assertTrue(any('API密钥为空' in warning for warning in result['warnings']))
    
    def test_validate_invalid_url_format(self):
        """测试验证无效的URL格式"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'not-a-url'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # 无效URL格式现在生成警告而不是错误
        self.assertTrue(any('基础URL' in warning for warning in result['warnings']))
    
    def test_validate_url_missing_scheme(self):
        """测试验证缺少协议的URL"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # 缺少协议的URL现在生成警告而不是错误
        self.assertTrue(any('缺少协议' in warning for warning in result['warnings']))
    
    def test_validate_url_unsupported_scheme(self):
        """测试验证不支持的协议"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'ftp://api.example.com'
        os.environ['LLM_MODEL'] = 'test-model'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # 不支持的协议现在生成警告而不是错误
        self.assertTrue(any('不支持的协议' in warning for warning in result['warnings']))
    
    def test_validate_model_name_empty(self):
        """测试验证空的模型名称"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = ''
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # 空模型名称现在生成警告而不是错误
        self.assertTrue(any('模型名称为空' in warning for warning in result['warnings']))
    
    def test_validate_model_name_short(self):
        """测试验证过短的模型名称"""
        os.environ['LLM_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_BASE_URL'] = 'https://api.example.com'
        os.environ['LLM_MODEL'] = 'ab'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')  # 过短模型名称现在生成警告而不是错误
        self.assertTrue(any('模型名称长度不足' in warning for warning in result['warnings']))
    
    def test_validate_numeric_params_invalid(self):
        """测试验证无效的数值参数"""
        config = LLMProviderConfig(
            api_key='sk-test123456789012345678901234567890123456789012',
            base_url='https://api.example.com',
            model='test-model',
            max_tokens=-100,  # 无效值
            temperature=3.0,   # 超出范围
            timeout=-10,       # 无效值
            weight=0           # 无效值
        )
        
        validator = self._create_validator()
        validator._validate_numeric_params(config, 'test_config')
        
        # 应该生成多个错误
        errors = validator.validation_errors
        self.assertTrue(any('max_tokens 必须是正整数' in error for error in errors))
        self.assertTrue(any('temperature 必须在' in error for error in errors))
        self.assertTrue(any('weight 必须是正整数' in error for error in errors))
    
    def test_validate_timeout_invalid(self):
        """测试验证无效的超时设置"""
        validator = self._create_validator()
        validator._validate_timeout(-5, 'test_config')
        
        errors = validator.validation_errors
        self.assertTrue(any('timeout 必须是正整数' in error for error in errors))
    
    def test_validate_multi_provider_configs(self):
        """测试验证多提供商配置"""
        os.environ['LLM_PROVIDER_1_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_PROVIDER_1_BASE_URL'] = 'https://api1.example.com'
        os.environ['LLM_PROVIDER_1_MODEL'] = 'model1'
        
        os.environ['LLM_PROVIDER_2_API_KEY'] = 'sk-test223456789012345678901234567890123456789012'
        os.environ['LLM_PROVIDER_2_BASE_URL'] = 'https://api2.example.com'
        os.environ['LLM_PROVIDER_2_MODEL'] = 'model2'
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_multi_provider_incomplete(self):
        """测试验证不完整的多提供商配置"""
        os.environ['LLM_PROVIDER_1_API_KEY'] = 'sk-test123456789012345678901234567890123456789012'
        os.environ['LLM_PROVIDER_1_BASE_URL'] = 'https://api1.example.com'
        # 缺少 MODEL
        
        validator = self._create_validator()
        result = validator.validate_all_configs()
        
        # 应该跳过不完整的provider，状态为valid因为没有错误
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
        
        self.assertEqual(result['status'], 'valid')
    
    def test_get_diagnostics_info(self):
        """测试获取诊断信息"""
        # 设置一些环境变量
        os.environ['LLM_API_KEY'] = 'test-key'
        os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
        
        diagnostics = get_configuration_diagnostics()
        
        self.assertIn('config_manager_available', diagnostics)
        self.assertIn('env_file_exists', diagnostics)
        self.assertIn('common_env_vars', diagnostics)
        self.assertIn('system_info', diagnostics)
        
        # 检查常见环境变量
        common_vars = diagnostics['common_env_vars']
        self.assertEqual(common_vars['LLM_API_KEY'], 'set')
        self.assertEqual(common_vars['NEO4J_URI'], 'set')
    
    def test_get_recommendations(self):
        """测试获取建议"""
        validator = self._create_validator()
        # 创建有错误的情况
        validator.validation_errors.append("测试错误")
        
        recommendations = validator._get_recommendations()
        
        self.assertTrue(any('修复所有配置错误' in rec for rec in recommendations))
    
    def test_check_common_env_vars(self):
        """测试检查常见环境变量"""
        os.environ['LLM_API_KEY'] = 'test-key'
        os.environ['LLM_PROVIDER_1_API_KEY'] = 'provider-key'
        os.environ['NEO4J_USER'] = 'neo4j'
        
        validator = self._create_validator()
        common_vars = validator._check_common_env_vars()
        
        self.assertEqual(common_vars['LLM_API_KEY'], 'set')
        self.assertEqual(common_vars['LLM_PROVIDER_1_API_KEY'], 'set')
        self.assertEqual(common_vars['NEO4J_USER'], 'set')
        self.assertEqual(common_vars['LLM_PROVIDER_2_API_KEY'], 'not_set')
    
    def test_get_system_info(self):
        """测试获取系统信息"""
        validator = self._create_validator()
        system_info = validator._get_system_info()
        
        self.assertIn('python_version', system_info)
        self.assertIn('platform', system_info)
        self.assertIn('cwd', system_info)
        self.assertIn('env_file_path', system_info)


class TestConfigValidatorWarnings(unittest.TestCase):
    """测试配置验证器的警告功能"""
    
    def setUp(self):
        """测试前置设置"""
        self.validator = ConfigValidator()
        self.original_env = dict(os.environ)
        # 清除相关的环境变量
        for key in list(os.environ.keys()):
            if key.startswith('LLM_'):
                del os.environ[key]
    
    def tearDown(self):
        """测试后置清理"""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_validate_large_max_tokens_warning(self):
        """测试验证过大的max_tokens警告"""
        config = LLMProviderConfig(
            api_key='sk-test123456789012345678901234567890123456789012',
            base_url='https://api.example.com',
            model='test-model',
            max_tokens=150000,  # 过大值
            temperature=0.1,
            timeout=30,
            weight=1
        )
        
        self.validator._validate_numeric_params(config, 'test_config')
        
        warnings = self.validator.validation_warnings
        self.assertTrue(any('max_tokens 值过大' in warning for warning in warnings))
    
    def test_validate_small_max_tokens_warning(self):
        """测试验证过小的max_tokens警告"""
        config = LLMProviderConfig(
            api_key='sk-test123456789012345678901234567890123456789012',
            base_url='https://api.example.com',
            model='test-model',
            max_tokens=50,  # 过小值
            temperature=0.1,
            timeout=30,
            weight=1
        )
        
        self.validator._validate_numeric_params(config, 'test_config')
        
        warnings = self.validator.validation_warnings
        self.assertTrue(any('max_tokens 值过小' in warning for warning in warnings))
    
    def test_validate_high_temperature_warning(self):
        """测试验证高temperature警告"""
        config = LLMProviderConfig(
            api_key='sk-test123456789012345678901234567890123456789012',
            base_url='https://api.example.com',
            model='test-model',
            max_tokens=1000,
            temperature=1.8,  # 高值
            timeout=30,
            weight=1
        )
        
        self.validator._validate_numeric_params(config, 'test_config')
        
        warnings = self.validator.validation_warnings
        self.assertTrue(any('temperature 值较高' in warning for warning in warnings))
    
    def test_validate_large_weight_warning(self):
        """测试验证大权重警告"""
        config = LLMProviderConfig(
            api_key='sk-test123456789012345678901234567890123456789012',
            base_url='https://api.example.com',
            model='test-model',
            max_tokens=1000,
            temperature=0.1,
            timeout=30,
            weight=15  # 大权重
        )
        
        self.validator._validate_numeric_params(config, 'test_config')
        
        warnings = self.validator.validation_warnings
        self.assertTrue(any('weight 值较大' in warning for warning in warnings))
    
    def test_validate_small_timeout_warning(self):
        """测试验证小超时警告"""
        self.validator._validate_timeout(3, 'test_config')
        
        warnings = self.validator.validation_warnings
        self.assertTrue(any('timeout 值过小' in warning for warning in warnings))
    
    def test_validate_large_timeout_warning(self):
        """测试验证大超时警告"""
        self.validator._validate_timeout(400, 'test_config')
        
        warnings = self.validator.validation_warnings
        self.assertTrue(any('timeout 值过大' in warning for warning in warnings))


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)