"""
配置验证和错误处理模块
提供配置验证、错误处理和诊断功能
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlparse
import os

try:
    from .config_manager import get_config_manager, LLMProviderConfig
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigValidator:
    """配置验证器"""
    
    # API密钥格式验证规则
    API_KEY_PATTERNS = {
        'openai': re.compile(r'^sk-[a-zA-Z0-9]{48}$'),
        'anthropic': re.compile(r'^sk-ant-[a-zA-Z0-9]{32,}$'),
        'google': re.compile(r'^[a-zA-Z0-9_-]{35,}$'),
        'generic': re.compile(r'^[a-zA-Z0-9_-]{20,}$')  # 通用格式
    }
    
    # 支持的模型格式
    SUPPORTED_MODEL_PATTERNS = {
        'openai': re.compile(r'^(gpt-4|gpt-3\.5-turbo|text-davinci-003|text-curie-001)$'),
        'anthropic': re.compile(r'^(claude-3-opus|claude-3-sonnet|claude-3-haiku|claude-2\.[0-9]|claude-2)$'),
        'google': re.compile(r'^(gemini-pro|gemini-pro-vision|text-bison|text-unicorn)$'),
        'generic': re.compile(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$')  # 通用格式，如: Qwen/Qwen2.5-7B-Instruct
    }
    
    # URL验证规则
    URL_REQUIREMENTS = {
        'min_length': 10,
        'max_length': 200,
        'required_schemes': ['http', 'https'],
        'forbidden_chars': [' ', '\t', '\n', '\r']
    }
    
    def __init__(self):
        """初始化配置验证器"""
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        self.suggestions: List[str] = []
    
    def validate_all_configs(self) -> Dict[str, Any]:
        """
        验证所有配置
        
        Returns:
            Dict[str, Any]: 验证结果，包含状态、错误、警告和建议
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        self.suggestions.clear()
        
        # 验证配置管理器
        try:
            if CONFIG_MANAGER_AVAILABLE:
                config_manager = get_config_manager()
                self._validate_config_manager(config_manager)
            else:
                self.validation_warnings.append("配置管理器不可用，将使用环境变量配置")
                self._validate_env_configs()
        except ImportError:
            # 配置管理器导入失败，使用环境变量配置
            self.validation_warnings.append("配置管理器导入失败，将使用环境变量配置")
            self._validate_env_configs()
        except Exception as e:
            # 其他异常，也使用环境变量配置
            self.validation_warnings.append(f"配置管理器初始化失败: {str(e)}，将使用环境变量配置")
            self._validate_env_configs()
        
        # 生成验证报告
        validation_result = {
            'status': 'valid' if not self.validation_errors else 'invalid',
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy(),
            'suggestions': self.suggestions.copy(),
            'summary': self._generate_summary()
        }
        
        return validation_result
    
    def _validate_config_manager(self, config_manager):
        """验证配置管理器"""
        try:
            # 检查是否加载了配置
            if not config_manager.is_config_loaded():
                self.validation_errors.append("配置管理器未成功加载配置")
                return
            
            # 获取所有LLM配置
            llm_configs = config_manager.get_llm_configs()
            
            if not llm_configs:
                self.validation_warnings.append("未配置任何LLM provider")
                self.suggestions.append("建议至少配置一个LLM provider以获得更好的性能")
                return
            
            # 验证每个配置
            for i, config_dict in enumerate(llm_configs):
                # 将字典转换为LLMProviderConfig对象
                config = LLMProviderConfig(**config_dict)
                self._validate_llm_config(config, f"provider_{i+1}")
                
        except Exception as e:
            self.validation_errors.append(f"配置管理器验证失败: {str(e)}")
            logger.error(f"配置管理器验证失败: {e}", exc_info=True)
    
    def _validate_llm_config(self, config: LLMProviderConfig, config_name: str):
        """验证单个LLM配置"""
        try:
            # 验证API密钥
            self._validate_api_key(config.api_key, config_name)
            
            # 验证基础URL
            self._validate_base_url(config.base_url, config_name)
            
            # 验证模型名称
            self._validate_model_name(config.model, config_name)
            
            # 验证数值参数
            self._validate_numeric_params(config, config_name)
            
            # 验证超时设置
            self._validate_timeout(config.timeout, config_name)
            
        except Exception as e:
            self.validation_errors.append(f"配置 {config_name} 验证失败: {str(e)}")
    
    def _validate_env_configs(self):
        """验证环境变量配置"""
        try:
            # 检查基本的API配置
            api_key = os.getenv('LLM_API_KEY')
            base_url = os.getenv('LLM_BASE_URL')
            model = os.getenv('LLM_MODEL')
            
            # 检查多provider配置
            has_multi_provider = self._check_multi_provider_config()
            
            # 如果没有API密钥且没有多provider配置，添加警告
            if not api_key and not has_multi_provider:
                self.validation_warnings.append("未配置API密钥 (LLM_API_KEY 或 LLM_PROVIDER_*_API_KEY)")
                return
            
            # 验证单个provider配置
            if api_key and base_url and model:
                self._validate_api_key(api_key, "env_single")
                self._validate_base_url(base_url, "env_single")
                self._validate_model_name(model, "env_single")
                self._validate_timeout(30, "env_single")  # 使用默认超时值
            elif has_multi_provider:
                # 验证多provider配置
                self._validate_multi_provider_configs()
            else:
                # 配置不完整的情况 - 验证已提供的参数，缺失的参数生成警告
                if api_key:
                    # 有API密钥但缺少其他必需参数
                    if not base_url:
                        self.validation_warnings.append("配置不完整: 缺少 BASE_URL")
                    if not model:
                        self.validation_warnings.append("配置不完整: 缺少 MODEL")
                    
                    # 验证已提供的参数（处理空值情况）
                    if api_key.strip():
                        self._validate_api_key(api_key, "env_single")
                    else:
                        self.validation_warnings.append("env_single: API密钥为空")
                    
                    if base_url and base_url.strip():
                        self._validate_base_url(base_url, "env_single")
                    elif base_url:  # 有值但为空字符串
                        self.validation_warnings.append("env_single: 基础URL为空")
                    
                    if model and model.strip():
                        self._validate_model_name(model, "env_single")
                    elif model:  # 有值但为空字符串
                        self.validation_warnings.append("env_single: 模型名称为空")
                    
                    self._validate_timeout(30, "env_single")  # 使用默认超时值
                # 注意：如果没有API密钥但有多provider配置，已经在上面处理了
                
        except Exception as e:
            self.validation_errors.append(f"环境变量配置验证失败: {str(e)}")
            logger.error(f"环境变量配置验证失败: {e}", exc_info=True)
    
    def _check_multi_provider_config(self) -> bool:
        """检查是否存在多provider配置"""
        return bool(os.getenv('LLM_PROVIDER_1_API_KEY'))
    
    def _validate_multi_provider_configs(self):
        """验证多provider配置"""
        provider_index = 1
        valid_providers = 0
        
        while True:
            api_key = os.getenv(f'LLM_PROVIDER_{provider_index}_API_KEY')
            if not api_key:
                break
            
            base_url = os.getenv(f'LLM_PROVIDER_{provider_index}_BASE_URL')
            model = os.getenv(f'LLM_PROVIDER_{provider_index}_MODEL')
            
            if base_url and model:
                config_name = f"multi_provider_{provider_index}"
                self._validate_api_key(api_key, config_name)
                self._validate_base_url(base_url, config_name)
                self._validate_model_name(model, config_name)
                self._validate_timeout(30, config_name)  # 使用默认超时值
                valid_providers += 1
            else:
                # 配置不完整的provider应该生成警告而不是错误
                self.validation_warnings.append(
                    f"Provider {provider_index} 配置不完整 (缺少BASE_URL或MODEL)"
                )
            
            provider_index += 1
        
        if valid_providers == 0:
            self.validation_warnings.append("多provider配置无效，没有完整的provider")
    
    def _validate_api_key(self, api_key: str, config_name: str):
        """验证API密钥"""
        if not api_key or not api_key.strip():
            self.validation_warnings.append(f"{config_name}: API密钥为空")
            return
        
        if len(api_key) < 10:
            self.validation_warnings.append(f"{config_name}: API密钥长度不足 (< 10字符)")
        
        # 检查API密钥格式
        matched_pattern = False
        for provider, pattern in self.API_KEY_PATTERNS.items():
            if pattern.match(api_key):
                matched_pattern = True
                break
        
        if not matched_pattern:
            self.validation_warnings.append(
                f"{config_name}: API密钥格式不符合已知模式，将使用通用验证"
            )
            # 使用通用模式进行最终检查
            if not self.API_KEY_PATTERNS['generic'].match(api_key):
                self.validation_warnings.append(f"{config_name}: API密钥格式无效")
    
    def _validate_base_url(self, base_url: str, config_name: str):
        """验证基础URL"""
        if not base_url or not base_url.strip():
            self.validation_warnings.append(f"{config_name}: 基础URL为空")
            return
        
        # 基本长度检查
        if len(base_url) < self.URL_REQUIREMENTS['min_length']:
            self.validation_warnings.append(f"{config_name}: 基础URL长度不足")
        
        if len(base_url) > self.URL_REQUIREMENTS['max_length']:
            self.validation_warnings.append(f"{config_name}: 基础URL过长")
        
        # 检查禁止字符
        for char in self.URL_REQUIREMENTS['forbidden_chars']:
            if char in base_url:
                self.validation_warnings.append(f"{config_name}: 基础URL包含非法字符 '{char}'")
        
        # URL格式验证
        try:
            parsed = urlparse(base_url)

            if not parsed.scheme:
                self.validation_warnings.append(f"{config_name}: 基础URL缺少协议 (http/https)")

            if parsed.scheme not in self.URL_REQUIREMENTS['required_schemes']:
                self.validation_warnings.append(
                    f"{config_name}: 不支持的协议 '{parsed.scheme}' 请使用 http 或 https"
                )

            if not parsed.netloc:
                self.validation_warnings.append(f"{config_name}: 基础URL缺少主机名")

            # 检查是否包含API路径
            if not parsed.path or parsed.path == '/':
                self.validation_warnings.append(
                    f"{config_name}: 基础URL可能缺少API路径，常见路径如 /v1 或 /api/v1"
                )

            # 检查常见API服务提供商
            if 'api.openai.com' in base_url:
                self.suggestions.append(f"{config_name}: 检测到OpenAI API，确保使用正确的模型")
            elif 'api.anthropic.com' in base_url:
                self.suggestions.append(f"{config_name}: 检测到Anthropic API，确保使用Claude模型")
            elif 'generativelanguage.googleapis.com' in base_url:
                self.suggestions.append(f"{config_name}: 检测到Google AI API，确保使用Gemini模型")
                
        except Exception as e:
            self.validation_warnings.append(f"{config_name}: 基础URL格式错误 - {str(e)}")
    
    def _validate_model_name(self, model: str, config_name: str):
        """验证模型名称"""
        if not model or not model.strip():
            self.validation_warnings.append(f"{config_name}: 模型名称为空")
            return
        
        if len(model) < 3:
            self.validation_warnings.append(f"{config_name}: 模型名称长度不足")
        
        # 检查模型格式
        matched_pattern = False
        for provider, pattern in self.SUPPORTED_MODEL_PATTERNS.items():
            if pattern.match(model):
                matched_pattern = True
                break
        
        if not matched_pattern:
            self.validation_warnings.append(
                f"{config_name}: 模型名称 '{model}' 不符合已知模式，将使用通用验证"
            )
            # 通用模型名称检查
            if not re.match(r'^[a-zA-Z0-9_/.-]+$', model):
                self.validation_warnings.append(f"{config_name}: 模型名称格式无效")
            # 检查模型系列
            if not any(series in model.lower() for series in ['gpt', 'claude', 'gemini', 'llama', 'qwen', 'yi']):
                self.validation_warnings.append(f"{config_name}: 模型名称可能不属于常见的大语言模型系列")
    
    def _validate_numeric_params(self, config: LLMProviderConfig, config_name: str):
        """验证数值参数"""
        # max_tokens 验证
        if hasattr(config, 'max_tokens') and config.max_tokens is not None:
            if not isinstance(config.max_tokens, (int, float)):
                self.validation_errors.append(f"{config_name}: max_tokens 必须是数值类型")
            elif config.max_tokens <= 0:
                self.validation_errors.append(f"{config_name}: max_tokens 必须是正整数")
            elif config.max_tokens > 100000:
                self.validation_warnings.append(f"{config_name}: max_tokens 值过大 ({config.max_tokens})，可能影响性能")
            elif config.max_tokens < 100:
                self.validation_warnings.append(f"{config_name}: max_tokens 值过小 ({config.max_tokens})，可能影响输出质量")
        
        # 验证temperature
        if hasattr(config, 'temperature') and config.temperature is not None:
            if not isinstance(config.temperature, (int, float)):
                self.validation_errors.append(f"{config_name}: temperature 必须是数值")
            elif not (0 <= config.temperature <= 2):
                self.validation_errors.append(f"{config_name}: temperature 必须在 0-2 范围内")
            elif config.temperature > 1.5:
                self.validation_warnings.append(f"{config_name}: temperature 值较高 ({config.temperature})，输出可能不稳定")
        
        # 验证weight
        if hasattr(config, 'weight') and config.weight is not None:
            if not isinstance(config.weight, int) or config.weight <= 0:
                self.validation_errors.append(f"{config_name}: weight 必须是正整数")
            elif config.weight > 10:
                self.validation_warnings.append(f"{config_name}: weight 值较大 ({config.weight})，可能导致负载不均")
    
    def _validate_timeout(self, timeout: int, config_name: str):
        """验证超时设置"""
        if not isinstance(timeout, int):
            self.validation_errors.append(f"{config_name}: 超时设置必须是整数")
        elif timeout <= 0:
            self.validation_errors.append(f"{config_name}: timeout 必须是正整数")
        elif timeout < 5:
            self.validation_warnings.append(f"{config_name}: timeout 值过小 ({timeout}s)，可能导致请求超时")
        elif timeout > 300:
            self.validation_warnings.append(f"{config_name}: timeout 值过大 ({timeout}s)，可能影响响应速度")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成验证摘要"""
        return {
            'total_errors': len(self.validation_errors),
            'total_warnings': len(self.validation_warnings),
            'total_suggestions': len(self.suggestions),
            'overall_health': 'good' if not self.validation_errors else 'poor',
            'recommendations': self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """获取配置建议"""
        recommendations = []
        
        if self.validation_errors:
            recommendations.append("修复所有配置错误后再继续使用")
        
        if len(self.validation_warnings) > 3:
            recommendations.append("考虑优化配置以减少警告")
        
        if not any('timeout' in error for error in self.validation_errors):
            recommendations.append("建议设置合理的超时时间 (30-60秒)")
        
        if not any('temperature' in warning for warning in self.validation_warnings):
            recommendations.append("建议根据使用场景调整temperature参数")
        
        return recommendations
    
    def get_diagnostics_info(self) -> Dict[str, Any]:
        """
        获取诊断信息
        
        Returns:
            Dict[str, Any]: 诊断信息
        """
        diagnostics = {
            'config_manager_available': CONFIG_MANAGER_AVAILABLE,
            'env_file_exists': os.path.exists('.env'),
            'common_env_vars': self._check_common_env_vars(),
            'system_info': self._get_system_info()
        }
        
        if CONFIG_MANAGER_AVAILABLE:
            try:
                config_manager = get_config_manager()
                diagnostics['config_manager_status'] = {
                    'is_loaded': config_manager.is_config_loaded(),
                    'provider_count': len(config_manager.get_llm_configs()),
                    'config_source': config_manager.get_config_source()
                }
            except Exception as e:
                diagnostics['config_manager_error'] = str(e)
        
        return diagnostics
    
    def _check_common_env_vars(self) -> Dict[str, str]:
        """检查常见的环境变量"""
        common_vars = [
            'LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL',
            'LLM_PROVIDER_1_API_KEY', 'LLM_PROVIDER_1_BASE_URL', 'LLM_PROVIDER_1_MODEL',
            'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD'
        ]
        
        result = {}
        for var in common_vars:
            result[var] = 'set' if os.getenv(var) else 'not_set'
        
        return result
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'platform': os.sys.platform,
            'cwd': os.getcwd(),
            'env_file_path': os.path.abspath('.env') if os.path.exists('.env') else None
        }


def validate_configuration() -> Dict[str, Any]:
    """
    验证当前配置的便捷函数
    
    Returns:
        Dict[str, Any]: 验证结果
    """
    validator = ConfigValidator()
    return validator.validate_all_configs()


def get_configuration_diagnostics() -> Dict[str, Any]:
    """
    获取配置诊断信息
    
    Returns:
        Dict[str, Any]: 诊断信息
    """
    validator = ConfigValidator()
    return validator.get_diagnostics_info()