"""
配置管理模块
提供统一的配置加载、验证和管理功能
支持.env文件自动加载和环境变量管理
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 设置日志级别为ERROR，减少日志输出
logger.setLevel(logging.ERROR)


@dataclass
class LLMProviderConfig:
    """LLM提供商配置"""
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: int = 30
    
    def validate(self) -> bool:
        """验证配置有效性"""
        return all([
            self.api_key.strip(),
            self.base_url.strip(),
            self.model.strip()
        ])


@dataclass
class ConfigManager:
    """配置管理器"""
    env_file_path: Path = field(default_factory=lambda: Path('.env'))
    providers: List[LLMProviderConfig] = field(default_factory=list)
    is_loaded: bool = False
    load_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后自动加载配置"""
        # 尝试多个位置查找.env文件
        possible_paths = [
            Path('.env'),                    # 当前目录
            Path('../.env'),                 # 上级目录
            Path(__file__).parent.parent / '.env',  # 项目根目录
            Path(__file__).parent / '.env'   # src目录
        ]
        
        for path in possible_paths:
            if path.exists():
                self.env_file_path = path
                break
        
        self.load_configuration()
    
    def load_configuration(self, force_reload: bool = False) -> bool:
        """
        加载配置文件
        
        Args:
            force_reload: 是否强制重新加载
            
        Returns:
            bool: 是否成功加载配置
        """
        if self.is_loaded and not force_reload:
            return True
        
        self.load_errors.clear()
        
        try:
            # 1. 加载.env文件
            self._load_env_file()
            
            # 2. 解析LLM提供商配置
            self._parse_llm_providers()
            
            # 3. 验证配置
            self._validate_configuration()
            
            self.is_loaded = True
            logger.info(f"配置加载成功，找到 {len(self.providers)} 个LLM提供商")
            return True
            
        except Exception as e:
            self.load_errors.append(f"配置加载失败: {str(e)}")
            logger.error(f"配置加载失败: {e}")
            return False
    
    def _load_env_file(self) -> None:
        """加载.env文件"""
        if not self.env_file_path.exists():
            self.load_errors.append(f"环境文件不存在: {self.env_file_path}")
            logger.warning(f"环境文件不存在: {self.env_file_path}")
            return
        
        try:
            with open(self.env_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            loaded_count = 0
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 解析键值对
                if '=' not in line:
                    self.load_errors.append(f"第{line_num}行格式错误: 缺少等号")
                    continue
                
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip()
                
                # 去除引号
                if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                    value = value[1:-1]
                
                # 设置环境变量
                os.environ[key] = value
                loaded_count += 1
            
            logger.info(f"成功加载 {loaded_count} 个环境变量")
            
        except Exception as e:
            self.load_errors.append(f"加载.env文件失败: {str(e)}")
            logger.error(f"加载.env文件失败: {e}")
    
    def _parse_llm_providers(self) -> None:
        """解析LLM提供商配置"""
        self.providers.clear()
        
        # 从环境变量读取配置（单提供商格式）
        api_key = os.getenv('LLM_API_KEY')
        base_url = os.getenv('LLM_BASE_URL')
        model = os.getenv('LLM_MODEL')
        
        if api_key and base_url and model:
            provider_config = LLMProviderConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
                temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
                timeout=int(os.getenv('LLM_TIMEOUT', '30'))
            )
            
            if provider_config.validate():
                self.providers.append(provider_config)
                logger.info("使用单提供商配置格式添加提供商")
        else:
            logger.warning("未找到有效的LLM提供商配置")
    
    def _validate_configuration(self) -> None:
        """验证配置有效性"""
        if not self.providers:
            self.load_errors.append("没有找到有效的LLM提供商配置")
            logger.warning("没有找到有效的LLM提供商配置")
        
        # 验证每个提供商
        for i, provider in enumerate(self.providers):
            if not provider.validate():
                self.load_errors.append(f"提供商 {i+1} 配置无效")
                logger.warning(f"提供商 {i+1} 配置无效")
    
    def get_providers(self) -> List[LLMProviderConfig]:
        """获取所有有效的提供商配置"""
        return self.providers.copy()
    
    def get_api_key(self) -> Optional[str]:
        """获取API密钥（主配置）"""
        if self.providers:
            return self.providers[0].api_key
        return None
    
    def get_base_url(self) -> Optional[str]:
        """获取基础URL（主配置）"""
        if self.providers:
            return self.providers[0].base_url
        return None
    
    def get_model(self) -> Optional[str]:
        """获取模型名称（主配置）"""
        if self.providers:
            return self.providers[0].model
        return None
    
    def get_max_tokens(self) -> int:
        """获取最大token数（主配置）"""
        if self.providers:
            return self.providers[0].max_tokens
        return 1000
    
    def get_timeout(self) -> int:
        """获取超时时间（主配置）"""
        if self.providers:
            return self.providers[0].timeout
        return 30
    
    def get_temperature(self) -> float:
        """获取温度参数（主配置）"""
        if self.providers:
            return self.providers[0].temperature
        return 0.1
    
    def is_config_loaded(self) -> bool:
        """检查配置是否已加载"""
        return self.is_loaded
    
    def get_config_source(self) -> str:
        """获取配置来源"""
        return str(self.env_file_path)
    
    def get_primary_provider(self) -> Optional[LLMProviderConfig]:
        """获取主要提供商配置"""
        if not self.providers:
            return None
        
        # 返回第一个提供商
        return self.providers[0]
    
    def get_provider_by_model(self, model: str) -> Optional[LLMProviderConfig]:
        """根据模型名称获取提供商配置"""
        for provider in self.providers:
            if provider.model == model:
                return provider
        return None
    
    def has_valid_configuration(self) -> bool:
        """检查是否有有效的配置"""
        return self.is_loaded and len(self.providers) > 0
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """获取配置状态信息"""
        return {
            'is_loaded': self.is_loaded,
            'providers_count': len(self.providers),
            'has_valid_config': self.has_valid_configuration(),
            'load_errors': self.load_errors.copy(),
            'provider_models': [p.model for p in self.providers],
            'primary_provider': self.get_primary_provider().model if self.get_primary_provider() else None
        }
    
    def reload_configuration(self) -> bool:
        """重新加载配置"""
        self.is_loaded = False
        self.providers.clear()
        self.load_errors.clear()
        return self.load_configuration(force_reload=True)


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_configuration(env_file_path: Optional[Union[str, Path]] = None) -> bool:
    """
    便捷函数：加载配置
    
    Args:
        env_file_path: 自定义.env文件路径
        
    Returns:
        bool: 是否成功加载配置
    """
    manager = get_config_manager()
    
    if env_file_path:
        manager.env_file_path = Path(env_file_path)
    
    return manager.load_configuration(force_reload=True)


def get_llm_providers() -> List[LLMProviderConfig]:
    """便捷函数：获取LLM提供商配置"""
    return get_config_manager().get_providers()


def has_llm_configuration() -> bool:
    """便捷函数：检查是否有LLM配置"""
    return get_config_manager().has_valid_configuration()


# 向后兼容：保持原有的配置常量
from .config import *