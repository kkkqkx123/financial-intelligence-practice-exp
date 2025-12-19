"""
LLM客户端 - 仅保留客户端连接功能
负责与LLM服务的连接和通信，不包含业务逻辑处理
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin
import urllib.request
import urllib.error

try:
    from .config_manager import get_config_manager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)

# 设置日志级别为WARNING，减少INFO级别的日志输出
logger.setLevel(logging.WARNING)


class LLMClientInterface(ABC):
    """LLM客户端接口 - 定义与LLM服务交互的基本方法"""
    
    @abstractmethod
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述"""
        pass
    
    @abstractmethod
    def resolve_entity_conflicts(self, conflicting_entities: List[Dict]) -> Dict[str, Any]:
        """解决实体冲突"""
        pass
    
    @abstractmethod
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系"""
        pass
    
    @abstractmethod
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业"""
        pass
    
    @abstractmethod
    def standardize_investor_name(self, investor_name: str, context: Dict) -> str:
        """标准化投资方名称"""
        pass


class OpenAICompatibleClient(LLMClientInterface):
    """OpenAI兼容的LLM客户端"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.1,
                 timeout: int = 30):
        """
        初始化OpenAI兼容客户端
        
        Args:
            api_key: API密钥，如果不提供则从环境变量读取
            base_url: 基础URL，如果不提供则从环境变量读取
            model: 模型名称，如果不提供则从环境变量读取
            max_tokens: 最大token数，如果不提供则从环境变量读取
            temperature: 温度参数
            timeout: 请求超时时间
        """
        # 优先使用显式参数，其次使用配置管理器，最后使用环境变量
        if CONFIG_MANAGER_AVAILABLE:
            config_manager = get_config_manager()
            self.api_key = api_key or config_manager.get_api_key() or os.getenv('LLM_API_KEY', '')
            self.base_url = base_url or config_manager.get_base_url() or os.getenv('LLM_BASE_URL', '')
            self.model = model or config_manager.get_model() or os.getenv('LLM_MODEL', '')
            self.max_tokens = max_tokens or config_manager.get_max_tokens() or int(os.getenv('LLM_MAX_TOKENS', '1000'))
            self.timeout = timeout or config_manager.get_timeout() or int(os.getenv('LLM_TIMEOUT', '30'))
        else:
            self.api_key = api_key or os.getenv('LLM_API_KEY', '')
            self.base_url = base_url or os.getenv('LLM_BASE_URL', '')
            self.model = model or os.getenv('LLM_MODEL', '')
            self.max_tokens = max_tokens or int(os.getenv('LLM_MAX_TOKENS', '1000'))
            self.timeout = timeout
            
        self.temperature = temperature
        
        # 验证配置
        if not self.api_key:
            raise ValueError("API密钥未提供，请设置LLM_API_KEY环境变量或在构造函数中提供")
        if not self.base_url:
            raise ValueError("基础URL未提供，请设置LLM_BASE_URL环境变量或在构造函数中提供")
        if not self.model:
            raise ValueError("模型名称未提供，请设置LLM_MODEL环境变量或在构造函数中提供")
        
        logger.info(f"初始化OpenAI兼容客户端，模型: {self.model}, URL: {self.base_url}")
    
    def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """发送请求到LLM API
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数，如temperature、max_tokens等
            
        Returns:
            str: LLM生成的响应内容
            
        Raises:
            RuntimeError: 当API调用失败时
        """
        # 构建请求payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": False  # 禁用流式响应
        }
        
        # 添加可选参数（仅当值不为None时）
        if kwargs.get('top_p') is not None:
            payload['top_p'] = kwargs['top_p']
        if kwargs.get('frequency_penalty') is not None:
            payload['frequency_penalty'] = kwargs['frequency_penalty']
        if kwargs.get('presence_penalty') is not None:
            payload['presence_penalty'] = kwargs['presence_penalty']
        
        # 确保base_url格式正确
        base_url = self.base_url.rstrip('/')
        if not base_url.endswith('/chat/completions'):
            if base_url.endswith('/v1'):
                url = f"{base_url}/chat/completions"
            elif base_url.endswith('/v1/'):
                url = f"{base_url}chat/completions"
            else:
                url = f"{base_url}/v1/chat/completions"
        else:
            url = base_url
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            logger.debug(f"发送请求到 {url}")
            logger.debug(f"请求头: {headers}")
            logger.debug(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
            
            data = json.dumps(payload).encode('utf-8')
            
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content'].strip()
                else:
                    raise ValueError(f"无效的API响应: {result}")
                    
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
            logger.error(f"HTTP错误 {e.code}: {error_body}")
            raise RuntimeError(f"LLM API调用失败: {error_body}")
        except Exception as e:
            logger.error(f"请求失败: {e}")
            raise RuntimeError(f"LLM请求失败: {str(e)}")
    
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述 - 委托给业务逻辑处理器"""
        # 这里应该调用llm_processor的业务逻辑
        # 临时返回简单的描述，实际应该抛出异常或返回错误
        logger.warning("llm_client.enhance_entity_description 被调用，但业务逻辑应该在llm_processor中实现")
        return f"{entity_name}的基本描述信息"
    
    def resolve_entity_conflicts(self, conflicting_entities: List[Dict]) -> Dict[str, Any]:
        """解决实体冲突 - 委托给业务逻辑处理器"""
        logger.warning("llm_client.resolve_entity_conflicts 被调用，但业务逻辑应该在llm_processor中实现")
        return {'resolved_entity': None, 'confidence': 0.0}
    
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系 - 委托给业务逻辑处理器"""
        logger.warning("llm_client.extract_relationships_from_text 被调用，但业务逻辑应该在llm_processor中实现")
        return []
    
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业 - 委托给业务逻辑处理器"""
        logger.warning("llm_client.classify_company_industry 被调用，但业务逻辑应该在llm_processor中实现")
        return ['其他']
    
    def standardize_investor_name(self, investor_name: str, context: Dict) -> str:
        """标准化投资方名称 - 委托给业务逻辑处理器"""
        logger.warning("llm_client.standardize_investor_name 被调用，但业务逻辑应该在llm_processor中实现")
        return investor_name.strip()
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """生成响应 - 使用OpenAI API
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他参数，如temperature、max_tokens等
            
        Returns:
            str: LLM生成的响应内容
        """
        # 构建消息列表
        messages = []
        
        # 添加系统消息（如果有）
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 添加用户消息
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # 使用内部的_make_request方法发送请求
        return self._make_request(messages, **kwargs)


class MockLLMClient(LLMClientInterface):
    """模拟LLM客户端 - 用于测试和演示"""
    
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0
        logger.info("初始化模拟LLM客户端")
    
    def enhance_entity_description(self, entity_name: str, context: Dict[str, Any]) -> str:
        """增强实体描述"""
        self.call_count += 1
        self.total_tokens += len(entity_name) // 4
        logger.info(f"[MOCK] 增强实体描述: {entity_name}")
        return f"这是一家专注于{entity_name}的领先企业，在相关领域具有显著优势。公司成立于2010年，总部位于深圳，在人工智能、云计算和大数据领域持续创新。"
    
    def resolve_entity_conflicts(self, conflicting_entities: List[Dict]) -> Dict[str, Any]:
        """解决实体冲突"""
        self.call_count += 1
        self.total_tokens += len(str(conflicting_entities)) // 4
        logger.info(f"[MOCK] 解决实体冲突，数量: {len(conflicting_entities)}")
        return {
            "resolved_entity": {
                "name": "合并后的实体名称",
                "description": "这是合并后的实体描述"
            },
            "confidence": 0.85
        }
    
    def extract_relationships_from_text(self, text: str) -> List[Dict]:
        """从文本中提取关系"""
        self.call_count += 1
        self.total_tokens += len(text) // 4
        logger.info(f"[MOCK] 提取关系，文本长度: {len(text)}")
        return [
            {
                "source": "公司A",
                "target": "公司B", 
                "relationship": "投资关系",
                "confidence": 0.9
            }
        ]
    
    def classify_company_industry(self, company_info: Dict) -> List[str]:
        """分类公司行业"""
        self.call_count += 1
        self.total_tokens += len(str(company_info)) // 4
        logger.info(f"[MOCK] 分类公司行业: {company_info.get('name', '未知公司')}")
        return ["科技", "互联网", "人工智能"]
    
    def standardize_investor_name(self, investor_name: str, context: Dict) -> str:
        """标准化投资方名称"""
        self.call_count += 1
        self.total_tokens += len(investor_name) // 4
        logger.info(f"[MOCK] 标准化投资方名称: {investor_name}")
        return "标准投资方名称"
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """生成模拟响应（用于兼容OpenAICompatibleProcessor）"""
        self.call_count += 1
        self.total_tokens += len(prompt) // 4
        
        logger.info(f"[MOCK] 生成模拟响应: {prompt[:50]}...")
        
        # 根据提示词内容返回相应的模拟响应
        if "描述" in prompt or "实体" in prompt:
            return "这是一家先进的科技公司，专注于创新技术和数字化转型。公司成立于2010年，总部位于深圳，在人工智能、云计算和大数据领域具有显著优势。"
        elif "冲突" in prompt or "合并" in prompt:
            return json.dumps({
                "resolved_entity": {
                    "name": "合并后的实体名称",
                    "description": "这是合并后的实体描述"
                },
                "confidence": 0.85
            }, ensure_ascii=False)
        elif "关系" in prompt:
            return json.dumps([
                {
                    "source": "公司A",
                    "target": "公司B", 
                    "relationship": "投资关系",
                    "confidence": 0.9
                }
            ], ensure_ascii=False)
        elif "行业" in prompt or "分类" in prompt:
            return json.dumps(["科技", "互联网", "人工智能"], ensure_ascii=False)
        elif "投资方" in prompt or "标准化" in prompt:
            return "标准投资方名称"
        else:
            return "这是一个模拟的LLM响应，用于测试目的。"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'call_count': self.call_count,
            'total_tokens': self.total_tokens,
            'client_type': 'mock'
        }


def get_llm_client() -> LLMClientInterface:
    """获取LLM客户端实例
    
    根据配置返回相应的LLM客户端：
    - 优先使用ConfigManager的配置
    - 如果没有配置，抛出异常
    """
    try:
        # 尝试使用配置管理器
        from .config_manager import ConfigManager
        config_manager = ConfigManager()
        
        if config_manager.is_config_loaded() and config_manager.get_providers():
            # 使用配置管理器中的配置
            logger.info("使用配置管理器中的LLM配置")
            return OpenAICompatibleClient()
    except Exception as e:
        logger.warning(f"配置管理器加载失败: {e}")
    
    # 回退到环境变量检查
    api_key = os.getenv('LLM_API_KEY', '').strip()
    base_url = os.getenv('LLM_BASE_URL', '').strip()
    model = os.getenv('LLM_MODEL', '').strip()
    
    if api_key and base_url and model:
        # 配置了完整的provider，使用OpenAI兼容客户端
        logger.info("使用环境变量中的LLM配置")
        return OpenAICompatibleClient()
    
    # 没有配置任何LLM provider，抛出异常
    raise ValueError("未配置任何LLM provider。请设置LLM_API_KEY、LLM_BASE_URL和LLM_MODEL环境变量，或使用ConfigManager加载配置文件")