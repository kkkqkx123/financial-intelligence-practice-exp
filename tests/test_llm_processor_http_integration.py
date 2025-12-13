"""
LLM处理器HTTP API集成测试模块
测试src/processors/llm_processor.py与真实HTTP API的集成
使用实际的LLM API进行端到端测试
"""

import os
import sys
import json
import time
import unittest
from typing import Dict, List, Any
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 优先加载配置管理器
try:
    from src.processors.config_manager import load_configuration
    config_loaded = load_configuration()
    print(f"✅ 配置管理器加载状态: {config_loaded}")
except Exception as e:
    print(f"⚠️ 配置管理器加载失败: {e}")
    # 回退到手动加载.env文件
    env_path = project_root / '.env'
    if env_path.exists():
        print(f"正在加载.env文件: {env_path}")
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ .env文件加载完成")
    else:
        print("⚠️ 未找到.env文件")

# 添加src目录到Python路径
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from processors.llm_processor import (
    OpenAICompatibleProcessor,
    BatchLLMProcessor,
    LLMEnhancementTracker,
    get_llm_processor,
    get_enhancement_tracker,
    get_batch_llm_processor
)
from processors.llm_client import OpenAICompatibleClient
from processors.config import LLM_CONFIG, BATCH_PROCESSING_CONFIG


class TestLLMProcessorHTTPIntegration(unittest.TestCase):
    """测试LLM处理器与真实HTTP API的集成"""
    
    @classmethod
    def setUpClass(cls):
        """类级别设置 - 检查是否配置了API密钥"""
        # 优先使用配置管理器获取配置
        try:
            from src.processors.config_manager import get_config_manager
            config_manager = get_config_manager()
            
            if config_manager.is_config_loaded() and config_manager.get_llm_configs():
                # 使用配置管理器中的配置
                primary_config = config_manager.get_llm_configs()[0]
                cls.api_key = primary_config.api_key
                cls.base_url = primary_config.base_url
                cls.model = primary_config.model
                print(f"✅ 使用配置管理器 - 检测到API配置: {cls.model} @ {cls.base_url}")
            else:
                # 回退到环境变量
                cls.api_key = os.getenv('LLM_API_KEY', '')
                cls.base_url = os.getenv('LLM_BASE_URL', '')
                cls.model = os.getenv('LLM_MODEL', '')
                print(f"✅ 使用环境变量 - 检测到API配置: {cls.model} @ {cls.base_url}")
        except Exception as e:
            # 回退到环境变量
            cls.api_key = os.getenv('LLM_API_KEY', '')
            cls.base_url = os.getenv('LLM_BASE_URL', '')
            cls.model = os.getenv('LLM_MODEL', '')
            print(f"✅ 使用环境变量 - 检测到API配置: {cls.model} @ {cls.base_url}")
        
        # 检查是否配置了真实的API
        cls.has_real_api = bool(cls.api_key and cls.base_url and cls.model)
        
        if not cls.has_real_api:
            print("⚠️  未配置真实的LLM API，将跳过HTTP集成测试")
            print("请在.env文件中设置以下环境变量：")
            print("- LLM_API_KEY")
            print("- LLM_BASE_URL") 
            print("- LLM_MODEL")
        else:
            print(f"✅ 检测到API配置: {cls.model} @ {cls.base_url}")
    
    def setUp(self):
        """每个测试方法前的设置"""
        if not self.has_real_api:
            self.skipTest("未配置真实的LLM API")
        
        # 创建真实的LLM客户端
        try:
            self.llm_client = OpenAICompatibleClient(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_tokens=500,  # 测试时限制token数
                temperature=0.1,
                timeout=30
            )
            self.processor = OpenAICompatibleProcessor(self.llm_client)
        except Exception as e:
            self.skipTest(f"无法初始化LLM客户端: {e}")
    
    def test_real_entity_description_enhancement(self):
        """测试真实的实体描述增强"""
        entity_name = "阿里巴巴集团"
        context = {
            "industry": "电子商务",
            "establish_date": "1999年",
            "description": "中国最大的电子商务公司"
        }
        
        try:
            result = self.processor.enhance_entity_description(entity_name, context)
            
            # 验证结果格式
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 50)  # 应该返回较长的描述
            self.assertIn("阿里巴巴", result)  # 应该包含公司名称
            
            print(f"实体描述增强结果: {result[:100]}...")
            
        except Exception as e:
            self.fail(f"实体描述增强失败: {e}")
    
    def test_real_entity_conflict_resolution(self):
        """测试真实的实体冲突解决"""
        conflicting_entities = [
            {"name": "腾讯科技", "confidence": 0.85, "source": "企查查"},
            {"name": "深圳市腾讯计算机系统有限公司", "confidence": 0.92, "source": "天眼查"},
            {"name": "腾讯公司", "confidence": 0.78, "source": "手动输入"}
        ]
        
        try:
            result = self.processor.resolve_entity_conflicts(conflicting_entities)
            
            # 验证结果格式
            self.assertIsInstance(result, dict)
            self.assertIn("resolved_entity", result)
            self.assertIn("confidence", result)
            self.assertIn("merged_aliases", result)
            
            # 验证置信度范围
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            
            print(f"冲突解决结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            self.fail(f"实体冲突解决失败: {e}")
    
    def test_real_relationship_extraction(self):
        """测试真实的关系提取"""
        text = """
        字节跳动在2021年完成了对 Pico 的收购，交易金额约为90亿元人民币。
        此次收购是字节跳动在VR领域的重要布局，显示了其对元宇宙概念的重视。
        同时，红杉资本中国基金参与了字节跳动的多轮融资，投资金额超过10亿美元。
        """
        
        try:
            result = self.processor.extract_relationships_from_text(text)
            
            # 验证结果格式
            self.assertIsInstance(result, list)
            
            if result:  # 如果有提取到关系
                for relationship in result:
                    self.assertIn("type", relationship)
                    self.assertIn("confidence", relationship)
                    self.assertIn("description", relationship)
                    
                    # 验证置信度范围
                    self.assertGreaterEqual(relationship["confidence"], 0.0)
                    self.assertLessEqual(relationship["confidence"], 1.0)
            
            print(f"关系提取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            self.fail(f"关系提取失败: {e}")
    
    def test_real_company_industry_classification(self):
        """测试真实的公司行业分类"""
        company_info = {
            "name": "深圳市腾讯计算机系统有限公司",
            "description": "腾讯是一家以互联网为基础的科技公司，提供社交、游戏、金融科技等服务",
            "website": "https://www.tencent.com",
            "establish_date": "1998年"
        }
        
        try:
            result = self.processor.classify_company_industry(company_info)
            
            # 验证结果格式
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)  # 应该至少有一个分类
            
            # 验证分类结果应该是中文
            for industry in result:
                self.assertIsInstance(industry, str)
                self.assertGreater(len(industry), 0)
            
            print(f"行业分类结果: {result}")
            
        except Exception as e:
            self.fail(f"行业分类失败: {e}")
    
    def test_real_investor_name_standardization(self):
        """测试真实的投资方名称标准化"""
        investor_name = "  红杉资本中国基金  "
        context = {
            "type": "VC",
            "scale": "大型",
            "preferred_rounds": ["A轮", "B轮"]
        }
        
        try:
            result = self.processor.standardize_investor_name(investor_name, context)
            
            # 验证结果格式
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            self.assertEqual(result.strip(), result)  # 应该去除首尾空格
            
            print(f"投资方名称标准化结果: '{result}'")
            
        except Exception as e:
            self.fail(f"投资方名称标准化失败: {e}")


class TestBatchLLMProcessorHTTPIntegration(unittest.TestCase):
    """测试批量LLM处理器与真实HTTP API的集成"""
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置"""
        cls.api_key = os.getenv('LLM_API_KEY', '')
        cls.base_url = os.getenv('LLM_BASE_URL', '')
        cls.model = os.getenv('LLM_MODEL', '')
        cls.has_real_api = bool(cls.api_key and cls.base_url and cls.model)
    
    def setUp(self):
        """每个测试方法前的设置"""
        if not self.has_real_api:
            self.skipTest("未配置真实的LLM API")
        
        try:
            self.llm_client = OpenAICompatibleClient(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_tokens=300,
                temperature=0.1,
                timeout=30
            )
            self.batch_processor = BatchLLMProcessor(self.llm_client)
        except Exception as e:
            self.skipTest(f"无法初始化批量LLM处理器: {e}")
    
    def test_batch_entity_description_enhancement(self):
        """测试批量实体描述增强"""
        entities = [
            {"name": "阿里巴巴集团", "context": {"industry": "电子商务"}},
            {"name": "腾讯科技", "context": {"industry": "互联网"}},
            {"name": "百度公司", "context": {"industry": "搜索引擎"}}
        ]
        
        try:
            results = self.batch_processor.batch_enhance_entity_descriptions(
                entities
            )
            
            # 验证结果格式
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), len(entities))
            
            for i, result in enumerate(results):
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 30)
                self.assertIn(entities[i]["name"], result)
            
            print(f"批量实体描述增强完成，处理了 {len(results)} 个实体")
            
        except Exception as e:
            self.fail(f"批量实体描述增强失败: {e}")
    
    def test_batch_company_industry_classification(self):
        """测试批量公司行业分类"""
        companies = [
            {"name": "字节跳动", "description": "短视频和资讯平台公司"},
            {"name": "美团", "description": "本地生活服务平台"},
            {"name": "滴滴出行", "description": "网约车和出行服务平台"}
        ]
        
        try:
            results = self.batch_processor.batch_classify_company_industries(
                companies
            )
            
            # 验证结果格式
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), len(companies))
            
            for i, result in enumerate(results):
                self.assertIsInstance(result, list)
                self.assertGreater(len(result), 0)
                print(f"{companies[i]['name']} 的行业分类: {result}")
            
        except Exception as e:
            self.fail(f"批量行业分类失败: {e}")


class TestLLMEnhancementTrackerHTTPIntegration(unittest.TestCase):
    """测试LLM增强跟踪器与真实HTTP API的集成"""
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置"""
        cls.api_key = os.getenv('LLM_API_KEY', '')
        cls.base_url = os.getenv('LLM_BASE_URL', '')
        cls.model = os.getenv('LLM_MODEL', '')
        cls.has_real_api = bool(cls.api_key and cls.base_url and cls.model)
    
    def setUp(self):
        """每个测试方法前的设置"""
        if not self.has_real_api:
            self.skipTest("未配置真实的LLM API")
        
        try:
            self.llm_client = OpenAICompatibleClient(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                max_tokens=300,
                temperature=0.1,
                timeout=30
            )
            self.processor = OpenAICompatibleProcessor(self.llm_client)
            self.tracker = LLMEnhancementTracker(self.processor)
        except Exception as e:
            self.skipTest(f"无法初始化增强跟踪器: {e}")
    
    def test_enhancement_tracker_with_real_api(self):
        """测试使用真实API的增强跟踪器"""
        # 添加一些增强请求
        self.tracker.add_enhancement_request(
            request_type="enhance_entity_description",
            entity_name="小米科技",
            context={"industry": "智能手机", "description": "中国智能手机制造商"}
        )
        
        self.tracker.add_enhancement_request(
            request_type="classify_company_industry",
            company_info={"name": "蔚来汽车", "description": "电动汽车制造商"}
        )
        
        # 处理请求
        try:
            results = self.tracker.process_batch_requests_smart(
                batch_size=2,
                max_workers=1  # 测试时限制并发数
            )
            
            # 验证结果
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            
            for result in results:
                self.assertIn("request_id", result)
                self.assertIn("result", result)
                self.assertIn("status", result)
                self.assertEqual(result["status"], "completed")
            
            print(f"增强跟踪器处理了 {len(results)} 个请求")
            
        except Exception as e:
            self.fail(f"增强跟踪器处理失败: {e}")


class TestFactoryFunctionsHTTPIntegration(unittest.TestCase):
    """测试工厂函数与真实HTTP API的集成"""
    
    def test_factory_functions_with_real_config(self):
        """测试使用真实配置的工厂函数"""
        # 设置环境变量（如果不存在）
        original_api_key = os.getenv('LLM_API_KEY', '')
        original_base_url = os.getenv('LLM_BASE_URL', '')
        original_model = os.getenv('LLM_MODEL', '')
        
        if not (original_api_key and original_base_url and original_model):
            self.skipTest("未配置真实的LLM API")
        
        try:
            # 测试工厂函数
            processor = get_llm_processor()
            self.assertIsInstance(processor, OpenAICompatibleProcessor)
            
            batch_processor = get_batch_llm_processor()
            self.assertIsInstance(batch_processor, BatchLLMProcessor)
            
            tracker = get_enhancement_tracker()
            self.assertIsInstance(tracker, LLMEnhancementTracker)
            
            # 测试基本功能
            result = processor.enhance_entity_description("测试公司", {"industry": "科技"})
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 20)
            
            print(f"工厂函数测试通过，处理器类型: {type(processor).__name__}")
            
        except Exception as e:
            self.fail(f"工厂函数测试失败: {e}")


class TestAPIConfiguration(unittest.TestCase):
    """测试API配置集成"""
    
    def test_config_loading(self):
        """测试配置加载"""
        # 测试从环境变量加载配置
        api_key = os.getenv('LLM_API_KEY', '')
        base_url = os.getenv('LLM_BASE_URL', '')
        model = os.getenv('LLM_MODEL', '')
        
        # 验证配置格式
        if api_key:
            self.assertIsInstance(api_key, str)
            self.assertGreater(len(api_key), 10)  # API密钥应该足够长
        
        if base_url:
            self.assertIsInstance(base_url, str)
            self.assertTrue(base_url.startswith('http'))  # 应该是HTTP URL
        
        if model:
            self.assertIsInstance(model, str)
            self.assertGreater(len(model), 0)
    
    def test_config_integration(self):
        """测试配置与处理器的集成"""
        api_key = os.getenv('LLM_API_KEY', '')
        base_url = os.getenv('LLM_BASE_URL', '')
        model = os.getenv('LLM_MODEL', '')
        
        if not (api_key and base_url and model):
            self.skipTest("未配置完整的API信息")
        
        try:
            # 创建客户端并验证配置
            client = OpenAICompatibleClient()
            self.assertEqual(client.api_key, api_key)
            self.assertEqual(client.base_url, base_url)
            self.assertEqual(client.model, model)
            
            print(f"配置集成测试通过，模型: {model}")
            
        except Exception as e:
            self.fail(f"配置集成测试失败: {e}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)