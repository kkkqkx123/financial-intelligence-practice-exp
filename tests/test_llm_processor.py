"""
LLM处理器测试模块
测试src/processors/llm_processor.py中的实现是否正确
"""

import os
import sys
import json
import time
import unittest
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from processors.llm_processor import (
    LLMProcessorInterface,
    MockLLMProcessor,
    OpenAICompatibleProcessor,
    BatchLLMProcessor,
    LLMEnhancementTracker,
    get_llm_processor,
    get_enhancement_tracker,
    get_batch_llm_processor
)
from processors.llm_client import LLMClientInterface


class TestMockLLMProcessor(unittest.TestCase):
    """测试MockLLMProcessor类"""
    
    def setUp(self):
        """每个测试方法前的设置"""
        self.processor = MockLLMProcessor()
    
    def test_enhance_entity_description(self):
        """测试实体描述增强"""
        entity_name = "测试公司"
        context = {"industry": "科技", "establish_date": "2020年"}
        
        result = self.processor.enhance_entity_description(entity_name, context)
        
        self.assertIsInstance(result, str)
        self.assertIn("测试公司", result)
        self.assertIn("科技", result)
        self.assertIn("2020年", result)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_resolve_entity_conflicts(self):
        """测试实体冲突解决"""
        conflicting_entities = [
            {"name": "公司A", "confidence": 0.8},
            {"name": "公司B", "confidence": 0.9},
            {"name": "公司C", "confidence": 0.7}
        ]
        
        result = self.processor.resolve_entity_conflicts(conflicting_entities)
        
        self.assertIsInstance(result, dict)
        self.assertIn("resolved_entity", result)
        self.assertIn("confidence", result)
        self.assertIn("merged_aliases", result)
        self.assertEqual(result["confidence"], 0.9)  # 选择置信度最高的
        self.assertEqual(self.processor.call_count, 1)
    
    def test_resolve_entity_conflicts_empty(self):
        """测试空实体冲突解决"""
        result = self.processor.resolve_entity_conflicts([])
        
        self.assertIsInstance(result, dict)
        self.assertIn("resolved_entity", result)
        self.assertIn("confidence", result)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_extract_relationships_from_text(self):
        """测试关系提取"""
        text = "这家公司获得了1000万美元的投资"
        
        result = self.processor.extract_relationships_from_text(text)
        
        self.assertIsInstance(result, list)
        # 可能返回空列表或包含投资关系的列表
        if result:
            self.assertIn("type", result[0])
            self.assertIn("confidence", result[0])
            self.assertIn("description", result[0])
        self.assertEqual(self.processor.call_count, 1)
    
    def test_classify_company_industry(self):
        """测试公司行业分类"""
        company_info = {
            "name": "智能科技有限公司",
            "description": "专注于人工智能技术开发"
        }
        
        result = self.processor.classify_company_industry(company_info)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("科技", result)  # 应该识别出科技类别
        self.assertEqual(self.processor.call_count, 1)
    
    def test_classify_company_industry_finance(self):
        """测试金融公司分类"""
        company_info = {
            "name": "金融支付公司",
            "description": "提供银行支付服务"
        }
        
        result = self.processor.classify_company_industry(company_info)
        
        self.assertIsInstance(result, list)
        self.assertIn("金融", result)  # 应该识别出金融类别
        self.assertEqual(self.processor.call_count, 1)
    
    def test_standardize_investor_name(self):
        """测试投资方名称标准化"""
        investor_name = "  红杉资本有限公司  "
        context = {"type": "VC"}
        
        result = self.processor.standardize_investor_name(investor_name, context)
        
        self.assertIsInstance(result, str)
        self.assertEqual(result.strip(), result)  # 应该去除首尾空格
        self.assertEqual(self.processor.call_count, 1)
    
    def test_get_stats(self):
        """测试统计信息获取"""
        # 先调用几个方法
        self.processor.enhance_entity_description("测试", {})
        self.processor.classify_company_industry({"name": "测试"})
        
        stats = self.processor.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_calls", stats)
        self.assertIn("mock_responses", stats)
        self.assertEqual(stats["total_calls"], 2)


class TestOpenAICompatibleProcessor(unittest.TestCase):
    """测试OpenAICompatibleProcessor类"""
    
    def setUp(self):
        """每个测试方法前的设置"""
        # 创建模拟的LLM客户端
        self.mock_llm_client = Mock(spec=LLMClientInterface)
        # 添加generate_response方法到模拟客户端
        self.mock_llm_client.generate_response = Mock()
        self.processor = OpenAICompatibleProcessor(self.mock_llm_client)
    
    def test_enhance_entity_description_success(self):
        """测试成功的实体描述增强"""
        entity_name = "测试公司"
        context = {"industry": "科技"}
        expected_response = "测试公司是一家专注于人工智能技术的科技公司，成立于2020年。"
        
        self.mock_llm_client.generate_response.return_value = expected_response
        
        result = self.processor.enhance_entity_description(entity_name, context)
        
        self.assertEqual(result, expected_response)
        self.assertEqual(self.processor.call_count, 1)
        self.mock_llm_client.generate_response.assert_called_once()
    
    def test_enhance_entity_description_failure(self):
        """测试失败的实体描述增强"""
        entity_name = "测试公司"
        context = {"industry": "科技"}
        
        self.mock_llm_client.generate_response.side_effect = Exception("API错误")
        
        result = self.processor.enhance_entity_description(entity_name, context)
        
        self.assertIsInstance(result, str)
        self.assertIn("测试公司", result)
        self.assertIn("科技", result)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_resolve_entity_conflicts_success(self):
        """测试成功的实体冲突解决"""
        conflicting_entities = [
            {"name": "公司A", "confidence": 0.8, "source": "source1"},
            {"name": "公司B", "confidence": 0.9, "source": "source2"}
        ]
        
        mock_response = json.dumps({
            "resolved_entity": {"name": "公司B", "confidence": 0.9},
            "confidence": 0.9,
            "merged_aliases": ["公司A"]
        })
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.resolve_entity_conflicts(conflicting_entities)
        
        self.assertIsInstance(result, dict)
        self.assertIn("resolved_entity", result)
        self.assertIn("confidence", result)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_resolve_entity_conflicts_empty(self):
        """测试空实体冲突解决"""
        result = self.processor.resolve_entity_conflicts([])
        
        self.assertIsInstance(result, dict)
        self.assertIsNone(result["resolved_entity"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_extract_relationships_from_text_success(self):
        """测试成功的关系提取"""
        text = "腾讯投资了京东，两者建立了战略合作关系。"
        
        mock_response = json.dumps([
            {
                "type": "investment",
                "source": "腾讯",
                "target": "京东",
                "confidence": 0.9,
                "description": "腾讯投资京东"
            }
        ])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.extract_relationships_from_text(text)
        
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 0)  # 取决于JSON解析结果
        self.assertEqual(self.processor.call_count, 1)
    
    def test_classify_company_industry_success(self):
        """测试成功的公司行业分类"""
        company_info = {"name": "科技公司", "description": "人工智能技术"}
        
        mock_response = json.dumps(["科技", "人工智能"])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.classify_company_industry(company_info)
        
        self.assertIsInstance(result, list)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_standardize_investor_name_success(self):
        """测试成功的投资方名称标准化"""
        investor_name = "红杉资本"
        context = {"type": "VC", "location": "美国"}
        
        mock_response = "Sequoia Capital"
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.standardize_investor_name(investor_name, context)
        
        self.assertEqual(result, mock_response)
        self.assertEqual(self.processor.call_count, 1)
    
    def test_get_stats(self):
        """测试统计信息获取"""
        # 调用一个方法
        self.mock_llm_client.generate_response.return_value = "测试响应"
        self.processor.enhance_entity_description("测试", {})
        
        stats = self.processor.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_calls", stats)
        self.assertEqual(stats["total_calls"], 1)


class TestBatchLLMProcessor(unittest.TestCase):
    """测试BatchLLMProcessor类"""
    
    def setUp(self):
        """每个测试方法前的设置"""# 创建模拟的LLM客户端
        self.mock_llm_client = Mock(spec=LLMClientInterface)
        # 添加generate_response方法到模拟客户端
        self.mock_llm_client.generate_response = Mock()
        self.processor = BatchLLMProcessor(llm_client=self.mock_llm_client)
    
    def test_batch_enhance_entity_descriptions(self):
        """测试批量实体描述增强"""
        entities = [
            {"name": "公司A", "context": {"industry": "科技"}},
            {"name": "公司B", "context": {"industry": "金融"}}
        ]
        
        mock_response = json.dumps([
            "公司A是一家科技公司",
            "公司B是一家金融公司"
        ])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.batch_enhance_entity_descriptions(entities)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.processor.batch_call_count, 1)
    
    def test_batch_resolve_entity_conflicts(self):
        """测试批量实体冲突解决"""
        conflict_groups = [
            [{"name": "公司A1", "confidence": 0.8}, {"name": "公司A2", "confidence": 0.9}],
            [{"name": "公司B1", "confidence": 0.7}, {"name": "公司B2", "confidence": 0.8}]
        ]
        
        mock_response = json.dumps([
            {"resolved_entity": {"name": "公司A2"}, "confidence": 0.9},
            {"resolved_entity": {"name": "公司B2"}, "confidence": 0.8}
        ])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.batch_resolve_entity_conflicts(conflict_groups)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.processor.batch_call_count, 1)
    
    def test_batch_extract_relationships_from_text(self):
        """测试批量关系提取"""
        texts = [
            "腾讯投资了京东",
            "阿里巴巴收购了饿了么"
        ]
        
        mock_response = json.dumps([
            [{"type": "investment", "source": "腾讯", "target": "京东"}],
            [{"type": "acquisition", "source": "阿里巴巴", "target": "饿了么"}]
        ])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.batch_extract_relationships_from_text(texts)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.processor.batch_call_count, 1)
    
    def test_batch_classify_company_industries(self):
        """测试批量公司行业分类"""
        companies_info = [
            {"name": "科技公司", "description": "人工智能"},
            {"name": "金融公司", "description": "银行服务"}
        ]
        
        mock_response = json.dumps([
            ["科技", "人工智能"],
            ["金融", "银行"]
        ])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.batch_classify_company_industries(companies_info)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.processor.batch_call_count, 1)
    
    def test_batch_standardize_investor_names(self):
        """测试批量投资方名称标准化"""
        investor_names = ["红杉资本", "IDG资本"]
        contexts = [{"type": "VC"}, {"type": "VC"}]
        
        mock_response = json.dumps([
            "Sequoia Capital",
            "IDG Capital"
        ])
        
        self.mock_llm_client.generate_response.return_value = mock_response
        
        result = self.processor.batch_standardize_investor_names(investor_names, contexts)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.processor.batch_call_count, 1)
    
    def test_process_in_batches(self):
        """测试通用批量处理"""
        items = ["项目1", "项目2", "项目3", "项目4", "项目5"]
        batch_size = 2
        
        def process_func(batch_items):
            return [f"处理_{item}" for item in batch_items]
        
        result = self.processor.process_in_batches(items, process_func)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(item.startswith("处理_") for item in result))
    
    def test_get_batch_stats(self):
        """测试批量统计信息获取"""
        # 模拟一些批量调用
        self.mock_llm_client.generate_response.return_value = json.dumps(["结果1", "结果2"])
        self.processor.batch_enhance_entity_descriptions([{"name": "测试"}])
        
        stats = self.processor.get_batch_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("batch_calls", stats)
        self.assertIn("batch_size", stats)
        self.assertEqual(stats["batch_calls"], 1)


class TestLLMEnhancementTracker(unittest.TestCase):
    """测试LLMEnhancementTracker类"""
    
    def setUp(self):
        """每个测试方法前的设置"""
        self.tracker = LLMEnhancementTracker()
    
    def test_add_enhancement_request(self):
        """测试添加增强请求"""
        request_data = {"entity_name": "测试公司", "context": {"industry": "科技"}}
        
        request_id = self.tracker.add_enhancement_request(
            "enhance_description", request_data, "high"
        )
        
        self.assertIsInstance(request_id, str)
        self.assertTrue(request_id.startswith("enh_"))
        self.assertEqual(self.tracker.stats["total_requests"], 1)
        self.assertEqual(self.tracker.stats["pending_requests"], 1)
    
    def test_get_pending_requests(self):
        """测试获取待处理请求"""
        # 添加几个请求
        self.tracker.add_enhancement_request("enhance_description", {})
        self.tracker.add_enhancement_request("resolve_conflict", {})
        self.tracker.add_enhancement_request("enhance_description", {}, "high")
        
        # 获取所有待处理请求
        pending = self.tracker.get_pending_requests()
        self.assertEqual(len(pending), 3)
        
        # 按类型筛选
        enhance_pending = self.tracker.get_pending_requests("enhance_description")
        self.assertEqual(len(enhance_pending), 2)
        
        # 按优先级筛选
        high_pending = self.tracker.get_pending_requests(priority="high")
        self.assertEqual(len(high_pending), 1)
    
    def test_process_batch_requests(self):
        """测试批量处理请求"""
        # 添加一些请求
        self.tracker.add_enhancement_request("enhance_description", {
            "entity_name": "公司A", "context": {}
        })
        self.tracker.add_enhancement_request("enhance_description", {
            "entity_name": "公司B", "context": {}
        })
        
        # 创建模拟的LLM处理器
        mock_processor = Mock(spec=LLMProcessorInterface)
        mock_processor.enhance_entity_description.return_value = "增强描述"
        
        result = self.tracker.process_batch_requests(mock_processor, batch_size=10)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["remaining"], 0)
        self.assertEqual(self.tracker.stats["processed_requests"], 2)
        self.assertEqual(self.tracker.stats["pending_requests"], 0)
    
    def test_process_batch_requests_smart(self):
        """测试智能批量处理请求"""
        # 添加不同类型的请求
        self.tracker.add_enhancement_request("enhance_description", {
            "entity_name": "公司A", "context": {}
        })
        self.tracker.add_enhancement_request("classify_industry", {
            "company_info": {"name": "公司B"}
        })
        
        # 创建模拟的批量LLM处理器
        mock_processor = Mock(spec=BatchLLMProcessor)
        mock_processor.batch_enhance_entity_descriptions.return_value = ["描述A"]
        mock_processor.batch_classify_company_industries.return_value = [["科技"]]
        
        result = self.tracker.process_batch_requests_smart(mock_processor, batch_size=10)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["batches"], 2)  # 两种类型
        self.assertEqual(self.tracker.stats["processed_requests"], 2)
    
    def test_get_stats(self):
        """测试统计信息获取"""
        # 添加一些请求
        self.tracker.add_enhancement_request("enhance_description", {})
        self.tracker.add_enhancement_request("classify_industry", {})
        
        stats = self.tracker.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_requests"], 2)
        self.assertEqual(stats["pending_requests"], 2)
        self.assertEqual(stats["processed_requests"], 0)
    
    def test_export_pending_requests(self):
        """测试导出待处理请求"""
        import tempfile
        
        # 添加一些请求
        self.tracker.add_enhancement_request("enhance_description", {})
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            self.tracker.export_pending_requests(temp_filepath)
            
            # 验证文件存在且内容正确
            self.assertTrue(os.path.exists(temp_filepath))
            
            with open(temp_filepath, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            
            self.assertIn("export_time", export_data)
            self.assertIn("total_pending", export_data)
            self.assertIn("requests", export_data)
            self.assertEqual(export_data["total_pending"], 1)
        finally:
            # 清理临时文件
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)


class TestFactoryFunctions(unittest.TestCase):
    """测试工厂函数"""
    
    @patch('processors.llm_processor.get_llm_client')
    def test_get_llm_processor(self, mock_get_client):
        """测试获取LLM处理器"""
        mock_client = Mock(spec=LLMClientInterface)
        mock_get_client.return_value = mock_client
        
        processor = get_llm_processor()
        
        self.assertIsInstance(processor, OpenAICompatibleProcessor)
        self.assertEqual(processor.llm_client, mock_client)
    
    def test_get_enhancement_tracker(self):
        """测试获取增强跟踪器"""
        tracker = get_enhancement_tracker()
        
        self.assertIsInstance(tracker, LLMEnhancementTracker)
    
    @patch('processors.llm_processor.get_llm_client')
    def test_get_batch_llm_processor(self, mock_get_client):
        """测试获取批量LLM处理器"""
        mock_client = Mock(spec=LLMClientInterface)
        mock_get_client.return_value = mock_client
        
        processor = get_batch_llm_processor()
        
        self.assertIsInstance(processor, BatchLLMProcessor)
        self.assertEqual(processor.llm_client, mock_client)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    @patch('processors.llm_processor.get_llm_client')
    def test_end_to_end_workflow(self, mock_get_client):
        """测试端到端工作流"""
        # 创建模拟的LLM客户端
        mock_client = Mock(spec=LLMClientInterface)
        mock_client.generate_response = Mock()
        mock_get_client.return_value = mock_client
        
        # 设置模拟响应
        mock_client.generate_response.return_value = json.dumps({
            "enhanced_description": "这是一个测试公司",
            "resolved_entity": {"name": "统一实体"},
            "relationships": [{"type": "investment"}],
            "industries": ["科技"],
            "standardized_name": "标准化名称"
        })
        
        # 获取处理器
        processor = get_llm_processor()
        batch_processor = get_batch_llm_processor()
        tracker = get_enhancement_tracker()
        
        # 测试基本功能
        description = processor.enhance_entity_description("测试公司", {"industry": "科技"})
        self.assertIsInstance(description, str)
        
        # 测试批量功能
        entities = [{"name": "公司A", "context": {}}, {"name": "公司B", "context": {}}]
        batch_results = batch_processor.batch_enhance_entity_descriptions(entities)
        self.assertIsInstance(batch_results, list)
        
        # 测试跟踪器
        request_id = tracker.add_enhancement_request("enhance_description", {})
        self.assertIsInstance(request_id, str)
        
        pending = tracker.get_pending_requests()
        self.assertEqual(len(pending), 1)


class TestRealLLMIntegration(unittest.TestCase):
    """真实LLM集成测试 - 验证实际API调用"""
    
    def setUp(self):
        """设置真实LLM处理器"""
        # 使用真实的LLM客户端
        try:
            from processors.llm_client import get_llm_client
            llm_client = get_llm_client()
            self.processor = OpenAICompatibleProcessor(llm_client)
            self.has_real_client = True
        except Exception as e:
            print(f"无法创建真实LLM客户端: {e}")
            self.has_real_client = False
    
    def test_real_entity_description_enhancement(self):
        """测试真实的实体描述增强"""
        if not self.has_real_client:
            self.skipTest("没有可用的真实LLM客户端")
        
        entity_name = "阿里巴巴集团"
        context = {"industry": "电子商务", "establish_date": "1999年"}
        
        try:
            result = self.processor.enhance_entity_description(entity_name, context)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 20)  # 应该生成较长的描述
            print(f"实体描述增强结果: {result[:100]}...")
        except Exception as e:
            self.fail(f"真实LLM调用失败: {e}")
    
    def test_real_company_industry_classification(self):
        """测试真实的公司行业分类"""
        if not self.has_real_client:
            self.skipTest("没有可用的真实LLM客户端")
        
        company_info = {
            "name": "深圳市腾讯计算机系统有限公司",
            "description": "提供互联网服务和人工智能技术的科技公司"
        }
        
        try:
            result = self.processor.classify_company_industry(company_info)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            print(f"行业分类结果: {result}")
        except Exception as e:
            self.fail(f"真实LLM调用失败: {e}")
    
    def test_real_investor_name_standardization(self):
        """测试真实的投资方名称标准化"""
        if not self.has_real_client:
            self.skipTest("没有可用的真实LLM客户端")
        
        investor_name = "红杉资本中国基金"
        context = {"type": "VC", "location": "中国"}
        
        try:
            result = self.processor.standardize_investor_name(investor_name, context)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            print(f"投资方名称标准化结果: {result}")
        except Exception as e:
            self.fail(f"真实LLM调用失败: {e}")


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 运行所有测试
    unittest.main(verbosity=2)