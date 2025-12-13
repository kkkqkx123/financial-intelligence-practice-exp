"""
批处理优化器测试文件
测试BatchOptimizer类的各种功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.processors.batch_optimizer import BatchOptimizer

def test_batch_optimizer_creation():
    """测试批处理优化器创建"""
    print("测试批处理优化器创建...")
    try:
        optimizer = BatchOptimizer()
        print("✓ 批处理优化器创建成功")
        return True
    except Exception as e:
        print(f"✗ 批处理优化器创建失败: {e}")
        return False

def test_entity_description_optimization():
    """测试实体描述优化"""
    print("\n测试实体描述优化...")
    try:
        optimizer = BatchOptimizer()
        
        # 创建测试实体数据
        test_entities = {
            'company_1': {
                'name': '测试公司1',
                'description': '',  # 空描述，需要增强
                'industry': '科技',
                'establish_date': '2020-01-01',
                'metadata': {'confidence': 0.3}
            },
            'company_2': {
                'name': '测试公司2',
                'description': '这是一家很好的公司',  # 已有描述，不需要增强
                'industry': '金融',
                'establish_date': '2019-01-01',
                'metadata': {'confidence': 0.8}
            }
        }
        
        # 测试描述优化
        result = optimizer.optimize_entity_descriptions(test_entities, 'company')
        
        if isinstance(result, dict) and len(result) == 2:
            print("✓ 实体描述优化功能正常")
            return True
        else:
            print(f"✗ 实体描述优化返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 实体描述优化测试失败: {e}")
        return False

def test_industry_classification():
    """测试行业分类优化"""
    print("\n测试行业分类优化...")
    try:
        optimizer = BatchOptimizer()
        
        # 创建测试公司数据
        test_companies = {
            'company_1': {
                'name': '测试科技公司',
                'industry': [],  # 空行业，需要分类
                'capital': '1000万人民币',
                'metadata': {'confidence': 0.4}
            },
            'company_2': {
                'name': '测试金融公司',
                'industry': ['金融', '科技'],  # 已有行业，不需要分类
                'capital': '500万人民币',
                'metadata': {'confidence': 0.9}
            }
        }
        
        # 测试行业分类
        result = optimizer.optimize_industry_classification(test_companies)
        
        if isinstance(result, dict) and len(result) == 2:
            print("✓ 行业分类优化功能正常")
            return True
        else:
            print(f"✗ 行业分类优化返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 行业分类优化测试失败: {e}")
        return False

def test_investor_name_standardization():
    """测试投资方名称标准化"""
    print("\n测试投资方名称标准化...")
    try:
        optimizer = BatchOptimizer()
        
        # 创建测试投资方名称
        test_names = {
            '腾讯投资',
            '腾讯资本',
            '阿里巴巴投资',
            '阿里资本',
            'IDG资本',
            'IDG'
        }
        
        # 测试名称标准化
        result = optimizer.optimize_investor_name_standardization(test_names)
        
        if isinstance(result, dict) and len(result) > 0:
            print("✓ 投资方名称标准化功能正常")
            return True
        else:
            print(f"✗ 投资方名称标准化返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 投资方名称标准化测试失败: {e}")
        return False

def test_optimization_stats():
    """测试优化统计功能"""
    print("\n测试优化统计功能...")
    try:
        optimizer = BatchOptimizer()
        
        # 获取统计信息
        stats = optimizer.get_optimization_stats()
        
        required_keys = [
            'entities_processed', 'relationships_processed', 'llm_calls_made',
            'batch_sizes', 'processing_time', 'efficiency_ratio',
            'smart_batches_used', 'traditional_batches_used', 'total_api_calls_saved'
        ]
        
        if isinstance(stats, dict) and all(key in stats for key in required_keys):
            print("✓ 优化统计功能正常")
            return True
        else:
            print(f"✗ 优化统计返回结果异常: {stats}")
            return False
            
    except Exception as e:
        print(f"✗ 优化统计测试失败: {e}")
        return False

def test_relationship_extraction():
    """测试关系提取优化"""
    print("\n测试关系提取优化...")
    try:
        optimizer = BatchOptimizer()
        
        # 创建测试文本源
        text_sources = [
            {
                'text': '腾讯投资了字节跳动，金额为10亿美元',
                'type': 'investment_news',
                'source': 'news'
            },
            {
                'text': '阿里巴巴收购了饿了么，交易金额达到95亿美元',
                'type': 'acquisition_news',
                'source': 'news'
            }
        ]
        
        # 测试关系提取
        result = optimizer.optimize_relationship_extraction(text_sources)
        
        if isinstance(result, list):
            print("✓ 关系提取优化功能正常")
            return True
        else:
            print(f"✗ 关系提取优化返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 关系提取优化测试失败: {e}")
        return False

def test_entity_conflict_resolution():
    """测试实体冲突解决"""
    print("\n测试实体冲突解决...")
    try:
        optimizer = BatchOptimizer()
        
        # 创建测试冲突数据
        conflicts = [
            {
                'id': 'entity_1',
                'name': '腾讯科技',
                'type': 'company',
                'confidence': 0.7
            },
            {
                'id': 'entity_2',
                'name': '腾讯科技有限公司',
                'type': 'company',
                'confidence': 0.8
            }
        ]
        
        # 测试冲突解决
        result = optimizer.optimize_entity_conflicts(conflicts)
        
        if isinstance(result, dict):
            print("✓ 实体冲突解决功能正常")
            return True
        else:
            print(f"✗ 实体冲突解决返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 实体冲突解决测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行批处理优化器测试")
    print("=" * 50)
    
    tests = [
        test_batch_optimizer_creation,
        test_entity_description_optimization,
        test_industry_classification,
        test_investor_name_standardization,
        test_optimization_stats,
        test_relationship_extraction,
        test_entity_conflict_resolution
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)