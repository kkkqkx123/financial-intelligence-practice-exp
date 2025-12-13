"""
实体匹配器测试文件
测试EntityMatcher类的各种匹配功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.processors.entity_matcher import EntityMatcher

def test_entity_matcher_creation():
    """测试实体匹配器创建"""
    print("测试实体匹配器创建...")
    try:
        matcher = EntityMatcher()
        print("✓ 实体匹配器创建成功")
        return True
    except Exception as e:
        print(f"✗ 实体匹配器创建失败: {e}")
        return False

def test_company_name_normalization():
    """测试公司名称标准化"""
    print("\n测试公司名称标准化...")
    try:
        matcher = EntityMatcher()
        
        # 测试各种公司名称格式
        test_cases = [
            ('深圳市腾讯计算机系统有限公司', '腾讯'),
            ('腾讯控股有限公司', '腾讯'),
            ('阿里巴巴集团控股有限公司', '阿里巴巴'),
            ('百度在线网络技术（北京）有限公司', '百度'),
            ('字节跳动科技有限公司', '字节跳动'),
            ('美团点评有限公司', '美团'),
            ('滴滴出行科技有限公司', '滴滴'),
            ('京东集团股份有限公司', '京东'),
            ('小米科技有限责任公司', '小米'),
            ('华为技术有限公司', '华为')
        ]
        
        for input_name, expected_short in test_cases:
            result = matcher.normalize_company_name(input_name)
            if result != expected_short:
                print(f"✗ 公司名称标准化失败: {input_name} -> {result} (期望: {expected_short})")
                return False
        
        print("✓ 公司名称标准化功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 公司名称标准化测试失败: {e}")
        return False

def test_investor_name_normalization():
    """测试投资方名称标准化"""
    print("\n测试投资方名称标准化...")
    try:
        matcher = EntityMatcher()
        
        # 测试各种投资方名称格式
        test_cases = [
            ('深圳市腾讯产业投资基金有限公司', '腾讯'),
            ('红杉资本中国基金', '红杉资本'),
            ('IDG资本投资顾问（北京）有限公司', 'IDG资本'),
            ('经纬创投（北京）投资管理顾问有限公司', '经纬创投'),
            ('高瓴资本管理有限公司', '高瓴资本'),
            ('真格基金（北京）投资管理有限公司', '真格基金'),
            ('创新工场（北京）企业管理股份有限公司', '创新工场'),
            ('金沙江创业投资管理有限公司', '金沙江创投'),
            ('启明维创投资咨询（上海）有限公司', '启明创投'),
            ('晨兴资本（中国）投资有限公司', '晨兴资本')
        ]
        
        for input_name, expected_short in test_cases:
            result = matcher.normalize_investor_name(input_name)
            if result != expected_short:
                print(f"✗ 投资方名称标准化失败: {input_name} -> {result} (期望: {expected_short})")
                return False
        
        print("✓ 投资方名称标准化功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 投资方名称标准化测试失败: {e}")
        return False

def test_exact_matching():
    """测试精确匹配"""
    print("\n测试精确匹配...")
    try:
        matcher = EntityMatcher()
        
        # 创建测试实体列表
        entities = [
            {'name': '腾讯控股有限公司', 'short_name': '腾讯'},
            {'name': '阿里巴巴集团控股有限公司', 'short_name': '阿里巴巴'},
            {'name': '百度在线网络技术（北京）有限公司', 'short_name': '百度'},
            {'name': '字节跳动科技有限公司', 'short_name': '字节跳动'},
            {'name': '美团点评有限公司', 'short_name': '美团'}
        ]
        
        # 测试精确匹配
        test_queries = [
            ('腾讯控股有限公司', '腾讯'),
            ('阿里巴巴集团控股有限公司', '阿里巴巴'),
            ('百度在线网络技术（北京）有限公司', '百度'),
            ('字节跳动科技有限公司', '字节跳动'),
            ('美团点评有限公司', '美团'),
            ('不存在的公司', None)  # 应该返回None
        ]
        
        for query, expected_match in test_queries:
            result = matcher.find_exact_match(query, entities)
            if result != expected_match:
                print(f"✗ 精确匹配失败: {query} -> {result} (期望: {expected_match})")
                return False
        
        print("✓ 精确匹配功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 精确匹配测试失败: {e}")
        return False

def test_fuzzy_matching():
    """测试模糊匹配"""
    print("\n测试模糊匹配...")
    try:
        matcher = EntityMatcher()
        
        # 创建测试实体列表
        entities = [
            {'name': '腾讯控股有限公司', 'short_name': '腾讯'},
            {'name': '阿里巴巴集团控股有限公司', 'short_name': '阿里巴巴'},
            {'name': '百度在线网络技术（北京）有限公司', 'short_name': '百度'},
            {'name': '字节跳动科技有限公司', 'short_name': '字节跳动'},
            {'name': '美团点评有限公司', 'short_name': '美团'}
        ]
        
        # 测试模糊匹配
        test_queries = [
            ('腾讯控股', '腾讯'),  # 部分匹配
            ('阿里巴巴', '阿里巴巴'),  # 完全匹配
            ('百度公司', '百度'),  # 添加后缀
            ('字节跳动科技', '字节跳动'),  # 部分匹配
            ('美团点评', '美团'),  # 简称匹配
            ('不相关的公司', None)  # 应该返回None
        ]
        
        for query, expected_match in test_queries:
            result = matcher.find_fuzzy_match(query, entities)
            if result != expected_match:
                print(f"✗ 模糊匹配失败: {query} -> {result} (期望: {expected_match})")
                return False
        
        print("✓ 模糊匹配功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 模糊匹配测试失败: {e}")
        return False

def test_similarity_calculation():
    """测试相似度计算"""
    print("\n测试相似度计算...")
    try:
        matcher = EntityMatcher()
        
        # 测试相似度计算
        test_pairs = [
            ('腾讯控股有限公司', '腾讯控股', 0.8),  # 高相似度
            ('阿里巴巴集团', '阿里巴巴', 0.8),  # 高相似度
            ('百度在线网络', '百度', 0.6),  # 中等相似度
            ('字节跳动科技', '字节跳动', 0.7),  # 中等相似度
            ('美团点评', '美团', 0.7),  # 中等相似度
            ('腾讯', '阿里巴巴', 0.1),  # 低相似度
            ('完全不相关的公司', '腾讯', 0.0)  # 极低相似度
        ]
        
        for name1, name2, min_similarity in test_pairs:
            similarity = matcher.calculate_similarity(name1, name2)
            if similarity < min_similarity:
                print(f"✗ 相似度计算失败: {name1} vs {name2} = {similarity} (期望 >= {min_similarity})")
                return False
        
        print("✓ 相似度计算功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 相似度计算测试失败: {e}")
        return False

def test_entity_matching_pipeline():
    """测试实体匹配管道"""
    print("\n测试实体匹配管道...")
    try:
        matcher = EntityMatcher()
        
        # 创建测试实体列表
        entities = [
            {'name': '腾讯控股有限公司', 'short_name': '腾讯', 'type': 'company'},
            {'name': '阿里巴巴集团控股有限公司', 'short_name': '阿里巴巴', 'type': 'company'},
            {'name': '百度在线网络技术（北京）有限公司', 'short_name': '百度', 'type': 'company'},
            {'name': '红杉资本中国基金', 'short_name': '红杉资本', 'type': 'investor'},
            {'name': 'IDG资本投资顾问（北京）有限公司', 'short_name': 'IDG资本', 'type': 'investor'}
        ]
        
        # 测试匹配管道
        test_queries = [
            ('深圳市腾讯计算机系统有限公司', '腾讯'),
            ('腾讯投资', '腾讯'),
            ('阿里巴巴集团', '阿里巴巴'),
            ('红杉资本中国', '红杉资本'),
            ('IDG资本投资', 'IDG资本'),
            ('不存在的实体', None)
        ]
        
        for query, expected_match in test_queries:
            result = matcher.match_entity(query, entities)
            if result != expected_match:
                print(f"✗ 实体匹配管道失败: {query} -> {result} (期望: {expected_match})")
                return False
        
        print("✓ 实体匹配管道功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 实体匹配管道测试失败: {e}")
        return False

def test_name_cleaning():
    """测试名称清理"""
    print("\n测试名称清理...")
    try:
        matcher = EntityMatcher()
        
        # 测试名称清理
        test_cases = [
            ('深圳市腾讯计算机系统有限公司', '腾讯计算机系统有限公司'),
            ('腾讯控股有限公司（香港）', '腾讯控股有限公司'),
            ('阿里巴巴（中国）有限公司', '阿里巴巴有限公司'),
            ('百度在线网络技术（北京）有限公司', '百度在线网络技术有限公司'),
            ('字节跳动科技有限公司', '字节跳动科技有限公司'),
            ('  腾讯控股有限公司  ', '腾讯控股有限公司'),  # 去除空格
            ('腾讯控股有限公司123', '腾讯控股有限公司')  # 去除数字
        ]
        
        for input_name, expected_cleaned in test_cases:
            result = matcher._clean_name(input_name)
            if result != expected_cleaned:
                print(f"✗ 名称清理失败: {input_name} -> {result} (期望: {expected_cleaned})")
                return False
        
        print("✓ 名称清理功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 名称清理测试失败: {e}")
        return False

def test_keyword_extraction():
    """测试关键词提取"""
    print("\n测试关键词提取...")
    try:
        matcher = EntityMatcher()
        
        # 测试关键词提取
        test_cases = [
            ('深圳市腾讯计算机系统有限公司', ['腾讯', '计算机', '系统']),
            ('腾讯控股有限公司', ['腾讯', '控股']),
            ('阿里巴巴集团控股有限公司', ['阿里巴巴', '集团', '控股']),
            ('百度在线网络技术（北京）有限公司', ['百度', '在线', '网络', '技术']),
            ('字节跳动科技有限公司', ['字节', '跳动', '科技']),
            ('美团点评有限公司', ['美团', '点评'])
        ]
        
        for input_name, expected_keywords in test_cases:
            result = matcher._extract_keywords(input_name)
            # 检查是否包含期望的关键词
            if not all(keyword in result for keyword in expected_keywords):
                print(f"✗ 关键词提取失败: {input_name} -> {result} (期望包含: {expected_keywords})")
                return False
        
        print("✓ 关键词提取功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 关键词提取测试失败: {e}")
        return False

def test_entity_type_detection():
    """测试实体类型检测"""
    print("\n测试实体类型检测...")
    try:
        matcher = EntityMatcher()
        
        # 测试实体类型检测
        test_cases = [
            ('深圳市腾讯计算机系统有限公司', 'company'),
            ('腾讯控股有限公司', 'company'),
            ('红杉资本中国基金', 'investor'),
            ('IDG资本投资顾问（北京）有限公司', 'investor'),
            ('百度在线网络技术（北京）有限公司', 'company'),
            ('创新工场（北京）企业管理股份有限公司', 'investor'),
            ('美团点评有限公司', 'company')
        ]
        
        for input_name, expected_type in test_cases:
            result = matcher._detect_entity_type(input_name)
            if result != expected_type:
                print(f"✗ 实体类型检测失败: {input_name} -> {result} (期望: {expected_type})")
                return False
        
        print("✓ 实体类型检测功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 实体类型检测测试失败: {e}")
        return False

def test_matching_confidence():
    """测试匹配置信度"""
    print("\n测试匹配置信度...")
    try:
        matcher = EntityMatcher()
        
        # 创建测试实体列表
        entities = [
            {'name': '腾讯控股有限公司', 'short_name': '腾讯'},
            {'name': '阿里巴巴集团控股有限公司', 'short_name': '阿里巴巴'}
        ]
        
        # 测试匹配置信度
        test_queries = [
            ('腾讯控股有限公司', 0.95),  # 高置信度
            ('腾讯控股', 0.8),  # 中等置信度
            ('腾讯', 0.9),  # 高置信度
            ('深圳市腾讯计算机系统有限公司', 0.7),  # 中等置信度
            ('阿里巴巴', 0.9),  # 高置信度
            ('不相关的公司', 0.1)  # 低置信度
        ]
        
        for query, min_confidence in test_queries:
            confidence = matcher.calculate_matching_confidence(query, entities)
            if confidence < min_confidence:
                print(f"✗ 匹配置信度计算失败: {query} = {confidence} (期望 >= {min_confidence})")
                return False
        
        print("✓ 匹配置信度功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 匹配置信度测试失败: {e}")
        return False

def test_matching_performance():
    """测试匹配性能"""
    print("\n测试匹配性能...")
    try:
        matcher = EntityMatcher()
        
        # 创建大量测试实体
        entities = []
        for i in range(100):
            entities.append({
                'name': f'测试公司{i}有限公司',
                'short_name': f'测试公司{i}'
            })
        
        # 测试匹配性能
        import time
        start_time = time.time()
        
        for i in range(10):
            query = f'测试公司{i}有限公司'
            result = matcher.match_entity(query, entities)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if execution_time < 5.0:  # 5秒内完成
            print(f"✓ 匹配性能良好: {execution_time:.2f}秒")
            return True
        else:
            print(f"✗ 匹配性能较差: {execution_time:.2f}秒")
            return False
        
    except Exception as e:
        print(f"✗ 匹配性能测试失败: {e}")
        return False

def test_stats_functionality():
    """测试统计功能"""
    print("\n测试统计功能...")
    try:
        matcher = EntityMatcher()
        
        # 先进行一些匹配操作
        entities = [
            {'name': '腾讯控股有限公司', 'short_name': '腾讯'},
            {'name': '阿里巴巴集团控股有限公司', 'short_name': '阿里巴巴'}
        ]
        
        matcher.match_entity('腾讯', entities)
        matcher.match_entity('阿里巴巴', entities)
        
        # 获取统计信息
        stats = matcher.get_stats()
        
        required_keys = [
            'total_entities_matched', 'exact_matches', 'fuzzy_matches',
            'failed_matches', 'average_similarity', 'matching_performance'
        ]
        
        if isinstance(stats, dict) and all(key in stats for key in required_keys):
            print("✓ 统计功能正常")
            return True
        else:
            print(f"✗ 统计功能返回结果异常: {stats}")
            return False
            
    except Exception as e:
        print(f"✗ 统计功能测试失败: {e}")
        return False

def test_matching_rules():
    """测试匹配规则"""
    print("\n测试匹配规则...")
    try:
        matcher = EntityMatcher()
        
        # 获取匹配规则
        rules = matcher.get_matching_rules()
        
        required_keys = [
            'company_keywords', 'investor_keywords', 'similarity_threshold',
            'fuzzy_matching_enabled', 'name_cleaning_rules'
        ]
        
        if isinstance(rules, dict) and all(key in rules for key in required_keys):
            print("✓ 匹配规则获取成功")
            return True
        else:
            print(f"✗ 匹配规则获取失败: {rules}")
            return False
        
    except Exception as e:
        print(f"✗ 匹配规则测试失败: {e}")
        return False

def test_custom_matching_rules():
    """测试自定义匹配规则"""
    print("\n测试自定义匹配规则...")
    try:
        matcher = EntityMatcher()
        
        # 设置自定义匹配规则
        custom_rules = {
            'similarity_threshold': 0.6,
            'company_keywords': ['科技', '技术', '有限公司'],
            'investor_keywords': ['资本', '基金', '投资'],
            'fuzzy_matching_enabled': True
        }
        
        matcher.set_matching_rules(custom_rules)
        
        # 测试自定义规则
        entities = [
            {'name': '腾讯控股有限公司', 'short_name': '腾讯'},
            {'name': '红杉资本中国基金', 'short_name': '红杉资本'}
        ]
        
        result1 = matcher.match_entity('腾讯科技', entities)
        result2 = matcher.match_entity('红杉基金', entities)
        
        if result1 and result2:
            print("✓ 自定义匹配规则生效")
            return True
        else:
            print("✗ 自定义匹配规则未生效")
            return False
            
    except Exception as e:
        print(f"✗ 自定义匹配规则测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行实体匹配器测试")
    print("=" * 50)
    
    tests = [
        test_entity_matcher_creation,
        test_company_name_normalization,
        test_investor_name_normalization,
        test_exact_matching,
        test_fuzzy_matching,
        test_similarity_calculation,
        test_entity_matching_pipeline,
        test_name_cleaning,
        test_keyword_extraction,
        test_entity_type_detection,
        test_matching_confidence,
        test_matching_performance,
        test_stats_functionality,
        test_matching_rules,
        test_custom_matching_rules
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