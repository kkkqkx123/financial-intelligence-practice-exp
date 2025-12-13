"""
数据解析器测试文件
测试DataParser类的各种功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.processors.data_parser import DataParser

def test_data_parser_creation():
    """测试数据解析器创建"""
    print("测试数据解析器创建...")
    try:
        parser = DataParser()
        print("✓ 数据解析器创建成功")
        return True
    except Exception as e:
        print(f"✗ 数据解析器创建失败: {e}")
        return False

def test_company_data_parsing():
    """测试公司数据解析"""
    print("\n测试公司数据解析...")
    try:
        parser = DataParser()
        
        # 测试字典格式数据解析
        test_dict_data = [
            {
                'short_name': '腾讯',
                'full_name': '腾讯控股有限公司',
                'description': '中国领先的互联网增值服务提供商',
                'registration_name': '深圳市腾讯计算机系统有限公司',
                'address': '深圳市南山区',
                'registration_id': '123456789012345',
                'establish_date': '1998-11-11',
                'legal_representative': '马化腾',
                'registered_capital': '1000万人民币',
                'credit_code': '91440300708461185T',
                'website': 'https://www.tencent.com'
            },
            {
                'name': '阿里巴巴',
                'description': '全球领先的电子商务公司',
                'contact_info': {'address': '杭州市余杭区'},
                'registration_date': '1999-09-09'
            }
        ]
        
        result = parser.parse_companies(test_dict_data)
        
        if isinstance(result, list) and len(result) == 2:
            print("✓ 公司数据解析功能正常")
            return True
        else:
            print(f"✗ 公司数据解析返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 公司数据解析测试失败: {e}")
        return False

def test_investment_event_parsing():
    """测试投资事件解析"""
    print("\n测试投资事件解析...")
    try:
        parser = DataParser()
        
        # 测试字典格式数据解析
        test_dict_data = [
            {
                'description': '腾讯投资字节跳动10亿美元',
                'investors': ['腾讯', '红杉资本'],
                'investee': '字节跳动',
                'investment_date': '2020-01-01',
                'round': 'C轮',
                'amount': '10亿美元'
            },
            {
                'event': '阿里巴巴收购饿了么',
                'company': '饿了么',
                'date': '2018-04-02',
                'funding_round': '并购',
                'funding_amount': '95亿美元'
            }
        ]
        
        result = parser.parse_investment_events(test_dict_data)
        
        if isinstance(result, list) and len(result) == 2:
            print("✓ 投资事件解析功能正常")
            return True
        else:
            print(f"✗ 投资事件解析返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 投资事件解析测试失败: {e}")
        return False

def test_investment_institution_parsing():
    """测试投资机构解析"""
    print("\n测试投资机构解析...")
    try:
        parser = DataParser()
        
        # 测试字典格式数据解析
        test_dict_data = [
            {
                'name': 'IDG资本',
                'description': '全球领先的投资机构',
                'industries': ['科技', '消费', '医疗'],
                'scale': '100亿美元',
                'preferred_rounds': ['A轮', 'B轮', 'C轮']
            },
            {
                'institution_name': '红杉资本',
                'introduction': '顶级风险投资机构',
                'sectors': ['互联网', '人工智能'],
                'fund_size': '80亿美元',
                'investment_stages': ['种子轮', '天使轮', 'A轮']
            }
        ]
        
        result = parser.parse_investment_institutions(test_dict_data)
        
        if isinstance(result, list) and len(result) == 2:
            print("✓ 投资机构解析功能正常")
            return True
        else:
            print(f"✗ 投资机构解析返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ 投资机构解析测试失败: {e}")
        return False

def test_date_parsing():
    """测试日期解析"""
    print("\n测试日期解析...")
    try:
        parser = DataParser()
        
        # 测试各种日期格式
        test_dates = [
            '2020-01-01',
            '2020/01/01',
            '2020年1月1日',
            '2020.01.01',
            '2020-13-01',  # 无效日期
            '',  # 空日期
            'invalid_date'  # 无效格式
        ]
        
        results = []
        for date_str in test_dates:
            result = parser._parse_date(date_str)
            results.append(result)
        
        # 检查前4个应该成功解析，后3个应该返回None
        # 注意：实际返回的格式可能是 2020-1-1 或 2020-01-01，都是有效的
        expected = ['2020-01-01', '2020-01-01', '2020-1-1', '2020-01-01', None, None, None]
        
        if results == expected:
            print("✓ 日期解析功能正常")
            return True
        else:
            print(f"✗ 日期解析结果异常: {results}")
            return False
            
    except Exception as e:
        print(f"✗ 日期解析测试失败: {e}")
        return False

def test_capital_normalization():
    """测试注册资本标准化"""
    print("\n测试注册资本标准化...")
    try:
        parser = DataParser()
        
        # 测试各种资本格式
        test_capitals = [
            '1000万人民币',
            '500万美元',
            '1亿人民币',
            '100万',
            '500',
            '',  # 空值
            'invalid'  # 无效格式
        ]
        
        results = []
        for capital in test_capitals:
            result = parser._normalize_capital(capital)
            results.append(result)
        
        # 检查是否返回数值或None
        valid_results = [r for r in results if r is None or isinstance(r, (int, float))]
        
        if len(valid_results) == len(test_capitals):
            print("✓ 注册资本标准化功能正常")
            return True
        else:
            print(f"✗ 注册资本标准化结果异常: {results}")
            return False
            
    except Exception as e:
        print(f"✗ 注册资本标准化测试失败: {e}")
        return False

def test_amount_normalization():
    """测试投资金额标准化"""
    print("\n测试投资金额标准化...")
    try:
        parser = DataParser()
        
        # 测试各种金额格式
        test_amounts = [
            '1000万人民币',
            '500万美元',
            '数千万人民币',
            '数百万美元',
            '数十万人民币',
            '未披露',
            '',  # 空值
            'invalid'  # 无效格式
        ]
        
        results = []
        for amount in test_amounts:
            result = parser._normalize_amount(amount)
            results.append(result)
        
        # 检查是否返回数值或None
        valid_results = [r for r in results if r is None or isinstance(r, (int, float))]
        
        if len(valid_results) == len(test_amounts):
            print("✓ 投资金额标准化功能正常")
            return True
        else:
            print(f"✗ 投资金额标准化结果异常: {results}")
            return False
            
    except Exception as e:
        print(f"✗ 投资金额标准化测试失败: {e}")
        return False

def test_round_normalization():
    """测试投资轮次标准化"""
    print("\n测试投资轮次标准化...")
    try:
        parser = DataParser()
        
        # 测试各种轮次格式
        test_rounds = [
            'A轮',
            'B+轮',
            'Pre-A',
            '种子轮',
            '天使轮',
            '',  # 空值
            '未知轮次'  # 未知轮次
        ]
        
        results = []
        for round_str in test_rounds:
            result = parser._normalize_round(round_str)
            results.append(result)
        
        # 检查是否返回字符串或None
        valid_results = [r for r in results if r is None or isinstance(r, str)]
        
        if len(valid_results) == len(test_rounds):
            print("✓ 投资轮次标准化功能正常")
            return True
        else:
            print(f"✗ 投资轮次标准化结果异常: {results}")
            return False
            
    except Exception as e:
        print(f"✗ 投资轮次标准化测试失败: {e}")
        return False

def test_investor_parsing():
    """测试投资方解析"""
    print("\n测试投资方解析...")
    try:
        parser = DataParser()
        
        # 测试各种投资方格式
        test_investors = [
            '腾讯、红杉资本、IDG资本',
            '阿里巴巴, 软银, 云峰基金',
            '腾讯 红杉 IDG',
            '腾讯投资；红杉资本；IDG资本',
            '',  # 空值
            '单个投资方'
        ]
        
        results = []
        for investors_str in test_investors:
            result = parser._parse_investors(investors_str)
            results.append(result)
        
        # 检查是否返回列表
        valid_results = [r for r in results if isinstance(r, list)]
        
        if len(valid_results) == len(test_investors):
            print("✓ 投资方解析功能正常")
            return True
        else:
            print(f"✗ 投资方解析结果异常: {results}")
            return False
            
    except Exception as e:
        print(f"✗ 投资方解析测试失败: {e}")
        return False

def test_stats_functionality():
    """测试统计功能"""
    print("\n测试统计功能...")
    try:
        parser = DataParser()
        
        # 先解析一些数据
        test_data = [
            {'name': '测试公司1', 'description': '测试描述1'},
            {'name': '测试公司2', 'description': '测试描述2'}
        ]
        
        parser.parse_companies(test_data)
        
        # 获取统计信息
        stats = parser.get_stats()
        
        required_keys = ['total_records', 'companies_processed', 'events_processed', 'investors_processed', 'errors']
        
        if isinstance(stats, dict) and all(key in stats for key in required_keys):
            print("✓ 统计功能正常")
            return True
        else:
            print(f"✗ 统计功能返回结果异常: {stats}")
            return False
            
    except Exception as e:
        print(f"✗ 统计功能测试失败: {e}")
        return False

def test_csv_parsing():
    """测试CSV解析"""
    print("\n测试CSV解析...")
    try:
        parser = DataParser()
        
        # 创建测试CSV内容
        csv_content = """公司名称,行业,成立时间,注册资本
腾讯,互联网,1998-11-11,1000万人民币
阿里巴巴,电商,1999-09-09,500万人民币
百度,搜索,2000-01-01,200万人民币"""
        
        result = parser.parse_csv_data(csv_content)
        
        if isinstance(result, list) and len(result) == 3:
            print("✓ CSV解析功能正常")
            return True
        else:
            print(f"✗ CSV解析返回结果异常: {result}")
            return False
            
    except Exception as e:
        print(f"✗ CSV解析测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行数据解析器测试")
    print("=" * 50)
    
    tests = [
        test_data_parser_creation,
        test_company_data_parsing,
        test_investment_event_parsing,
        test_investment_institution_parsing,
        test_date_parsing,
        test_capital_normalization,
        test_amount_normalization,
        test_round_normalization,
        test_investor_parsing,
        test_stats_functionality,
        test_csv_parsing
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