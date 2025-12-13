"""
数据验证器测试文件
测试DataValidator类的各种验证功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.processors.data_validator import DataValidator

def test_data_validator_creation():
    """测试数据验证器创建"""
    print("测试数据验证器创建...")
    try:
        validator = DataValidator()
        print("✓ 数据验证器创建成功")
        return True
    except Exception as e:
        print(f"✗ 数据验证器创建失败: {e}")
        return False

def test_company_data_validation():
    """测试公司数据验证"""
    print("\n测试公司数据验证...")
    try:
        validator = DataValidator()
        
        # 测试有效的公司数据
        valid_company = {
            '公司名称': '腾讯控股有限公司',
            '公司描述': '中国领先的互联网增值服务提供商',
            '工商注册名称': '深圳市腾讯计算机系统有限公司',
            '公司地址': '深圳市南山区',
            '工商注册id': '123456789012345',
            '成立时间': '1998-11-11',
            '法定代表人': '马化腾',
            '注册资金': '1000万人民币',
            '统一信用代码': '91440300708461185T',
            '网址': 'https://www.tencent.com'
        }
        
        result = validator.validate_company_data([valid_company])
        if result['valid_records'] > 0:
            print("✓ 有效公司数据验证通过")
        else:
            print(f"✗ 有效公司数据验证失败: {result['validation_errors']}")
            return False
        
        # 测试无效的公司数据
        invalid_company = {
            '公司名称': '',  # 空名称
            '公司描述': '',  # 空描述
            '工商注册id': '123',  # 过短的注册号
            '成立时间': '2025-01-01',  # 未来日期
            '统一信用代码': '123',  # 无效的信用代码
            '网址': 'not-a-url'  # 无效的网址
        }
        
        result = validator.validate_company_data([invalid_company])
        if result['invalid_records'] > 0:
            print("✓ 无效公司数据正确拒绝")
        else:
            print("✗ 无效公司数据应该被拒绝")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 公司数据验证测试失败: {e}")
        return False

def test_investment_event_validation():
    """测试投资事件验证"""
    print("\n测试投资事件验证...")
    try:
        validator = DataValidator()
        
        # 测试有效的投资事件
        valid_event = {
            'description': '腾讯投资字节跳动10亿美元',
            '投资方': '腾讯、红杉资本',
            '融资方': '字节跳动',
            '融资时间': '2020-01-01',
            '轮次': 'C轮',
            '金额': '10亿美元'
        }
        
        result = validator.validate_investment_event_data([valid_event])
        if result['valid_records'] > 0:
            print("✓ 有效投资事件验证通过")
        else:
            print(f"✗ 有效投资事件验证失败: {result['validation_errors']}")
            return False
        
        # 测试无效的投资事件
        invalid_event = {
            'description': '',  # 空描述
            '投资方': '',  # 空投资方
            '融资方': '',  # 空融资方
            '融资时间': '2025-01-01',  # 未来日期
            '轮次': 'Z轮',  # 无效轮次
            '金额': 'invalid'  # 无效金额
        }
        
        result = validator.validate_investment_event_data([invalid_event])
        if result['invalid_records'] > 0:
            print("✓ 无效投资事件正确拒绝")
        else:
            print("✗ 无效投资事件应该被拒绝")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 投资事件验证测试失败: {e}")
        return False

def test_investment_institution_validation():
    """测试投资机构验证"""
    print("\n测试投资机构验证...")
    try:
        validator = DataValidator()
        
        # 测试有效的投资机构
        valid_institution = {
            '机构名称': 'IDG资本',
            '介绍': '全球领先的投资机构',
            '行业': ['科技', '消费', '医疗'],
            '规模': '100亿美元',
            '轮次': ['A轮', 'B轮', 'C轮'],
            '成立时间': 1993,
            '总部': '北京',
            '网址': 'https://www.idg.com'
        }
        
        result = validator.validate_investor_data([valid_institution])
        if result['valid_records'] > 0:
            print("✓ 有效投资机构验证通过")
        else:
            print(f"✗ 有效投资机构验证失败: {result['validation_errors']}")
            return False
        
        # 测试无效的投资机构
        invalid_institution = {
            '机构名称': '',  # 空名称
            '介绍': '',  # 空描述
            '行业': [],  # 空行业列表
            '规模': 'invalid',  # 无效规模
            '轮次': ['Z轮'],  # 无效轮次
            '成立时间': 2050,  # 未来年份
            '网址': 'not-a-url'  # 无效网址
        }
        
        result = validator.validate_investor_data([invalid_institution])
        if result['invalid_records'] > 0:
            print("✓ 无效投资机构正确拒绝")
        else:
            print("✗ 无效投资机构应该被拒绝")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 投资机构验证测试失败: {e}")
        return False

def test_name_validation():
    """测试名称验证"""
    print("\n测试名称验证...")
    try:
        validator = DataValidator()
        
        # 测试有效名称
        valid_names = [
            '腾讯控股有限公司',
            '阿里巴巴（中国）有限公司',
            '百度在线网络技术（北京）有限公司',
            '字节跳动科技有限公司',
            '美团点评'
        ]
        
        for name in valid_names:
            result = validator.validate_name(name)
            if not result['is_valid']:
                print(f"✗ 有效名称验证失败: {name}")
                return False
        
        print("✓ 有效名称验证通过")
        
        # 测试无效名称
        invalid_names = [
            '',  # 空名称
            '   ',  # 只有空格
            'a',  # 过短
            'x' * 201,  # 过长
            '腾讯@控股',  # 包含特殊字符
            '腾讯控股有限公司12345'  # 包含数字
        ]
        
        for name in invalid_names:
            result = validator.validate_name(name)
            if result['is_valid']:
                print(f"✗ 无效名称应该被拒绝: {name}")
                return False
        
        print("✓ 无效名称正确拒绝")
        return True
        
    except Exception as e:
        print(f"✗ 名称验证测试失败: {e}")
        return False

def test_date_validation():
    """测试日期验证"""
    print("\n测试日期验证...")
    try:
        validator = DataValidator()
        
        # 测试有效日期
        valid_dates = [
            '1998-11-11',
            '2020-01-01',
            '2023-12-31',
            '2000-02-29',  # 闰年
            '1990-01-01'
        ]
        
        for date_str in valid_dates:
            result = validator.validate_date(date_str)
            if not result['is_valid']:
                print(f"✗ 有效日期验证失败: {date_str}")
                return False
        
        print("✓ 有效日期验证通过")
        
        # 测试无效日期
        invalid_dates = [
            '',  # 空日期
            '2026-01-01',  # 未来日期
            '2023-13-01',  # 无效月份
            '2023-01-32',  # 无效日期
            '2023-02-29',  # 非闰年
            'invalid-date',  # 无效格式
            '2023/01/01'  # 错误格式
        ]
        
        for date_str in invalid_dates:
            result = validator.validate_date(date_str)
            if result['is_valid']:
                print(f"✗ 无效日期应该被拒绝: {date_str}")
                return False
        
        print("✓ 无效日期正确拒绝")
        return True
        
    except Exception as e:
        print(f"✗ 日期验证测试失败: {e}")
        return False

def test_amount_validation():
    """测试金额验证"""
    print("\n测试金额验证...")
    try:
        validator = DataValidator()
        
        # 测试有效金额
        valid_amounts = [
            '1000万人民币',
            '500万美元',
            '1亿人民币',
            '数千万人民币',
            '数百万美元',
            '数十万人民币',
            '100万',
            '50万美元'
        ]
        
        for amount in valid_amounts:
            result = validator.validate_amount(amount)
            if not result['is_valid']:
                print(f"✗ 有效金额验证失败: {amount}")
                return False
        
        print("✓ 有效金额验证通过")
        
        # 测试无效金额
        invalid_amounts = [
            '',  # 空金额
            'invalid',  # 无效格式
            '1000万',  # 缺少货币单位
            '人民币1000万',  # 错误顺序
            '1000万人民',  # 不完整单位
            '1000万日元'  # 不支持货币
        ]
        
        for amount in invalid_amounts:
            result = validator.validate_amount(amount)
            if result['is_valid']:
                print(f"✗ 无效金额应该被拒绝: {amount}")
                return False
        
        print("✓ 无效金额正确拒绝")
        return True
        
    except Exception as e:
        print(f"✗ 金额验证测试失败: {e}")
        return False

def test_round_validation():
    """测试轮次验证"""
    print("\n测试轮次验证...")
    try:
        validator = DataValidator()
        
        # 测试有效轮次
        valid_rounds = [
            '种子轮',
            '天使轮',
            'Pre-A',
            'A轮',
            'A+轮',
            'B轮',
            'C轮',
            'D轮',
            'E轮',
            'F轮',
            'Pre-IPO',
            'IPO',
            '并购',
            '战略投资',
            '股权转让'
        ]
        
        for round_str in valid_rounds:
            result = validator.validate_round(round_str)
            if not result['is_valid']:
                print(f"✗ 有效轮次验证失败: {round_str}")
                return False
        
        print("✓ 有效轮次验证通过")
        
        # 测试无效轮次
        invalid_rounds = [
            '',  # 空轮次
            'Z轮',  # 无效轮次
            'Pre-Z',  # 无效Pre轮次
            'A++轮',  # 过多+号
            '轮次A',  # 错误顺序
            'A轮B轮'  # 多个轮次
        ]
        
        for round_str in invalid_rounds:
            result = validator.validate_round(round_str)
            if result['is_valid']:
                print(f"✗ 无效轮次应该被拒绝: {round_str}")
                return False
        
        print("✓ 无效轮次正确拒绝")
        return True
        
    except Exception as e:
        print(f"✗ 轮次验证测试失败: {e}")
        return False

def test_website_validation():
    """测试网址验证"""
    print("\n测试网址验证...")
    try:
        validator = DataValidator()
        
        # 测试有效网址
        valid_websites = [
            'https://www.tencent.com',
            'http://www.alibaba.com',
            'https://www.baidu.com',
            'http://www.jd.com',
            'https://www.meituan.com'
        ]
        
        for website in valid_websites:
            result = validator.validate_website(website)
            if not result['is_valid']:
                print(f"✗ 有效网址验证失败: {website}")
                return False
        
        print("✓ 有效网址验证通过")
        
        # 测试无效网址
        invalid_websites = [
            '',  # 空网址
            'not-a-url',  # 无效格式
            'www.tencent.com',  # 缺少协议
            'http://',  # 只有协议
            'https://invalid',  # 无效域名
            'ftp://www.tencent.com'  # 不支持协议
        ]
        
        for website in invalid_websites:
            result = validator.validate_website(website)
            if result['is_valid']:
                print(f"✗ 无效网址应该被拒绝: {website}")
                return False
        
        print("✓ 无效网址正确拒绝")
        return True
        
    except Exception as e:
        print(f"✗ 网址验证测试失败: {e}")
        return False

def test_batch_validation():
    """测试批量验证"""
    print("\n测试批量验证...")
    try:
        validator = DataValidator()
        
        # 测试批量公司数据验证
        companies = [
            {
                '公司名称': '腾讯控股有限公司',
                '工商注册id': '440301103455666',
                '统一信用代码': '9144030071526726XG',
                '注册资金': '1000万人民币',
                '成立时间': '1998-11-11',
                '网址': 'https://www.tencent.com'
            },
            {
                '公司名称': '阿里巴巴（中国）有限公司',
                '工商注册id': '91330100699855058B',
                '统一信用代码': '91330100699855058B',
                '注册资金': '500万人民币',
                '成立时间': '1999-09-09',
                '网址': 'https://www.alibaba.com'
            }
        ]
        
        results = validator.validate_companies_batch(companies)
        
        if len(results) == 2 and all(result['is_valid'] for result in results):
            print("✓ 批量公司数据验证通过")
        else:
            print(f"✗ 批量公司数据验证失败: {results}")
            return False
        
        # 测试批量投资事件验证
        events = [
            {
                '投资方': '腾讯',
                '融资方': '字节跳动',
                '融资时间': '2020-01-01',
                '轮次': 'C轮',
                '金额': '10亿美元'
            },
            {
                '投资方': '阿里巴巴',
                '融资方': '饿了么',
                '融资时间': '2018-04-02',
                '轮次': '并购',
                '金额': '95亿美元'
            }
        ]
        
        results = validator.validate_investment_events_batch(events)
        
        if len(results) == 2 and all(result['is_valid'] for result in results):
            print("✓ 批量投资事件验证通过")
        else:
            print(f"✗ 批量投资事件验证失败: {results}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 批量验证测试失败: {e}")
        return False

def test_validation_rules():
    """测试验证规则"""
    print("\n测试验证规则...")
    try:
        validator = DataValidator()
        
        # 获取验证规则
        rules = validator.get_validation_rules()
        
        required_keys = [
            'company_rules', 'investment_event_rules', 'investment_institution_rules',
            'name_rules', 'date_rules', 'amount_rules', 'round_rules', 'website_rules'
        ]
        
        if isinstance(rules, dict) and all(key in rules for key in required_keys):
            print("✓ 验证规则获取成功")
            return True
        else:
            print(f"✗ 验证规则获取失败: {rules}")
            return False
        
    except Exception as e:
        print(f"✗ 验证规则测试失败: {e}")
        return False

def test_stats_functionality():
    """测试统计功能"""
    print("\n测试统计功能...")
    try:
        validator = DataValidator()
        
        # 先验证一些数据
        test_data = {
            '公司名称': '测试公司',
            '公司描述': '测试描述'
        }
        validator.validate_company_data([test_data])
        
        # 获取统计信息
        stats = validator.get_stats()
        
        required_keys = [
            'total_validated', 'valid_records', 'invalid_records',
            'validation_errors', 'validation_rules_applied'
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

def test_custom_validation_rules():
    """测试自定义验证规则"""
    print("\n测试自定义验证规则...")
    try:
        validator = DataValidator()
        
        # 添加自定义验证规则
        custom_rules = {
            'min_company_name_length': 5,
            'max_company_name_length': 50,
            'required_company_fields': ['公司名称', '公司描述', '工商注册名称'],
            'allowed_currencies': ['人民币', '美元', '港元']
        }
        
        validator.set_custom_rules(custom_rules)
        
        # 测试自定义规则
        test_company = {
            '公司名称': '腾讯',  # 长度小于5，应该失败
            '公司描述': '测试公司',
            '工商注册名称': '腾讯科技有限公司'
        }
        
        result = validator.validate_company_data([test_company])
        if result['invalid_records'] > 0:
            print("✓ 自定义验证规则生效")
            return True
        else:
            print("✗ 自定义验证规则未生效")
            return False
            
    except Exception as e:
        print(f"✗ 自定义验证规则测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行数据验证器测试")
    print("=" * 50)
    
    tests = [
        test_data_validator_creation,
        test_company_data_validation,
        test_investment_event_validation,
        test_investment_institution_validation,
        test_name_validation,
        test_date_validation,
        test_amount_validation,
        test_round_validation,
        test_website_validation,
        test_batch_validation,
        test_validation_rules,
        test_stats_functionality,
        test_custom_validation_rules
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