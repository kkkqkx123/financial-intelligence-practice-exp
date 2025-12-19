import sys
import asyncio
import traceback
sys.path.append('src')
from processors.data_parser import DataParser
from processors.data_validator import DataValidator

async def main():
    try:
        print("开始测试数据处理...")
        parser = DataParser()
        validator = DataValidator()
        
        # 测试公司数据解析
        print("解析公司数据...")
        companies = parser.parse_companies("src/dataset/company_data.csv")
        print(f"解析到 {len(companies)} 家公司")
        
        # 测试投资事件数据解析
        print("解析投资事件数据...")
        events = parser.parse_investment_events("src/dataset/investment_events.csv")
        print(f"解析到 {len(events)} 个投资事件")
        
        # 测试投资机构数据解析
        print("解析投资机构数据...")
        institutions = parser.parse_investment_institutions("src/dataset/investment_structure.csv")
        print(f"解析到 {len(institutions)} 家投资机构")
        
        # 验证数据
        print("验证数据...")
        company_validation = validator.validate_company_data(companies)
        event_validation = validator.validate_investment_event_data(events)
        institution_validation = validator.validate_investor_data(institutions)
        
        print(f"公司数据验证结果: {company_validation['valid_records']}/{company_validation['total_records']} 有效")
        print(f"投资事件数据验证结果: {event_validation['valid_records']}/{event_validation['total_records']} 有效")
        print(f"投资机构数据验证结果: {institution_validation['valid_records']}/{institution_validation['total_records']} 有效")
        
        print("数据处理测试完成!")
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())