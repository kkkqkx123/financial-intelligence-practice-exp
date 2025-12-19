"""
数据处理器测试脚本
用于测试DataParser和DataValidator的功能
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from processors.data_parser import DataParser
from processors.data_validator import DataValidator
from utils.logger import get_logger

logger = get_logger(__name__)

async def test_data_processing():
    """测试数据处理流程"""
    print("开始测试数据处理...")
    
    # 初始化处理器
    parser = DataParser()
    validator = DataValidator()
    
    # 测试公司数据
    print("\n=== 测试公司数据 ===")
    try:
        # 读取CSV文件内容
        with open("src/dataset/company_data.csv", "r", encoding="utf-8") as f:
            companies_content = f.read()
        
        # 解析公司数据
        companies = parser.parse_companies(companies_content)
        print(f"解析到 {len(companies)} 家公司")
        
        # 验证公司数据
        company_result = validator.validate_company_data(companies)
        print(f"公司数据验证: {'通过' if company_result.get('data_quality_score', 0) > 0.5 else '失败'}")
        if company_result.get('validation_errors'):
            print(f"错误: {company_result['validation_errors'][:5]}...")  # 只显示前5个错误
            
    except Exception as e:
        print(f"公司数据测试失败: {e}")
        logger.exception("公司数据测试异常")
    
    # 测试投资事件数据
    print("\n=== 测试投资事件数据 ===")
    try:
        # 读取CSV文件内容
        with open("src/dataset/investment_events.csv", "r", encoding="utf-8") as f:
            events_content = f.read()
        
        print(f"投资事件文件内容长度: {len(events_content)}")
        print(f"投资事件文件前100个字符: {events_content[:100]}")
        
        # 解析投资事件数据
        events = parser.parse_investment_events(events_content)
        print(f"解析到 {len(events)} 个投资事件")
        
        # 验证投资事件数据
        event_result = validator.validate_investment_event_data(events)
        print(f"投资事件数据验证: {'通过' if event_result.get('data_quality_score', 0) > 0.5 else '失败'}")
        if event_result.get('validation_errors'):
            print(f"错误: {event_result['validation_errors'][:5]}...")  # 只显示前5个错误
            
    except Exception as e:
        print(f"投资事件数据测试失败: {e}")
        logger.exception("投资事件数据测试异常")
    
    # 测试投资机构数据
    print("\n=== 测试投资机构数据 ===")
    try:
        # 读取CSV文件内容
        with open("src/dataset/investment_structure.csv", "r", encoding="utf-8") as f:
            institutions_content = f.read()
        
        # 解析投资机构数据
        institutions = parser.parse_investment_institutions(institutions_content)
        print(f"解析到 {len(institutions)} 家投资机构")
        
        # 验证投资机构数据
        institution_result = validator.validate_investor_data(institutions)
        print(f"投资机构数据验证: {'通过' if institution_result.get('data_quality_score', 0) > 0.5 else '失败'}")
        if institution_result.get('validation_errors'):
            print(f"错误: {institution_result['validation_errors'][:5]}...")  # 只显示前5个错误
            
    except Exception as e:
        print(f"投资机构数据测试失败: {e}")
        logger.exception("投资机构数据测试异常")
    
    # 打印解析统计
    print("\n=== 解析统计 ===")
    stats = parser.get_stats()
    print(f"总处理记录数: {stats.get('total_records', 0)}")
    print(f"处理公司数: {stats.get('companies_processed', 0)}")
    print(f"处理事件数: {stats.get('events_processed', 0)}")
    print(f"处理机构数: {stats.get('institutions_processed', 0)}")
    print(f"错误数: {stats.get('errors', 0)}")
    
    print("\n数据处理测试完成!")

async def main():
    """主函数"""
    await test_data_processing()

if __name__ == "__main__":
    asyncio.run(main())