#!/usr/bin/env python3
"""
金融知识图谱构建系统示例脚本
演示如何使用系统构建知识图谱
"""

import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """创建示例数据"""
    logger.info("创建示例数据...")
    
    # 创建数据集目录
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # 公司数据
    companies = [
        {
            "company_name": "腾讯科技",
            "industry": "互联网",
            "founded_year": 1998,
            "description": "中国领先的互联网增值服务提供商",
            "website": "https://www.tencent.com",
            "employees": 50000
        },
        {
            "company_name": "阿里巴巴",
            "industry": "电商",
            "founded_year": 1999,
            "description": "全球最大的电子商务公司之一",
            "website": "https://www.alibabagroup.com",
            "employees": 100000
        },
        {
            "company_name": "字节跳动",
            "industry": "互联网",
            "founded_year": 2012,
            "description": "全球化的移动互联网平台",
            "website": "https://www.bytedance.com",
            "employees": 60000
        }
    ]
    
    # 投资方数据
    investors = [
        {
            "investor_name": "红杉资本中国",
            "investor_type": "风险投资",
            "founded_year": 2005,
            "aum": "300亿美元",
            "headquarters": "北京",
            "description": "中国领先的风险投资机构"
        },
        {
            "investor_name": "IDG资本",
            "investor_type": "私募股权投资",
            "founded_year": 1993,
            "aum": "200亿美元",
            "headquarters": "北京",
            "description": "全球领先的投资机构"
        }
    ]
    
    # 投资事件数据
    investment_events = [
        {
            "event_id": "INV001",
            "company_name": "字节跳动",
            "investor_name": "红杉资本中国",
            "investment_amount": "10亿美元",
            "investment_round": "D轮",
            "investment_date": "2018-10-20",
            "valuation": "750亿美元"
        },
        {
            "event_id": "INV002",
            "company_name": "阿里巴巴",
            "investor_name": "IDG资本",
            "investment_amount": "5亿美元",
            "investment_round": "战略投资",
            "investment_date": "2020-03-15",
            "valuation": "5000亿美元"
        }
    ]
    
    # 保存数据文件
    with open(dataset_dir / "companies.json", 'w', encoding='utf-8') as f:
        json.dump(companies, f, ensure_ascii=False, indent=2)
    
    with open(dataset_dir / "investors.json", 'w', encoding='utf-8') as f:
        json.dump(investors, f, ensure_ascii=False, indent=2)
    
    with open(dataset_dir / "investment_events.json", 'w', encoding='utf-8') as f:
        json.dump(investment_events, f, ensure_ascii=False, indent=2)
    
    logger.info(f"示例数据创建完成，共 {len(companies)} 家公司, {len(investors)} 个投资方, {len(investment_events)} 个投资事件")

def run_simple_demo():
    """运行简单演示"""
    logger.info("开始运行简单演示...")
    
    try:
        # 导入系统
        sys.path.append(str(Path(__file__).parent / "src"))
        from main import KnowledgeGraphPipeline
        
        # 创建数据
        create_sample_data()
        
        # 初始化流水线
        pipeline = KnowledgeGraphPipeline(
            data_dir="dataset",
            output_dir="output"
        )
        
        # 运行完整流程
        logger.info("运行知识图谱构建流程...")
        results = pipeline.run_full_pipeline(save_intermediate=True)
        
        # 输出结果摘要
        if results.get('success'):
            logger.info("✅ 知识图谱构建成功！")
            
            # 显示统计信息
            stats = results.get('statistics', {})
            logger.info(f"处理时间: {stats.get('total_time', 0):.2f}秒")
            
            # 显示实体统计
            entities = results.get('final_knowledge_graph', {}).get('entities', {})
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            logger.info(f"总实体数: {total_entities}")
            
            # 显示关系统计
            relations = results.get('final_knowledge_graph', {}).get('relations', {})
            total_relations = sum(len(relation_list) for relation_list in relations.values())
            logger.info(f"总关系数: {total_relations}")
            
            # 显示数据质量
            quality = results.get('validation_results', {}).get('overall_quality', {})
            logger.info(f"数据质量评分: {quality.get('quality_score', 0):.1f}")
            
            logger.info("演示完成！请查看 output 目录中的结果文件。")
            
        else:
            logger.error("❌ 知识图谱构建失败")
            
    except Exception as e:
        logger.error(f"演示运行失败: {str(e)}")
        logger.info("请确保已安装所有依赖: pip install -r requirements.txt")

if __name__ == "__main__":
    import sys
    run_simple_demo()