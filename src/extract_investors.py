#!/usr/bin/env python3
"""
从投资事件数据中提取投资方信息，生成投资方数据文件
"""

import csv
import os
import json
from collections import defaultdict, Counter

def extract_investors_from_events(input_file, output_file):
    """
    从投资事件CSV文件中提取投资方信息
    
    Args:
        input_file: 投资事件CSV文件路径
        output_file: 输出的投资方数据文件路径
    """
    # 存储投资方信息
    investors = {}
    investor_count = Counter()
    investor_rounds = defaultdict(set)
    investor_industries = defaultdict(set)
    
    # 读取投资事件数据
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            investors_str = row.get('投资方', '').strip()
            round_type = row.get('轮次', '').strip()
            
            # 跳过没有投资方信息或投资方为"未披露机构"的记录
            if not investors_str or investors_str == '未披露机构':
                continue
                
            # 分割多个投资方
            investor_names = split_investors(investors_str)
            
            for name in investor_names:
                name = name.strip()
                if not name:
                    continue
                    
                # 统计投资方出现次数
                investor_count[name] += 1
                
                # 记录投资方参与的轮次
                if round_type and round_type != '未披露':
                    investor_rounds[name].add(round_type)
                
                # 如果有融资方信息，尝试推断行业
                company = row.get('融资方', '').strip()
                if company:
                    # 这里可以添加更复杂的行业推断逻辑
                    # 目前简单使用公司名作为行业标识
                    investor_industries[name].add(company)
    
    # 构建投资方数据
    for name, count in investor_count.items():
        investors[name] = {
            'name': name,
            'investment_count': count,
            'preferred_rounds': list(investor_rounds[name]),
            'industries': list(investor_industries[name])[:5],  # 限制行业数量
            'description': f'{name}是一家投资机构，已参与{count}次投资事件。'
        }
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(investors.values()), f, ensure_ascii=False, indent=2)
    
    print(f"成功提取{len(investors)}个投资方信息，保存到{output_file}")
    return investors

def split_investors(investors_str):
    """
    分割投资方字符串，支持多种分隔符
    
    Args:
        investors_str: 投资方字符串
        
    Returns:
        投资方名称列表
    """
    # 常见分隔符
    separators = [' ', '、', '，', ',', ';', '；']
    
    # 尝试使用不同分隔符分割
    for sep in separators:
        if sep in investors_str:
            return [inv.strip() for inv in investors_str.split(sep) if inv.strip()]
    
    # 如果没有找到分隔符，返回整个字符串
    return [investors_str]

def main():
    # 输入输出文件路径
    input_file = 'dataset/investment_events.csv'
    output_file = '../dataset/investors.json'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件{input_file}不存在")
        return
    
    # 提取投资方信息
    extract_investors_from_events(input_file, output_file)

if __name__ == '__main__':
    main()