#!/usr/bin/env python3
"""
数据集筛选脚本
将数据集各自仅保留50条数据用于测试
"""

import pandas as pd
import os
import shutil
from pathlib import Path

def filter_dataset(input_file, output_file, sample_size=50, random_state=42):
    """筛选数据集，仅保留指定数量的记录"""
    try:
        # 读取数据
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"原始数据: {input_file} - {len(df)} 条记录")
        
        # 如果数据量小于等于sample_size，直接返回原数据
        if len(df) <= sample_size:
            print(f"数据量已小于等于{sample_size}，无需筛选")
            return False
        
        # 随机采样指定数量的数据
        sampled_df = df.sample(n=sample_size, random_state=random_state)
        
        # 保存筛选后的数据
        sampled_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"筛选后数据: {output_file} - {len(sampled_df)} 条记录")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")
        return False

def backup_original_files(dataset_dir):
    """备份原始文件"""
    backup_dir = dataset_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "company_data.csv",
        "investment_events.csv", 
        "investment_structure.csv"
    ]
    
    for file_name in files_to_backup:
        src_file = dataset_dir / file_name
        if src_file.exists():
            dst_file = backup_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"备份文件: {file_name} -> backup/{file_name}")

def main():
    """主函数"""
    # 设置数据集目录
    dataset_dir = Path("d:/Source/torch/financial-intellgience/src/dataset")
    
    print("=" * 50)
    print("数据集筛选工具")
    print("=" * 50)
    
    # 备份原始文件
    print("正在备份原始文件...")
    backup_original_files(dataset_dir)
    
    # 定义要处理的文件
    files_to_process = [
        ("investment_events.csv", 50),  # 投资事件数据保留50条
        ("investment_structure.csv", 50),  # 投资结构数据保留50条
    ]
    
    # 注意：company_data.csv 被屏蔽了，所以不处理
    
    print(f"\n开始筛选数据集，每个文件保留50条记录...")
    
    success_count = 0
    for file_name, sample_size in files_to_process:
        input_file = dataset_dir / file_name
        
        if not input_file.exists():
            print(f"文件不存在: {file_name}")
            continue
            
        # 直接覆盖原文件（因为有备份）
        output_file = input_file
        
        if filter_dataset(input_file, output_file, sample_size):
            success_count += 1
    
    print(f"\n筛选完成！成功处理 {success_count} 个文件")
    print("原始文件已备份到 dataset/backup/ 目录")
    print("=" * 50)

if __name__ == "__main__":
    main()