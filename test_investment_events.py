import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

from processors.data_parser import DataParser

def test_investment_events():
    """测试投资事件数据解析"""
    print("开始测试投资事件数据解析...")
    
    # 初始化处理器
    parser = DataParser()
    
    try:
        # 读取CSV文件内容
        with open("src/dataset/investment_events.csv", "r", encoding="utf-8") as f:
            events_content = f.read()
        
        print(f"投资事件文件内容长度: {len(events_content)}")
        print(f"投资事件文件前100个字符: {events_content[:100]}")
        
        # 解析投资事件数据
        events = parser.parse_investment_events(events_content)
        print(f"解析到 {len(events)} 个投资事件")
        
        # 打印解析统计
        stats = parser.get_stats()
        print(f"处理事件数: {stats.get('events_processed', 0)}")
        print(f"错误数: {stats.get('errors', 0)}")
        
    except Exception as e:
        print(f"投资事件数据测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_investment_events()