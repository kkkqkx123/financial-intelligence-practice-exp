import sys
import traceback
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    # 导入主模块
    import main
    
    # 设置命令行参数（使用新的简化格式）
    sys.argv = ['main.py', '--verbose']
    
    # 运行主函数
    main.main()
    
except Exception as e:
    print('=== ERROR ===')
    print(f'Error: {e}')
    traceback.print_exc()
    
    # 将错误写入文件
    error_log_file = "D:/Source/torch/financial-intellgience/src/logs/error_log.txt"
    with open(error_log_file, 'w', encoding='utf-8') as f:
        f.write(f'Error: {e}\n')
        traceback.print_exc(file=f)