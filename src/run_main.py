import sys
import traceback
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    # 导入主模块
    import main
    
    # 设置命令行参数
    sys.argv = ['main.py', '--data-dir', 'dataset', '--enable-neo4j', '--neo4j-uri', 'bolt://localhost:7687', '--neo4j-user', 'neo4j', '--neo4j-password', '1234567kk', '--verbose']
    
    # 运行主函数
    main.main()
    
except Exception as e:
    print('=== ERROR ===')
    print(f'Error: {e}')
    traceback.print_exc()
    
    # 将错误写入文件
    with open('error_log.txt', 'w') as f:
        f.write(f'Error: {e}\n')
        traceback.print_exc(file=f)