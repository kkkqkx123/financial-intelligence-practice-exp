import json
import os

def extract_code_from_notebook(notebook_path, output_path):
    """
    从Jupyter笔记本中提取Python代码块并写入.py文件
    """
    try:
        # 读取notebook文件
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # 提取代码单元
        code_cells = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # 将代码行合并为字符串
                code = ''.join(cell['source'])
                if code.strip():  # 只添加非空代码
                    code_cells.append(code)
        
        # 将所有代码合并，用双换行分隔
        all_code = '\n\n'.join(code_cells)
        
        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_code)
        
        print(f"成功提取 {len(code_cells)} 个代码块")
        print(f"代码已写入: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        return False

if __name__ == "__main__":
    # 定义文件路径
    notebook_file = "d:/学习/AI应用/经济学/金融智能理论与实践/作业/2/数据预处理-jupyter模板.ipynb"
    output_file = "d:/学习/AI应用/经济学/金融智能理论与实践/作业/2/main.py"
    
    # 提取代码
    extract_code_from_notebook(notebook_file, output_file)