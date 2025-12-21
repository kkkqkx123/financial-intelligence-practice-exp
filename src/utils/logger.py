"""
日志配置模块
提供统一的日志配置和格式化输出
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 保存原始级别名称
        original_levelname = record.levelname
        
        # 如果是终端输出，添加颜色
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, '')
            if color:
                record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # 格式化日志
        formatted = super().format(record)
        
        # 恢复原始级别名称
        record.levelname = original_levelname
        
        return formatted


def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
    
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 创建格式化器
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file_output and log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别的日志
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已配置的日志器"""
    return logging.getLogger(name)


# 创建默认日志器
default_logger = setup_logger(
    name='financial_kg',
    level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file=os.getenv('LOG_FILE', 'D:/Source/torch/financial-intellgience/src/logs/financial_kg.log'),
    console_output=True,
    file_output=True
)