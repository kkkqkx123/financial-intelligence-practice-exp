import random

import torch
import numpy as np

from computing.eval.portfolio_eval import PortfolioEval
from computing.proxy.metric_proxy import MetricProxy
from experiments.config.config_util_func import get_path_config, config_data


def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def config_metric(path_type="remote33"):
    main_dir = get_path_config(path_type, "metric")

    # Metric的参数 以及 实例生成
    metrics_class = {'PortfolioEval': PortfolioEval}
    metrics_name = {'PortfolioEval': ["cm_reward", "max_draw", "volatility", "sharpe_ratio",
                                      "calmar_ratio", "turnovers",
                                      "annualized_percentage_yield", "weight"]}

    metric_params = {
        "metrics_class": metrics_class,  # 表示要用那些eval类
        "metrics_name": metrics_name,  # 表示要用保存哪些指标
        "metrics_disable_mode": {'PortfolioEval': {'PreTraining', 'Training', 'Updating', 'Validation'}
                                 },  # 表示要disable哪些mode
        "transaction_cost": 0,
        "frequency": 252
    }

    metric = MetricProxy(**metric_params)
    return metric, main_dir


def exp_data_split(data_name, init_index):
    data_split = {
        "ashare_sse300": {
        "start_index": init_index,
        "train": 242 + 1 - init_index,
        "test": 360 - 242
        }
    }
    return data_split[data_name]


def get_data(data_name, path_type, window_size):
    data_split = exp_data_split(data_name, window_size)
    data, stock_num = config_data(data_name=data_name,
                                  path_type=path_type,
                                  context_menu=[],
                                  window_size=window_size,
                                  context_window_size=window_size)
    metric, main_dir = config_metric(path_type=path_type)
    return data, stock_num, data_split, metric, main_dir