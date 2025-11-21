import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set cuda device, default 0

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + '/compute_platform')
sys.path.append(PARENT_DIR + '/experiments')

from computing.proxy.execute_proxy import DRLExecuteProxy
from computing.proxy.execute_proxy_set import ExecuteProxySet
from computing.proxy.model_proxy import ModelProxy
from experiments.fintech_class_experiments import get_data

from computing.base_strategy.modules.network.rat import RAT
from computing.base_strategy.modules.buffer.drl_buffer import DRLBuffer
from computing.base_strategy.containers.portfolio.portfolio_rat import RATAgent, PortfolioRAT


def run(data_name, path_type, save=False, device="cuda", buffer_size=64, lr=0.0001, n_train_epoch=10, window_size=20):
    print('data_name', data_name, 'save', save)
    data, stock_num, data_split, metric, main_dir = get_data(data_name, path_type, window_size)

    ###########################################
    # todo: 生成portfolio实例。需要自己添加参数
    net = RAT()
    buffer = DRLBuffer(buffer_size=buffer_size, sample_size=int(buffer_size / 2))
    agent = RATAgent(modules_dict={"net": net}, lr=lr, weight_decay=1e-3)
    portfolio = PortfolioRAT(modules_dict={"agent": agent, "buffer": buffer})
    ###########################################
    model = ModelProxy(model=portfolio)


    # Execute参数 以及 实例生成
    execute_set_params = {
        "start_index": data_split["start_index"],
        "n_pretrain": 0,
        "n_train": data_split["train"],
        "n_validate": 0,
        "n_test": data_split["test"]
    }
    execute_set = ExecuteProxySet(**execute_set_params)

    execute_params = {"execute_proxy_class": "DRLExecuteProxy",
                      "result_dir": os.path.join(main_dir, "results/fintech_class_experiments/" + "rat"),
                      "save": save,
                      "n_train_epoch": 1,
                      "device": device}
    execute = DRLExecuteProxy(model=model, data=data, metric=metric, **execute_params)
    execute_set.add_from_instance(execute)

    n_train_epoch = n_train_epoch
    for i in range(n_train_epoch):
        execute_set.execute(mode='PreTraining', batch_size=1, start_index=data_split["start_index"],
                            end_index=data_split["start_index"] + 1)
        execute_set.execute(mode='Training', batch_size=1, start_index=data_split["start_index"],
                            end_index=data_split["start_index"] + data_split["train"])
        execute_set.execute(mode='OnlineDecision', batch_size=1)


if __name__ == "__main__":
    data_name = "ashare_sse300"
    path_type = "local"

    run(data_name, path_type)
