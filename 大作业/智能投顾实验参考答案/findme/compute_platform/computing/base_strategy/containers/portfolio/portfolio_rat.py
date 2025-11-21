from collections import OrderedDict

import torch as torch
import torch.optim as optim

from computing.core.container import Container
from computing.core.dtype import ContainerProperty
from computing.utils.data_struct import ContextSeriesTensorData


class RATAgent(Container):
    def __init__(self, container_id=-1, modules_dict={"net": None}, lr=0.01, weight_decay=0, **kwargs):
        super(RATAgent, self).__init__(container_id=container_id, modules_dict=modules_dict)
        self.optim = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_sch = optim.lr_scheduler.ExponentialLR(self.optim, gamma=1)

    def train(self, round_data: ContextSeriesTensorData, mini_batch, **kwargs):
        # 获得 action
        self.net("forward", mini_batch["state"])
        weight = self.net.decide()["portfolio_vector"]  # [b, s]

        close_price = mini_batch["reward"]
        loss = self.calculate_loss(weight, close_price)
        print("loss", round_data.init_index, loss)

        # 回传 loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.lr_sch.step()

    def calculate_loss(self, weight, price_relative_vector):  # w:[b,s+1]  r:[b,s]
        ######################
        # todo: 计算loss
        ######################
        return loss


class PortfolioRAT(Container):
    def __init__(self, container_id=-1, modules_dict={"buffer": None, "agent": None}, **kwargs):
        super(PortfolioRAT, self).__init__(container_id=container_id, modules_dict=modules_dict)

    def decide(self, round_data: ContextSeriesTensorData, **kwargs):
        result = self.action(round_data)

        # 封装 portfolio
        portfolio = OrderedDict()
        series_index_map = round_data.index_map_dict['series_index_map']
        for i in range(len(series_index_map)):
            stock = series_index_map[i]
            w = result["action"][0, i + 1]
            portfolio[stock] = w if w > 0 else 0
        result['portfolio'] = portfolio
        return result   # 回测框架会自动取result['portfolio']来执行回测

    def update(self, result, round_data: ContextSeriesTensorData, **kwargs):
        # 更新buffer
        reward = round_data.reward
        self.buffer("update", result["state"], result["action"][:, 1:], reward)

    def action(self, round_data: ContextSeriesTensorData):
        result = {}

        ######################################
        # todo: 历史价格数据归一化

        # todo：从buffer里取前一天的weight（注意：由于第一天决策时，我们没有前一天的weight，所以此时需要自己初始化一个weight，一般都是现金权重为0、股票权重为1/N）

        ######################################

        # 封装state
        state = {"price_series": price_series, "previous_w": previous_w}
        result["state"] = state

        # 获取 action
        with torch.no_grad():
            self.agent.net("forward", state)
            action = self.agent.net.decide()["portfolio_vector"]  # [b, 1+s]
            result["action"] = action
        return result

    def pre_train(self, data, **kwargs):
        self.buffer("update", release=True)

    def train(self, round_data: ContextSeriesTensorData, **kwargs):
        result = self.action(round_data)
        self.update(result, round_data)

        # 更新buffer，当buffer满了时则训练
        if self.buffer.decide()["is_full"]:
            mini_batch = self._get_buffer_data(["sample_state", "sample_action", "sample_reward"])
            self.agent(data=round_data, mini_batch=mini_batch, mode="Training")

    def _get_buffer_data(self, buffer_list):
        mini_batch = {}
        for buffer_name in buffer_list:
            mini_batch[buffer_name[7:]] = self.buffer.decide()[buffer_name]
        return mini_batch
