from computing.core.container import Container
import torch as torch
import numpy
from collections import OrderedDict
from computing.utils.data_struct import SeriesDictData, RoundDictData, ContextSeriesTensorData
from computing.base_strategy.modules.optimizor.mvo import MVO


class PortfolioMVO(Container):
    """
    在线MVO的框架，主要承载需要划窗计算mvo的算法
    """

    def __init__(self, container_id=-1, mvo_module=MVO(), **kwargs):
        super(PortfolioMVO, self).__init__(container_id=container_id, modules_dict={"mvo_module": mvo_module})

    def decide(self, tensor_data: ContextSeriesTensorData, **kwargs):
        portfolio = OrderedDict()
        self.mvo_module("update", context=tensor_data.series_context)
        weight = self.mvo_module.decide()["weight"]
        i = 0
        for stock in tensor_data.index_map_dict.get("series_index_map"):
            portfolio[stock] = weight[i]
            i += 1
        result = {'portfolio': portfolio}
        return result

