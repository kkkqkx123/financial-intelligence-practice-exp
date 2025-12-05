from computing.core.container import Container
import torch as torch
from collections import OrderedDict
from computing.utils.data_struct import SeriesDictData, RoundDictData


class Portfolio(Container):
    r"""
    投资组合算法的顶层容器，装载非树状结构的、只有单个模块的投资组合算法。

    Args:
        container_id (int or str): 容器的ID。Default: -1
        module: module or container
        **kwargs: 子模块所需参数

    Note:
        该容器下，建议配置的模块如下（不同子模型所需参数请跳转至子模型注释页面）：

        * :class:`~computing.base_strategy.modules.ons.ONS`: ONS 算法，Ref: AGARWAL, A., HAZAN, E., KALE, S., AND SCHAPIRE, R. E. 2006. Algorithms for portfolio management based on the newton method. In Proceedings of International Conference on Machine Learning. 9–16.

        * :class:`~computing.base_strategy.modules.eg.EG`: EG 算法，Ref: HELMBOLD, D. P., SCHAPIRE, R. E., SINGER, Y., AND WARMUTH, M. K. 1996. On-line portfolio selection using multiplicative updates. In Proceedings of the International Conference on Machine Learning. 243–251.

        * :class:`~computing.base_strategy.modules.ew.EW`: EW 算法，Ref: 每次均返回等权重

    """

    def __init__(self, container_id=-1, opt_module=None, **kwargs):
        super(Portfolio, self).__init__(container_id=container_id, modules_dict={"opt_module": opt_module})

    def decide(self, dict_data: SeriesDictData, **kwargs):
        portfolio = OrderedDict()

        weight = self.opt_module.decide()["weight"]
        i = 0
        round_data = dict_data[0]
        for stock in round_data.reward.keys():
            portfolio[stock] = weight[i]
            i += 1
        result = {'portfolio': portfolio}
        return result

    def update(self, results, dict_data: SeriesDictData, **kwargs):
        result = results['portfolio']
        round_data = dict_data[0]
        price_relative = torch.tensor(list(round_data.reward.values()))
        last_weight = torch.tensor(list(result.values()))
        last_weight_hat = last_weight * price_relative / torch.dot(last_weight.double(), price_relative.double())
        series_context = round_data.series_context
        series_name = list(round_data.reward.keys())
        self.opt_module("update", price_relative=price_relative, last_weight=last_weight,
                        last_weight_hat=last_weight_hat, series_context=series_context, series_name=series_name)

