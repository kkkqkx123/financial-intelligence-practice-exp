import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from computing.core.module import Module
from computing.core.dtype import LearningVariable


# ================================================== main model ========================================================
class RAT(Module):
    def __init__(self, module_id=-1):
        super(RAT, self).__init__(module_id=module_id)

        #########################################
        # todo: 网络init

        #########################################

        # 要传给contrainer的变量需要套上LearningVariable数据类型，然后注册
        self.portfolio_vector = LearningVariable(None)
        self.register_decide_hooks(['portfolio_vector'])

    def forward(self, state):
        """
        Args:
            state: dict, {"price_series": "shape is [batch_size,stock_num,window_size,feature_num]",
                          "previous_w": "shape is [batch_size, stock_num]"}
        """
        #########################################
        # todo: 计算投资组合权重，输出维度是[batch_size, 1+stock_num]。注意投资组合第0维是现金，1～N维是股票。

        #########################################
        self.portfolio_vector = weight   # [b,s+1]


