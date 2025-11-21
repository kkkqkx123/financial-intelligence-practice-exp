# -*- coding: utf-8 -*-
from computing.eval.evaluator import Evaluator
from computing.utils.data_struct import RoundDictData
from computing.config import EVAL_DISPLAY_BATCH
import numpy
from scipy.optimize import fsolve
import torch
from collections import OrderedDict

__all__ = ["PortfolioEval", "PortfolioRegretEval", "SparsePortfolioEval", "IntradayPortfolioEval",
           "MultipointIntradayPortfolioEval"]


class PortfolioEval(Evaluator):
    def __init__(self, file_dir, method="", dataset="", frequency=252., transaction_cost=0.0025, account=1., **kwargs):
        """

        Args:
            file_dir: metric存放路径
            method: 算法名称
            dataset: 数据集名称
            frequency: 交易频率，按照一年有多少次交易为准，例如日交易252，月交易12
            transaction_cost: 交易费率
            account: 账户金额（一般不用）
            **kwargs:
        """
        super().__init__(file_dir, method, dataset)
        self.cm_reward = [torch.tensor(1.), ]
        self.annualized_percentage_yield = torch.tensor(0.)
        self.account = account
        self.net_reward = []
        self.weight = []
        self.weight_o = []
        self.max_draw = torch.tensor(0.)
        self.volatility = torch.tensor(0.)
        self.sharpe_ratio = torch.tensor(0.)
        self.calmar_ratio = torch.tensor(0.)
        self.transaction_cost = torch.tensor(transaction_cost)
        self.turnovers = torch.tensor(0.)
        self.frequency = torch.tensor(frequency)
        self.decision_result = None

    def compute_cm(self):
        self.cm_reward.append(self.cm_reward[-1] * self.net_reward[-1])

    def compute_sharpe(self, reset_metric):
        non_risk_rate = torch.tensor(0.)  # 无风险利率为0.02
        if self.volatility == 0 or self.cm_reward[-1] < 0:
            self.sharpe_ratio = torch.tensor(0.)
        else:
            self.sharpe_ratio = (self.annualized_percentage_yield - non_risk_rate) / self.volatility
        print('Final sharpe_ratio for {}_{} is: {}'.format(self._dataset, self._method, self.sharpe_ratio))

    def compute_calmar(self, reset_metric):
        if self.max_draw == 0.:
            self.calmar_ratio = torch.tensor(0.)
        else:
            self.calmar_ratio = self.annualized_percentage_yield / self.max_draw
        print('Final calmar_ratio for {}_{} is: {}'.format(self._dataset, self._method, self.calmar_ratio))

    def compute_max_drawdown(self, reset_metric):
        """最大回撤率"""
        i = numpy.argmax((numpy.maximum.accumulate(self.cm_reward) - self.cm_reward) / numpy.maximum.accumulate(
            self.cm_reward))  # 结束位置
        if i == 0:
            self.max_draw = torch.tensor(0.)
        else:
            j = numpy.argmax(self.cm_reward[:i])  # 开始位置
            self.max_draw = (self.cm_reward[int(j)] - self.cm_reward[int(i)]) / (self.cm_reward[int(j)])
        print('Final max_draw for {}_{} is: {}'.format(self._dataset, self._method, self.max_draw))

    def compute_volatility(self, reset_metric):
        net_return = torch.tensor(self.net_reward) - torch.tensor(1.)
        return_std = torch.std(net_return)
        self.volatility = torch.sqrt(self.frequency) * return_std
        print('Final volatility for {}_{} is: {}'.format(self._dataset, self._method, self.volatility))

    def compute_turnovers(self, reset_metric):
        self.turnovers /= self._times
        print('Final turnovers for {}_{} is: {}'.format(self._dataset, self._method, self.turnovers))

    def update_net(self, portfolio: dict, reward: torch.Tensor, **kwargs):
        """
        main trading mechanism
        Args:
            portfolio: 投资组合权重
            reward: 收益

        Returns:

        """
        w = torch.tensor(list(portfolio.values()))
        if torch.sum(w) == 0.:
            net_reward = 1.
            self.turnovers += torch.tensor(0.)
            w_o = torch.cat([w, torch.tensor([1.])])
        else:
            w = torch.cat([w, torch.tensor([1. - torch.sum(w)])])
            if len(self.weight_o) == 0.:
                w_o = w
            else:
                w_o = self.weight_o[-1]

            r = torch.cat([reward, torch.tensor([1.])])

            def f3(x):
                return numpy.array(
                    x - 1 + self.transaction_cost.numpy() * numpy.sum(numpy.abs(w_o.numpy() - w.numpy() * x)))

            net_proportion = torch.tensor(fsolve(f3, numpy.array(1.))[0], dtype=torch.float32)
            net_reward = torch.dot(w.double(), r.double()) * net_proportion
            self.turnovers += torch.sum(torch.abs(w * net_proportion - w_o))
            w_o = w * r / torch.dot(w.double(), r.double())

        self.weight.append(list(w.numpy()))
        self.weight_o.append(w_o)
        self.net_reward.append(net_reward)

    def final_cm(self):
        print('Final cm for {}_{} is: {}'.format(self._dataset, self._method, self.cm_reward[-1]))

    def eval(self, result: dict, data: RoundDictData, **kwargs):
        # 判断格式合法性
        if "portfolio" in result.keys():  # 决策和结果都有
            self.decision_result = result["portfolio"]
        else:
            raise ModuleNotFoundError("Can't find portfolio in result. Please check result item!")

        # 每轮的计算
        reward = torch.tensor(list(result["eval_reward"].values())) \
            if "eval_reward" in result.keys() else torch.tensor(list(data.reward.values()))

        self.update_net(self.decision_result, reward)
        self.compute_cm()

        # 每轮的显示
        if self._times % EVAL_DISPLAY_BATCH == 0:
            acc_cm = self.cm_reward[-1]
            print('cm for {} after {} iterations on date {} is: {}'.format(self._method, self._times, data.round,
                                                                           acc_cm))

    def after_eval(self, reset_metric=True, *args, **kwargs):
        rounds = len(self.cm_reward)
        self.final_cm()
        self.annualized_percentage_yield = torch.pow(self.cm_reward[-1], torch.tensor(self.frequency / rounds)) - 1
        print('Final annualized_percentage_yield for {}_{} is: {}'.format(self._dataset, self._method,
                                                                          self.annualized_percentage_yield))
        self.compute_max_drawdown(reset_metric=reset_metric)
        self.compute_volatility(reset_metric=reset_metric)
        self.compute_sharpe(reset_metric=reset_metric)
        self.compute_calmar(reset_metric=reset_metric)

        self.compute_turnovers(reset_metric=reset_metric)


class PortfolioRegretEval(Evaluator):
    def after_eval(self, reset_metric=True, *args, **kwargs):
        pass

    def __init__(self, file_dir, method="Default", dataset="Default", regret_type="OLU_BCRP"):

        super().__init__(file_dir, method, dataset)
        self.regret = []
        self.final_regret = torch.tensor(0)
        self.regret_type = regret_type
        self.cardinality = None
        self.bcrp = None
        self.weight = None
        self.arm_times_dict = OrderedDict()
        self.arm_times = []

    def set_bcrp(self, bcrp_weight):
        object.__setattr__(self, "bcrp", bcrp_weight)
        self.cardinality = bcrp_weight.shape[0]

    def update_regret(self, result, bandit_data):
        if self.regret_type == "BCRP":
            if self.bcrp is None:
                raise AttributeError("cannot compute weak regret before max_arm setting.")
            weight = torch.tensor(list(result.values()))
            stock_return = torch.tensor(list(bandit_data.arm_reward.values()))
            regret = torch.dot(self.bcrp, stock_return[:self.cardinality]) - torch.dot(weight, stock_return)
        elif self.regret_type == "OLU_BCRP":
            if self.bcrp is None:
                raise AttributeError("cannot compute weak regret before max_arm setting.")
            weight = torch.tensor(list(result.values()))
            stock_return = torch.tensor(list(bandit_data.arm_reward.values()))
            if self.weight is None:
                # save hat weight
                object.__setattr__(self, "weight", weight)
            regret = torch.dot(self.bcrp, stock_return[:self.cardinality]) - torch.dot(weight,
                                                                                       stock_return) + torch.abs(
                weight - self.weight).sum()
            object.__setattr__(self, "weight", weight * stock_return / torch.dot(weight, stock_return))
        else:
            raise TypeError("cannot compute regret by {} type.".format(self.regret_type))
        if len(self.regret) == 0:
            self.regret.append(regret)
        else:
            self.regret.append(self.regret[-1] + regret)
        self.final_regret = self.regret[-1]
        if self._times % EVAL_DISPLAY_BATCH == 0:
            print('regret for {} after {} iterations is: {}.'.format(self._method, self._times, self.regret[-1]))

    def eval(self, result: dict, data: RoundDictData, **kwargs):
        if "portfolio" in result.keys():
            self.update_regret(result["portfolio"], data)
        else:
            raise ModuleNotFoundError("Can't find portfolio in result. Please check result item!")


class SparsePortfolioEval(PortfolioEval):
    def __init__(self, file_dir, method: str = "", dataset: str = "", frequency: float = 252.,
                 transaction_cost: float = 0.0025, account: float = 1,
                 group_split: dict = {}, **kwargs):
        """

        Args:
            file_dir:
            method:
            dataset:
            frequency:
            transaction_cost:
            account:
            group_split: {sector: [stock_name, stock_name],sector: [], sector: []}
            **kwargs:
        """
        super(SparsePortfolioEval, self).__init__(file_dir, method,
                                                  dataset, frequency, transaction_cost, account, **kwargs)
        self.active_stock_num = []
        self.active_group_num = []
        self.active_stock = []
        self.active_group = []
        self.group_split = group_split
        self.sparsity = torch.zeros(1)

    def compute_active(self, portfolio):
        min_weight = 0.001

        active_stock_dict = {}
        active_stock_num = 0
        for stock in portfolio.keys():
            if portfolio[stock] > min_weight:
                active_stock_dict[stock] = portfolio[stock]
                active_stock_num += 1
        self.active_stock.append(active_stock_dict)
        self.active_stock_num.append(active_stock_num)
        # print("active_stock_num:", active_stock_num)

        active_group_dict = {}
        active_group_num = 0
        for group, stocks in self.group_split.items():
            group_weight = 0
            for stock in stocks:
                if portfolio.get(stock) > min_weight:
                    group_weight += portfolio.get(stock)
            if group_weight > 0:
                active_group_dict[group] = group_weight
                active_group_num += 1
        self.active_group.append(active_group_dict)
        self.active_group_num.append(active_group_num)

        # print("active_group_num", active_group_num)

    def update_net(self, portfolio: dict, reward: torch.Tensor, **kwargs):
        super(SparsePortfolioEval, self).update_net(portfolio, reward)
        self.compute_active(portfolio)

    def after_eval(self, reset_metric=True, *args, **kwargs):
        super(SparsePortfolioEval, self).after_eval(reset_metric)
        self.compute_sparsity(reset_metric)

    def compute_sparsity(self, reset_metric):
        stock_num = len(self.weight[0]) - 1
        sparsity = (stock_num - numpy.array(self.active_stock_num)) / stock_num
        self.sparsity = torch.tensor(numpy.mean(sparsity))
        print('Final sparsity for {}_{} is: {}'.format(self._dataset, self._method, self.sparsity))


class IntradayPortfolioEval(PortfolioEval):
    def __init__(self, file_dir, method="", dataset="", frequency=252., transaction_cost=0.0025, account=1., **kwargs):
        super().__init__(file_dir, method, dataset, frequency=frequency, transaction_cost=transaction_cost,
                         account=account, **kwargs)
        self.last_close_price = []

    @staticmethod
    def calculate_normalizing(w, trading_price, close_price, w_o, last_close_price, transaction_cost=0.0025):
        if torch.sum(w) == 0.:
            net_reward = 1.
            turnovers = torch.tensor(0.)
            w_close = torch.cat([w, torch.tensor([1.])])
        else:
            w = torch.cat([w, torch.tensor([1. - torch.sum(w)])])
            if w_o is None:
                w_o = w
                is_not_first = False
            else:
                is_not_first = True

            r_before_trading = torch.cat([trading_price / last_close_price, torch.tensor([1.])])
            w_pre = w_o
            w_cur = w
            w_pre_o = w_o * r_before_trading / torch.dot(w_o.double(), r_before_trading.double())

            def f3(x):
                return numpy.array(
                    x - 1 + transaction_cost.numpy() * numpy.sum(numpy.abs(w_pre.numpy() - w_cur.numpy() * x)))

            net_proportion = torch.tensor(fsolve(f3, numpy.array(1.))[0], dtype=torch.float32)
            net_reward_before_trading = torch.dot(w.double(), r_before_trading.double()) * net_proportion

            r_after_trading = torch.cat([close_price / trading_price, torch.tensor([1.])])
            w_close = w * r_after_trading / torch.dot(w.double(), r_after_trading.double())
            net_reward_after_trading = torch.dot(w_close.double(), r_after_trading.double())

            turnovers = torch.sum(torch.abs(w_close - w_pre_o))
            net_reward = net_reward_before_trading * net_reward_after_trading if is_not_first else 1
        return w_close, net_reward, turnovers

    def update_net(self, portfolio: dict, trading_price: torch.Tensor, close_price: torch.Tensor, **kwargs):
        """
        main trading mechanism
        Args:
            portfolio: 投资组合权重
            trading_price: 交易时刻的价格
            close_price: 当天的收盘价格
        Returns:

        """
        w = torch.tensor(list(portfolio.values()))
        if len(self.weight_o) == 0.:
            w_o = None
        else:
            w_o = self.weight_o[-1]

        w_close, net_reward, turnovers = self.calculate_normalizing(w, trading_price, close_price, w_o,
                                                                    self.last_close_price, self.transaction_cost)
        self.turnovers += turnovers
        self.weight.append(list(w.numpy()))
        self.weight_o.append(w_close)
        self.net_reward.append(net_reward)
        self.last_close_price = close_price

    def eval(self, result: dict, data: RoundDictData, **kwargs):
        # 判断格式合法性
        if "portfolio" in result.keys() and "trading_timing" in result.keys():  # 决策和结果都有
            self.decision_result = result["portfolio"]
            trading_timing = result["trading_timing"].cpu()
        else:
            raise ModuleNotFoundError("Can't find portfolio or timing in result. Please check result item!")

        # 每轮的计算
        total_price = torch.stack(list(data.reward.values()), dim=0).cpu()

        close_price = total_price[:, -1]
        trading_price = torch.tensor([total_price[i, trading_timing[i]] for i in range(total_price.shape[0])])
        if len(self.last_close_price) == 0:
            self.last_close_price = total_price[:, -1]

        self.update_net(portfolio=self.decision_result, trading_price=trading_price, close_price=close_price)
        self.compute_cm()

        # 每轮的显示
        if self._times % EVAL_DISPLAY_BATCH == 0:
            acc_cm = self.cm_reward[-1]
            print('cm for {} after {} iterations on date {} is: {}'.format(self._method, self._times, data.round,
                                                                           acc_cm))


class MultipointIntradayPortfolioEval(IntradayPortfolioEval):
    def __init__(self, file_dir, method="", dataset="", frequency=252., transaction_cost=0.0025, account=1.,
                 candidate_size=10, **kwargs):
        super().__init__(file_dir, method, dataset, frequency=frequency, transaction_cost=transaction_cost,
                         account=account, **kwargs)
        self.candidate_size = candidate_size
        self.recall_list = []
        self.recall = torch.zeros(1)

        self.trading_points = []
        self.trading_direction = []
        self.prediction = []

    def compute_round_recall(self, trading_points, trading_direction, total_price):
        # [0:卖, 1:买]
        recall = 0
        for s in range(total_price.shape[0]):
            if trading_direction[s] == 1:
                _, true_points = torch.topk(total_price[s], self.candidate_size, largest=False, sorted=True)
            else:
                _, true_points = torch.topk(total_price[s], self.candidate_size, largest=True, sorted=True)
            trading_points_list = trading_points[s].cpu().numpy().tolist()
            true_points = true_points.cpu().numpy().tolist()

            recall += len(set(trading_points_list) & set(true_points)) / 10
        self.recall_list.append(recall / total_price.shape[0])

    def compute_recall(self, reset_metric):
        self.recall = torch.tensor(numpy.mean(self.recall_list))
        print('Final recall for {}_{} is: {}'.format(self._dataset, self._method, self.recall))

    def calculate_normalizing_weight(self, current_weight: torch.Tensor, trading_points, total_price, pre_weight, last_close_price):
        """

        Args:
            current_weight: tensor, # s
            trading_points: # s,k
            total_price: # s,m
            pre_weight: # s
            last_close_price:s

        Returns:
            normalizing_weight
            trading_point
        """
        close_price, trading_price, trading_point = self.calculate_prices(trading_points, total_price)
        normalizing_weight, _, _ = self.calculate_normalizing(current_weight, trading_price, close_price, pre_weight, last_close_price)
        return normalizing_weight, trading_point

    @staticmethod
    def calculate_prices(trading_points, total_price):
        """

        Args:
            trading_points:
            total_price:

        Returns:
            close_price: 1D-dimension vector (s) of close price
            trading_price: 1D-dimension vector (s) of trading price
            trading_point: 1D-dimension vector (s) to store the index
        """
        close_price = []
        for s in range(total_price.shape[0]):
            close_price_s = None
            for k in range(total_price.shape[1]):
                if total_price[s, total_price.shape[1] - k -1] != 0:
                    close_price_s = total_price[s, total_price.shape[1] - k -1]
                    break
            close_price.append(close_price_s)
        close_price = torch.tensor(close_price).to(trading_points.device)

        trading_price = []
        trading_point = []
        for s in range(total_price.shape[0]):
            trading_price_s = None
            trading_point_s = None
            for k in range(trading_points.shape[1]):
                success_rate = torch.rand(1)
                success_rate = 0
                if total_price[s, trading_points[s, k]] != 0 and success_rate < 0.9:
                    trading_price_s = (total_price[s, trading_points[s, k]])
                    trading_point_s = trading_points[s, k]
                    break
            if trading_price_s is None:
                trading_price_s = close_price[s]
                trading_point_s = total_price.shape[1]
            trading_price.append(trading_price_s)
            trading_point.append(trading_point_s)

        trading_price = torch.tensor(trading_price).to(trading_points.device)
        trading_point = torch.tensor(trading_point).to(trading_points.device)
        return close_price, trading_price, trading_point

    def eval(self, result: dict, data: RoundDictData, **kwargs):
        # 判断格式合法性
        if "portfolio" in result.keys() and "trading_points" in result.keys():  # 决策和结果都有
            self.decision_result = result["portfolio"]
            trading_points = result["trading_points"].cpu() # s, k
            trading_direction = result["direction"].cpu() # s, [0:卖, 1:买]
        else:
            raise ModuleNotFoundError("Can't find portfolio or timing in result. Please check result item!")

        self.trading_points.append(trading_points.tolist())
        self.trading_direction.append(trading_direction.tolist())
        self.prediction.append(result["prediction"].cpu())

        # 每轮的计算
        total_price = torch.stack(list(data.reward.values()), dim=0).cpu()
        close_price, trading_price, _ = self.calculate_prices(trading_points, total_price)
        if len(self.last_close_price) == 0:
            self.last_close_price = close_price

        self.update_net(portfolio=self.decision_result, trading_price=trading_price, close_price=close_price)
        self.compute_round_recall(trading_points=trading_points, trading_direction=trading_direction,
                                  total_price=total_price)
        self.compute_cm()

        # 每轮的显示
        if self._times % EVAL_DISPLAY_BATCH == 0:
            acc_cm = self.cm_reward[-1]
            print('cm for {} after {} iterations on date {} is: {}'.format(self._method, self._times, data.round,
                                                                           acc_cm))

    def after_eval(self, reset_metric=True, *args, **kwargs):
        super(MultipointIntradayPortfolioEval, self).after_eval(reset_metric)
        self.compute_recall(reset_metric)
