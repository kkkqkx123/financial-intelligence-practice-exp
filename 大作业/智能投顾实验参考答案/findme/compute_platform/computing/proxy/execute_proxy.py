import json
import os
import pickle
import time
from collections import Iterable

import numpy as np
import torch

from computing.proxy.data_proxy import DataProxy
from computing.proxy.metric_proxy import MetricProxy
from computing.proxy.model_proxy import ModelProxy
from computing.proxy import ALL_MODE_TYPE


class RandomPolicy:
    @staticmethod
    def decide(arm_set):
        return np.random.choice(arm_set)


class ExecuteProxy:
    def __init__(self, model: ModelProxy, data: DataProxy, metric: MetricProxy, result_dir=None, random_seed=0,
                 n_train_epoch=1, n_pretrain_epoch=1, load_execute_dir=None, load_model_mode=None,
                 save_file_dir=None,
                 device='cpu', save=False, **kwargs):
        """
        Args:
            model:ModelProxy
            data:DataProxy
            metric:MetricProxy
            result_dir: 若该运行保存的路径为自动生成路径，则需要指定一个大的路径
            random_seed: 用于给DRL一些拓展
            n_train_epoch: 训练的重复次数
            n_pretrain_epoch: 预训练的重复次数
            load_execute_dir:
                读取某次运行的参数。如该参数非None，则将从 load_execute_dir + model 中读取模型，从 load_execute_dir + metric
                中读取矩阵。
            load_model_mode:
                读取指定的模型，输入的是mode，详见ALL_MODE_TYPE
            save_file_dir:
                模型的保存路径。模型保存路径的优先级为 save_file_dir > load_execute_dir > 自动生成的路径，即：
                如该save_file_dir非None，则保存路径为save_file_dir；
                否则，如该 load_execute_dir 非None，则保存路径为 load_execute_dir；
                否则，自动生成保存路径
            device: 可选 'cuda' 或 'cpu'
            save: 可选 True 或 False，表示是否保存该运行
            **kwargs:
        """
        # =============== set model\data\metric proxy ================================
        self.load_execute_dir = load_execute_dir if load_execute_dir else None
        self.model = model if not load_execute_dir else self.load_model(absolute_file_mode=load_model_mode)
        self.data = data
        self.metrics = metric if not load_execute_dir else pickle.load(
            open(os.path.join(load_execute_dir, 'metric', 'metric.pt'), 'rb'))
        self.result_dir = result_dir

        # =============== set device ================================================
        if device not in ["cuda", "cpu"]:
            raise TypeError("We do not support %s device" % device)
        else:
            self.device = device
        self.model.set_device(self.device)
        self.data.set_device(self.device)

        # =============== set execute params (计数类的) =================================
        self.start_index = self.data.get_max_window_size()
        self.n_pretrain = 0
        self.n_train = 0
        self.n_validate = 0
        self.n_test = None
        if n_train_epoch < 0 or n_pretrain_epoch < 0:
            raise TypeError("Epoch must > 0!")
        else:
            self.n_train_epoch = n_train_epoch
            self.n_pretrain_epoch = n_pretrain_epoch
        self.dataset_len = self.data.get_dataset_len()

        # =============== set execute mode and data type =================================
        # 以下根据不同的pipeline子类自动设置
        self.mode_mapping = {}
        self.data_type_mapping = {}
        for mode in ALL_MODE_TYPE:
            self.mode_mapping[mode] = mode
            self.data_type_mapping[mode] = "dict"
        self.mode_mapping["Validation"] = "OnlineDecision"
        self.data_type_mapping["PreTraining"] = "tensor"

        # =============== set logs ==================================================
        # 记录运行的日志
        self.call_log = []
        self.execute_statistics = {}

        # 以下根据传入的execute_params控制
        # init n_mode_call，用于统计mode被call的次数，只有reset metric 才会加1
        if not load_execute_dir:
            self.n_mode_call = {}
            for mode in ALL_MODE_TYPE:
                self.n_mode_call[mode] = 0  # count before mapping mode !!
        else:
            self.n_mode_call = pickle.load(open(os.path.join(load_execute_dir, 'metric', 'n_mode_call.pt'), 'rb'))

        # =============== set save ==================================================
        self.save = save  # 控制是否要保存此次运行结果

        # 设置 save_file_dir 下的文件夹
        if self.save:
            # =============== set save root path ==================================================
            # 设置 save_file_dir，优先级 save_file_dir > load_execute_dir > 自动生成的路径
            if save_file_dir:
                self.save_file_dir = save_file_dir
            elif load_execute_dir:
                self.save_file_dir = load_execute_dir
            else:
                self.save_file_dir = os.path.join(result_dir,
                                                  self.model.get_model_name() + '_' + self.data.get_dataset_name()
                                                  + '_' + str(time.time()))
                print("file_hash:", self.save_file_dir.split("_")[-1])
            self.load_execute_dir = self.save_file_dir
            # =============== set save file path ==================================================
            # 设置 Metric 文件夹
            self.metric_save_dir = os.path.join(self.save_file_dir, "metric")
            if not os.path.exists(self.metric_save_dir):
                os.makedirs(self.metric_save_dir)

            # 设置 Model 文件夹
            self.model_save_path = os.path.join(self.save_file_dir, "model")
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

            # 设置 params 文件夹
            self.params_dir = os.path.join(self.save_file_dir, "params")
            if not os.path.exists(self.params_dir):
                os.makedirs(self.params_dir)
            json.dump(self.model.get_model_params_dict(), open(os.path.join(self.params_dir, "model_params.json"), 'w'))
            json.dump(self.model.get_model_tree_struct_from_model(),
                      open(os.path.join(self.params_dir, "model_tree_struct.json"), 'w'))
            json.dump(self.data.get_data_params_dict(), open(os.path.join(self.params_dir, "data_params.json"), 'w'))
            json.dump(self.get_execute_params_dict(), open(os.path.join(self.params_dir, "execute_params.json"), 'w'))

    def get_execute_params_dict(self):
        execute_dict = {}
        for k, v in self.__dict__.items():
            if not callable(v) and (
                    isinstance(v, Iterable) or isinstance(v, int) or isinstance(v, float)) and k not in ['call_log',
                                                                                                         'execute_statistics']:
                execute_dict[k] = v
        return execute_dict

    def set_execute_set_params(self, start_index=None, n_pretrain=None, n_train=None, n_validate=None, n_test=None,
                               n_pretrain_ratio=None, n_train_ratio=None, n_validate_ratio=None, n_test_ratio=None):
        def set_one_param(count, ratio, length, old):
            if count is not None and ratio is not None:
                raise
            elif count is not None:
                return count
            elif ratio is not None:
                return int(ratio * length)
            else:
                return old

        # =============== set execute params  ==================================================
        if type(start_index) == str and start_index is not None:
            date = self.data.data_reader.dataset.index_map_dict[
                'round_index_map']  # todo check data_proxy.data_reader.dataset.index_map_dict
            start_index = date.index(start_index)

        dataset_len = self.dataset_len
        self.n_pretrain = set_one_param(n_pretrain, n_pretrain_ratio, dataset_len, self.n_pretrain)
        self.n_train = set_one_param(n_train, n_train_ratio, dataset_len, self.n_train)
        self.n_validate = set_one_param(n_validate, n_validate_ratio, dataset_len, self.n_validate)
        self.n_test = set_one_param(n_test, n_test_ratio, dataset_len, self.n_test)

        self.start_index = start_index if start_index else self.start_index

        if self.n_test:
            if self.n_pretrain + self.n_train + self.n_validate + self.n_test > self.dataset_len:
                raise EOFError(
                    'The setting of self.n_pretrain + self.n_train + self.n_validate + self.n_test exceed the max '
                    'dataset length, please reset them!')
        else:
            if self.n_pretrain + self.n_train + self.n_validate > self.dataset_len:
                raise EOFError(
                    'The setting of self.n_pretrain + self.n_train + self.n_validate + self.n_test exceed the max '
                    'dataset len, please reset them!')
        if self.save:
            json.dump(self.get_execute_params_dict(), open(os.path.join(self.params_dir, "execute_params.json"), 'w'))

    def get_instance_name(self):
        return self.model.get_model_name()

    def execute(self, start_index, end_index, batch_size, mode, data_type, reset_metric, dataset=None,
                input_result=None, after_eval=True):
        """

        Args:
            start_index:
            end_index:
            batch_size:
            mode: 这个mode是面上的mode. 详见ALL_MODE_TYPE
                  {PreTraining：预训练, Training：正式模型训练, OnlineDecision：在线决策与更新, Decision:Updating, Validation}
            data_type: tensor or dict
            reset_metric:
            input_result:

        Returns:

        """
        if type(start_index) == str and start_index is not None:
            date = self.data.data_reader.dataset.index_map_dict[
                'round_index_map']  # todo check data_proxy.data_reader.dataset.index_map_dict，
            # todo 这里的date str -> start_index(int) 可能是不需要的
            start_index = date.index(start_index)
        results = None
        container_mode = self.mode_mapping[mode]
        # =============== fetch data ==============================================================
        self.data.set_fetch_batch(start_index=start_index, end_index=end_index, batch_size=batch_size)
        data_batch = self.data.fetch_next_batch(data_type=data_type)
        # =============== execute for batch size ==================================================
        n_executed_data = start_index
        while data_batch != -1:
            n_executed_data += batch_size
            results = self.model(data=data_batch, result=input_result, mode=container_mode,
                                 external_data=self.data.external_data)
            self.metrics.update(results, data_batch, mode=mode)
            data_batch = self.data.fetch_next_batch(data_type=data_type)
        # =============== execute for last batch ==================================================
        if n_executed_data < end_index:
            self.data.set_fetch_batch(start_index=n_executed_data, end_index=None,
                                      batch_size=end_index - n_executed_data)
            data_batch = self.data.fetch_next_batch()
            results = self.model(data=data_batch, mode=container_mode, external_data=self.data.external_data)
            self.metrics.update(results, data_batch, mode=mode)
        # =============== save model and metrics ==================================================
        if after_eval:
            self.after_eval(reset_metric=reset_metric, mode=mode)
        if self.save:
            self.save_metric(reset_metric=reset_metric, mode=mode)
            self.save_model(mode=mode)
        return results

    def after_eval(self, reset_metric=True, mode=None):
        self.metrics.after_eval(reset_metric=reset_metric, mode=mode)

    def save_model(self, mode=None):
        if mode:
            self.model.save(os.path.join(self.model_save_path, 'model_{}.pt'.format(mode)))
        self.model.save(os.path.join(self.model_save_path, 'model_complete.pt'))

    def save_metric(self, reset_metric=True, mode=None):
        self.metrics.save(reset_metric=reset_metric, mode=mode)

    def load_model(self, absolute_file_mode=None):
        # absolute_file_name 可以选择 ALL_MODE_TYPE, complete
        latest_modify = 0
        latest_filepath = None
        load_model_dir = os.path.join(self.load_execute_dir, "model")
        # =============== save model and metrics ==================================================
        if absolute_file_mode in ALL_MODE_TYPE + ["complete"]:
            latest_filepath = os.path.join(load_model_dir, "model_" + absolute_file_mode)
        elif absolute_file_mode is None:
            # 列出self.load_model_dir的文件
            for filename in os.listdir(load_model_dir):
                file_path = os.path.join(load_model_dir, filename)
                modify_time = os.path.getmtime(file_path)
                if modify_time > latest_modify:
                    latest_modify = modify_time
                    latest_filepath = file_path
        else:
            raise FileNotFoundError("Can not find such file : %s" % "model_" + absolute_file_mode)
        if latest_filepath:
            model = torch.load(latest_filepath)
            return ModelProxy(model=model)

    def __call__(self, mode, batch_size=None, start_index=None, end_index=None, reset_metric=True, result=None,
                 dataset=None, after_eval=True):
        """
        call function，作为分发执行的入口
        Args:
            mode: 详见ALL_MODE_TYPE
                  {PreTraining：预训练, Training：正式模型训练, OnlineDecision：在线决策与更新, Decision:Updating, Validation}
            batch_size:
            start_index: 初始的index，是数据集的index
            end_index: 结束的index，并不会取到这个index的数据
            reset_metric: 是否要重置metric，默认重置
            result: 专门用于Updating给一个传入

        Returns:

        """
        results = None

        # =============== handle reset metric ==================================================
        # 所有模式调用之前，metric都要初始化，并更改metric目录
        if reset_metric or self.n_mode_call[mode] == 0:
            if self.save:
                self.n_mode_call[mode] += 1
                metric_save_dir = self.metric_save_dir if self.metric_save_dir else ""
                self.metrics.set_metric(
                    file_dir=os.path.join(metric_save_dir, mode + '_' + str(self.n_mode_call[mode])), mode=mode,
                    method=self.model.get_model_name())
            else:
                self.n_mode_call[mode] += 1
                self.metrics.set_metric(file_dir=None, mode=mode, method=self.model.get_model_name())

        else:
            self.n_mode_call[mode] += 1


        # if self.save:
        #     self.metrics.change_file_dir(
        #         file_dir=os.path.join(self.metric_save_dir, mode + '_' + str(self.n_mode_call[mode])), mode=mode)
        if self.save and reset_metric:
            self.metrics.change_file_dir(
                file_dir=os.path.join(self.metric_save_dir, mode + '_' + str(self.n_mode_call[mode])), mode=mode)

        batch_size = batch_size if batch_size else self.data.get_batch_size()

        if self.save or self.load_execute_dir:
            self.load_model()

        # =============== handle all modes ==================================================
        if type(start_index) == str and start_index is not None:
            date = self.data.data_reader.dataset.index_map_dict['round_index_map']  # todo check data_proxy.data_reader.dataset.index_map_dict
            start_index = date.index(start_index)
        s = start_index
        e = end_index

        # =============== handle PreTraining：预训练 ==================================================
        if mode == "PreTraining":
            s = start_index if (start_index and start_index > self.data.get_max_window_size()) else self.start_index
            e = end_index if (end_index and self.data.get_max_window_size() < end_index < self.dataset_len) \
                else self.start_index + self.n_pretrain
            if self.n_pretrain > 0 or s < e:
                for i_epoch in range(self.n_pretrain_epoch):
                    results = self.execute(batch_size=batch_size, start_index=s, end_index=e,
                                           mode=mode, dataset=dataset, after_eval=after_eval,
                                           data_type=self.data_type_mapping[mode], reset_metric=reset_metric)

        # =============== handle Training：正式模型训练  ==================================================
        elif mode == "Training":
            s = start_index if (start_index and start_index > self.data.get_max_window_size()) \
                else self.start_index + self.n_pretrain
            e = end_index if (end_index and self.data.get_max_window_size() < end_index < self.dataset_len) \
                else self.start_index + self.n_pretrain + self.n_train
            if self.n_train > 0 or s < e:
                for i_epoch in range(self.n_train_epoch):
                    results = self.execute(batch_size=batch_size, start_index=s, end_index=e,
                                           mode=mode, dataset=dataset, after_eval=after_eval,
                                           data_type=self.data_type_mapping[mode], reset_metric=reset_metric)

        # =============== handle Validation：校验 ==================================================
        elif mode == "Validation":
            s = start_index if (
                    start_index and start_index > self.data.get_max_window_size() and end_index < self.dataset_len) \
                else self.start_index + self.n_pretrain + self.n_train
            e = end_index if (end_index and self.data.get_max_window_size() < end_index < self.dataset_len) \
                else self.start_index + self.n_pretrain + self.n_train + self.n_validate
            if self.n_validate > 0 or s < e:
                results = self.execute(batch_size=batch_size, start_index=s, end_index=e, mode=mode,
                                       reset_metric=reset_metric, dataset=dataset, after_eval=after_eval,
                                       data_type=self.data_type_mapping[mode])

        elif mode == "OnlineDecision" or mode == "Decision":
            s = start_index if (start_index and start_index >= self.data.get_max_window_size()) \
                else self.start_index + self.n_pretrain + self.n_train + self.n_validate
            if self.n_test:
                e = end_index if (end_index and self.data.get_max_window_size() < end_index < self.dataset_len) \
                    else self.start_index + self.n_pretrain + self.n_train + self.n_validate + self.n_test
            else:
                e = end_index if (end_index and self.data.get_max_window_size() <= end_index < self.dataset_len) \
                    else self.dataset_len

            results = self.execute(batch_size=batch_size, start_index=s, end_index=e, mode=mode, dataset=dataset,
                                   data_type=self.data_type_mapping[mode], reset_metric=reset_metric,
                                   after_eval=after_eval)

        # =============== handle Updating：更新模型 ==================================================
        elif mode == "Updating":
            if start_index is None or end_index is None:
                raise TypeError("Can't Updating without start_index and end_index. Please set start and end index!")
            s = start_index
            e = end_index
            results = self.execute(batch_size=batch_size, start_index=s, end_index=e, mode=mode,
                                   data_type=self.data_type_mapping[mode], dataset=dataset, after_eval=after_eval,
                                   reset_metric=reset_metric, input_result=result)

        if mode not in ALL_MODE_TYPE:
            raise NameError("The given mode in unavailable. Please choose mode in {}".format(ALL_MODE_TYPE))

        # =============== handle metrics and logs saving ==================================================
        if self.save:
            self.save_metric(mode=mode, reset_metric=reset_metric)
            pickle.dump(self.metrics, open(os.path.join(self.metric_save_dir, 'metric.pt'), 'wb'))
            pickle.dump(self.n_mode_call, open(os.path.join(self.metric_save_dir, 'n_mode_call.pt'), 'wb'))

            # 保留log
            self.call_log.append(['mode={}, start_index={}, end_index={}'.format(mode, s, e)])
            # 保留统计值
            if mode in self.execute_statistics.keys():
                self.execute_statistics[mode]['length'] += e - s
            else:
                self.execute_statistics[mode] = {}
                self.execute_statistics[mode]['start_index'] = s
                self.execute_statistics[mode]['length'] = e - s
            execute_info = {'execute_statistics': self.execute_statistics, 'call_log': self.call_log}
            json.dump(self.get_execute_params_dict(), open(os.path.join(self.params_dir, "execute_params.json"), 'w'))
            json.dump(execute_info, open(os.path.join(self.params_dir, "execute_log.json"), 'w'))

        return results


class OnlineExecuteProxy(ExecuteProxy):
    def __init__(self, model: ModelProxy, data: DataProxy, metric: MetricProxy, **kwargs):
        super(OnlineExecuteProxy, self).__init__(model, data, metric, **kwargs)
        self.mode_mapping["Training"] = "OnlineDecision"
        self.mode_mapping["Validation"] = "OnlineDecision"


class OnlineTensorExecuteProxy(ExecuteProxy):
    """
    ExecuteProxy for online learning algorithm using tensors
    """

    def __init__(self, model: ModelProxy, data: DataProxy, metric: MetricProxy, **kwargs):
        super(OnlineTensorExecuteProxy, self).__init__(model, data, metric, **kwargs)
        self.mode_mapping["Training"] = "OnlineDecision"
        self.mode_mapping["Validation"] = "OnlineDecision"
        for mode in self.data_type_mapping.keys():
            self.data_type_mapping[mode] = "tensor"



class DRLExecuteProxy(ExecuteProxy):
    """
    ExecuteProxy for Deep Reinforcement Learning
    """

    def __init__(self, model: ModelProxy, data: DataProxy, metric: MetricProxy, **kwargs):
        super(DRLExecuteProxy, self).__init__(model, data, metric, **kwargs)
        self.mode_mapping["Validation"] = "OnlineDecision"
        for mode in self.data_type_mapping.keys():
            self.data_type_mapping[mode] = "tensor"