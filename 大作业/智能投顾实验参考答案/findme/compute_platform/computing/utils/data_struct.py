import os
import json
from pathlib import Path
from typing import Union, Optional
from collections import UserList

import torch
from torch.utils.data import Dataset

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
DATA_TYPE = json.load(open(Path(PARENT_DIR + "/data_type_config.json")))

ENVIRONMENT_KEYS_TYPE = DATA_TYPE["environment_keys_type"]
SERIES_KEYS_TYPE = DATA_TYPE["series_keys_type"]


class SeriesDictData(UserList):
    def __init__(self, index_map_dict: Optional[dict] = None, data_list=[], window_size=1, context_window_size=1,
                 pre_dataset=None, init_index=0, packed: bool = True):
        '''
        :param index_map_dict:
        :param l: 存放了多个RoundDictData的列表
        :param pre_dataset:
        '''
        super(SeriesDictData, self).__init__(initlist=data_list)
        self.index_map_dict = index_map_dict
        self.window_size = window_size
        self.context_window_size = context_window_size
        self.pre_dataset = pre_dataset
        self.packed = packed
        self.init_index = init_index

        if self.pre_dataset and self.pre_dataset.pre_dataset is not None:
            self.pre_dataset.pre_dataset = None

    def get_total_length(self):
        return len(self.index_map_dict["round_index_map"])

    def set_pre_dataset(self, pre_dataset):
        if pre_dataset is None:
            self.pre_dataset = None
        elif not isinstance(pre_dataset, self.__class__):
            raise TypeError('The type of pre_dataset can only be {}'.format(self.__class__))
        else:
            self.pre_dataset = pre_dataset

    def __getitem__(self, item: Union[slice, int]):
        """
        :param item:
        :return:
        实现思路：
        - 获取 slice 的start, end, step，获取最大的 max_window_size
        - 如果start < max_window_size - 1, 则获取从 pre_dataset 中获取缺失的数据列表: pre_dataset.get_packed_item([: window_size - 1 - start], return_tyoe="list")
        - 对于每个item生成一个RoundDictData，其中
            - reward 按照 window size 获取 pre_dataset 和 self.data_list 中的 reward。具体方法: 调用两个数据集的 dataset.get_packed_item(slice, 'list').reward 方法，
                循环获得的data，获取reward，并拼接为tensor
            - series_context: 循环series_context的key，按照上述 reward 的方法，获得 context。然后，如果是list元素，拼接为list；如果是tensor或special，则拼接为tensor
            - environment_context：同上
            - round: 查找当前的 Round
        """

        # ===========================  如果本身数据是 unpack 的则直接返回数据 ==========================
        if not self.packed:
            if isinstance(item, slice):
                init_index = self.init_index + item.start if isinstance(item, slice) else self.init_index + item
                return_data = self.__class__(index_map_dict=self.index_map_dict, data_list=self.data[item],
                                             packed=False,
                                             window_size=self.window_size, context_window_size=self.context_window_size,
                                             pre_dataset=None, init_index=init_index)
            else:
                return_data = self.data[item]
            return return_data

        # ===========================  如果本身数据是 pack 的则执行以下操作 ==========================
        # ===========================  获取本次 getitem 的数据列表，即 query_data ==========================
        max_window_size = max(self.window_size, self.context_window_size)
        query_data = None
        # 获取 slice 的start, end, step，获取最大的 max_window_size
        input_start, input_stop, input_step = [item.start, item.stop, item.step] if isinstance(item, slice) \
            else [item, item + 1, None]
        # 如果start < max_window_size - 1, 则获取从 pre_dataset 中获取缺失的数据列表:
        # pre_dataset.get_packed_item([: window_size - 1 - start], return_type="list"
        if input_start < max_window_size:
            if self.pre_dataset is None:
                raise EOFError('Pre_dataset is None and input_start is less then max_window_size, '
                               'so cannot get unpacked samples.')
            previous_data = self.pre_dataset.get_packed_item(
                slice(len(self.pre_dataset) - max_window_size - 1 + input_start, len(self.pre_dataset)))
            current_data = self.get_packed_item(slice(input_start, input_stop))
            query_data = previous_data
            query_data.extend(current_data)
        else:
            query_data = self.get_packed_item(slice(input_start - max_window_size, input_stop))

        # =========================== 对于每个item生成一个RoundDictData =================================
        init_index = self.init_index + item.start if isinstance(item, slice) else self.init_index + item
        return_data = self.__class__(index_map_dict=self.index_map_dict, data_list=[], packed=False,
                                     window_size=self.window_size, context_window_size=self.context_window_size,
                                     pre_dataset=None, init_index=init_index)

        for sample_index, round_dict_data in enumerate(query_data.data[max_window_size:]):
            reward = round_dict_data.reward
            round = round_dict_data.round

            # unpack 处理 series_context 数据，
            series_contexts = {}
            window_query_data = query_data.data[
                                sample_index: sample_index + max_window_size + 1]  # 总共 前面三个window_size的数据+当前数据
            # 循环每一个 context 类型
            for context_key, context_value in round_dict_data.series_context.items():
                context_data = {}
                # 循环每一个 series
                for series_key, series_context in context_value.items():
                    series_context = []
                    if SERIES_KEYS_TYPE[context_key] == 'list':
                        for i in range(self.context_window_size):
                            series_context.append(window_query_data[-self.context_window_size + i + 1].
                                                  series_context[context_key][series_key])
                        context_data[series_key] = series_context
                    elif SERIES_KEYS_TYPE[context_key] == 'tensor':
                        for i in range(self.context_window_size):
                            series_context.append(window_query_data[-self.context_window_size + i + 1].
                                                  series_context[context_key][series_key].unsqueeze(0))
                        context_data[series_key] = torch.cat(series_context, dim=0)
                    elif SERIES_KEYS_TYPE[context_key] == 'special':
                        for i in range(self.window_size):
                            series_context.append(window_query_data[-self.window_size + i + 1].
                                                  series_context[context_key][series_key])
                        context_data[series_key] = torch.cat(series_context, dim=0)
                series_contexts[context_key] = context_data

            # unpack 处理 environment_context
            environment_contexts = {}
            # 循环每个 context 类型
            for context_key, context_value in round_dict_data.environment_context.items():
                environment_context = []
                for i in range(self.context_window_size):
                    environment_context.append(
                        window_query_data[-self.context_window_size + i + 1].environment_context[context_key])
                environment_contexts[context_key] = environment_context

            # 生成数据实例
            data = RoundDictData(round_=round, series_context=series_contexts,
                                 environment_context=environment_contexts, reward=reward)
            return_data.append(data)

        # ========================================== 返回结果 ==========================================
        if len(return_data) <= 0:
            return None
        elif isinstance(item, int):
            return return_data.data[0]
        else:
            return return_data

    def get_packed_item(self, item: Union[int, slice], return_type="list"):
        # ========================================== 异常检测 ===========================================
        # 检查，unpack 数据不能调用该函数
        if not self.packed:
            raise TypeError('Unpack item cannot call get_packed_item()')

        # 检查：pre_dataset 必须为 packed
        if self.pre_dataset and not self.pre_dataset.packed:
            raise TypeError('pre_dataset must be packed')

        # ==================================== 更新 pre_dataset =====================================
        input_start = item.start if isinstance(item, slice) else item
        # 如果 pre_dataset 是空，则 pre_dataset 为 item 前的数据
        if self.pre_dataset is None:
            pre_dataset_list = self.data[0: input_start]
            init_index = self.init_index
        # 如果 pre_dataset 非空，则原来的 pre_dataset 拼接当前数据
        else:
            pre_dataset_list = self.pre_dataset.data + self.data[0: input_start]
            init_index = self.pre_dataset.init_index
        pre_dataset = self.__class__(index_map_dict=self.index_map_dict, data_list=pre_dataset_list, packed=True,
                                     window_size=self.window_size, context_window_size=self.context_window_size,
                                     pre_dataset=None, init_index=init_index)

        # =========================================== 返回数据 ======================================
        if isinstance(item, int):
            return super().__getitem__(item)
        else:
            round_dict_data_list = self.data[item]
            return self.__class__(index_map_dict=self.index_map_dict, data_list=round_dict_data_list, packed=True,
                                  window_size=self.window_size, context_window_size=self.context_window_size,
                                  pre_dataset=pre_dataset, init_index=self.init_index + item.start)

    def set_window_size(self, window_size: dict):
        self.window_size = window_size["window_size"] if 'window_size' in window_size else self.window_size
        self.context_window_size = window_size["context_window_size"] if 'context_window_size' in window_size \
            else self.context_window_size


class RoundDictData(object):
    """
    round: int or str
    reward: dict, include {key:value}, key is str, value is tensor
    series_context: dict, include {key:value}, key is series_name (str), value is tensor
    environment_context: dict, include {key:value}, key is series_name (str), value is tensor
    """

    def __init__(self, round_, reward: Optional[dict] = None, series_context: Optional[dict] = None,
                 environment_context: Optional[dict] = None):
        self.round = round_
        self.reward = reward
        self.series_context = series_context
        self.environment_context = environment_context

    def convert_reward_to_tensor(self, index_map_dict: Optional[dict], reward: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param index_map_dict: 包含series和round两个index的list
        :param reward: shape 为 (n_round, n_series) 大小的 tensor
        :return: shape 为 (n_round + 1, n_series) 大小的 tensor

        实现思路：
        - 计算当前数据的 reward tensor
        - 如果reward是空，则返回 (n_series, reward) 的 tensor
        - 如果reward非空，则返回 拼接原有 reward 和 (n_series, reward) 的 tensor

        """
        # =============================== 计算当前数据的 reward tensor =============================
        rewards_list = []
        for series in index_map_dict['series_index_map']:
            if series not in self.reward.keys():
                raise KeyError('Keys do not match index_map_dict, conversion failed!')
            series_reward = self.reward[series]
            rewards_list.append(series_reward)
        tensor_rewards = torch.cat(rewards_list).unsqueeze(0)

        # =============================== 返回拼接的 reward tensor =============================
        if reward is None:
            return tensor_rewards
        else:
            return torch.cat((reward, tensor_rewards), dim=0)

    def convert_environment_context_to_tensor(self, environment_context):
        """
        实现思路：
        - 计算当前数据的 context tensor
        - 如果reward是空，① 如果是 packed 模式，则返回 (, news) 的 tensor ② 如果是 unpacked 模式，则返回 (n_series, reward)
        - 如果reward非空，① 如果是 packed 模式，则返回 (n_series, reward) 的 tensor ② 如果是 unpacked 模式，则返回 (n_series, reward)
        """
        for context_key, context_value in self.environment_context.items():
            context = self.environment_context[context_key]
            if ENVIRONMENT_KEYS_TYPE[context_key] == 'list':
                if context_key in environment_context:
                    environment_context[context_key].append(context)
                else:
                    environment_context[context_key] = [context]
            else:
                if context_key in environment_context:
                    environment_context[context_key] = torch.cat((environment_context[context_key],
                                                                  context.unsqueeze(0)), dim=0)
                else:
                    environment_context[context_key] = context.unsqueeze(0)

        return environment_context

    def convert_series_context_to_tensor(self, index_map_dict: Optional[dict], series_context, packed):
        """
        实现思路：
        - 计算当前数据的 context tensor
        - 如果reward是空，① 如果是 packed 模式，则返回 (, news) 的 tensor ② 如果是 unpacked 模式，则返回 (n_series, reward)
        - 如果reward非空，① 如果是 packed 模式，则返回 (n_series, reward) 的 tensor ② 如果是 unpacked 模式，则返回 (n_series, reward)
        """
        # 遍历 context 类型
        for context_key, context_value in self.series_context.items():
            # if context_key != 'news':
            #     return {}
            context = self.series_context[context_key]
            # 遍历 series
            series_context_list = []
            if SERIES_KEYS_TYPE[context_key] == 'list':
                for series in index_map_dict['series_index_map']:
                    if series not in self.series_context[context_key].keys():
                        raise KeyError('Keys do not match index_map_dict, conversion failed!')
                    series_context_list.append(context[series])
                if context_key in series_context:
                    series_context[context_key].append(series_context_list)
                else:
                    series_context[context_key] = [series_context_list]
            elif SERIES_KEYS_TYPE[context_key] == 'tensor':
                for series in index_map_dict['series_index_map']:
                    if series not in self.series_context[context_key].keys():
                        raise KeyError('Keys do not match index_map_dict, conversion failed!')
                    series_context_list.append(context[series].unsqueeze(0))
                series_context_tensor = torch.cat(series_context_list, dim=0).unsqueeze(0)
                if packed:
                    series_context_tensor = series_context_tensor.unsqueeze(1)
                else:
                    # x1, x2, x3, x4 = series_context_tensor.shape
                    # series_context_tensor = series_context_tensor.reshape((x1, x3, x2, x4))
                    series_context_tensor = series_context_tensor.transpose(1, 2).contiguous()
                if context_key in series_context:
                    series_context[context_key] = torch.cat((series_context[context_key], series_context_tensor), dim=0)
                else:
                    series_context[context_key] = series_context_tensor
            else:
                for series in index_map_dict['series_index_map']:
                    if series not in self.series_context[context_key].keys():
                        raise KeyError('Keys do not match index_map_dict, conversion failed!')
                    series_context_list.append(context[series].unsqueeze(0))
                series_context_tensor = torch.cat(series_context_list, dim=0).unsqueeze(0)
                if packed:
                    series_context_tensor = series_context_tensor.unsqueeze(1)
                    series_context_tensor = series_context_tensor.squeeze(3)
                else:
                    series_context_tensor = series_context_tensor.transpose(1, 2).contiguous()
                if context_key in series_context:
                    series_context[context_key] = torch.cat((series_context[context_key], series_context_tensor), dim=0)
                else:
                    series_context[context_key] = series_context_tensor
        return series_context


class SeriesTensorData(Dataset):
    """
    maintain the information of series data in the form of Tensor

    """

    def __init__(self, index_map_dict: dict, reward: Optional[torch.Tensor] = None, iter_index: int = 0,
                 window_size: int = 1, pre_dataset=None, init_index: int = 0, packed: bool = True):
        """

        Args:
            index_map_dict(Dict): an important file for formulate rounds and series
            reward:(Tensor): the label or the reward that your algorithm refers to
            iter_index(int): change the dim of iterating
            window_size(int): set the size of sliding window
            pre_dataset: mount previous dataset for handling slide window
            init_index: the current round for the first record
            packed: the data is packed or unpacked by slide window
        """

        super(SeriesTensorData, self).__init__()
        self.reward = reward
        self.index_map_dict = index_map_dict
        self.iter_index = iter_index
        self.window_size = window_size
        self.pre_dataset = pre_dataset  # 前一个seg的数据拼接上这个seg的一部分信息，用于解决slide window的
        self.init_index = init_index  # 用于记录第一个index在整个index_map_dict中的位置
        self.packed = packed

    def set_iter_mode(self, iter_index):
        # todo: 没有支持
        assert iter_index == 0 or iter_index == 1
        self.iter_index = iter_index

    def set_window_size(self, window_size: int):
        self.window_size = window_size

    def convert_reward_to_dict(self):
        if self.reward is None:
            return {}

        round_index_map = self.index_map_dict.get('round_index_map')
        series_index_map = self.index_map_dict.get('series_index_map')
        reward_dic = {}

        if series_index_map is None:  # 虽然不允许发生，但是还是支持一下
            for i in range(self.reward.shape[0]):
                round_ = round_index_map[i + self.init_index]
                one_round_reward_dic = {}
                for j in range(self.reward.shape[1]):
                    one_round_reward_dic[j] = self.reward[i][j]
                reward_dic[round_] = one_round_reward_dic
        else:
            for i in range(self.reward.shape[0]):
                round_ = round_index_map[i + self.init_index]
                one_round_reward_dic = {}
                for j in range(self.reward.shape[1]):
                    series = series_index_map[j]
                    one_round_reward_dic[series] = self.reward[i][j]
                reward_dic[round_] = one_round_reward_dic

        return reward_dic

    def convert_series_context_to_dict(self):
        return None

    def convert_environment_context_to_dict(self):
        return None

    def append(self, data):
        if data.__class__.__name__ == "RoundDictData":
            self.index_map_dict['round_index_map'] = data.convert_round_index_map(
                self.index_map_dict['round_index_map'])
            self.reward = data.convert_reward_to_tenosr(self.reward)

        elif data.__class__.__name__ == self.__class__.__name__:
            if self.reward is not None:
                self.reward = torch.cat([self.reward, data.reward], dim=0)
        return self

    def convert_device(self, data: Union[list, dict, torch.Tensor], device) -> Union[list, dict, torch.Tensor]:
        if isinstance(data, list):
            device_data = []
            for item in data:
                device_item = self.convert_device(item, device)
                device_data.append(device_item)
        elif isinstance(data, dict):
            device_data = {}
            for key, value in data.items():
                device_data[key] = self.convert_device(value, device)
        elif isinstance(data, torch.Tensor):
            if device == "cuda":
                device_data = data.cuda()
            else:
                device_data = data.cpu()
        else:
            device_data = data
        return device_data

    def cuda(self):
        self.reward = self.convert_device(self.reward, "cuda")
        return self

    def cpu(self):
        self.reward = self.convert_device(self.reward, "cpu")

    def __len__(self):
        return len(self.reward)

    def get_total_length(self):
        return len(self.index_map_dict["round_index_map"])

    def get_sample_init_index(self, item, window_size=0):
        # 寻找sample_init_index
        if isinstance(item, slice):
            if item.start is None:
                if self.packed:
                    item_start = self.init_index
                else:
                    item_start = 0
            else:
                item_start = item.start
            if item.stop is None:
                if self.packed:
                    item_end = self.get_total_length()
                else:
                    item_end = len(self)
            else:
                item_end = item.stop
            if self.packed:
                item = slice(item_start - self.init_index, item_end - self.init_index, item.step)
                sample_init_index = item_start
            else:
                sample_init_index = item_start + self.init_index
        elif isinstance(item, int):
            item_start = item
            item_end = item + 1
            if self.packed:
                item = item - self.init_index
                sample_init_index = item_start
            else:
                sample_init_index = item_start + self.init_index
        else:
            raise TypeError("item is not slice or int!")

        # 处理一些溢出和slide window size不合法的问题
        # 取的最后一个数据比这个seg的数据更大，需要切换seg
        if item_end - 1 >= self.init_index + len(self):
            raise StopIteration("item end是%d，现在的长度是%d，溢出了！！！" % (item_end - 1, self.init_index + len(self)))
        # 如果是未展开的数据，需要计算window_size的长度是否都可以取到
        if self.packed and self.pre_dataset is None and item_start < window_size:
            raise IndexError("The window_size is too large! Please check the start index.")
        if self.pre_dataset is not None:
            pre_length = len(self.pre_dataset)
            # print(item_start)
            # print(pre_length)
            # print(window_size)
            if self.packed and pre_length + item_start < window_size:
                raise IndexError("The window_size is too large! Please check the start index.")
        return item, sample_init_index

    def get_sample_tensor(self, tensor: torch.Tensor, item: Union[int, slice], window_size, pad_num=0, pre_tensor=None):
        def tensor_pad(pad_size, current_tensor, pad_num_=0, pre_tensor_=None):
            pad_shape = [current_tensor.shape]
            pad_shape[0] = pad_size
            if pre_tensor_ is not None:
                pad = pre_tensor_[-pad_size:]
            else:
                if pad_num_ == 1:
                    pad = torch.ones(pad_shape)
                else:
                    pad = torch.zeros(pad_shape)
            return torch.cat([pad, current_tensor], dim=0)

        def get_window_size_tensor(packed, current_index, window_size_, current_tensor, previous_tensor):
            """

            Args:
                packed:
                current_index:
                window_size_:
                current_tensor:
                previous_tensor:

            Returns:

            """
            if current_index == -1 and previous_tensor is None:
                raise IndexError("The window_size is too large! Please check the start index.")
            if current_index == -1:
                if packed is False:
                    return previous_tensor[-1].unsqueeze(0)
                else:
                    return previous_tensor[-window_size_:]
            # 已经slide window展开的，则返回一条数据
            if packed is False:
                return current_tensor[current_index].unsqueeze(0)
            # 没有slide window展开的，返回slide window展开的数据
            else:
                if index < window_size_ - 1:
                    return tensor_pad(window_size_ - (current_index + 1),
                                      current_tensor[0:current_index + 1],
                                      pad_num,
                                      previous_tensor)
                else:
                    return current_tensor[current_index - window_size + 1: current_index + 1]

        if isinstance(item, int):
            sample_tensor = get_window_size_tensor(self.packed, item, window_size, tensor, pre_tensor)
            if window_size > 1 and self.packed:
                sample_tensor = sample_tensor.unsqueeze(0)
            return sample_tensor
        elif isinstance(item, slice):
            tensor_list = []
            new_item = item
            if item.start == -1:
                window_tensor = get_window_size_tensor(self.packed, item.start, window_size, tensor, pre_tensor)
                if window_size > 1 and self.packed:
                    window_tensor = window_tensor.unsqueeze(0)
                tensor_list.append(window_tensor)
                new_item = slice(0, item.stop, item.step)
            for index in range(len(tensor))[new_item]:
                window_tensor = get_window_size_tensor(self.packed, index, window_size, tensor, pre_tensor)
                if window_size > 1 and self.packed:
                    window_tensor = window_tensor.unsqueeze(0)
                tensor_list.append(window_tensor)
            sample_tensor = torch.cat(tensor_list)
            return sample_tensor

    def get_sample_list(self, list_data: list, item, window_size: int, pre_list: list = []) -> list:
        def get_window_size_list(packed, current_index, window_size_, current_list, previous_list):
            """
            处理两种情况：如果已经slide window展开的，则返回一条数据；否则返回slide window展开的数据；
            Args:
                packed: 是否packed
                current_index: 当前要取的数据
                window_size_: slide window size
                current_list: 当前的数据
                previous_list: 上一seg的数据

            Returns:

            """
            if current_index == -1 and len(previous_list) == 0:
                raise IndexError("The window_size is too large! Please check the start index.")
            if current_index == -1:
                if packed is False:
                    return previous_list[-1]
                else:
                    return previous_list[-window_size_:]
            # 已经slide window展开的，则返回一条数据
            if packed is False:
                return current_list[current_index]
            # 没有slide window展开的，返回slide window展开的数据
            else:
                if index < window_size_ - 1:
                    if len(previous_list) > 0:
                        l1 = previous_list[- window_size_ + (current_index + 1):]
                    else:
                        l1 = [None] * window_size_ - (current_index + 1)
                    l2 = current_list[0:current_index + 1]
                    return l1 + l2
                else:
                    return current_list[current_index - window_size + 1: current_index + 1]

        if isinstance(item, int):
            return [get_window_size_list(self.packed, item, window_size, list_data, pre_list)]
        elif isinstance(item, slice):
            merge_list = []
            new_item = item
            if item.start == -1:
                merge_list.append(get_window_size_list(self.packed, item.start, window_size, list_data, pre_list))
                new_item = slice(0, item.stop, item.step)
            for index in range(len(list_data))[new_item]:
                merge_list.append(get_window_size_list(self.packed, index, window_size, list_data, pre_list))
            return merge_list

    @staticmethod
    def get_pre_tensor(tensor: torch.Tensor, item: slice, pre_tensor=None) -> Optional[torch.Tensor]:
        if pre_tensor is not None:
            pre_tensor = torch.cat([pre_tensor, tensor])
        elif tensor is not None:
            pre_tensor = tensor[:item.start]
        else:
            pre_tensor = None
        return pre_tensor

    @staticmethod
    def get_pre_list(list_data: list, item: slice, pre_list: list = None) -> Optional[list]:
        if list_data is None:
            pre_list = pre_list
        elif pre_list is not None:
            pre_list += list_data[:item.start]
        else:
            pre_list = list_data[:item.start]
        return pre_list

    def get_packed_item(self, item: slice):
        if self.packed is False:
            raise TypeError("Can not get packed item from unpacked data!")
        # 压缩数据，减少内存损耗
        sample_reward = None
        pre_dataset = self.pre_dataset

        if self.pre_dataset is None:
            pre_reward = None
            if self.reward is not None:
                sample_reward = self.reward[item]
                pre_reward = self.get_pre_tensor(self.reward, item, None)
            pre_dataset = SeriesTensorData(index_map_dict=self.index_map_dict,
                                           reward=pre_reward,
                                           iter_index=self.iter_index,
                                           window_size=self.window_size,
                                           pre_dataset=None, packed=self.packed)
        else:
            if self.reward is not None:
                sample_reward = self.reward[item]
                if pre_dataset.packed is False:
                    raise TypeError("Can not get packed item from unpacked data!")
                pre_dataset.reward = self.get_pre_tensor(self.reward, item,
                                                         self.pre_dataset.reward)
        return SeriesTensorData(index_map_dict=self.index_map_dict,
                                reward=sample_reward,
                                iter_index=self.iter_index,
                                window_size=self.window_size,
                                pre_dataset=pre_dataset, init_index=item.start, packed=self.packed)

    def __getitem__(self, item):
        item, sample_init_index = self.get_sample_init_index(item, 0)

        sample_reward = None

        if self.reward is not None:
            sample_reward = self.get_sample_tensor(self.reward, item, 1, 1,
                                                   self.pre_dataset.reward)
        return SeriesTensorData(index_map_dict=self.index_map_dict,
                                reward=sample_reward,
                                iter_index=self.iter_index,
                                window_size=self.window_size, init_index=sample_init_index, packed=False)


class ContextSeriesTensorData(SeriesTensorData):
    """
    maintain the information of series data in the form of Tensor with context

    """

    def __init__(self, index_map_dict: dict, series_context: Optional[dict] = None,
                 environment_context: Optional[dict] = None,
                 reward: Optional[torch.Tensor] = None, init_index: int = 0,
                 window_size=1, context_window_size=1, pre_dataset=None, packed=True):
        """

        Args:
            index_map_dict(Dict): an important file for formulate rounds and series
            series_context(Dict): the input series features, context and so on,
                                  value shape [series, rounds, context_features]
            environment_context(Dict): the input environment features, context, value shape [rounds, context_features]
            reward(Tensor): the label or the reward that your algorithm refers to
            window_size(Dict): a dict of some size of sliding window
            pre_dataset: mount previous dataset for handling slide window
            init_index(int): the current round for the first record
        """

        super(ContextSeriesTensorData, self).__init__(index_map_dict=index_map_dict,
                                                      init_index=init_index,
                                                      window_size=window_size,
                                                      reward=reward, pre_dataset=pre_dataset, packed=packed)
        # =============== set environment_context ================================
        self.environment_context = {}
        for key in ENVIRONMENT_KEYS_TYPE.keys():
            if key in environment_context.keys() and environment_context[key] is not None:
                self.environment_context[key] = environment_context[key]

        # =============== set series_context ====================================
        self.series_context = {}
        for key in SERIES_KEYS_TYPE.keys():
            if key in series_context.keys() and series_context[key] is not None:
                self.series_context[key] = series_context[key]

        # =============== set context_window_size ================================
        self.context_window_size = context_window_size

    def set_window_size(self, window_size: dict):
        self.window_size = window_size["window_size"]
        self.context_window_size = window_size["context_window_size"]

    def append(self, data):
        # if list, add key; if tensor, add torch.cat
        if data.__class__.__name__ == self.__class__.__name__:
            # =============== append reward ====================================
            if self.reward is not None:
                self.reward = torch.cat([self.reward, data.reward], dim=0)
            # =============== append environment_context ====================================
            if len(self.environment_context) > 0:
                for context_key in self.environment_context.keys():
                    if ENVIRONMENT_KEYS_TYPE[context_key] == "list":
                        self.environment_context[context_key] = self.environment_context[context_key] + \
                                                                data.environment_context[context_key]
                    elif ENVIRONMENT_KEYS_TYPE[context_key] == "tensor":
                        self.environment_context[context_key] = torch.cat([self.environment_context[context_key],
                                                                           data.environment_context[context_key]],
                                                                          dim=0)
                    else:
                        self.environment_context[context_key] = None
            # =============== append series_context ====================================
            if len(self.series_context) > 0:
                for context_key in self.series_context.keys():
                    if SERIES_KEYS_TYPE[context_key] == "list":
                        self.series_context[context_key] = self.series_context[context_key] + \
                                                           data.series_context[context_key]
                    elif SERIES_KEYS_TYPE[context_key] in ["tensor", "special"]:
                        self.series_context[context_key] = torch.cat([self.series_context[context_key],
                                                                      data.series_context[context_key]],
                                                                     dim=0)
                    else:
                        self.series_context[context_key] = None
        else:
            raise TypeError("Can not append different class for ContextSeriesTensorData")
        return self

    def convert_series_context_to_dict(self):
        if len(self.series_context) == 0:
            return {}
        round_index_map = self.index_map_dict['round_index_map']
        series_index_map = self.index_map_dict['series_index_map']
        series_context_dic = {}
        for i in range(len(self)):
            round_ = round_index_map[i + self.init_index]
            one_round_series_context_dic = {}
            for key in self.series_context.keys():
                item = self.series_context[key][i]
                for j in range(len(series_index_map)):
                    series = series_index_map[j]
                    if SERIES_KEYS_TYPE[key] == "list":
                        one_round_series_context_dic[series] = {key: item[j]}
                    elif SERIES_KEYS_TYPE[key] in ["tensor", "special"]:
                        if self.packed:
                            one_round_series_context_dic[series] = {key: item[j, :]}
                        else:
                            one_round_series_context_dic[series] = {key: item[:, j]}
                    else:
                        self.series_context[key] = None
            series_context_dic[round_] = one_round_series_context_dic
        return series_context_dic

    def convert_environment_context_to_dict(self):
        if len(self.environment_context) == 0:
            return {}
        round_index_map = self.index_map_dict['round_index_map']
        environment_context_dic = {}
        for i in range(len(self)):
            round_ = round_index_map[i + self.init_index]
            environment_context_dic[round_] = {}
            for key in self.environment_context.keys():
                if ENVIRONMENT_KEYS_TYPE[key] == ["tensor", "special", "list"]:
                    environment_context_dic[round_][key] = self.environment_context[key][i]
                else:
                    environment_context_dic[round_][key] = None
        return environment_context_dic

    def get_packed_item(self, item: slice):
        # 压缩数据，减少内存损耗
        if self.packed is False:
            raise TypeError("Can not get packed item from unpacked data!")
        sample_environment_context = {}
        sample_series_context = {}
        sample_reward = None
        pre_dataset = self.pre_dataset
        if self.pre_dataset.reward is None:
            pre_environment_context = {}
            for key in self.environment_context.keys():
                sample_environment_context[key] = self.environment_context[key][item]
                if ENVIRONMENT_KEYS_TYPE[key] == "list":
                    pre_environment_context[key] = self.get_pre_list(self.environment_context[key], item, None)
                elif ENVIRONMENT_KEYS_TYPE[key] == "tensor":
                    pre_environment_context[key] = self.get_pre_tensor(self.environment_context[key], item, None)
                else:
                    pre_environment_context[key] = None
            pre_series_context = {}
            for key in self.series_context.keys():
                if self.series_context[key] is not None:
                    sample_series_context[key] = self.series_context[key][item]
                    if SERIES_KEYS_TYPE[key] == "list":
                        pre_series_context[key] = self.get_pre_list(self.series_context[key], item, None)
                    elif SERIES_KEYS_TYPE[key] == "tensor":
                        pre_series_context[key] = self.get_pre_tensor(self.series_context[key], item, None)
                    else:
                        pre_series_context[key] = None
                else:
                    sample_series_context[key] = None
            pre_reward = None
            if self.reward is not None:
                sample_reward = self.reward[item]
                pre_reward = self.get_pre_tensor(self.reward, item, None)
            pre_dataset = ContextSeriesTensorData(index_map_dict=self.index_map_dict,
                                                  series_context=pre_series_context,
                                                  environment_context=pre_environment_context,
                                                  reward=pre_reward,
                                                  pre_dataset=None,
                                                  packed=self.packed,
                                                  window_size=self.window_size,
                                                  context_window_size=self.context_window_size)
        else:
            if pre_dataset.packed is False:
                raise TypeError("Can not get packed item from unpacked data!")
            for key in self.environment_context.keys():
                sample_environment_context[key] = self.environment_context[key][item]
                if ENVIRONMENT_KEYS_TYPE[key] == "list":
                    pre_dataset.environment_context[key] = \
                        self.get_pre_list(self.environment_context[key], item,
                                          self.pre_dataset.environment_context[key])
                elif ENVIRONMENT_KEYS_TYPE[key] == "tensor":
                    pre_dataset.environment_context[key] = \
                        self.get_pre_tensor(self.environment_context[key], item,
                                            self.pre_dataset.environment_context[key])
                else:
                    sample_environment_context[key] = None
                    pre_dataset.environment_context[key] = None

            for key in self.series_context.keys():
                if self.series_context[key] is not None:
                    sample_series_context[key] = self.series_context[key][item]
                    if SERIES_KEYS_TYPE[key] == "list":
                        pre_dataset.series_context[key] = self.get_pre_list(self.series_context[key], item,
                                                                            self.pre_dataset.series_context[key])
                    elif SERIES_KEYS_TYPE[key] == "tensor":
                        pre_dataset.series_context[key] = self.get_pre_tensor(self.series_context[key], item,
                                                                              self.pre_dataset.series_context[key])
                    else:
                        sample_series_context[key] = None
                        pre_dataset.series_context[key] = None

            if self.reward is not None:
                sample_reward = self.reward[item]
                pre_dataset.reward = self.get_pre_tensor(self.reward, item,
                                                         self.pre_dataset.reward)
        return ContextSeriesTensorData(index_map_dict=self.index_map_dict,
                                       series_context=sample_series_context,
                                       environment_context=sample_environment_context,
                                       reward=sample_reward,
                                       window_size=self.window_size,
                                       context_window_size=self.context_window_size,
                                       pre_dataset=pre_dataset, init_index=item.start, packed=self.packed)

    def __getitem__(self, item):
        item, sample_init_index = self.get_sample_init_index(item,
                                                             max(self.context_window_size, self.window_size))
        sample_environment_context = {}
        sample_series_context = {}
        sample_reward = None

        # 这里取数据是用slide window展开的
        if len(self.environment_context) > 0:
            for key in self.environment_context.keys():
                if ENVIRONMENT_KEYS_TYPE[key] == "list":
                    pre_environment_context = self.pre_dataset.environment_context[key] \
                        if self.pre_dataset is not None else []
                    sample_environment_context[key] = \
                        self.get_sample_list(self.environment_context[key],
                                             item,
                                             self.context_window_size,
                                             pre_environment_context)
                elif ENVIRONMENT_KEYS_TYPE[key] == "tensor":
                    pre_environment_context = self.pre_dataset.pre_environment_context[key] \
                        if self.pre_dataset is not None else []
                    sample_environment_context[key] = \
                        self.get_sample_tensor(self.environment_context[key],
                                               item,
                                               self.context_window_size,
                                               pre_environment_context)
                else:
                    sample_environment_context[key] = None
        if self.window_size > 0:
            self.series_context["history_price"] = None
        if len(self.series_context) > 0:
            for key in self.series_context.keys():
                if SERIES_KEYS_TYPE[key] == "list":
                    pre_series_context = self.pre_dataset.series_context[key] if self.pre_dataset is not None else []
                    sample_series_context[key] = self.get_sample_list(list_data=self.series_context[key],
                                                                      item=item,
                                                                      window_size=self.context_window_size,
                                                                      pre_list=pre_series_context)
                    # 调整数据的结构，变成纵切
                    batch_merge_list = []
                    for one_series_context in sample_series_context[key]:
                        merge_dict = {}
                        for one_day_series_context in one_series_context:
                            series_idx = 0
                            for series in one_day_series_context:
                                if series_idx in merge_dict.keys():
                                    merge_dict[series_idx].append(series)
                                else:
                                    merge_dict[series_idx] = [series]
                                series_idx += 1
                        batch_merge_list.append(list(merge_dict.values()))
                    sample_series_context[key] = batch_merge_list
                elif SERIES_KEYS_TYPE[key] == "tensor":
                    pre_series_context = self.pre_dataset.series_context[key] if self.pre_dataset is not None else None
                    sample_series_context[key] = self.get_sample_tensor(tensor=self.series_context[key],
                                                                        item=item,
                                                                        window_size=self.context_window_size,
                                                                        pad_num=1,
                                                                        pre_tensor=pre_series_context)
                    if self.context_window_size == 1:
                        sample_series_context[key] = sample_series_context.get(key).unsqueeze(1)
                elif SERIES_KEYS_TYPE[key] == "special":
                    if self.window_size != 0:
                        pre_reward = self.pre_dataset.reward if self.pre_dataset is not None else None
                        sample_series_context[key] = self.get_sample_tensor(tensor=self.reward,
                                                                            item=item,
                                                                            window_size=self.window_size + 1,
                                                                            pad_num=1,
                                                                            pre_tensor=pre_reward)
                        sample_series_context[key] = sample_series_context.get(key)[:, :-1, :]
                else:
                    sample_series_context[key] = None
        if self.reward is not None:
            pre_reward = self.pre_dataset.reward if self.pre_dataset is not None else []
            sample_reward = self.get_sample_tensor(tensor=self.reward, item=item, window_size=1, pad_num=1,
                                                   pre_tensor=pre_reward)

        return ContextSeriesTensorData(index_map_dict=self.index_map_dict,
                                       series_context=sample_series_context,
                                       environment_context=sample_environment_context,
                                       reward=sample_reward,
                                       window_size=self.window_size,
                                       context_window_size=self.context_window_size,
                                       init_index=sample_init_index,
                                       packed=False)

    def cuda(self):
        self.reward = self.convert_device(self.reward, "cuda")
        for key in self.environment_context.keys():
            self.environment_context[key] = self.convert_device(self.environment_context[key], "cuda")
        for key in self.series_context.keys():
            self.series_context[key] = self.convert_device(self.series_context[key], "cuda")
        return self

    def cpu(self):
        self.reward = self.convert_device(self.reward, "cpu")
        for key in self.environment_context.keys():
            self.environment_context[key] = self.convert_device(self.environment_context[key], "cpu")
        for key in self.series_context.keys():
            self.series_context[key] = self.convert_device(self.series_context[key], "cpu")
        return self


