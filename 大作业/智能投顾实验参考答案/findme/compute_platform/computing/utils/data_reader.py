import copy
import sys
import json
import torch
import os
from queue import Queue

from computing.utils.load_constructor import dict_load_constructor, default_load_tensor_constructor
from computing.utils.data_struct import SeriesDictData
from computing.utils.util_functions import get_seg_index

sys.path.append('..')


# ======================================== jsonl类型的数据获取 ===================================================
class DictDataReader(object):
    def __init__(self, file_dir=None, file_path=None, dataset_: SeriesDictData = None, window_size=1,
                 context_window_size=1, init_index=0, split_type="year", constructor=dict_load_constructor):
        # TODO: 修改load constructor，
        # TODO：写一个和 tensor 数据格式对应的 dict 文件来测试

        self.dataloader = None

        # dict reader 参数
        self.file_dir = file_dir  # 数据文件所在的目录
        self.file_path = file_path  # 每个数据格式对应的文件名
        self.seg = 1  # 当前所在的文件
        self.split_type = split_type  # 文件的划分方法（用于直接通过 index_map_dict 获得文件的起止 index）
        self.init_index = init_index  # 当前 dataset 的第一条数据，对应全局数据的第几条
        self.constructor = constructor  # 数据构造器
        self.window_size = window_size
        self.context_window_size = context_window_size

        # fetch batch 所需要的参数
        self.current_dataset = None
        self.batch_size = None
        self.remaining_size = None  #

        self.dataset = None
        if dataset_ is not None:
            self.dataset = dataset_
            self.index_map_dict = self.dataset.index_map_dict
        elif file_dir is not None:
            self.index_map_dict = self.load(file_dir=file_dir, file_path=file_path["index_map_dict"],
                                            constructor=self.constructor)
            self.dataset = self.load_dataset(file_dir=file_dir, seg=0, file_path=self.file_path)
        else:
            raise FileNotFoundError("Either file_dir or dataset should be provided.")

    @staticmethod
    def load(file_dir, file_path, constructor):
        if file_path == 'index_map_dict.pt':
            return torch.load(os.path.join(file_dir, file_path))
        else:
            round_dict_data_list = []
            try:
                fptr = open(os.path.join(file_dir, file_path))
            except EOFError:
                return None
            for line in fptr.readlines():
                if line:
                    x = json.loads(line)
                    round_dict_data = constructor(x)
                    round_dict_data_list.append(round_dict_data)
            fptr.close()
            return round_dict_data_list

    def load_dataset(self, file_dir, file_path, seg=1):
        """
        Get dataset from file, set the dataset to be self.dataset, and return the dataset.
        Returns: dataset (list): a list of SeriesDictData instance containing all data from file
        """
        pre_dataset = None
        # 设置前一个数据集
        if seg > 2:
            pre_round_dict_data_list = self.load(file_dir=file_dir, file_path=file_path["dict_data"] + '_' + seg - 1,
                                                 constructor=self.constructor)
            if len(pre_round_dict_data_list) > 0:
                init_index, _ = get_seg_index(round_index_map=self.index_map_dict, seg=self.seg, split_type=self.split_type)
                pre_dataset = SeriesDictData(index_map_dict=self.index_map_dict, data_list=pre_round_dict_data_list,
                                             window_size=self.window_size, context_window_size=self.context_window_size,
                                             pre_dataset=None, init_index=init_index, packed=False)
        else:
            pre_dataset = None

        # 设置当前数据集
        round_dict_data_list = self.load(file_dir=file_dir, file_path=file_path["dict_data"] + '_' + seg,
                                         constructor=self.constructor)

        # 挂载前一个数据集
        if pre_dataset is not None:
            init_index, _ = get_seg_index(round_index_map=self.index_map_dict, seg=self.seg,
                                          split_type=self.split_type)
            dataset = SeriesDictData(index_map_dict=self.index_map_dict, data_list=round_dict_data_list,
                                     window_size=self.window_size, context_window_size=self.context_window_size,
                                     pre_dataset=pre_dataset, init_index=init_index, packed=False)
        else:
            dataset = -1
        # 返回加载的数据集
        return dataset

    def get_dataset_len(self):
        return self.dataset.get_total_length()

    def get_dataset(self):
        return self.dataset

    def get_round_index_map(self):
        return self.dataset.index_map_dict["round_index_map"]

    def set_data_loader(self, dataset=None, batch_size=1, start_index=0, update_iter=True):
        """
        Args:
            update_iter:
            start_index:
            dataset:
            batch_size: 1 for fetch one
        Returns:
        """
        # todo: 检测start_index所在的seg
        assert dataset is not None
        index = range(start_index, dataset.get_total_length(), batch_size)
        del self.current_dataset
        self.current_dataset = copy.deepcopy(dataset)
        if update_iter:
            self.dataloader = iter(index)

    def set_window_size(self, window_size):
        self.window_size = window_size["window_size"] if 'window_size' in window_size else self.window_size
        self.context_window_size = window_size["context_window_size"] if 'context_window_size' in window_size \
            else self.context_window_size
        self.dataset.set_window_size(window_size)

    def set_fetch_batch(self, batch_size, start_index=0, end_index=None, window_size=None, dataset=None):
        if window_size is not None:
            self.set_window_size(window_size)
        self.batch_size = batch_size
        self.seg = 1
        if end_index is None:
            self.remaining_size = self.get_dataset_len() - start_index
        else:
            self.remaining_size = end_index - start_index
        self.set_data_loader(dataset=self.dataset.get_packed_item(slice(0, len(self.dataset) + 1)),
                             batch_size=self.batch_size, start_index=start_index)

    def fetch_next_batch(self, start_index=None):
        if self.remaining_size < self.batch_size:
            return -1
        if self.dataloader is not None:
            if start_index is None:
                start_index = next(self.dataloader)
            try:
                data = self.current_dataset[start_index:start_index + self.batch_size]
            except StopIteration:
                if self.file_dir is None:
                    return -1
                elif self.remaining_size <= 0:
                    return -1
                else:
                    # 需要判断所有出错的情况
                    # 首先拿到这个seg的首末位置
                    seg_start_index, seg_end_index = get_seg_index(round_index_map=self.get_round_index_map(),
                                                                   seg=self.seg, split_type=self.split_type)
                    pre_part_data = None
                    # 有一部分落在这个区间里
                    if seg_end_index >= start_index > seg_start_index:
                        pre_part_data = self.current_dataset[
                                        start_index:min(start_index + self.batch_size, seg_end_index + 1)]
                    # 获取下一个seg的数据
                    self.seg += 1
                    dataset = self.load_dataset(file_dir=self.file_dir, file_path=self.file_path, seg=self.seg)
                    # 拿到这个seg的首末位置
                    seg_start_index, seg_end_index = get_seg_index(round_index_map=self.get_round_index_map(),
                                                                   seg=self.seg,
                                                                   split_type=self.split_type)
                    # 数据集不存在
                    if dataset == -1:
                        return -1
                    # 尽管可能有数据在区间里，但不是和上一seg拼接的形式
                    elif pre_part_data is None and seg_start_index < start_index:
                        self.dataset = dataset
                        self.set_data_loader(dataset=self.dataset, batch_size=self.batch_size, start_index=start_index,
                                             update_iter=False)
                        data = self.fetch_next_batch(start_index=start_index)
                    # 这个数据集里有开头的数据
                    else:
                        self.dataset = dataset
                        self.set_data_loader(dataset=self.dataset, batch_size=self.batch_size, start_index=start_index,
                                             update_iter=False)
                        if pre_part_data is None:
                            data = self.current_dataset[seg_start_index: seg_start_index + self.batch_size]
                        else:
                            after_data = self.current_dataset[seg_start_index:start_index + self.batch_size]
                            data = pre_part_data.append(after_data)
                        self.remaining_size -= self.batch_size
                    return data
            else:
                self.remaining_size -= self.batch_size
                return data
        else:
            raise NotImplementedError("The data loader is not set.")


# ======================================== tensor类型的数据获取 ===================================================
class TensorDataReader(object):
    def __init__(self, file_dir=None, iter_index=0, split_type="year",
                 file_path={"series_context": "series_context", "environment_context": "environment_context",
                            "reward": "reward", "index_map_dict": "index_map_dict"}, dataset_=None,
                 constructor=default_load_tensor_constructor, window_size={"window_size": 1}):
        """
        Args:
            file_dir: the stored tensor dataset dir
            file_path: the stored tensor file
            dataset_: the dataset
            loader: call for dataset
            :rtype: object
        """
        self.dataset = None  # keep the dataset to be None at first for assert
        self.current_dataset = None
        self.dataloader = None
        self.batch_size = None
        self.remaining_size = None
        self.seg = 1

        self.window_size = window_size
        self.iter_index = iter_index

        self.file_dir = file_dir
        self.file_path = file_path
        self.split_type = split_type
        self.constructor = constructor

        if dataset_ is not None:
            self.dataset = dataset_
        if file_dir is not None:
            self.dataset = self.load_dataset(constructor=self.constructor, file_dir=self.file_dir,
                                             file_path=self.file_path, seg=self.seg,
                                             iter_index=self.iter_index, window_size=self.window_size)
        if self.dataset == -1:
            raise FileNotFoundError("Load dataset failed!!")

    @staticmethod
    def load(file_dir, file_path):
        """
        Args:
            file_dir:
            file_path: load tensor file

        Returns:
        """
        try:
            obj = torch.load(os.path.join(file_dir, file_path))
        except IOError:
            return None
        else:
            return obj

    def load_dataset_by_seg(self, file_dir, file_path, seg: int = 1):
        reward_data = self.load(file_dir, file_path["reward"] + "_" + str(seg) + ".pt")
        series_context_data = {}
        environment_context_data = {}
        for file_key, file_name in file_path.items():
            if "series" in file_key:
                key = file_key[7:]
                series_context_data[key] = self.load(file_dir, file_name + "_" + str(seg) + ".pt")
            elif "environment" in file_key:
                key = file_key[12:]
                environment_context_data[key] = self.load(file_dir, file_name + "_" + str(seg) + ".pt")
        if "series_context" in file_path.keys():
            series_context_data = self.load(file_dir, file_path["series_context"] + "_" + str(seg) + ".pt")
        if "environment_context" in file_path.keys():
            environment_context_data = self.load(file_dir, file_path["environment_context"] + "_" + str(seg) + ".pt")
        return reward_data, series_context_data, environment_context_data

    def load_dataset(self, constructor, file_dir, file_path, seg: int = 1, iter_index: int = 1,
                     window_size: dict = {"window_size": 1}):
        """
        Get dataset from file, set the dataset to be self.dataset, and return the dataset.
        Returns: dataset (list): a list of SeriesDictData instance containing all data from file
        """

        index_map_dict_data = self.load(file_dir, file_path["index_map_dict"] + ".pt")
        if seg > 1:
            reward_data, series_context_data, environment_context_data = self.load_dataset_by_seg(file_dir, file_path,
                                                                                                  seg - 1)

            pre_dataset = constructor(series_context_data, environment_context_data, reward_data,
                                      index_map_dict_data, iter_index, window_size, None, seg, self.split_type)
        else:
            pre_dataset = None
        reward_data, series_context_data, environment_context_data = self.load_dataset_by_seg(file_dir, file_path, seg)

        return constructor(series_context_data, environment_context_data, reward_data,
                           index_map_dict_data, iter_index, window_size, pre_dataset, seg, self.split_type)

    def get_dataset_len(self):
        return self.dataset.get_total_length()

    def get_dataset(self):
        return self.dataset

    def get_rewards(self):
        reward = Queue()
        [reward.put(i) for i in self.dataset]
        return reward

    def change_iter_dimension(self, iter_index):
        assert self.dataset is not None
        self.dataset.set_iter_mode(iter_index)

    def set_data_loader(self, dataset=None, batch_size=1, start_index=0, update_iter=True):
        """
        Args:
            batch_size: 1 for fetch one
        Returns:
        """
        # 检测start_index所在的seg
        assert dataset is not None
        index = range(start_index, dataset.get_total_length(), batch_size)
        del self.current_dataset
        self.current_dataset = copy.deepcopy(dataset)
        # print(len(self.current_dataset),len(self.current_dataset.pre_dataset), len(self.dataset),len(self.dataset.pre_dataset))
        if update_iter:
            self.dataloader = iter(index)

    def get_round_index_map(self):
        return self.dataset.index_map_dict["round_index_map"]

    def set_fetch_batch(self, batch_size, start_index=0, end_index=None, window_size=None, dataset=None):
        if window_size is not None:
            self.window_size = window_size
            self.dataset.set_window_size(window_size)
        self.batch_size = batch_size
        self.seg = 1
        dataset = self.load_dataset(constructor=self.constructor, file_dir=self.file_dir,
                          file_path=self.file_path, seg=self.seg,
                          iter_index=self.iter_index, window_size=self.window_size)
        if end_index is None:
            self.remaining_size = self.get_dataset_len() - start_index
        else:
            self.remaining_size = end_index - start_index
        self.set_data_loader(dataset=dataset,
                             batch_size=self.batch_size, start_index=start_index)

    def fetch_next_batch(self, start_index=None):
        if self.remaining_size < self.batch_size:
            return -1
        if self.dataloader is not None:
            if start_index is None:
                start_index = next(self.dataloader)
            try:
                data = self.current_dataset[start_index:start_index + self.batch_size]
            except StopIteration:
                if self.file_dir is None:
                    return -1
                elif self.remaining_size <= 0:
                    return -1
                else:
                    # 需要判断所有出错的情况
                    # 首先拿到这个seg的首末位置
                    seg_start_index, seg_end_index = get_seg_index(round_index_map=self.get_round_index_map(),
                                                                   seg=self.seg,
                                                                   split_type=self.split_type)
                    pre_part_data = None
                    # 有一部分落在这个区间里
                    if seg_end_index >= start_index > seg_start_index:
                        pre_part_data = self.current_dataset[
                                        start_index:min(start_index + self.batch_size, seg_end_index + 1)]
                    # 获取下一个seg的数据
                    self.seg += 1
                    dataset = self.load_dataset(constructor=self.constructor, file_dir=self.file_dir,
                                                file_path=self.file_path, seg=self.seg,
                                                iter_index=self.iter_index, window_size=self.window_size)
                    # 拿到这个seg的首末位置
                    seg_start_index, seg_end_index = get_seg_index(round_index_map=self.get_round_index_map(),
                                                                   seg=self.seg,
                                                                   split_type=self.split_type)
                    # 数据集不存在
                    if dataset == -1:
                        return -1
                    # 尽管可能有数据在区间里，但不是和上一seg拼接的形式
                    elif pre_part_data is None and seg_start_index < start_index:
                        self.dataset = dataset
                        self.set_data_loader(dataset=self.dataset, batch_size=self.batch_size, start_index=start_index,
                                             update_iter=False)
                        data = self.fetch_next_batch(start_index=start_index)
                    # 这个数据集里有开头的数据
                    else:
                        self.dataset = dataset
                        self.set_data_loader(dataset=self.dataset, batch_size=self.batch_size, start_index=start_index,
                                             update_iter=False)
                        if pre_part_data is None:
                            data = self.current_dataset[seg_start_index: seg_start_index + self.batch_size]
                        else:
                            after_data = self.current_dataset[seg_start_index:start_index + self.batch_size]
                            data = pre_part_data.append(after_data)
                        self.remaining_size -= self.batch_size
                    return data
            else:
                self.remaining_size -= self.batch_size
                return data
        else:
            raise NotImplementedError("The data loader is not set.")


