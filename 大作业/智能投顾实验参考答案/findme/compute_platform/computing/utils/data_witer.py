import os

import torch

from computing.utils import BATCH_SIZE
from computing.utils.data_struct import SeriesTensorData, SeriesDictData


class DictDataWriter(object):
    def __init__(self, out_file_dir=None, dataset_: SeriesDictData = None, batch_size=BATCH_SIZE):
        """
        根据dict_data所在的文件夹目录，设置所有数据的读取
        Args:
            out_file_dir: dict_data所在的文件夹目录
            dataset_: 数据集
        """
        self.batch_size = batch_size
        self.dataset = dataset_
        self.out_file_dir = out_file_dir

    def get_dataset(self):
        return self.dataset

    def write(self, data_dir):
        """
        批量写入文件，按照BATCH_SIZE来分割文件
        :param data_dir: 写入的目录
        :return:
        """
        batch_count = 0
        f_bandit_data = open(self.get_file_path(data_dir, "/dict_data_" + str(batch_count)), "w")
        for i in range(len(self.dataset)):
            if i % self.batch_size == 0:
                f_bandit_data.close()
                f_bandit_data = open(self.get_file_path(data_dir, "/dict_data_" + str(batch_count)), "w")
                batch_count += 1
            f_bandit_data.write(self.dataset[i] + "\n")

    @staticmethod
    def get_file_path(dir_base, arg_name):
        """
        写入数据
        :param arg_name: 数据对应的文件名
        :param dir_base: 写入的目录
        :return: 文件名称
        """
        file_path = dir_base + '/' + arg_name + '.jsonl'
        is_exists = os.path.exists(dir_base)
        if not is_exists:
            os.makedirs(dir_base)
        return file_path


class TensorDataWriter(object):
    def __init__(self, out_file_dir=None, dataset_: SeriesTensorData = None, save_tensor_name: list = None, batch_size=BATCH_SIZE):
        """
        根据dict_data所在的文件夹目录，设置所有数据的读取
        Args:
            out_file_dir: dict_data所在的文件夹目录
            dataset_: 数据集, class:SeriesTensorData
        """
        self.batch_size = batch_size
        self.dataset = dataset_
        self.save_tensor_name = save_tensor_name
        self.out_file_dir = out_file_dir

    def get_dataset(self):
        return self.dataset

    def write(self):
        for tensor_name in self.save_tensor_name:
            tensor_data = self.dataset.__getattribute__(tensor_name)
            self.batch_write(self.out_file_dir, tensor_data, tensor_name)

    def batch_write(self, data_dir, tensor_data, tensor_name):
        """
        批量写入文件，按照BATCH_SIZE来分割文件
        :param tensor_data: 需要写入的数据
        :param tensor_name: 需要写入的数据的名称
        :param data_dir: 写入的目录
        :return:
        """
        batch_count = 0
        for batch_count in range(len(tensor_data) / self.batch_size):
            batch_tensor = tensor_data[batch_count:batch_count+self.batch_size]
            torch.save(batch_tensor, self.get_file_path(data_dir, "/" + tensor_name + "_" + str(batch_count) + ".pt"))
        batch_count += 1
        batch_tensor = tensor_data[batch_count:]
        torch.save(batch_tensor, self.get_file_path(data_dir, "/" + tensor_name + "_" + str(batch_count) + ".pt"))

    @staticmethod
    def get_file_path(dir_base, arg_name):
        """
        写入数据
        :param arg_name: 数据对应的文件名
        :param dir_base: 写入的目录
        :return: 文件名称
        """
        file_path = dir_base + '/' + arg_name
        is_exists = os.path.exists(dir_base)
        if not is_exists:
            os.makedirs(dir_base)
        return file_path
