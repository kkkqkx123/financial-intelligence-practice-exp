import os
from computing.utils.data_reader import DictDataReader, TensorDataReader
from computing.utils.data_struct import SeriesTensorData, SeriesDictData
from computing.utils.data_converter import series_tensor_to_dict, series_dict_to_tensor
import torch


class DataProxy:
    def __init__(self, dataset_name, file_dir=None, file_path=None, iter_index=0, batch_size=1,
                 shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 split_type="year", window_size={"default": 0}, device='cpu',
                 constructor='default_load_tensor_constructor', external_data_path=None,
                 default_data_type=None,
                 **kwargs):
        self.dataset = None
        self.dataset_name = None
        self.windows_size = window_size
        self.default_data_type = default_data_type
        self.iter_index = iter_index
        self.dataset_kwargs = kwargs
        self.shuffle, self.sampler, self.batch_sampler, self.num_workers \
            = [shuffle, sampler, batch_sampler, num_workers]
        self.new_batch = True
        self.batch_size = batch_size
        self.data_reader, self.dataset_name = self.set_data_reader(dataset_name, file_dir, file_path, window_size,
                                                                   constructor, split_type)
        self.device = device
        self.external_data = None

    def set_device(self, device):
        self.device = device

    def get_batch_size(self):
        return self.batch_size

    def get_data_params_dict(self):
        data_dict = {}
        for k, v in self.__dict__.items():
            if not callable(v) and k != "external_data" and k != "data_reader" and k != "dataset_kwargs":
                data_dict[k] = v
        return data_dict

    def set_data_reader(self, dataset_name: str, file_dir: str, file_path: dict, window_size, constructor, split_type):
        """
        根据输入的参数值，设置data_reader
        Args:
            dataset_name:
            file_dir:
            file_path:
            window_size:
            constructor:
            split_type:

        Returns:

        """
        # detect exception
        files_in_dataset_dir = os.listdir(file_dir)
        if len(files_in_dataset_dir) == 0:
            raise FileExistsError("No file exists in this directory.")

        # load data
        data_file_postfix = os.path.splitext(files_in_dataset_dir[0])[1]
        # if data_file_postfix == ".jsonl":
        if self.default_data_type == 'dict':
            data_reader = DictDataReader(file_dir)
        # elif data_file_postfix == ".pt":
        elif self.default_data_type == "tensor":
            exec("from computing.utils.load_constructor import {}".format(constructor))
            eval_str = "TensorDataReader(file_dir=file_dir, file_path=file_path, constructor={},\
                                           window_size=self.windows_size, iter_index=self.iter_index,\
                                           split_type=split_type)".format(constructor)
            data_reader = eval(eval_str)
        else:
            data_reader = eval("MiddlePlatformDataReader(**self.dataset_kwargs)")

        return data_reader, dataset_name

    def set_fetch_batch(self, start_index, end_index, batch_size=None, dataset=None):
        self.data_reader.set_fetch_batch(batch_size=batch_size, start_index=start_index, end_index=end_index,
                                         dataset=dataset)

    def fetch_next_batch(self, data_type="tensor"):
        # data_reader.fetch_next_batch() 根据设置好的 fetch_batch 参数，获取下一个batch
        data = self.data_reader.fetch_next_batch()
        if data_type == "tensor" and isinstance(data, SeriesDictData):
            data = series_dict_to_tensor(data)
        elif data_type == "dict" and isinstance(data, SeriesTensorData):
            data = series_tensor_to_dict(data)

        if self.device == 'cuda' and data != -1:
            return data.cuda()
        else:
            return data

    def set_fetch_one(self, start_index, end_index=None):
        # data_reader.fetch_batch() 设置fetch的头、尾index以及batch_size
        self.data_reader.set_fetch_batch(batch_size=1, start_index=start_index, end_index=end_index)

    def get_dataset_name(self):
        return str(self.dataset_name)

    def get_dataset_len(self):
        return self.data_reader.get_dataset_len()

    def get_max_window_size(self):
        window_sizes = self.windows_size.values()
        return max(window_sizes)


class SeriesDictDataProxy(DataProxy):
    def __init__(self, dataset_name, file_dir=None, context_file_path=None, reward_file_path=None,
                 index_map_file_path=None, iter_index=1, windows_size=0, shuffle=False, batch_size=1,
                 sampler=None, batch_sampler=None, num_workers=0, **kwargs):
        super(SeriesDictDataProxy, self).__init__(dataset_name=dataset_name,
                                                  file_dir=file_dir,
                                                  context_file_path=context_file_path,
                                                  reward_file_path=reward_file_path,
                                                  index_map_file_path=index_map_file_path,
                                                  iter_index=iter_index,
                                                  windows_size=windows_size,
                                                  shuffle=shuffle,
                                                  sampler=sampler,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=num_workers,
                                                  default_data_type='dict',
                                                  batch_size=batch_size, **kwargs)

    def simulate_data(self, dataset_name, dataset_dir, loader, data_reader):
        # TODO: 之后设计＆补充模拟数据生成方法
        return dataset_name, dataset_dir, loader, data_reader


class SeriesTensorDataProxy(DataProxy):
    def __init__(self, dataset_name=None, file_path=None, file_dir=None, iter_index=0, shuffle=False, batch_size=1,
                 sampler=None, batch_sampler=None, num_workers=0, window_size=None, split_type="year",
                 external_data_path=None,
                 constructor='news_load_tensor_constructor', **kwargs):
        super(SeriesTensorDataProxy, self).__init__(dataset_name=dataset_name,
                                                    file_dir=file_dir,
                                                    file_path=file_path,
                                                    iter_index=iter_index,
                                                    window_size=window_size,
                                                    shuffle=shuffle,
                                                    sampler=sampler,
                                                    batch_sampler=batch_sampler,
                                                    num_workers=num_workers,
                                                    batch_size=batch_size,
                                                    split_type=split_type,
                                                    constructor=constructor,
                                                    default_data_type='tensor',
                                                    **kwargs)
        self.external_data = torch.load(external_data_path) if external_data_path else None

    def simulate_data(self, dataset_name, dataset_dir, loader, data_reader):
        # TODO: 之后设计＆补充模拟数据生成方法
        return dataset_name, dataset_dir, loader, data_reader


