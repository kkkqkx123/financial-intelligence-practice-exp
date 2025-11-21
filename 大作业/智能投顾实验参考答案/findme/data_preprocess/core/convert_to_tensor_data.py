"""
将vector信息转成高维度tensor同时存成.pt形式
"""
import os
import time
from typing import Union
from dateutil.relativedelta import relativedelta

import torch
import pandas as pd

from computing.utils.util_functions import get_seg_index, str2datetime

# zero_embedding = '0.,' * 768
# zero_embedding = zero_embedding[:-1]


class TensorDatePreprocessor(object):
    def __init__(self, raw_dataset_dir, tensor_dataset_dir, reload=True, series_=None, round_=None, split_type="1y",
                 *args, **kwargs):
        self.raw_dataset_dir = raw_dataset_dir
        self.tensor_dataset_dir = tensor_dataset_dir
        self.split_type = split_type
        self.index_map_dict = self.generate_index_map_dict(reload, series_, round_, *args, **kwargs)

    def load_round_index_map(self, *args, **kwargs) -> list:
        return []

    def load_series_index_map(self, *args, **kwargs) -> list:
        return []

    def generate_index_map_dict(self, reload, series, round, *args, **kwargs):
        file_path = os.path.join(self.tensor_dataset_dir, 'index_map_dict.pt')
        if reload:
            try:
                index_map_dict = torch.load(file_path)
            except IOError:
                raise IOError("Error: 没有找到文件或读取文件失败")
        else:
            index_map_dict = {}
            if round is None:
                index_map_dict['round_index_map'] = self.load_round_index_map(*args, **kwargs)
            else:
                index_map_dict['round_index_map'] = round
            if series is None:
                index_map_dict['series_index_map'] = self.load_series_index_map(*args, **kwargs)
            else:
                index_map_dict['series_index_map'] = series

            torch.save(index_map_dict, file_path)
        return index_map_dict

    def read_csv(self, path_name, usecols=None, header=None, dtype_=None, delimiter=None, index_col=None):
        return pd.read_csv(os.path.join(self.raw_dataset_dir, path_name), usecols=usecols, header=header, dtype=
        dtype_, delimiter=delimiter, index_col=index_col)

    def get_round_context(self, round_df, data_name, column_type, zero_embedding=None,
                          auxiliary_round_df=None) -> Union[list, torch.Tensor]:
        data_type = column_type[0]
        round_value = []

        if data_type == "float":
            column_name = column_type[3]
            column_data = round_df[column_name].values
            round_value = torch.tensor(column_data)

        elif (data_type == "float list") or (data_type == "float tensor"):
            col_name_list = column_type[3]
            column_data = round_df[col_name_list].values.tolist()
            round_value = torch.tensor(column_data)
            round_value = round_value.squeeze(0)
        elif data_type == "str_emb":
            column_name = column_type[3]
            embeddings = round_df[column_name].values
            for embedding in embeddings:
                column_data = [float(s) for s in embedding.split(',')]
                round_value.append(column_data)
            round_value = torch.tensor(round_value)
        elif data_type == "str":
            column_name = column_type[3]
            strings = round_df[column_name].values
            for column_data in strings:
                round_value.append(column_data)
            round_value = torch.tensor(round_value)
        elif data_type == "time":
            column_name = column_type[3]
            time_round = round_df[column_name].values
            for index, timestamp in enumerate(time_round):
                round_value.append(timestamp - zero_embedding)
            round_value = torch.tensor(round_value)
        elif data_type == "id":
            column_name = column_type[3]
            column_data = list(round_df[column_name].values)
            round_value = column_data
        elif data_type == "function" or data_type == "function list":
            function_name = list(column_type[3].keys())[0]
            import_str = "from data_preprocess.utils.utils_for_features import " + function_name
            exec(import_str)
            function_params = column_type[3].get(function_name)
            eval_str = function_name + "(auxiliary_raw_df=auxiliary_round_df,"
            for param_name, param in function_params.items():
                eval_str += param_name + "=" + str(param) + ","
            eval_str = eval_str[:-1] + ")"

            column_data = eval(eval_str)
            round_value = column_data
        else:
            raise TypeError("The type of column is " + str(type(column_type)) + ", which is invalid!")

        return round_value

    def context_to_tensor_by_series(self, raw_df, series_index, column_name, column_type, zero_embedding=None,
                                    auxiliary_raw_df=None) -> list:
        round_value = []
        data_type = column_type[0]
        if len(series_index) == 1:
            series_round_value = self.get_round_context(raw_df, column_name, column_type, zero_embedding,
                                                        auxiliary_raw_df)
            if data_type == "str" or data_type == "str_emb" or data_type == "time" or data_type == "id":
                round_value.append(series_round_value)
            else:
                round_value = series_round_value.unsqueeze(0)
            return round_value
        for index, series in enumerate(series_index):
            if "Name" in raw_df.columns:
                name = "Name"
            elif "stock_id" in raw_df.columns:
                name = "stock_id" # 处理news embedding的部分,因为走的时候mapping.csv
            else:
                raise LookupError("Can't find series mark in data!")
            series_df = raw_df[(raw_df[name] == series)]
            if auxiliary_raw_df is None:
                auxiliary_series_df = None
            else:
                if name == "Name":
                    auxiliary_series_df = auxiliary_raw_df[(auxiliary_raw_df[name] == series)]
                else:
                    auxiliary_series_df = auxiliary_raw_df[(auxiliary_raw_df.stock_id == index)]

            if series_df.empty is False:
                series_round_value = self.get_round_context(series_df, column_name, column_type, zero_embedding,
                                                            auxiliary_series_df)
            else:
                # 处理缺省值，以1代替（relate price有效）
                if data_type == "time" or data_type == "str" or data_type == "str_emb":
                    series_round_value = None
                else:
                    series_round_value = zero_embedding

            round_value.append(series_round_value)
        if data_type == "str" or data_type == "str_emb" or data_type == "time" or data_type == "id":
            return round_value
        else:
            round_value = torch.stack(round_value).float()
            return round_value

    def context_to_tensor_by_all(self, raw_df, column_name, column_type, zero_embedding=None, auxiliary_raw_df=None) \
            -> Union[list, torch.Tensor]:
        """

        Args:
            auxiliary_raw_df:
            raw_df:
            column_name: feature name
            column_type: a tuple of (data_type, context_type, col_name_list)
            (data_type: float, float list, float tensor, str (convert to float list), tensor, ...)
            (context_type: environment, series)
            (col_name_list: if float list and float tensor, use this one)
            zero_embedding:

        Returns:

        """
        round_value = self.get_round_context(raw_df, column_name, column_type, zero_embedding, auxiliary_raw_df)
        return round_value

    def context_to_tensor(self, data_df: pd.DataFrame, keep_values_col, seg=1, sort_by="Date", auxiliary_window_size=0):
        """
        转换的函数的入口，用于选择时间，分配转换的函数是哪个，整合转换后的结果
        Args:
            sort_by: the name of index column
            seg: the file seg
            auxiliary_window_size: 对raw data的窗口
            data_df: source dataframe to convert tensor data
            keep_values_col: values columns, dict, including column_name and column_type
                             column_type is a tuple of (data_type, context_type, default value, col_name_list/function param)
                             (data_type: float, float list, float tensor, str (convert to float list), tensor, ...)
                             (context_type: environment, series)
                             (default value: if data is empty, set default vale)
                             (col_name_list: if float list and float tensor, use this one)
                             (function param: if function, use this one)
        Returns: Tensor like [round, series_name, values(1*D, M*N)]
        """
        total_date_index = self.index_map_dict["round_index_map"]
        start_index, end_index = get_seg_index(round_index_map=total_date_index, seg=seg, split_type=self.split_type)

        # 设置初始的日期
        if seg == 1:
            pre_date = total_date_index[0]
            datetime_type = "time" if len(pre_date) > 10 else "date"
            pre_date_datetime = str2datetime(pre_date, datetime_type)
            pre_date_datetime = pre_date_datetime - relativedelta(days=1)
            pre_date = pre_date_datetime.strftime(
                "%Y-%m-%d") if datetime_type == "date" else pre_date_datetime.strftime("%Y-%m-%d %H:%M:%S")

        else:
            pre_date = total_date_index[start_index - 1]
        # 选择需要convert的日期
        seg_date_index = total_date_index[start_index:end_index + 1]  # 左右闭集

        # 初始化输出数据
        convert_data = {}  # 最终输出的文件，每个key一个预处理的任务
        for column_name, column_type in keep_values_col.items():
            convert_data[column_name] = []

        for index, date in enumerate(seg_date_index):
            if index % 1000 == 0:
                print("handling date:", date)
            zero_timestamp = 0  # for joint time = True
            # 实现周末新闻归于周五；reward是当天的数据，其余都是前一天的数据
            raw_df = data_df.loc[(data_df.index >= pre_date) & (data_df.index < date)]
            if auxiliary_window_size == 0:
                auxiliary_raw_df = raw_df
            elif auxiliary_window_size > 0:
                auxiliary_pre_date = total_date_index[max(0, index + start_index - auxiliary_window_size)]
                auxiliary_raw_df = data_df.loc[(data_df.index >= auxiliary_pre_date) & (data_df.index < date)]
            else:  # 需要泄露一些信息给当天使用，特殊情况，用于做日内即时计算
                if index + start_index - auxiliary_window_size >= len(total_date_index):
                    print("endddddd")
                    auxiliary_raw_df = data_df.loc[(data_df.index >= date)]
                else:
                    auxiliary_next_date = total_date_index[index + start_index - auxiliary_window_size]
                    auxiliary_raw_df = data_df.loc[(data_df.index >= date) & (data_df.index < auxiliary_next_date)]

            if raw_df.empty:
                for column_name, column_type in keep_values_col.items():
                    default_value = column_type[2]
                    context_type = column_type[1]
                    if context_type == "environment":
                        convert_data.get(column_name).append(default_value)
                    else:  # series
                        series_index = self.index_map_dict["series_index_map"]
                        round_value = []
                        for series in series_index:
                            round_value.append(default_value)
                        if isinstance(default_value, torch.Tensor):
                            round_value = torch.stack(round_value)
                        convert_data.get(column_name).append(round_value)
                pre_date = date
                continue
            if "time" in str(keep_values_col.values()):
                # 先转换基准的timestamp，
                zero_timestamp = time.strptime(pre_date, "%Y-%m-%d")
                zero_timestamp = int(time.mktime(zero_timestamp))
            else:
                zero_timestamp = None

            # 需要做日内排序, 做所有当天数据的排序
            raw_df = raw_df.sort_values(by=sort_by, axis=0, ascending=True)
            auxiliary_raw_df = auxiliary_raw_df.sort_values(by=sort_by, axis=0, ascending=True)

            round_value = None

            for column_name, column_type in keep_values_col.items():
                data_type = column_type[0]
                context_type = column_type[1]
                if context_type == "environment":
                    if data_type == "time":
                        round_value = self.context_to_tensor_by_all(raw_df, column_name, column_type, zero_timestamp)
                    elif data_type == "function" or data_type == "function list":
                        round_value = self.context_to_tensor_by_all(raw_df, column_name, column_type,
                                                                    auxiliary_raw_df=auxiliary_raw_df)
                    else:
                        round_value = self.context_to_tensor_by_all(raw_df, column_name, column_type)
                    if index % 10000 == 0:
                        print(column_name, ": environment context shape is", len(round_value))
                elif context_type == "series":
                    series_index = self.index_map_dict["series_index_map"]
                    if data_type == "time":
                        round_value = self.context_to_tensor_by_series(raw_df, series_index, column_name, column_type,
                                                                       zero_embedding=zero_timestamp,
                                                                       auxiliary_raw_df=auxiliary_raw_df)
                    elif data_type == "function" or data_type == "function list":
                        round_value = self.context_to_tensor_by_series(raw_df, series_index, column_name, column_type,
                                                                       auxiliary_raw_df=auxiliary_raw_df)
                    else:
                        round_value = self.context_to_tensor_by_series(raw_df, series_index, column_name, column_type)

                    if index % 10000 == 0:
                        print(column_name, ": series number is", len(round_value), "series context shape is",
                              len(round_value))

                convert_data.get(column_name).append(round_value)
            pre_date = date
        for column_name, column_type in keep_values_col.items():
            data_type = column_type[0]
            context_type = column_type[1]
            if data_type == "str" or data_type == "str_emb" or data_type == "time" \
                    or data_type == "id" or data_type == "float list" or data_type == "function list":
                print("round_list_shape of ", column_name, ":", len(convert_data.get(column_name)))
            else:
                values_tensor = convert_data.get(column_name)
                values_tensor = torch.stack(values_tensor).float()
                print("round_list_shape of ", column_name, ":", values_tensor.shape)
                convert_data[column_name] = values_tensor

        return convert_data

    def save_reward_tensor_data(self, reward_file_name: str, seg: int):
        """
        用于处理reward，生成reward.pt，
        :param reward_file_name: 生成reward的文件名称
        :param seg:
        :return:
        """
        raise NotImplementedError
