import os
import re

import pandas as pd

from computing.utils.util_functions import get_seg, get_seg_range
from data_preprocess import get_config


def sort_one_file(INIT_DATE, file_path, FILE_DIR_PATH, split_type="1y", datetime_type="date", time_column_name="Date"):
    """
    @param INIT_DATE: str
    @param file_path: 待排序的文件
    @param FILE_DIR_PATH: 排序好的文件存储的文件夹路径
    @param split_type: str
    @return: list, 被写入的文件的seg
    """
    data = pd.read_csv(file_path, header=0)
    dates = data[time_column_name].unique()

    sorted_file = [get_seg(date, INIT_DATE, split_type, datetime_type) for date in dates]
    sorted_file = list(set(sorted_file))
    if -1 in sorted_file:
        sorted_file.remove(-1)

    for seg in sorted_file:
        start, end = get_seg_range(seg, INIT_DATE, split_type, datetime_type)
        df = data[(start <= data[time_column_name]) & (data[time_column_name] <= end)]

        name = os.path.basename(file_path)
        name = '_'.join(name.split("_")[:-1])
        name = 'sort_' + name + '_' + str(seg) + ".csv"
        write_path = FILE_DIR_PATH + "/"+ name
        if not os.path.exists(write_path):
            df.to_csv(write_path, mode='w', header=True, index=None)
        else:
            df.to_csv(write_path, mode='a', header=False, index=None)
    return sorted_file


def resort(dataset, file_list, split_type="1y", empty=True, datetime_type="date", time_column_name="Date"):
    """
    @param dataset:数据集名称，也是文件夹名称
    @param file_list: 支持多个文件名的一起sort
    @param split_type: 划分时间，在config里面有
    @param empty: 是否在resort前清空旧文件，默认清空
    @param datetime_type: date or time
    @return:
    """
    FILE_DIR_PATH = get_config(dataset, 'data_source_path')
    INIT_DATE = get_config(dataset, "init_date")
    for context_file in file_list:
        sorted_file_cursor = 0
        path = FILE_DIR_PATH + context_file + "_" + str(sorted_file_cursor + 1) + '.csv'
        sorted_file = []

        if empty:
            file_name_list = os.listdir(FILE_DIR_PATH)
            match_file = "sort_" + context_file + "_*"
            for file_name in file_name_list:
                if re.match(match_file, file_name):
                    os.remove(FILE_DIR_PATH + file_name)

        while os.path.exists(path):
            sorted_file += sort_one_file(INIT_DATE, path, FILE_DIR_PATH, split_type=split_type,
                                         datetime_type=datetime_type, time_column_name=time_column_name)
            print("finish sorting: " + context_file + "_" + str(sorted_file_cursor + 1) + '.csv')
            sorted_file_cursor += 1
            path = FILE_DIR_PATH + "/" + context_file + "_" + str(sorted_file_cursor + 1) + '.csv'


# 生成round内的相对index
def relative_index_generation(dataset, file_list, datetime_column="date", sort_by="timestamp"):
    """
    @param dataset: 数据集名称，也是文件夹名称
    @param file_list:支持多个文件名的一起sort
    @param datetime_column: date
    @return:
    """
    FILE_DIR_PATH = get_config(dataset, 'data_source_path')
    for context_file in file_list:
        sorted_file_cursor = 0
        path = FILE_DIR_PATH + "sort_" + context_file + "_" + str(sorted_file_cursor + 1) + '.csv'

        while os.path.exists(path):
            data = pd.read_csv(path, header=0)
            # 检测是否有relative_index这一列，有的话，删除这一列
            if "relative_index" in data.columns:
                del data['relative_index']
            # 排序整个文件
            data = data.sort_values(by=sort_by, axis=0, ascending=True)
            # 生成relative_index列
            relative_index = []

            group_date_list = data[datetime_column].unique()
            count_list = data[datetime_column].value_counts()
            for cur_date in group_date_list:
                relative_index += list(range(count_list[cur_date]))

            data["relative_index"] = relative_index

            write_path = FILE_DIR_PATH + "/" + "sort_" + context_file + "_" + str(sorted_file_cursor + 1) + '.csv'
            data.to_csv(write_path, index=False)
            print("finish generating index: " + context_file + "_" + str(sorted_file_cursor + 1) + '.csv')
            sorted_file_cursor += 1
            path = FILE_DIR_PATH + "/" + "sort_" + context_file + "_" + str(sorted_file_cursor + 1) + '.csv'
            del data