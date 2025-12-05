import json
import datetime

from dateutil.relativedelta import relativedelta
from datetime import datetime


def obj2json(python_obj):
    """
    将一个class转成json
    :param python_obj: python中自定义的一个class
    :return: json
    """
    return json.dumps(python_obj.__dict__)


def str2datetime(str, datetime_type="date"):
    if datetime_type == "date":
        return datetime.strptime(str[0:10], "%Y-%m-%d")
    elif datetime_type == "time":
        if len(str) == 10:
            str = str + " 00:00:00"
        elif len(str) == 16:
            str = str + ":00"
        return datetime.strptime(str[0:19], "%Y-%m-%d %H:%M:%S")


def get_seg(round, init_round, split_type="1y", datetime_type=None):
    """
    @param round: str
    @param init_round: str
    @param split_type: str , which common format is split length + split scale.
                             For example, "7d" means 7 days, "1m" means 1 month, and "3y" means 3 years.
                             Moreover, there are some special type, which currently supported is "half_month".
    @return: the index of seg, which start from 1
    """
    if datetime_type is None:
        datetime_type = "date" if len(round) == 10 else "time"
    round = str2datetime(round, datetime_type)
    init_round = str2datetime(init_round, datetime_type)

    if round < init_round:
        return -1

    # get seg: special type
    if split_type == "half_month":
        month_interval = (round.year * 12 + round.month) - (init_round.year * 12 + init_round.month)
        if init_round.day <= 15:
            seg = month_interval * 2 + 1 if round.day <= 15 else month_interval * 2 + 2
        else:
            seg = month_interval * 2 if round.day <= 15 else month_interval * 2 + 1
        return seg

    # get seg: common type
    if split_type[-1] == 'd':
        interval = (round - init_round).days
    elif split_type[-1] == "m":
        interval = (round.year * 12 + round.month) - (init_round.year * 12 + init_round.month)
    elif split_type[-1] == "y":
        interval = round.year - init_round.year
    seg = interval // int(split_type[0:-1]) + 1
    return seg


def get_seg_range(seg, init_round, split_type="1y", datetime_type=None):
    """
    @param seg: the index of seg, which start from 1
    @param init_round: str
    @param split_type: str , which common format is split length + split scale.
                             For example, "7d" means 7 days, "1m" means 1 month, and "3y" means 3 years.
                             Moreover, there are some special type, which currently supported is "half_month".
    @return: str, the start date and end date of the seg
    """
    if datetime_type is None:
        raise IOError("Please set datetime_type!")
    init_round = str2datetime(init_round, datetime_type)

    if split_type == "half_month":
        if init_round.day <= 15:
            if seg % 2 == 1:
                start_round = init_round.replace(day=1) + relativedelta(months=seg // 2)
                end_round = start_round.replace(day=15)
            else:
                start_round = init_round.replace(day=16) + relativedelta(months=(seg-1) // 2)
                end_round = start_round.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
        else:
            if seg % 2 == 1:
                start_round = init_round.replace(day=16) + relativedelta(months=seg // 2)
                end_round = start_round.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
            else:
                start_round = init_round.replace(day=1) + relativedelta(months=seg // 2)
                end_round = start_round.replace(day=15)
    else:   # common type
        length = int(split_type[0:-1])
        if split_type[-1] == 'd':
            start_round = init_round + relativedelta(days=(seg - 1) * length)
            end_round = init_round + relativedelta(days=seg * length - 1)
        elif split_type[-1] == "m":
            start_round = init_round.replace(day=1) + relativedelta(months=(seg - 1) * length)
            end_round = init_round.replace(day=1) + relativedelta(months=seg * length) - relativedelta(days=1)
        elif split_type[-1] == "y":
            start_round = init_round.replace(month=1, day=1) + relativedelta(years=(seg - 1) * length)
            end_round = init_round.replace(month=1, day=1) + relativedelta(years=seg * length) - relativedelta(days=1)

    if datetime_type == "date":
        start_round_str = init_round.strftime("%Y-%m-%d") if seg == 1 else start_round.strftime("%Y-%m-%d")
        end_round_str = end_round.strftime("%Y-%m-%d")
    elif datetime_type == "time":
        start_round_str = init_round.strftime("%Y-%m-%d %H:%M:%S") if seg == 1 \
            else start_round.strftime("%Y-%m-%d") + " 00:00:00"
        end_round_str = end_round.strftime("%Y-%m-%d") + " 23:59:59"
    return start_round_str, end_round_str


def get_round_index(round_index_map, round, type=">="):
    if round <= round_index_map[0]:
        return 0
    if round >= round_index_map[-1]:
        return len(round_index_map) - 1

    i, j = 0, len(round_index_map) - 1
    index = -1
    while i < j:    # 二分查找
        k = (i + j) // 2
        if round < round_index_map[k]:
            if round > round_index_map[k-1]:
                index = k
                break
            elif round == round_index_map[k - 1]:
                index = k-1
                break
            else:
                j = k - 1
        elif round > round_index_map[k]:
            if round <= round_index_map[k + 1]:
                index = k+1
                break
            else:
                i = k + 1
        else:
            index = k
            break

    if index != -1 and type == "<=":
        if round != round_index_map[index]:
            index -= 1
    # elif index != len(round_index_map)-1 and type == ">=":
    #     if round != round_index_map[index]:
    #         index += 1

    return index


def get_seg_index(round_index_map, seg, split_type="half_month", datetime_type=None):
    """
    @param round_index_map: a list of round
    @param seg: the index of seg, which start from 1
    @param split_type: str , which common format is split length + split scale.
                             For example, "7d" means 7 days, "1m" means 1 month, and "3y" means 3 years.
                             Moreover, there are some special type, which currently supported is "half_month".
    @return: the start round index and end round index of the seg
    """
    if datetime_type is None:
        datetime_type = "date" if len(round_index_map[0]) == 10 else "time"
    init_round = round_index_map[0]
    target_start_round, target_end_index = get_seg_range(seg, init_round, split_type, datetime_type)
    if target_end_index > round_index_map[-1]:
        target_end_index = round_index_map[-1]

    target_start_index = get_round_index(round_index_map, target_start_round, ">=")
    target_end_index = get_round_index(round_index_map, target_end_index, "<=")

    return target_start_index, target_end_index

