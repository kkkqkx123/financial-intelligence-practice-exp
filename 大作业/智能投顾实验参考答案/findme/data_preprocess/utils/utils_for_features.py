import torch
import pandas as pd
import numpy as np
from copy import deepcopy
import torchvision.transforms as transforms
from PIL import Image


# todo: input中有一个auxiliary_raw_df，基于auxiliary_raw_df做函数处理的。auxiliary_raw_df是一个pandas.dataframe。
def generate_historical_price_change(auxiliary_raw_df: pd.DataFrame, win_sizes=None):
    assert win_sizes is not None, ('The win_sizes is not allowed to be None ')
    # todo: 从auxiliary_raw_df取出p_C
    p_C_df = auxiliary_raw_df['Close']
    p_C = torch.from_numpy(np.array(p_C_df)).reshape(-1, 1)

    # if win_sizes is None:
    #     win_sizes = list(range(1, 45)) + [3 * 60, 5 * 60, 24 * 60, 3 * 24 * 60, 10 * 24 * 60]
    len_win_size = len(win_sizes)
    if p_C.shape[0] < max(win_sizes):
        return torch.zeros(len_win_size)
    else:
        p_C_feature = torch.zeros(len_win_size)

        for idx, win_size in enumerate(win_sizes):
            p_C_win = deepcopy(p_C)
            p_C_win[win_size:] = p_C_win[win_size:] - p_C_win[:-win_size]
            p_C_win[:win_size] = 0
            p_C_feature[idx] = p_C_win[-1]
            del p_C_win
        return p_C_feature


def generate_bc_action(auxiliary_raw_df: pd.DataFrame, win_size=0):
    win_size -= 1
    p_O_df = auxiliary_raw_df[['Open']]
    p_O = torch.from_numpy(np.array(p_O_df)).reshape(-1, 1)
    # bc_action = torch.zeros_like(p_O)
    arg_min_t = torch.argmin(p_O)
    arg_max_t = torch.argmax(p_O)
    if arg_min_t == win_size:
        action = 1  # long, order
    elif arg_max_t == win_size:
        action = -1  # short, order
    elif arg_max_t == win_size and arg_min_t == win_size:
        action = 0
    else:
        action = 0
    return torch.tensor(action)  # .reshape(1,-1)


def dual_thrust(auxiliary_raw_df: pd.DataFrame, K1=0.5, K2=0.5, date_index_map=None, trading_per_day=4 * 60):
    # todo: 从auxiliary_raw_df取出OHLC

    if len(auxiliary_raw_df) <= trading_per_day:
        return torch.zeros(2)
    else:
        data_indices_df = pd.DataFrame({'round_index_map': date_index_map})
        auxiliary_raw_df = auxiliary_raw_df.rename_axis('index').reset_index().rename(columns={'index': 'Date'})
        t_current = auxiliary_raw_df.iloc[-1]['Date']
        t_current_idx = date_index_map.index(t_current)
        t_start = data_indices_df.loc[(data_indices_df['round_index_map'].str.contains(t_current[:11]))].values[0]
        t_start_idx = date_index_map.index(t_start)  # ['round_index_map'][:11]
        last_day = date_index_map[t_start_idx - 1][:11]
        last_day_start = data_indices_df.loc[(data_indices_df['round_index_map'].str.contains(last_day))].values[
            0]  # 找前一天的日期，然后进而找到开始的位置
        last_day_start_idx = date_index_map.index(last_day_start)
        open_of_today = torch.tensor(auxiliary_raw_df.iloc[t_start_idx - t_current_idx - 1]['Open'])
        OHLC_df = auxiliary_raw_df.iloc[last_day_start_idx - t_current_idx - 1: t_start_idx - t_current_idx - 1][
            ['Open', 'Close', 'High', 'Low']]

        OHLC = torch.from_numpy(np.array(OHLC_df))
        HH = torch.max(OHLC[:, 1])  # the highest of the highest price
        LC = torch.min(OHLC[:, 3])
        HC = torch.max(OHLC[:, 3])
        LL = torch.min(OHLC[:, 2])
        range = torch.max((HH - LC), (HC - LL))

        buyLine = open_of_today + K1 * range
        sellLine = open_of_today - K2 * range
        return torch.stack([buyLine, sellLine]).reshape(-1, 1).squeeze(1)


def transform_picture(auxiliary_raw_df: pd.DataFrame, resize: int, out_file: str):
    image_name_list = auxiliary_raw_df['picture_name']
    value_list = []
    for image_name in image_name_list:
        try:
            img = Image.open(out_file + image_name).convert('RGB')
            transform = transforms.Compose([transforms.Resize(size=(resize, resize)), transforms.ToTensor()])
            value_list.append(transform(img))
        except:
            print("error picture:", image_name)
        finally:
            continue
    return value_list
