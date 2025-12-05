import torch

from computing.utils.data_struct import RoundDictData
from computing.utils.data_struct import SeriesTensorData, ContextSeriesTensorData
from computing.utils.util_functions import get_seg_index


def dict2tensor(dicts):
    if dicts:
        for key, value in dicts.items():
            dicts[key] = torch.tensor(value)
        return dicts
    else:
        return None


def dict_load_constructor(json_object):
    # print(json_object)
    environment_context = {"bandit_id": json_object["bandit_id"]}
    if json_object["bandit_context"] != "":
        environment_context["bandit_context"] = torch.tensor(json_object["bandit_context"])

    return RoundDictData(round_=json_object["round"],
                         reward=dict2tensor(json_object["reward"]),
                         series_context=dict2tensor(json_object["series_context"]),
                         environment_context=environment_context['environment_context'])


def default_load_tensor_constructor(reward, index_map_dict_data, iter_index=1,
                                    window_size={"window_size": 1}, pre_dataset=None, seg=0, split_type="year"):
    start_index, _ = get_seg_index(round_index_map=index_map_dict_data["round_index_map"], seg=seg, split_type=split_type)
    return SeriesTensorData(reward=reward,
                            index_map_dict=index_map_dict_data,
                            iter_index=iter_index, window_size=window_size["window_size"],
                            pre_dataset=pre_dataset, init_index=start_index)


def multi_context_load_tensor_constructor(series_context, environment_context, reward, index_map_dict_data, iter_index=1,
                                          window_size={"window_size": 1, "context_window_size": 1},
                                          pre_dataset=None, seg=0, split_type="year"):
    start_index, _ = get_seg_index(round_index_map=index_map_dict_data["round_index_map"], seg=seg, split_type=split_type)
    return ContextSeriesTensorData(series_context=series_context,
                                   environment_context=environment_context,
                                   reward=reward,
                                   index_map_dict=index_map_dict_data,
                                   window_size=window_size["window_size"],
                                   context_window_size=window_size["context_window_size"],
                                   pre_dataset=pre_dataset, init_index=start_index)
