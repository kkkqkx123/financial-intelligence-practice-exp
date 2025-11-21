from typing import Union, Optional
from computing.utils.data_struct import SeriesDictData, RoundDictData, SeriesTensorData, ContextSeriesTensorData


def series_tensor_to_dict(series_tensor_data: Union[None, SeriesTensorData, ContextSeriesTensorData]) \
        -> Union[None, SeriesDictData]:
    if series_tensor_data is None:
        return None

    series_dict_date_list = []
    reward_dic = series_tensor_data.convert_reward_to_dict()
    series_context_dic = series_tensor_data.convert_series_context_to_dict()
    environment_context_dic = series_tensor_data.convert_environment_context_to_dict()

    for round_, reward in reward_dic.items():
        series_context = series_context_dic.get(round_)
        environment_context = environment_context_dic.get(round_)
        series_dict_date_list.append(
            RoundDictData(round_=round_, environment_context=environment_context, series_context=series_context,
                          reward=reward))
    context_window_size = 0 if series_tensor_data.__class__.__name__ == "SeriesTensorData" \
        else series_tensor_data.context_window_size

    if series_tensor_data.pre_dataset is None:
        pre_dataset = None
    else:
        series_tensor_data.pre_dataset.pre_dataset = None  # 确保截断循环挂载
        pre_dataset = series_tensor_to_dict(series_tensor_data.pre_dataset)

    return SeriesDictData(index_map_dict=series_tensor_data.index_map_dict, data_list=series_dict_date_list,
                          window_size=series_tensor_data.window_size,
                          context_window_size=context_window_size,
                          packed=series_tensor_data.packed,
                          pre_dataset=pre_dataset)


def series_dict_to_tensor(series_dict_data: Optional[SeriesDictData]) \
        -> Union[None, SeriesTensorData, ContextSeriesTensorData]:
    if series_dict_data is None or len(series_dict_data) == 0:
        return None

    # confirm context or not context
    test_round_dict_data = series_dict_data.data[0]
    has_no_context = test_round_dict_data.series_context is None and test_round_dict_data.environment_context is None

    if series_dict_data.pre_dataset and series_dict_data.pre_dataset.pre_dataset:
        series_dict_data.pre_dataset.pre_dataset = None
    pre_dataset = series_dict_to_tensor(series_dict_data.pre_dataset) if series_dict_data.pre_dataset else None
    index_map_dict = series_dict_data.index_map_dict
    packed = series_dict_data.packed

    reward = None
    series_context = {}
    environment_context = {}

    for round_dict_data in series_dict_data.data:
        reward = round_dict_data.convert_reward_to_tensor(index_map_dict=index_map_dict, reward=reward)
        if has_no_context is False:
            series_context = round_dict_data.convert_series_context_to_tensor(index_map_dict=index_map_dict,
                                                                              series_context=series_context,
                                                                              packed=packed)
            environment_context = round_dict_data.convert_environment_context_to_tensor(environment_context)
    # if len(reward) != len(date_index_dict):
    #     raise EOFError('dates do not match date_index_dict, conversion failed!')
    if has_no_context:
        return SeriesTensorData(reward=reward, index_map_dict=index_map_dict,
                                pre_dataset=pre_dataset, init_index=series_dict_data.init_index,
                                packed=series_dict_data.packed, window_size=series_dict_data.window_size)
    else:
        return ContextSeriesTensorData(environment_context=environment_context, series_context=series_context,
                                       pre_dataset=pre_dataset, init_index=series_dict_data.init_index,
                                       reward=reward, index_map_dict=series_dict_data.index_map_dict,
                                       packed=packed, window_size=series_dict_data.window_size,
                                       context_window_size=series_dict_data.context_window_size)
