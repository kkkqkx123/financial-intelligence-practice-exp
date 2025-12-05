from computing.utils.data_reader import TensorDataReader
from computing.utils.load_constructor import multi_context_load_tensor_constructor
from data_preprocess import get_config

def tensor_data_view(dateset_name, file_path, window_size=0, context_window_size=0,
                     batch_size=16, start_index=None, end_index=None):
    tensor_file_path = get_config(dateset_name, "data_target_path")
    split_type = get_config(dateset_name, "split_type")
    reader = TensorDataReader(file_dir=tensor_file_path,
                              file_path=file_path,
                              dataset_=None,
                              constructor=multi_context_load_tensor_constructor,
                              window_size={"window_size": window_size, "context_window_size": context_window_size},
                              split_type=split_type)

    if start_index is None: start_index = max(window_size, context_window_size)
    start_index = max(window_size, context_window_size, start_index)
    reader.set_fetch_batch(batch_size=batch_size, start_index=start_index, end_index=end_index)
    batch_data = reader.fetch_next_batch()
    while batch_data != -1:
        print(batch_data)
        batch_data = reader.fetch_next_batch()


if __name__ == "__main__":
    dateset_name = "demo"
    file_path = {"environment_news": "news_environment_context",
                 "series_news_index": "news_index_series_context",
                 "environment_audio": "audio_environment_context",
                 "series_audio_index": "audio_index_series_context",
                 "reward": "reward_tensor", "index_map_dict": "index_map_dict"}
    tensor_data_view(dateset_name, file_path, context_window_size=5)

