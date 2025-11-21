import json
import os
from pathlib import Path
from computing.proxy.data_proxy import SeriesTensorDataProxy
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
PATH_FILE = Path(PARENT_DIR + "/path_config.json")
CONTEXT_MENU_FILE = Path(PARENT_DIR + "/context_menu_config.json")
DATASET_FILE = Path(PARENT_DIR + "/dataset_config.json")


def get_path_config(service, key, dataset=""):
    """
    :param service: server name
    :param dataset: dataset name
    :param key: dataset/ metric
    :return:
    """
    paths = json.load(open(PATH_FILE))
    path = paths[key][service]
    path = path.format(dataset)
    return path


def get_dataset_config(dataset="djia", key=""):
    """
    :param dataset: dataset name
    :param key: dataset/ metric
    :return:
    """
    paths = json.load(open(DATASET_FILE))
    value = paths[dataset][key]
    return value


def get_context_menu_all_config():
    all_context_menus = json.load(open(CONTEXT_MENU_FILE))
    return all_context_menus


def get_context_menu_config(context_name):
    all_context_menus = json.load(open(CONTEXT_MENU_FILE))
    all_context_menu = all_context_menus[context_name]
    return all_context_menu


def update_context_menu_config(context_name, file_name):
    all_context_menus = json.load(open(CONTEXT_MENU_FILE))
    all_context_menus[context_name] = file_name
    with open(CONTEXT_MENU_FILE, 'w') as outfile:
        json.dump(all_context_menus, outfile)


def update_path_config(service, key):
    paths = json.load(open(PATH_FILE))
    paths[key] = service
    with open(PATH_FILE, 'w') as outfile:
        json.dump(paths, outfile)


def config_data(data_name="djia", path_type="remote33", reward_file_name="reward_tensor", context_menu=None, window_size=30, context_window_size=5,
                external_data_path=None):
    FILE_DIR = get_path_config(path_type, 'dataset', data_name)

    file_path = {"reward": reward_file_name, "index_map_dict": "index_map_dict"}
    if context_menu is None:
        file_path += get_context_menu_all_config()
    else:
        for context_name in context_menu:
            file_path[context_name] = get_context_menu_config(context_name)

    data_params = {"dataset_name": data_name,
                   "file_dir": FILE_DIR,
                   "file_path": file_path,
                   "split_type": get_dataset_config(data_name, "split_type"),
                   "constructor": "multi_context_load_tensor_constructor",
                   "window_size": {"window_size": window_size, "context_window_size": context_window_size},
                   "data_proxy_class": get_dataset_config(data_name, "data_proxy_class")}
    if external_data_path:
        data_params["external_data_path"] = external_data_path

    data = SeriesTensorDataProxy(**data_params)
    stock_num = get_dataset_config(data_name, "stock_num")

    # index_map_dict = torch.load(FILE_DIR + "/index_map_dict.pt")
    return data, stock_num
