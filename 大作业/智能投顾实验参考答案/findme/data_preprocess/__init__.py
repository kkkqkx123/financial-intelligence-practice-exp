import json
import os
from pathlib import Path
from typing import Union

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
STATUS_FILE = Path(PARENT_DIR + "/preprocess_config.json")


def update_config(key, value):
    status = json.load(open(STATUS_FILE))
    status[key] = value
    with open(STATUS_FILE, 'w') as outfile:
        json.dump(status, outfile)


def get_config(dataset, key) -> Union[int, str, list]:
    status = json.load(open(STATUS_FILE))
    dataset_config = status.get(dataset)
    config = dataset_config[key]
    return config

