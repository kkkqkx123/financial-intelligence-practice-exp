# 为了以后扩展新的mode
ALL_MODE_TYPE = ["PreTraining", "Training", "Validation", "OnlineDecision", "Decision", "Updating"]

from .model_proxy import ModelProxy
from .data_proxy import DataProxy
from .execute_proxy import ExecuteProxy
from .metric_proxy import MetricProxy
