from collections import OrderedDict

import torch
import pandas as pd

from computing.utils.data_struct import SeriesDictData, SeriesTensorData
from computing.utils.data_converter import series_tensor_to_dict



class Evaluator(object):
    def __init__(self, file_dir, method="Default", dataset="Default"):
        self._file_dir = file_dir

        self._metrics = OrderedDict()
        self._method = method
        self._dataset = dataset
        self._times = 0

    def eval(self, result, dict_data, *args, **kwargs):
        raise NotImplementedError

    def after_eval(self, reset_metric=True, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, result, data, *args, **kwargs):
        self._times += 1
        if isinstance(data, SeriesTensorData):
            if len(data) == 1:
                eval_data = series_tensor_to_dict(data)[0]
            else:
                eval_data = data
        elif isinstance(data, SeriesDictData):
            if len(data) == 1:
                eval_data = data[0]
            else:
                eval_data = data
        else:
            raise TypeError("cannot assign '{}' object to SeriesDictData object ".format(type(data)))
        if result is None:
            raise ModuleNotFoundError("Can't find results. Please check !")

        # eval_data : Union[RoundDictData, SeriesTensorData, SeriesDictData]
        self.eval(result, eval_data, *args, **kwargs)

    def change_file_dir(self, file_dir):
        self._file_dir = file_dir

    def save(self, keywords):
        """
        save metrics by metric name
        :param keywords: name of metrics
        """
        file_name = self._file_dir + '_{dataset}_{keyword}_{method}.csv'
        for keyword in keywords:
            if not isinstance(self._metrics[keyword], list):
                metric = pd.DataFrame([self._metrics[keyword]])
            else:
                metric = pd.DataFrame(self._metrics[keyword])
            metric.to_csv(file_name.format(dataset=self._dataset, method=self._method, keyword=keyword), index=True,
                          sep=',')

    def register_metric(self, name, metric):
        if '_metrics' not in self.__dict__:
            raise AttributeError(
                "cannot assign metric before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("metric name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("metric name can't contain \".\"")
        elif name == '':
            raise KeyError("metric name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._metrics:
            raise KeyError("attribute '{}' already exists".format(name))

        if metric is None:
            self._metrics[name] = None
        elif not (isinstance(metric, list) or isinstance(metric, torch.Tensor)):
            raise TypeError("cannot assign '{}' object to list '{}' "
                            "(list or None required)"
                            .format(type(metric), name))
        else:
            self._metrics[name] = metric

    def __getattr__(self, name):
        if '_metrics' in self.__dict__:
            _metrics = self.__dict__['_metrics']
            if name in _metrics:
                return _metrics[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        metrics = self.__dict__.get('_metrics')
        if isinstance(value, list) or isinstance(value, torch.Tensor):
            if metrics is None:
                raise AttributeError(
                    "cannot assign metrics before Evaluator.__init__() call")
            self.register_metric(name, value)
        elif metrics is not None and name in metrics:
            if value is not None:
                raise TypeError("cannot assign '{}' as metric '{}' "
                                "(list or None expected)"
                                .format(torch.typename(value), name))
            self.register_metric(name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._metrics:
            del self._metrics[name]
        else:
            object.__delattr__(self, name)

