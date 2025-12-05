import os
import pickle
import datetime
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from computing.core.dtype import LearningVariable, HyperParameter, BufferParameter


class Module(nn.Module):
    r"""
    Module是所有子模块的基类。它无法包含其他Container或 Module，通常作为树结构的叶子节点。

    Args:
        module_id (int or str): 模块的ID

    Note:
        Container的实例可以直接调用，调用时会进行一轮模型更新、决策用时更新、参数更新。具体代码如下::

            class Container(object):
                def __call__(self, *args, **kwargs):
                    self._call_time = datetime.datetime.now()
                    self.update(*args, **kwargs)
                    self.register_decide_hooks(self._decide_hooks.keys())
                    self._call_time = datetime.datetime.now() - self._call_time
    """

    def __init__(self, module_id, device="cpu"):
        """
        Initializes internal Module state.
        """
        super(Module, self).__init__()
        self.__id = module_id
        self._learning_variables = OrderedDict()
        self._hyper_parameters = OrderedDict()
        self._buffer_parameters = OrderedDict()
        self._decide_hooks = OrderedDict()
        self._call_time = 0
        self.device = device

    def get_model_tree_struct(self, nodes={}):
        d = {}
        d['module'] = None
        d['sub_modules'] = None
        d['sub_module_template'] = None
        d['file'] = "modules/{}".format(self.get_class_file_name())
        d['class'] = self.get_class_name()
        key = self.get_class_name()
        nodes[key] = d

    def set_device(self, device):
        if self.device == 'cuda':
            self.cuda()
        self.device = device

    def set_id(self, id):
        """
        重设模块的ID。

        Args:
            id (int or str): 新的模块ID。
        """
        self.__id = id

    def get_id(self):
        """
        获取模块的ID。
        Returns:
            int or str: 模块的ID
        """
        return self.__id

    def get_time(self):
        """
        获取该容器最近一次调用（包含决策和更新）所需要的时间。

        Returns:
            datetime: 调用耗时
        """
        return self._call_time

    def forward(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        """
        根据模型给出的决策，以及环境中的反馈，更新模型参数。该函数为虚函数，待子类实现。
        """
        return None

    def __call__(self, mode="update", *args, **kwargs):
        """

        Args:
            mode: 只能有"update","forward","both"
            *args:
            **kwargs:

        Returns:

        """
        self._call_time = datetime.datetime.now()
        if mode == "update":
            self.update(*args, **kwargs)
        elif mode == "forward":
            self.forward(*args, **kwargs)
        elif mode == "both":
            self.forward(*args, **kwargs)
            self.update(*args, **kwargs)
        else:
            self.update(mode, *args, **kwargs)

        self.register_decide_hooks(self._decide_hooks.keys())
        self._call_time = datetime.datetime.now() - self._call_time

    def decide(self, *args, **kwargs):
        r"""
        根据当前的市场数据，给出决策。该函数为虚函数，待子类实现。
        """
        return self._decide_hooks

    def register_hyper_parameter(self, name, param):
        """
        注册模块的buffer。

        Args:
            name (str): hyper_parameter的名字
            param (Any): hyper_parameter的值
        """
        if '_hyper_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign hyper_parameters before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("hyper_parameter's name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("hyper_parameter's name can't contain \".\"")
        elif name == '':
            raise KeyError("hyper_parameter's name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._hyper_parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._hyper_parameters[name] = None
        else:
            param = param.data
            if isinstance(param, torch.Tensor) and self.device != param.device:
                self._hyper_parameters[name] = param.to(self.device)
            else:
                self._hyper_parameters[name] = param

    def register_buffer_parameter(self, name, param):
        """
        注册模块的buffer。

        Args:
            name (str): buffer_parameter的名字
            param (Any): buffer_parameter的值
        """
        if '_buffer_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer_parameters before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer_parameter's name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("buffer_parameter's name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer_parameter's name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffer_parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._buffer_parameters[name] = None
        else:
            param = param.data
            if isinstance(param, torch.Tensor):
                self._buffer_parameters[name] = param.to("cpu")
            else:
                self._buffer_parameters[name] = param

    def register_learning_variable(self, name, param):
        """
        注册模块的 learning_variable。

        Args:
            name (str): learning_variable 的名字
            param (Any): learning_variable 的值
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign learning_variable before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("learning_variable's name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("learning_variable's name can't contain \".\"")
        elif name == '':
            raise KeyError("learning_variable name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._learning_variables:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._learning_variables[name] = None
        else:
            param = param.data
            if isinstance(param, torch.Tensor) and self.device != param.device:
                self._learning_variables[name] = param.to(self.device)
            else:
                self._learning_variables[name] = param

    def register_decide_hooks(self, name_list):
        """
        注册/更新模块中需要被外部调用的参数。

        Args:
            name_list (Any iterable): 需要注册/更新的参数名字
        """
        if '_decide_hooks' not in self.__dict__:
            raise AttributeError(
                "cannot assign decide hook before Module.__init__() call")
        else:
            for name in name_list:
                if name in self._learning_variables.keys():
                    self._decide_hooks[name] = self._learning_variables[name]
                elif name in self._hyper_parameters.keys():
                    self._decide_hooks[name] = self._hyper_parameters[name]
                elif name in self._buffer_parameters.keys():
                    self._decide_hooks[name] = self._buffer_parameters[name]
                elif name in self.__dict__.keys():
                    self._decide_hooks[name] = self.__dict__[name]
                else:
                    raise KeyError("attribute '{}' is not in hyper_parameters, learning_variables, "
                                   "and buffer_parameters".format(name))

    def __getattr__(self, name):
        """
        获取参数值。

        Args:
            name (str): 参数名

        Returns:
            any: 参数值
        """
        if '_learning_variables' in self.__dict__:
            _learning_variables = self.__dict__['_learning_variables']
            if name in _learning_variables:
                return _learning_variables[name]
        if '_hyper_parameters' in self.__dict__:
            _hyper_parameters = self.__dict__['_hyper_parameters']
            if name in _hyper_parameters:
                return _hyper_parameters[name]
        if '_buffer_parameters' in self.__dict__:
            _buffer_parameters = self.__dict__['_buffer_parameters']
            if name in _buffer_parameters:
                return _buffer_parameters[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]

        return super().__getattr__(name)


    def __setattr__(self, name, value):
        """
        设置参数值。

        Args:
            name (str): 参数名
            value (any): 参数值
        """

        def remove_from(*dicts):
            for one_dict in dicts:
                if name in one_dict:
                    del one_dict[name]

        module_default_parameters = ["__id", "_call_time", "device", "_learning_variables", "_hyper_parameters", "_buffer_parameters", "_decide_hooks"]
        for default_parameter in module_default_parameters:
            if default_parameter in name:
                object.__setattr__(self, name, value)
                return

        if isinstance(value, LearningVariable):
            learning_variables = self.__dict__.get('_learning_variables')
            if learning_variables is None:
                raise AttributeError(
                    "cannot assign learning_variables before Module.__init__() call")
            remove_from(self.__dict__, self._hyper_parameters, self._buffer_parameters)
            self.register_learning_variable(name, value)
        elif isinstance(value, HyperParameter):
            hyper_parameters = self.__dict__.get('_hyper_parameters')
            if hyper_parameters is None:
                raise AttributeError(
                    "cannot assign buffers before Module.__init__() call")
            remove_from(self.__dict__, self._learning_variables, self._buffer_parameters)
            self.register_hyper_parameter(name, value)
        elif isinstance(value, BufferParameter):
            buffer_parameters = self.__dict__.get('_buffer_parameters')
            if buffer_parameters is None:
                raise AttributeError(
                    "cannot assign buffers before Module.__init__() call")
            remove_from(self.__dict__, self._learning_variables, self._hyper_parameters)
            self.register_buffer_parameter(name, value)
        elif isinstance(value, nn.Module):
            remove_from(self.__dict__)
            super().__setattr__(name, value)
            return
        else:
            nn_module_default_parameters = nn.Module().__dict__
            if name in nn_module_default_parameters.keys():
                super().__setattr__(name, value)
                return
            elif name in self._hyper_parameters.keys():
                self._hyper_parameters[name] = value
                return
            elif name in self._learning_variables.keys():
                self._learning_variables[name] = value
                return
            elif name in self._buffer_parameters.keys():
                self._buffer_parameters[name] = value
                return
            else:
                super().__setattr__(name, value)
                warnings.warn(message="Attribute '{}' is assigned by nn.Module, please type declaration for Module "
                              "parameters, such as LearningVariable, HyperParameter, and BufferParameter".format(name),
                              category=DeprecationWarning)

    def __delattr__(self, name):
        """
        删除参数值。

        Args:
            name (str): 参数名
        """
        if name in self._learning_variables:
            del self._learning_variables[name]
        elif name in self._hyper_parameters:
            del self._hyper_parameters[name]
        elif name in self._buffer_parameters:
            del self._buffer_parameters[name]
        else:
            object.__delattr__(self, name)

    def get_class_name(self):
        return self.__class__.__name__

    def get_class_file_name(self):
        file_name = str(os.path.basename(__file__))
        return file_name.split('.')[0]

    def get_hyper_parameters(self, name_space=False):
        """
        获取所有 hyper_parameters。

        Returns:
            OrderedDict: hyper_parameters
        """
        hp = self._hyper_parameters
        hp_namespace = {}
        if name_space:
            for k, v in hp.items():
                hp_namespace[self.__class__.__name__ + '_' + str(k)] = v
            return hp_namespace
        else:
            return self._hyper_parameters

    def get_learning_variables(self):
        """
        获取所有 _learning_variables。

        Returns:
            OrderedDict: _learning_variables
        """
        return self._learning_variables

    def get_buffer_parameters(self):
        """
        获取所有 _learning_variables。

        Returns:
            OrderedDict: _learning_variables
        """
        return self._buffer_parameters

    def save(self, file_folder_path, prev_path=''):
        if '_learning_variables' not in self.__dict__.keys():
            raise AttributeError(
                "cannot assign learning_variables before {}.__init__() call".format(type(self).__name__))

        content_dict = {}
        if prev_path:
            raise NotImplementedError

        else:
            self_dict = self.__dict__
            file_name = type(self).__name__ + '@' + \
                        str(datetime.datetime.now()).replace('-', '_').replace(' ', '_')
            file_path = file_folder_path + '/' + file_name

            # save parameters
            for key in self_dict.keys():
                content_dict[key] = self_dict[key]

            # save as json
            content_dict['class_type'] = 'Module'
            with open(file_path, 'wb') as f:
                pickle.dump(content_dict, f)

        return file_name

    def convert_device(self, device="cpu"):
        for key, item in self._hyper_parameters.items():
            if isinstance(item, torch.Tensor):
                if device == "cuda":
                    self._hyper_parameters[key] = item.cuda()
                else:
                    self._hyper_parameters[key] = item.cpu()
        for key, item in self._learning_variables.items():
            if isinstance(item, torch.Tensor):
                if device == "cuda":
                    self._learning_variables[key] = item.cuda()
                else:
                    self._learning_variables[key] = item.cpu()
        for key, item in self._decide_hooks.items():
            if isinstance(item, torch.Tensor):
                if device == "cuda":
                    self._decide_hooks[key] = item.cuda()
                else:
                    self._decide_hooks[key] = item.cpu()

        if device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"

