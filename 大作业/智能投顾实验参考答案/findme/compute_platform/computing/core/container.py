from computing.core.module import Module
from computing.core.dtype import ContainerProperty
from computing.utils.data_struct import SeriesDictData, RoundDictData, SeriesTensorData
import torch
from collections import OrderedDict
import datetime
import copy
import os
from typing import Union


class Container(object):
    r"""
    Container是所有容器的基类。它可以包含其他Container或 Module，允许使用树结构嵌入他们。你可以将子模块赋值给模型属性。

    Args:
        container_id (int or str): 容器的ID
        modules_dict (:class:`~computing.core.container.Container` or :class:`~computing.core.module.Module`): 容器中所包含的子模块，允许有多个

    Note:
        Container的实例可以直接调用，调用时会根据模式进行一轮操作，基础操作包括：训练、决策、更新模型。
    """

    def __init__(self, container_id: int = -1, modules_dict: dict = None, module_template=None, **kwargs):
        self.__id = container_id
        self._modules = OrderedDict()
        self._container_properties = OrderedDict()
        self._call_time = 0
        self._train_time = 0
        self._decision_time = 0
        self._update_time = 0
        self.device = "cpu"
        if modules_dict is not None:
            for name, module in modules_dict.items():
                self.add_module(module=module, name=name)
        if module_template is not None:
            if not (isinstance(module_template, Module) or isinstance(module_template, Container)):
                raise TypeError("cannot assign '{}' object to Module object "
                                .format(type(module_template)))
            self._module_template = module_template
        else:
            self._module_template = None

    def set_device(self, device):
        if self._modules is not None:
            for k, v in self._modules.items():
                self._modules[k].set_device(device)

        if device == 'cuda':
            self.cuda()
        else:
            self.cpu()
        self.device = device

    def get_hyper_parameters(self, name_space=False):
        module_hyper_parameters = OrderedDict()
        module_hyper_parameters_namespace = OrderedDict()
        if self._modules is not None:
            for k, v in self._modules.items():
                module_hyper_parameters.update(self._modules[k].get_hyper_parameters(name_space=name_space))
            if name_space:
                for k, v in module_hyper_parameters.items():
                    module_hyper_parameters_namespace[self.__class__.__name__ + '_' + str(k)] = v

        hyper_parameters = {**module_hyper_parameters} if not name_space else {
            **module_hyper_parameters_namespace}
        return hyper_parameters

    def __getstate__(self):
        # 必须返回字典
        return self.__dict__

    def __getattr__(self, name):
        """
        重写本函数，为了快捷获取模型。

        Args:
            name (str): 参数名

        Returns:
            any: 参数值
        """
        if '_modules' and '_container_properties' in self.__dict__:
            _modules = self.__dict__['_modules']
            _container_properties = self.__dict__['_container_properties']
            if name in _modules:
                return _modules[name]
            elif name in _container_properties:
                return _container_properties[name]
            elif name in self.__dict__:
                return self.__dict__[name]
            else:
                return self.__dict__
        else:
            super().__getattribute__(name)

    def register_container_properties(self, name, param):
        """
        注册容器的container_properties。

        Args:
            name (str): hyper_parameter的名字
            param (Any): hyper_parameter的值
        """
        if '_container_properties' not in self.__dict__:
            raise AttributeError(
                "cannot assign container_properties before Container.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("container_properties's name should be a string. "
                            "Got {}".format(type(name)))
        elif '.' in name:
            raise KeyError("container_property's name can't contain \".\"")
        elif name == '':
            raise KeyError("container_property's name can't be empty string \"\"")
        elif name in self.__dict__ and name not in self._container_properties:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._container_properties[name] = None
        else:
            param = param.data
            if isinstance(param, torch.Tensor) and self.device != param.device:
                self._container_properties[name] = param.to(self.device)
            else:
                self._container_properties[name] = param

    def re_init(self, new_arm_set, new_w):
        """
        当 arm set 变化的时候，Container 默认的 re_init 方法为调用 self._module 和 self._modules 的 re_init。

        Args:
            new_arm_set (set): 新的 arm set
            new_w (torch.Tensor): 平滑后的投资组合权重项链 w
        """
        if self._modules is not None and len(self._modules) > 0:
            for i, _ in self._modules.items():
                self._modules[i].re_init(new_arm_set, new_w)

    def get_id(self):
        """
        获取容器的ID。
        Returns:
            int or str: 容器的ID
        """
        return self.__id

    def set_id(self, container_id=-1):
        """
        重设容器的ID。

        Args:
            container_id (int or str): 新的容器ID。
        """
        self.__id = container_id

    def get_time(self):
        """
        获取该容器最近一次调用（包含决策和更新）所需要的时间。

        Returns:
            datetime: 调用耗时
        """
        return self._call_time, self._train_time, self._decision_time, self._update_time

    def pre_train(self, data, **kwargs):
        return None

    def train(self, data, **kwargs):
        return None

    def update(self, result: dict, data, **kwargs):
        """
        根据模型给出的决策，以及环境中的反馈，更新模型参数。该函数为虚函数，待子类实现。

        Args:
            result (dict): 模型给出的决策，字段包含但不限于 ``portfolio`` (类型为 dict), ``recommendation``,
                和 ``explanation``，其他决策输出的字段可由用户自定义。
            data (:class:`~computing.utils.data_struct.BanditData`): 环境数据
        """
        return None

    def decide(self, data: Union[SeriesDictData, SeriesTensorData, RoundDictData], **kwargs):
        r"""
        根据当前的round数据，给出决策。该函数为虚函数，待子类实现。

        Args:
            data (:class:`~computing.utils.data_struct.SeriesDictData`
            or :class:`~computing.utils.data_struct.SeriesTensorData`
            or :class:`~computing.utils.data_struct.RoundDictData`)

        Returns:
            dict: 模型给出的决策，字段包含但不限于 ``portfolio`` (类型为 dict), ``recommendation``, 和 ``explanation``，其他决策输出的字段可由用户自定义。

        """
        return None

    def __call__(self, data: Union[SeriesDictData, SeriesTensorData, RoundDictData], result=None, mode="Decision",
                 external_data=None, **kwargs):
        r"""
        根据mode类型完成模型的不同操作。

        Args:
            mode (string): 不同的模式，包括 ``Decision``（决策或者测试）, ``PreTraining`` （预训练）, ``Training`` （训练）,
            ``OninleDecision`` （决策并更新）, 和``Updating`` （更新）

        Returns:

        """
        self._call_time = datetime.datetime.now()

        assert isinstance(data, SeriesDictData) or isinstance(data, SeriesTensorData) or isinstance(data, RoundDictData)

        if mode == "Decision":
            self._decision_time = datetime.datetime.now()
            result = self.decide(data, external_data=external_data, **kwargs)
            self._call_time = datetime.datetime.now() - self._call_time
            self._decision_time = datetime.datetime.now() - self._decision_time
            self._train_time = 0
            self._update_time = 0
            if result is None:
                result = {}
            result["call_time"] = self._call_time
            return result
        elif mode == "PreTraining":
            result = {}
            self.pre_train(data, external_data=external_data, **kwargs)
            self._call_time = datetime.datetime.now() - self._call_time
            self._decision_time = 0
            self._train_time = 0
            self._update_time = 0
            if result is None:
                result = {}
            result["call_time"] = self._call_time
            return result
        elif mode == "Training":
            self._train_time = datetime.datetime.now()
            result = self.train(data, external_data=external_data, **kwargs)
            self._call_time = datetime.datetime.now() - self._call_time
            self._decision_time = 0
            self._train_time = datetime.datetime.now() - self._train_time
            self._update_time = 0
            if result is None:
                result = {}
            result["call_time"] = self._call_time
            return result
        elif mode == "OnlineDecision":
            self._decision_time = datetime.datetime.now()
            result = self.decide(data, external_data=external_data, **kwargs)
            self._decision_time = datetime.datetime.now() - self._decision_time
            self._update_time = datetime.datetime.now()
            self.update(result, data, external_data=external_data, **kwargs)
            self._update_time = datetime.datetime.now() - self._update_time
            self._call_time = datetime.datetime.now() - self._call_time
            self._train_time = 0
            if result is None:
                result = {}
            result["call_time"] = self._call_time
            return result
        elif mode == "Updating":
            self._update_time = datetime.datetime.now()
            self.update(result, data, external_data=external_data, **kwargs)
            self._update_time = datetime.datetime.now() - self._update_time
            self._call_time = datetime.datetime.now() - self._call_time
            self._train_time = 0
            self._decision_time = 0
            if result is None:
                result = {}
            result["call_time"] = self._call_time
            return result
        return None

    def children(self):
        """Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self._modules.items():
            yield module

    def apply(self, fn):
        """
        递归对所有子模型执行相同操作。

        Args:
            fn: 执行操作的函数

        Returns:
            :class:`computing.core.container.Container`: self
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def __setattr__(self, name, value):
        """
        设置参数值和模型。

        Args:
            name (str): 参数名
            value (any): 参数值
        """

        def remove_from(*dicts):
            for one_dict in dicts:
                if name in one_dict:
                    del one_dict[name]

        if "__id" in name or "_call_time" in name or "_train_time" in name or "_decision_time" in name or "_update_time" in name or "device" in name:
            super().__setattr__(name, value)
            return
        container_properties = self.__dict__.get('_container_properties')
        if isinstance(value, ContainerProperty):
            if container_properties is None:
                raise AttributeError(
                    "cannot assign container_properties before Container.__init__() call")
            remove_from(self.__dict__, self._modules)
            self.register_container_properties(name, value)
            return
        elif isinstance(value, Container) or isinstance(value, Module):
            modules = self.__dict__.get('_modules')
            if modules is None:
                raise AttributeError(
                    "cannot assign modules before Container.__init__() call")
            remove_from(self.__dict__, self._container_properties)
            if name in modules.keys():
                modules[name] = value
                return
            else:
                self.add_module(name=name, module=value)
                return
        elif container_properties is not None and name in self._container_properties:
            self._container_properties[name] = value
            return
        else:
            super().__setattr__(name, value)
            return

    def add_module(self, module=None, name=None):
        """
        添加子模型。

        Args:
            name: 模型的别名，缺省为模型的id
            module (:class:`~computing.core.container.Container` or :class:`~computing.core.module.Module`): 待添加的子模块
        """
        if name in self._modules.keys():
            raise KeyError("module '{}' already exists".format(name))
        elif name is None and module.get_id() in self._modules.keys():
            raise KeyError("module '{}' already exists".format(module.get_id()))
        else:
            if not (isinstance(module, Module) or isinstance(module, Container)):
                raise TypeError("cannot assign '{}' object to Module object "
                                .format(type(module)))
            if name is None:
                self._modules[module.get_id()] = module
            else:
                self._modules[name] = module

    def remove_module(self, module=None, name=None):
        """
        根据参数module(用id构建的)或name，移除响应的模块。

        Args:
            module (:class:`~computing.core.container.Container` or :class:`~computing.core.module.Module`): 待移除的子模块
            name (int or str): 待移除的子模块的ID或者名字
        """
        if module is not None:
            if module.get_id() in self._modules.keys():
                self._modules.pop(module.get_id())
                return
            else:
                print("Bandit Warning: This arm was already removed.")
                return
        if name is not None:
            if name in self._modules.keys():
                self._modules.pop(name)
                return
            else:
                print("Bandit Warning: This arm was already removed.")
                return

    def set_module_template(self, module_template):
        """
        设置子模块模板。

        Args:
            module_template (:class:`~computing.core.container.Container` or :class:`~computing.core.module.Module`): 待添加的子模块模板

        Returns:

        """
        if module_template is not None:
            if not (isinstance(module_template, Module) or isinstance(module_template, Container)):
                raise TypeError("cannot assign '{}' object to Module template object "
                                .format(type(module_template)))
            self._module_template = module_template
        else:
            print("Bandit Warning: There is a null template.")
            return

    def get_module_from_template(self):
        """
        根据子模块模板，获得实例。

        Returns:
            (:class:`~computing.core.container.Container` or :class:`~computing.core.module.Module`): 子模块实例。
        """
        return copy.deepcopy(self._module_template)

    def cuda(self):
        self.device = "cuda"

        for module in self._modules.values():
            if isinstance(module, Container):
                module.cuda()
            elif isinstance(module, Module):
                module.cuda()
                module.convert_device(device="cuda")

        for v in self._container_properties.values():
            if isinstance(v, torch.Tensor):
                v.cuda()

        if isinstance(self._module_template, Container):
            self._module_template.cuda()
        elif isinstance(self._module_template, Module):
            self._module_template.cuda()
            self._module_template.convert_device(device="cuda")
        return self

    def cpu(self):
        self.device = "cpu"

        for module in self._modules.values():
            if isinstance(module, Container):
                module.cpu()
            elif isinstance(module, Module):
                module.cpu()
                module.convert_device(device="cpu")

        for v in self._container_properties.values():
            if isinstance(v, torch.Tensor):
                v.cpu()

        if isinstance(self._module_template, Container):
            self._module_template.cpu()
        elif isinstance(self._module_template, Module):
            self._module_template.cpu()
            self._module_template.convert_device(device="cpu")
        return self

    def get_class_name(self):
        return self.__class__.__name__

    def get_class_file_name(self):
        file_name = str(os.path.basename(__file__))
        return file_name.split('.')[0]

    def get_model_tree_struct(self, nodes={}, key=None):
        d = {}

        if self._modules is not None:
            d['modules'] = []
            for k, v in self._modules.items():
                self._modules[k].get_model_tree_struct(nodes)
                d['modules'].append(self._modules[k].get_class_name())
        else:
            d['modules'] = None

        if self._module_template is not None:
            self._module_template.get_model_tree_struct(nodes)
            d['module_template'] = self._module_template.get_class_name()

        d['file'] = "containers/{}".format(self.get_class_file_name())
        d['class'] = self.get_class_name()
        k = key if key else self.get_class_name()
        nodes[k] = d
