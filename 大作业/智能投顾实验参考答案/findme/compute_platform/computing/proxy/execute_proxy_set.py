from computing.proxy.execute_proxy import ExecuteProxy
from computing.proxy.model_proxy import ModelProxy
from computing.proxy.metric_proxy import MetricProxy



class ExecuteProxySet:
    def __init__(self, **execute_set_params):
        self.execute_set_params = execute_set_params
        self.instances = {}

    def add_from_instance(self, execute_instance: ExecuteProxy):
        self.instances[execute_instance.get_instance_name()] = execute_instance
        self.instances[execute_instance.get_instance_name()].set_execute_set_params(**self.execute_set_params)

    def add_from_template(self, model=None, metric=None, data=None,
                          data_params={}, model_params={}, metric_params={}, execute_params={}):
        # 新增数据集代理
        if data is not None:
            data_proxy = data
        elif data is None and len(data_params) > 0:
            exec("from computing.proxy.data_proxy import {}".format(data_params['data_proxy_class']))
            data_proxy = eval(data_params['data_proxy_class'] + "(**data_params)")
            eval(data_params['data_proxy_class'] + "(**data_params)")
        else:
            raise FileNotFoundError("No data information is provided")

        # 新增模型代理
        if model is not None:
            models_proxy = [model]
        elif model is None and len(model_params) > 0:
            models_proxy = self.config_models_from_json(model_params=model_params['model_tree_params'],
                                                        model_tree_struct=model_params['model_tree_struct'], )
        else:
            raise FileNotFoundError("No model information is provided")

        # 新增评价代理
        if metric is not None:
            metric_proxy = metric
        elif metric is None and len(metric_params) > 0:
            metric_proxy = eval("MetricProxy(**metric_params)")
        else:
            metric_proxy = MetricProxy()

        # 添加Execute实例
        for m in models_proxy:
            exec("from computing.proxy.execute_proxy import {}".format(execute_params['execute_proxy_class']))
            execute_proxy = eval(
                execute_params['execute_proxy_class'] + "(model=model, data=data, metric=metric, **execute_params)")
            self.add_from_instance(execute_proxy)

        return

    def parameter_config(self, params_dict_list, paras, paras_exec=""):
        '''
        输入为原始参数文件读取出来的参数params，有可能一个参数名下有多个参数值，用列表来存储。
        针对这种情况，对params进行拆分，把每个参数不包含一对多关系的参数字典存储到params_dict_list中。
        Args:
            params_dict_list:
            paras:
            paras_exec:

        Returns:

        '''
        if not paras:
            dict_str = "{" + paras_exec[:-1] + "}"
            d = eval(dict_str)
            params_dict_list.append(d)
            return
        para_name = list(paras.keys())[0]
        paras_new = paras.copy()
        para_values = paras_new.pop(para_name)
        if isinstance(para_values, list) and para_name != "layers":
            for para in para_values:
                paras_exec_new = paras_exec + '"' + para_name + '"' + ":" + str(para) + ","
                self.parameter_config(params_dict_list, paras_new, paras_exec_new)
        else:
            para = para_values
            paras_exec_new = paras_exec + '"' + para_name + '"' + ":" + str(para) + ","
            self.parameter_config(params_dict_list, paras_new, paras_exec_new)
        return

    def config_models_from_json(self, model_params=None, model_tree_struct=None):
        params_dict_list = []
        models = []
        # self.parameter_config(params_dict_list, json.load(open(model_params_path, 'r', encoding='UTF-8')))
        self.parameter_config(params_dict_list, model_params)
        for params in params_dict_list:
            model = ModelProxy(model_tree_struct=model_tree_struct, model_params=params)
            models.append(model)
        return models

    def execute(self, mode="OnlineDecision", reset_metric=True, **kwargs):
        """
        Args:
            mode:
            # PreTraining：预训练
            # Training：正式模型训练
            # OnlineDecision：在线决策与更新
            # Decision:
            # Updating
            # Validation
        """
        results = None
        for instance_name, instance in self.instances.items():
            results = instance(mode, reset_metric=reset_metric, **kwargs)
        return results

    def compare(self):
        raise NotImplementedError

    def visuslize(self):
        raise NotImplementedError
