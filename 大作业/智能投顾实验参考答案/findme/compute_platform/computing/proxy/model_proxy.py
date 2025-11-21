import torch


class ModelProxy:
    def __init__(self, model_tree_struct=None, model_params=None, model=None, device="cpu"):
        # self.add_models(model_file_dir, model_class, params)
        self.__model = model if model else self.gen_model_tree(model_tree_struct, "0", model_params)
        self.model_tree_struct = model_tree_struct if model_tree_struct else self.get_model_tree_struct_from_model()
        self.model_params = model_params if model_params else self.get_model_params_dict()
        self.__model_name = self.model_tree_struct["0"]['class']

    def set_device(self, device):
        self.__model.set_device(device)

    def get_model_params_dict(self):
        model_params = self.__model.get_hyper_parameters(name_space=True)
        for k, v in model_params.items():
            if isinstance(v, torch.Tensor):
                model_params[k] = model_params[k].cpu().numpy().tolist()
        return model_params

    def get_model_tree_struct_from_model(self):
        model_tree_struct = {}
        self.__model.get_model_tree_struct(nodes=model_tree_struct, key="0")
        return model_tree_struct

    def gen_model_tree(self, model_tree_struct, node_id, params):
        """
        Args:
            model_tree_struct.json 样例如下：
                {
                  "0": {
                    "file": "containers/portfolio_cand",
                    "class": "PortfolioCVaR",
                    "sub_modules_template": "oco_news_template",
                    "module": "1",
                    "sub_modules": null
                  },
                  "1": {
                    "file": "modules/cvar_news",
                    "class": "CVaRNews",
                    "sub_modules_template": null,
                    "module": null,
                    "sub_modules": null
                  },
                  "oco_news_template": {
                    "file": "modules/oco",
                    "class": "OCONews",
                    "sub_modules_template": null,
                    "module": null,
                    "sub_modules": null
                  }
                }
            params.json示例如下:
                {
                  "k":10,
                  "PortfolioCVaR.CVaRNews.lamda": 0.5,
                  "PortfolioCVaR.OCONews.lamda": 0.5
                }
        Returns:

        """
        # 获取该node的信息
        module_type, module_file = model_tree_struct[node_id]["file"].split('/')
        module_class_name = model_tree_struct[node_id]["class"]
        module_node_id = model_tree_struct[node_id]["module"]
        sub_modules_nodes_id = model_tree_struct[node_id]["sub_modules"]
        sub_module_template_id = model_tree_struct[node_id]["sub_modules_template"]
        exec("from computing.base_strategy.{}.{} import {}".format(module_type, module_file, module_class_name))

        # 按照name space拆分params
        new_params = {}
        for param_name, param_value in params.items():
            if module_class_name in param_name:
                new_params[param_name[param_name.find(module_class_name) + len(module_class_name) + 1:]] = param_value
            else:
                new_params[param_name] = param_value

        # 根据node信息和params构建node
        params_str = ""
        for param_name, param_value in new_params.items():
            params_str += "{}={},".format(param_name, str(param_value))

        # 如果为叶子node，直接返回node
        if model_tree_struct[node_id]["module"] is None and model_tree_struct[node_id]["sub_modules"] is None:
            node = eval("{}({})".format(module_class_name, params_str))
            return node
        # 如果为非叶子node，继续递归构建node的module和sub_modules再返回
        else:
            module = sub_modules = sub_module_template = None
            if sub_module_template_id:
                params_str += "sub_module_template={{'file':'{}', 'name':'{}'}}".format(
                    model_tree_struct[sub_module_template_id]["file"],
                    model_tree_struct[sub_module_template_id]["class"]
                )
                # node = eval("{}({})".format(module_class_name, params_str))
                sub_module_template = self.gen_model_tree(model_tree_struct, sub_module_template_id, new_params)
                # node.set_sub_module_template(self.gen_model_tree(model_tree_struct, sub_module_template_id, new_params))
            if module_node_id:
                module = self.gen_model_tree(model_tree_struct, module_node_id, new_params)
                # node.set_module(self.gen_model_tree(model_tree_struct, module_node_id, new_params))
            if sub_modules_nodes_id:
                sub_modules = {}
                for id, node_id in enumerate(sub_modules_nodes_id):
                    sub_modules[node_id] = self.gen_model_tree(model_tree_struct, node_id, new_params)
                    # node.add_sub_module(self.gen_model_tree(model_tree_struct, node_id, new_params))
            node = eval("{}(module=module, sub_modules=sub_modules, sub_module_template=sub_module_template, "
                        "{})".format(module_class_name, params_str))
            return node

    def get_models(self):
        return self.__model

    def get_model_name(self):
        return str(self.__model_name)

    def save(self, path):
        torch.save(self.__model, path)

    def load(self, path):
        self.__model = torch.load(path)

    def get_params(self):
        return self.model_params

    def get_model_tree_struct(self):
        return self.model_tree_struct

    def __call__(self, mode, data, result=None, **kwargs):
        return self.__model(data=data, mode=mode, result=result, **kwargs)
