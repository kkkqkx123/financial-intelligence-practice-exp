from computing.proxy import ALL_MODE_TYPE


class MetricProxy:
    def __init__(self, metrics_class={}, metrics_name={},
                 all_modes=ALL_MODE_TYPE,
                 metrics_disable_mode={}, **metric_params):

        """
        根据 result_file_dir 确定存放结果的目录，根据 result_type 确定存放结果的子目录.
        result_types 可以存放 "train", "validate", "online_decide"
        如果result_types中包含空字符串以外的字符，则会自动新建文件夹，并自动对metric按照名字区分，ExecuteProxy也可以按照名字来选择更新哪种type的矩阵
        Args:
            metrics_class: 如 {'PortfolioEval': PortfolioEval},
            metrics_name: 如 {'PortfolioEval': ["cm_reward"]}
            all_modes: 如pre_train, train, validate, online_decide
            metrics_disable_mode: 如 {'PortfolioEval': {"pre_train", "train", "validate"}}
            model_name:
            dataset_name:
            result_file_dir:
        """
        self.metrics_class = metrics_class
        self.metrics_name = metrics_name
        self.all_modes = set(all_modes)
        self.metrics_available_modes = {}
        self.metrics = {}
        self.metric_params = metric_params
        self.file_dir = None

        # 设置 self.metrics_available_modes，表示每个metric支持哪些mode下的更新
        for name in metrics_class.keys():
            self.metrics_available_modes[name] = self.all_modes - metrics_disable_mode[
                name] if name in metrics_disable_mode else self.all_modes

    def get_all_modes(self):
        return self.all_modes

    def set_metric(self, file_dir, mode, method=''):
        # 初始化metric中的子文件
        """
        根据结果的保存路径，新建文件夹，新建多个metric
        Args:
            file_dir:
            mode:
            method:

        Returns:

        """
        self.file_dir = file_dir
        for k, metric in self.metrics_class.items():
            if mode in self.metrics_available_modes[k]:
                if file_dir:
                    self.metrics[k + "_" + mode] = metric(file_dir=file_dir + '_' + str(k), method=method,
                                                          **self.metric_params)
                else:
                    self.metrics[k + "_" + mode] = metric(file_dir=None, method=method, **self.metric_params)
        return

    def get_result_file_dir(self):
        return self.file_dir

    def update(self, results, input_data, mode="", **kwargs):
        """
        根据 type来决定更新哪些参数
        """
        for k, metric in self.metrics_class.items():
            if mode in self.metrics_available_modes[k]:
                self.metrics[k + "_" + mode](results, input_data, **kwargs)

    def change_file_dir(self, file_dir, mode=''):
        for k, metric in self.metrics_class.items():
            if mode in self.metrics_available_modes[k]:
                self.metrics[k + "_" + mode].change_file_dir(file_dir=file_dir)

    def after_eval(self, mode='', reset_metric=True):
        for metric_name, save_metric_list in self.metrics_name.items():
            if mode in self.metrics_available_modes[metric_name]:
                self.metrics[metric_name + "_" + mode].after_eval(reset_metric=reset_metric)

    def save(self, mode='', reset_metric=True):
        for metric_name, save_metric_list in self.metrics_name.items():
            if mode in self.metrics_available_modes[metric_name]:
                self.metrics[metric_name + "_" + mode].save(save_metric_list)

