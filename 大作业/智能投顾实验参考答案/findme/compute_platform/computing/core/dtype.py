class LearningVariable(object):
    """
    参数类，用于装载 学习量 类型的参数。

    """

    def __init__(self, data=None):
        self.data = data


class HyperParameter(object):
    """
    参数类，用于装载 超参 类型的参数。

    """

    def __init__(self, data=None):
        self.data = data


class BufferParameter(object):
    """
    参数类，用于装载 buffer信息 类型的参数。

    """

    def __init__(self, data=None):
        self.data = data


class ContainerProperty(object):
    """
    参数类，用于装载 容器属性 的参数。

    """

    def __init__(self, data=None):
        self.data = data

