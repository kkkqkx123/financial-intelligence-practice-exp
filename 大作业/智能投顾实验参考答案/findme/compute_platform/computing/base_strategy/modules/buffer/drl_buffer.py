import torch
import numpy as np

from computing.core.dtype import BufferParameter
from computing.core.module import Module


class Buffer(Module):
    """
    Buffer基类，需要指定buffer_size和首先buffer更新函数
    """
    def __init__(self, buffer_size=128, module_id=-1, **kwargs):
        """
        Args:
            buffer_size: buffer支持的最大容量
            current_load_num: 记录buffer目前已经装载的数量
            is_full: buffer是否装满
        """
        super(Buffer, self).__init__(module_id=module_id)
        self.buffer_size = BufferParameter(buffer_size)
        self.current_load_num = BufferParameter(0)
        self.is_full = BufferParameter(False)

    def update(self, *args, **kwargs):
        raise NotImplementedError("A Buffer module must have a update function")


class DRLBuffer(Buffer):
    """
    DRLBuffer class, which is a data structure of buffer for batch training
     "X": input state data [batch, feature, stock];
     "y": future relative price [batch, norm_feature, coin];
     "last_w:" a tensor with shape [batch_size, assets];
    """

    def __init__(self, buffer_size=128, sample_size=64, module_id=-1, transaction_cost=0.025, **kwargs):
        super(DRLBuffer, self).__init__(buffer_size=buffer_size, module_id=module_id)
        self.sample_size = BufferParameter(sample_size)

        self.buffer_state = BufferParameter(None)
        self.buffer_reward = BufferParameter(None)
        self.buffer_action = BufferParameter(None)
        self.sample_state = BufferParameter(None)
        self.sample_reward = BufferParameter(None)
        self.sample_action = BufferParameter(None)
        self.transaction_cost = BufferParameter(torch.tensor(transaction_cost))

        self.register_decide_hooks(["sample_state", "sample_reward", "sample_action",
                                    "buffer_state", "buffer_reward", "buffer_action",
                                    "is_full", "transaction_cost"])

    def update(self, state=None, action=None, reward=None, release=False, *args, **kwargs):
        if release:
            self._release()
        else:
            self.current_load_num = min(self.current_load_num+1, self.buffer_size)
            self.buffer_state = self._update_single_data(self.buffer_state, state)
            self.buffer_action = self._update_single_data(self.buffer_action, action)
            self.buffer_reward = self._update_single_data(self.buffer_reward, reward)

            sample_index = self._sample()
            if sample_index is not None:
                self.sample_state = self._sample_single_data(self.buffer_state, sample_index)
                self.sample_reward = self._sample_single_data(self.buffer_reward, sample_index)
                self.sample_action = self._sample_single_data(self.buffer_action, sample_index)

    # ================================== update function ==================================
    def _update_single_data(self, buffer_x, x):
        if buffer_x is None:
            return x

        data_type = type(x)
        if data_type is torch.Tensor:
            start_index = 1 if buffer_x.shape[0] == self.buffer_size else 0
            buffer_x = torch.cat([buffer_x[start_index:], x])
        elif data_type == list:
            if len(buffer_x) == self.buffer_size:
                buffer_x.pop(0)
            buffer_x.append(x[0])
        elif data_type == dict:
            for key, value in x.items():
                buffer_x[key] = self._update_single_data(buffer_x[key], value)
        else:
            raise IOError("Please check element type, which only supports tensor, list and dict.")

        return buffer_x

    def _release(self):
        print("the buffer has been released.")
        self.buffer_state = BufferParameter(None)
        self.buffer_reward = BufferParameter(None)
        self.buffer_action = BufferParameter(None)
        self.sample_state = BufferParameter(None)
        self.sample_reward = BufferParameter(None)
        self.sample_action = BufferParameter(None)
        self.is_full = BufferParameter(False)
        self.current_load_num = BufferParameter(0)

    # ================================== sample function ==================================
    def _get_sample_indices(self):
        sample_bias = 0.1
        ran = np.random.geometric(sample_bias)
        while ran > self.current_load_num - self.sample_size:
            ran = np.random.geometric(sample_bias)
        sample_indices = range(ran, ran + self.sample_size)
        return sample_indices

    def _sample(self):
        if self.current_load_num < self.sample_size:
            return None
        self.is_full = True

        if self.current_load_num - self.sample_size == 0:
            return range(0, self.sample_size)
        sample_indices = self._get_sample_indices()
        return sample_indices

    def _sample_single_data(self, buffer_x, sample_index):
        if buffer_x is None:
            return None

        data_type = type(buffer_x)
        if data_type is torch.Tensor:
            sample_x = buffer_x[sample_index]
        elif data_type == list:
            sample_x = [buffer_x[index] for index in sample_index]
        elif data_type == dict:
            sample_x = {}
            for key, value in buffer_x.items():
                sample_x[key] = self._sample_single_data(buffer_x[key], sample_index)
        else:
            raise IOError("Please check element type, which only supports tensor, list and dict.")
        return sample_x

    # ======================================================================================