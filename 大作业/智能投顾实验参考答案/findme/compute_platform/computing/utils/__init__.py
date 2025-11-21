import torch as torch

from computing.utils.data_struct import SeriesDictData

# global constants
BATCH_SIZE = 100000
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

EVAL_DISPLAY_BATCH = 1
