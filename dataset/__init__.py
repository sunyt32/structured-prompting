from .BaseTask import BaseTask
from .BoolQ import BoolQ
from .CB import CB
from .COPA import COPA
from .MultiRC import MultiRC
from .WSC import WSC
from .SST2 import SST2
from .RTE import RTE

dataset_dict = {
    'cb': CB,
    'copa': COPA,
    'wsc': WSC,
    'boolq': BoolQ,
    'multirc': MultiRC,
    'sst2': SST2,
    'rte': RTE
}

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    return dataset_dict[dataset](*args, **kwargs)


