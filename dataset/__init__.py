from .BaseTask import BaseTask
from .BoolQ import BoolQ
from .CB import CB
from .COPA import COPA
from .MultiRC import MultiRC
from .WSC import WSC
from .SST2 import SST2
from .RTE import RTE
from .AGNews import AGNews
from .SST5 import SST5
from .Subj import Subj
from .TREC import TREC


dataset_dict = {
    'cb': CB,
    'copa': COPA,
    'sst2': SST2,
    'rte': RTE,
    'agnews': AGNews,
    'sst5': SST5,
    'subj': Subj,
    'trec': TREC,
    # 'wsc': WSC,
    # 'boolq': BoolQ,
    # 'multirc': MultiRC,
}

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    return dataset_dict[dataset](*args, **kwargs)


