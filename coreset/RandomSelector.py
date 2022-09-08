from . import CoreSet
from utils import validate

import random

class RandomSelector(CoreSet):
    def __init__(self, dataset_train, model=None, tokenizer=None, device=None, val=False):
        self.dataset_train = dataset_train
        self.indices = list(range(len(dataset_train)))
        self.val = val
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_demo_indices(self, demo_num):
        return random.sample(self.indices, demo_num)
        # random.shuffle(self.indices)
        # demo_each_label = demo_num / self.dataset_train.class_num
        # label_count = [0 for _ in range(self.dataset_train.class_num)]
        # indices = []
        # for index in self.indices:
        #     _, _, label = self.dataset_train.examples[index]
        #     if label_count[label] < demo_each_label:
        #         indices.append(index)     
        #         label_count[label] += 1

        # return indices

    def get_demo_indices(self, demo_num):
        if self.val:
            acc_max = 0
            for _ in range(10):
                indices = self._get_demo_indices(demo_num)
                self.dataset_train.demo = self.dataset_train.get_demo_from_indices(indices)
                acc = validate(self.model, self.dataset_train, self.tokenizer, self.device)
                if acc > acc_max:
                    acc_max = acc
                    best_indices = indices

            return best_indices
        else:
            return self._get_demo_indices(demo_num)
        
        