from . import CoreSet

import random

class RandomSelector(CoreSet):
    def __init__(self, dataset_train):
        self.dataset_train = dataset_train

    def get_demo_indices(self, demo_num):
        random.shuffle(self.dataset_train.examples)
        demo_each_label = demo_num / self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        indices = []
        for index, _, _, label in enumerate(self.dataset_train.examples):
            if label_count[label] < demo_each_label:
                indices.append(index)     
                label_count[label] += 1

        return indices
        