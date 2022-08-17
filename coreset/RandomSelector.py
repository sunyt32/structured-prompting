from . import CoreSet

import random

class RandomSelector(CoreSet):
    def __init__(self, dataset_train):
        self.dataset_train = dataset_train

    def get_demo_indices(self, demo_num):
        all_indices = list(range(len(self.dataset_train)))
        random.shuffle(all_indices)
        demo_each_label = demo_num / self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        indices = []
        for index in all_indices:
            _, _, label = self.dataset_train.examples[index]
            if label_count[label] < demo_each_label:
                indices.append(index)     
                label_count[label] += 1

        return indices
        