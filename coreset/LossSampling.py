from . import CoreSet
from utils import validate

import torch


class LossSampling(CoreSet):
    @torch.no_grad()
    def __init__(self, args, model, tokenizer, device, dataset_train):
        model.eval()
        print("Begin to calculate loss...")
        _, loss = validate(model, dataset_train, tokenizer, device, output_loss=True)
        _, indices = torch.sort(loss, dim=0, descending=True)
        self.indices = indices.cpu().tolist()
        self.dataset_train = dataset_train
    
    def get_demo_indices(self, demo_num):
        demo_each_label = demo_num // self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        label_max = [demo_each_label for _ in range(self.dataset_train.class_num)]
        for index in range(demo_num - self.dataset_train.class_num * demo_each_label):
            label_max[index] += 1

        final_indices = []
        for index in self.indices:
            _, _, label = self.dataset_train.examples[index]
            if label_count[label] < label_max[label]:
                final_indices.append(index)     
                label_count[label] += 1

        return final_indices

