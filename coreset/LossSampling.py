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
        self.indices = indices.cpu().tolist()[:args.coreset_size]
        self.dataset_train = dataset_train


