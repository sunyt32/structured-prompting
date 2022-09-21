from . import CoreSet
from utils import validate
from utils.kmeans import KMeans

import random

import torch


class AlignEmbedding(CoreSet):
    def __init__(self, args, model, tokenizer, device, dataset_train, val=False):
        self.dataset_train = dataset_train
        self.val = val
        self.device = device
        # _, loss = validate(model, dataset_train, tokenizer, device, output_loss=True)
        all_embeddings = None
        for train_input_str, train_output_str, train_answer in dataset_train:
            train_input_encoding = tokenizer(
                [template.replace("{sentence}", train_input_str + train_output_str[train_answer] + '\n') for template in self.templates()],
                truncation=True,
                padding=True,
                return_tensors='pt',
            ).to(device)
            with torch.no_grad():
                hidden_states = model(
                    input_ids=train_input_encoding.input_ids,
                    output_hidden_states=True
                    ).hidden_states

            hidden_states = hidden_states[-1].to(device)
            hidden_states = hidden_states[torch.arange(len(self.templates())).to(device), torch.sum(train_input_encoding.attention_mask, dim=1) - 1].reshape(1, -1)
            if all_embeddings is None:
                all_embeddings = hidden_states
            else:
                all_embeddings = torch.cat((all_embeddings, hidden_states), dim=0)

        self.all_embeddings = all_embeddings
        metric = len(all_embeddings) * all_embeddings.square().mean(dim=1) - 2 * all_embeddings.mul(all_embeddings.sum(dim=0)).mean(dim=1)
        _, indices = torch.sort(metric, dim=0)
        self.indices = indices.cpu().tolist()[:args.coreset_size]

    def templates(self):
        return [
            "{sentence} is ",
            "{sentence} means ",
            "This sentence : \" {sentence} \" means ", 
            "This {sentence} should be "
        ]
    
    def get_demo_indices(self, demo_num):
        if self.val:
            acc_max = 0
            for _ in range(10):
                indices = random.sample(self.indices, demo_num)
                self.dataset_train.demo = self.dataset_train.get_demo_from_indices(indices)
                acc = validate(self.model, self.dataset_train, self.tokenizer, self.device)
                if acc > acc_max:
                    acc_max = acc
                    best_indices = indices

            return best_indices
        else:
            return super().get_demo_indices(demo_num)

