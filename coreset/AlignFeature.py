from . import CoreSet
from utils import validate
from utils.functional import expand_past_key_value

import random

import torch
from torch.utils.data import DataLoader


class AlignFeature(CoreSet):
    def __init__(self, args, model, tokenizer, device, dataset_train, dataset_val, val=False, dynamic=False):
        self.dataset_train = dataset_train
        self.dynamic = dynamic
        self.val = val
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if not dynamic:
            dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
            # _, loss = validate(model, dataset_train, tokenizer, device, output_loss=True)
            metric = torch.zeros(len(self.dataset_train)).to(device)
            model.eval()
            print("Begin to calculate feature metrics for training set size {}, val set size {}...".format(len(dataset_train), len(dataset_val)))
            for index, (val_input_str, _, _) in enumerate(dataloader_val):
                if index >= args.sample_num:
                    break
                
                metric += self.get_metric(model, device, tokenizer, dataset_train, val_input_str)
                
            _, align_indices = torch.sort(metric, dim=0)
            self.indices = align_indices.cpu().tolist()[:args.coreset_size]
    
    @staticmethod
    def get_metric(model, device, tokenizer, dataset_train, val_input_str, weight_vector = None):
        metric = torch.zeros(len(dataset_train)).to(device)
        if weight_vector is None:
            weight_vector = torch.ones(len(dataset_train)).to(device)

        val_input_encoding = tokenizer(
            list(val_input_str),
            padding=True,
            return_tensors='pt',
        ).to(device)
        all_past_hidden_states = []
        for train_input_str, train_output_str, train_answer in dataset_train:
            train_input_encoding = tokenizer(
                [train_input_str + train_output_str[train_answer] + '\n'],
                return_tensors='pt',
            ).to(device)
            with torch.no_grad():
                past_key_values = model(
                    input_ids=train_input_encoding.input_ids, 
                    use_cache=True
                    ).past_key_values
                val_output = model(
                    input_ids=val_input_encoding.input_ids,
                    past_key_values=expand_past_key_value(past_key_values, len(val_input_str)), 
                    output_hidden_states=True
                    )

            all_past_hidden_states.append(val_output.hidden_states[-1].div(1e5).half().to(device)[torch.where(val_input_encoding.attention_mask)].flatten().cpu())

        sum_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(device) # dim
        norm_square = torch.empty(0).to(device)
        for weight, hidden_states in zip(weight_vector, all_past_hidden_states):
            hidden_states = hidden_states.to(device)
            sum_hidden_states += weight * hidden_states
            norm_square = torch.cat((norm_square, hidden_states.square().sum().unsqueeze(0)), dim=0)

        sum_hidden_states /= torch.sum(weight)
        metric += norm_square
        for index, hidden_states in enumerate(all_past_hidden_states):
            hidden_states = hidden_states.to(device)
            metric[index] -= 2 * torch.sum(sum_hidden_states * hidden_states)

        return metric
    
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

