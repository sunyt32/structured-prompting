from . import CoreSet
from utils import validation
from utils.functional import expand_past_key_value

import random

import torch
from torch.utils.data import DataLoader


class AlignFeature(CoreSet):
    def __init__(self, args, model, tokenizer, device, dataset_train, dataset_val, validation=False, dynamic=False):
        self.dataset_train = dataset_train
        self.dynamic = dynamic
        self.validation = validation
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if not dynamic:
            dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
            metric = torch.zeros(len(self.dataset_train)).to(device)
            model.eval()
            print("Begin to calculate feature metrics for training set size {}, val set size {}...".format(len(dataset_train), len(dataset_val)))
            for index, (val_input_str, _, _) in enumerate(dataloader_val):
                if index >= args.sample_num:
                    break
                
                metric += self.get_metric(model, device, tokenizer, dataset_train, val_input_str)
                
            values, indices = torch.sort(metric, dim=0)
            self.indices = self._get_demo_indices(args.coreset_size, indices.cpu().tolist())
            core_values = []
            values /= 1e10
            for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
                if index in self.indices:
                    core_values.append(value)

            print(torch.var(values), torch.var(torch.Tensor(core_values)))

    def _get_demo_indices(self, demo_num, indices):
        demo_each_label = demo_num // self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        label_max = [demo_each_label for _ in range(self.dataset_train.class_num)]
        for index in range(demo_num - self.dataset_train.class_num * demo_each_label):
            label_max[index] += 1

        final_indices = []
        for index in indices:
            _, _, label = self.dataset_train.examples[index]
            if label_count[label] < label_max[label]:
                final_indices.append(index)     
                label_count[label] += 1

        return final_indices
    
    @staticmethod
    def get_metric(model, device, tokenizer, dataset_train, val_input_str):
        metric = torch.zeros(len(dataset_train)).to(device)
        val_input_encoding = tokenizer(
            list(val_input_str),
            truncation=True,
            padding=True,
            return_tensors='pt',
        ).to(device)
        all_past_hidden_states = []
        for train_input_str, train_output_str, train_answer in dataset_train:
            train_input_encoding = tokenizer(
                [train_input_str + train_output_str[train_answer]],
                truncation=True,
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

            all_past_hidden_states.append(val_output.hidden_states[-1].half().to(device)[torch.where(val_input_encoding.attention_mask)].flatten().cpu())

        sum_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(device) # dim
        norm_square = torch.empty(0).to(device)
        for hidden_states in all_past_hidden_states:
            hidden_states = hidden_states.to(device)
            sum_hidden_states += hidden_states
            norm_square = torch.cat((norm_square, hidden_states.square().sum().unsqueeze(0)), dim=0)

        metric += norm_square * len(dataset_train) + torch.sum(norm_square)
        for index, hidden_states in enumerate(all_past_hidden_states):
            hidden_states = hidden_states.to(device)
            metric[index] -= 2 * torch.sum(sum_hidden_states * hidden_states)

        return metric

    def get_dynamic_indices(self, val_input_str, demo_num):
        metric = self.get_metric(self.model, self.device, self.tokenizer, self.dataset_train, val_input_str)
        _, indices = torch.sort(metric, dim=0)
        return self.get_demo_indices(demo_num, indices.cpu().tolist())
    
    def get_demo_indices(self, demo_num):
        if self.validation:
            acc_max = 0
            for _ in range(10):
                random.shuffle(self.indices)
                indices = self._get_demo_indices(demo_num, self.indices)
                self.dataset_train.demo = self.dataset_train.get_demo_from_indices(indices)
                acc = validation(self.model, self.dataset_train, self.tokenizer, self.device)
                if acc > acc_max:
                    acc_max = acc
                    best_indices = indices

            return best_indices
        else:
            random.shuffle(self.indices)
            indices = self._get_demo_indices(demo_num, self.indices)
            return indices
            # final_indices = [] # reranking
            # for index in self.indices:
            #     if index in indices:
            #         final_indices.append(index)

            # final_indices.reverse()
            # return final_indices

