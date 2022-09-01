from . import CoreSet
from utils.functional import expand_past_key_value

import random

import torch


class BalanceAlignFeature(CoreSet):
    @torch.no_grad()
    def __init__(self, model, tokenizer, device, dataset_train, dynamic=True):
        self.dataset_train = dataset_train
        self.dynamic = dynamic
        if dynamic:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        else:
            raise NotImplementedError()

    @staticmethod
    def get_hidden_states(model, device, tokenizer, dataset_train, val_input_str):
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

        return all_past_hidden_states

    def get_dynamic_indices(self, val_input_str, demo_num):
        if not self.dynamic:
            raise NotImplementedError()

        all_past_hidden_states = self.get_hidden_states(self.model, self.device, self.tokenizer, self.dataset_train, val_input_str)
        mean_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(self.device)
        for hidden_states in all_past_hidden_states:
            hidden_states = hidden_states.to(self.device)
            mean_hidden_states += hidden_states

        mean_hidden_states /= len(all_past_hidden_states)

        current_sum_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(self.device)
        current_indices = []
        
        demo_each_label = demo_num // self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        label_max = [demo_each_label for _ in range(self.dataset_train.class_num)]
        for index in range(demo_num - self.dataset_train.class_num * demo_each_label):
            label_max[index] += 1

        for _ in range(demo_num):
            min_bias = 1e20
            min_index = None
            min_hidden_states = None
            min_label = None
            for index, hidden_states in enumerate(all_past_hidden_states):
                _, _, label = self.dataset_train.examples[index]
                if index in current_indices or label_count[label] >= label_max[label]:
                    continue

                hidden_states = hidden_states.to(self.device)
                bias = ((current_sum_hidden_states + hidden_states) / (len(current_indices) + 1) - mean_hidden_states).abs().sum().item()
                if bias < min_bias:
                    min_bias = bias
                    min_index = index
                    min_hidden_states = hidden_states
                    min_label = label

            current_indices.append(min_index)
            current_sum_hidden_states += min_hidden_states
            label_count[min_label] += 1

        return current_indices



    def get_demo_indices(self):
        raise NotImplementedError()
