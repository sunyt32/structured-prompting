from . import CoreSet
from utils import expand_past_key_value

import random

import torch
from torch.utils.data import DataLoader


class AlignFeature(CoreSet):
    @torch.no_grad()
    def __init__(self, args, model, tokenizer, device, dataset_train, dataset_val):
        self.dataset_train = dataset_train
        dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
        metric = torch.zeros(len(self.dataset_train)).to(device)
        model.eval()
        print("Begin to calculate feature metrics for training set size {}, val set size {}...".format(len(self.dataset_train), len(self.dataset_val)))
        for index, (val_input_str, _, _) in enumerate(dataloader_val):
            if index >= args.sample_num:
                break
            
            val_input_encoding = tokenizer(
                list(val_input_str),
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(device)
            all_past_hidden_states = []
            for index, (train_input_str, train_output_str, train_answer) in enumerate(self.dataset_train):
                train_input_encoding = tokenizer(
                    [train_input_str + train_output_str[train_answer]],
                    truncation=True,
                    max_length=args.max_length,
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

                all_past_hidden_states.append(val_output.hidden_states[-1].to(device)[torch.where(val_input_encoding.attention_mask)].cpu())
                # hidden_states = torch.empty(0)
                # for current in val_output.hidden_states:
                #     hidden_states = torch.cat((hidden_states, current.flatten().cpu()))

                # all_past_hidden_states.append(hidden_states)

            sum_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(device) # length * dim
            norm_square = torch.empty(0, sum_hidden_states.shape[0]).to(device) # size * length
            for hidden_states in all_past_hidden_states:
                hidden_states = hidden_states.to(device)
                sum_hidden_states += hidden_states
                norm_square = torch.cat((norm_square, hidden_states.square().sum(dim=-1).unsqueeze(0)), dim=0)

            metric += torch.sum(norm_square * len(self.dataset_train) + torch.sum(norm_square, dim=0, keepdim=True), dim=1)
            for index, hidden_states in enumerate(all_past_hidden_states):
                hidden_states = hidden_states.to(device)
                metric[index] -= 2 * torch.sum(sum_hidden_states * hidden_states)

        _, indices = torch.sort(metric, dim=0)
        self.indices = indices.cpu().tolist()
        # self.indices = self.get_demo_indices(32, False)

    def get_demo_indices(self, demo_num, shuffle=False):
        if shuffle:
            random.shuffle(self.indices)

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
