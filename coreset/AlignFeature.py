from . import CoreSet
from utils import expand_past_key_value

import torch
from torch.utils.data import DataLoader


class AlignFeature(CoreSet):
    def __init__(self, args, model, tokenizer, device, dataset_train, dataset_val):
        self.sample_num = args.sample_num
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

    
    @torch.no_grad()
    def get_demo_indices(self, demo_num):
        dataloader_val = DataLoader(self.dataset_val, self.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
        metric = torch.zeros(len(self.dataset_train)).to(self.device)
        self.model.eval()
        print("Begin to calculate feature metrics for size {}...".format(len(self.dataset_train)))
        for index, (val_input_str, _, _) in enumerate(dataloader_val):
            if index >= self.sample_num:
                break
            
            val_input_encoding = self.tokenizer(
                list(val_input_str),
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self.device)
            all_past_hidden_states = []
            for index, (train_input_str, train_output_str, train_answer) in enumerate(self.dataset_train):
                train_input_encoding = self.tokenizer(
                    [train_input_str + train_output_str[train_answer]],
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                ).to(self.device)
                past_key_values = self.model(
                    input_ids=train_input_encoding.input_ids, 
                    use_cache=True
                    ).past_key_values
                val_output = self.model(
                    input_ids=val_input_encoding.input_ids,
                    past_key_values=expand_past_key_value(past_key_values, len(val_input_str)), 
                    output_hidden_states=True
                    )

                all_past_hidden_states.append(val_output.hidden_states[-1].mul(val_input_encoding.attention_mask.unsqueeze(-1)).cpu())
                # hidden_states = torch.empty(0)
                # for current in val_output.hidden_states:
                #     hidden_states = torch.cat((hidden_states, current.flatten().cpu()))

                # all_past_hidden_states.append(hidden_states)

            sum_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(self.device)
            norm_square = torch.empty(0).to(self.device)
            for hidden_states in all_past_hidden_states:
                hidden_states = hidden_states.to(self.device)
                sum_hidden_states += hidden_states
                norm_square = torch.cat((norm_square, hidden_states.square().sum().unsqueeze(0)))

            metric += norm_square * len(self.dataset_train) + torch.sum(norm_square)
            for index, hidden_states in enumerate(all_past_hidden_states):
                hidden_states = hidden_states.to(self.device)
                metric[index] -= 2 * torch.sum(sum_hidden_states * hidden_states)

        indices = metric.sort(dim=0).indices
        demo_each_label = demo_num / self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        final_indices = []
        for index in indices:
            _, _, label = self.dataset_train.examples[index]
            if label_count[label] < demo_each_label:
                final_indices.append(index)     
                label_count[label] += 1

        return final_indices
