from .BaseTask import BaseTask
from .BoolQ import BoolQ
from .CB import CB
from .COPA import COPA
from .MultiRC import MultiRC
from .WSC import WSC

import torch
from torch.utils.data import DataLoader

from utils import expand_past_key_value

dataset_dict = {
    'cb': CB,
    'copa': COPA,
    'wsc': WSC,
    'boolq': BoolQ,
    'multirc': MultiRC
}

def get_dataset(dataset, *args, **kwargs) -> BaseTask:
    return dataset_dict[dataset](*args, **kwargs)

@torch.no_grad()
def align_feature_demo(args, model, tokenizer, device, dataset_train, dataset_val):
    dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x)))
    metric = torch.zeros(len(dataset_train), len(dataset_train)).to(device)
    model.eval()
    print("Begin to calculate feature metrics...")
    for val_input_str, _, _ in dataloader_val:
        val_input_encoding = tokenizer(
            list(val_input_str),
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors='pt',
        ).to(device)
        all_past_hidden_states = []
        for index, (train_input_str, train_output_str, train_answer) in enumerate(dataset_train):
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

            for past_hidden_states in all_past_hidden_states:
                for past_index, (current, past) in enumerate(zip(val_output.hidden_states, past_hidden_states)):
                    metric[index][past_index] += (current - past).mul(val_input_encoding.attention_mask.unsqueeze(-1)).square().sum().item()

            all_past_hidden_states.append(val_output.hidden_states)

    metric = torch.sum(metric + metric.transpose(1, 0), dim=1)
    indices = metric.sort(dim=0).indices

    demo_each_label = args.demo_num / dataset_train.class_num
    label_count = [0 for _ in range(dataset_train.class_num)]
    final_indices = []
    for index in indices:
        _, _, label = dataset_train.examples[index]
        if label_count[label] < demo_each_label:
            final_indices.append(index)     
            label_count[label] += 1

    return dataset_train.get_demo_from_indices(final_indices)
