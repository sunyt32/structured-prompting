from . import CoreSet
from utils import expand_past_key_value

import random

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class LossPartition(CoreSet):
    @torch.no_grad()
    def __init__(self, args, model, tokenizer, device, dataset_train):
        model.eval()
        dataloader_train = DataLoader(dataset_train, 1, shuffle=False, collate_fn=lambda x: list(zip(*x)))
        criterion = CrossEntropyLoss(reduction='none')
        print("Begin to calculate loss...")
        loss = torch.empty(0).to(device)
        for train_input_str, train_output_str, train_answer in dataloader_train:
            train_input_encoding = tokenizer(
                list(train_input_str),
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(device)
            logits = model(
                input_ids=train_input_encoding.input_ids
                ).logits
            answer_encoding = tokenizer(
                train_output_str[0],
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(device)
            answer = torch.LongTensor(train_answer).to(device)
            answer_shape = answer_encoding.input_ids.shape

            output = model(
                input_ids=train_input_encoding.input_ids, 
                use_cache=True
                )
            input_logits = output.logits

            if answer_shape[1] > 1: # multi-choice
                answer_logits = model(
                    input_ids=answer_encoding.input_ids,
                    past_key_values=expand_past_key_value(output.past_key_values, answer_shape[0]), 
                    ).logits
                logits = torch.cat((input_logits[0, -1:].repeat(answer_shape[0], 1, 1), answer_logits[:, :-1]), dim=1).log_softmax(dim=-1)
            else:
                logits = input_logits[0, -1:].repeat(answer_shape[0], 1, 1).log_softmax(dim=-1)

            # select answer
            logits = logits.view(answer_shape[0] * answer_shape[1], -1)[torch.arange(answer_shape[0] * answer_shape[1]).to(device), answer_encoding.input_ids.flatten()].view(answer_shape)
            logits = logits.mul(answer_encoding.attention_mask).sum(dim=1).unsqueeze(0)
            loss = torch.cat((loss, criterion(logits, answer)))
            
        indices = loss.sort(dim=0).indices
        self.indices = indices.cpu()
        self.dataset_train = dataset_train
    
    def get_demo_indices(self, demo_num):
        demo_each_label = demo_num // self.dataset_train.class_num
        label_class = torch.arange(self.dataset_train.class_num).repeat(demo_each_label)
        while True:
            random.shuffle(label_class)
            final_indices = []
            for sub_indices, expect_label in zip(self.indices.chunk(demo_num), label_class):
                for index in sub_indices:
                    _, _, label = self.dataset_train.examples[index]
                    if label == expect_label.item():
                        final_indices.append(index)
                        break

            if len(final_indices) == demo_num:
                return final_indices

