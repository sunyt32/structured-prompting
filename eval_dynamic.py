import argparse
import os
import json

import torch
from torch.utils.data import DataLoader


from models import OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM, GPT2Tokenizer

from dataset import get_dataset, dataset_dict
from coreset import AlignFeature, BalanceAlignFeature
from utils.functional import expand_past_key_value


@torch.no_grad()
def eval(args, model, data_loader, tokenizer, device, selector):
    model.eval()
    correct = 0
    total = 0
    for batch_input_str, batch_output_str, batch_answer in data_loader:
        indices = selector.get_dynamic_indices(batch_input_str, args.demo_num)
        demo = selector.dataset_train.get_demo_from_indices(indices)
        for input_str, output_str, answer in zip(batch_input_str, batch_output_str, batch_answer):
            input_encoding = tokenizer(
                [demo + input_str],
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(device)
            answer_encoding = tokenizer(
                output_str,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors='pt',
            ).to(device)
            answer = torch.LongTensor([answer]).to(device)
            answer_shape = answer_encoding.input_ids.shape

            output = model(
                input_ids=input_encoding.input_ids, 
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
            preds = logits.mul(answer_encoding.attention_mask).sum(dim=1).argmax(dim=-1)
            correct += preds.eq(answer).sum().item()
            total += len(answer)

    return correct / total      

def main():
    parser = argparse.ArgumentParser()
    # Model setting
    parser.add_argument('--model', type=str, default="gpt-j-6B")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--parallelize', action='store_true')
    # Data setting
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--log_path', type=str, default="./log/log.json")
    parser.add_argument('--sample_num', type=int, default=512)
    parser.add_argument('--select_method', type=str)
    # Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--demo_num', type=int, default=6)
    parser.add_argument('--repeat_num', type=int, default=20)
    parser.add_argument('--max_train_num', type=int, default=512)
    parser.add_argument('--max_val_num', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=2048)
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, "model", args.model)
    if args.model.startswith('opt'):
        model = OPTForCausalLM.from_pretrained(model_path) # add model parallel
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False)

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.parallelize:
        model.parallelize()
        device = torch.device(model.lm_head.weight.device)
    else:
        device = torch.device(args.device)
        model = model.to(device)

    print("Model initialized.")
    
    if args.task:
        dataset_list = [args.task]
    else:
        dataset_list = dataset_dict.keys()

    for dataset in dataset_list:
        dataset_train = get_dataset(dataset, is_train=True, max_data_num=args.max_train_num)
        dataset_val = get_dataset(dataset, is_train=False, max_data_num=args.max_val_num)
        dataloader_val = DataLoader(dataset_val, args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
        if args.select_method == "align_feature":
            selector = AlignFeature(args, model, tokenizer, device, dataset_train, dataset_val, dynamic=True)
        elif args.select_method == "balance_align_feature":
            selector = BalanceAlignFeature(model, tokenizer, device, dataset_train, dynamic=True)
        else:
            raise NotImplementedError()

        acc_list = []
        for _ in range(args.repeat_num):
            acc = eval(args, model, dataloader_val, tokenizer, device, selector)
            acc_list.append({
                "acc": acc
            })
            print(acc)
 
        log_dict = {
            "acc": torch.Tensor([item["acc"] for item in acc_list]).mean().item(),
            "details": acc_list
        }
        print(args)
        print(log_dict)
        with open(args.log_path, 'w') as fp:
            fp.write(json.dumps(log_dict))


if __name__ == "__main__":
    main()
