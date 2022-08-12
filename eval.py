import argparse
import os

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoForCausalLM, GPT2Tokenizer

from dataset import get_dataset, align_feature_demo, dataset_dict
from utils import expand_past_key_value


@torch.no_grad()
def eval(args, model, data_loader, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    for input_str, output_str, answer in data_loader:
        input_encoding = tokenizer(
            list(input_str),
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt',
        ).to(device)
        answer_encoding = tokenizer(
            output_str[0],
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors='pt',
        ).to(device)
        answer = torch.LongTensor(answer).to(device)
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--parallelize', action='store_true')
    # Data setting
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str, default="./data")
    # Parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--demo_num', type=int, default=6)
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=8192)
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, "model", args.model)
    config = AutoConfig.from_pretrained(model_path)
    config.max_position_embeddings = args.max_length

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        config=config, 
        use_fast=False)

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.parallelize:
        model.parallelize()
        device = torch.device(model.transformer.first_device)
    else:
        device = torch.device(args.device)
        model = model.to(device)

    print("Model initialized.")
    
    if args.task:
        dataset_list = [args.task]
    else:
        dataset_list = dataset_dict.keys()

    for dataset in dataset_list:
        dataset_train = get_dataset(dataset, is_train=True)
        dataset_val = get_dataset(dataset, is_train=False)
        dataloader_val = DataLoader(dataset_val, 1, shuffle=False, collate_fn=lambda x: list(zip(*x)))
        acc_list = []
        for _ in range(args.repeat_num):
            dataset_val.demo = align_feature_demo(args, model, tokenizer, device, dataset_train, dataset_val)
            # dataset_val.demo=dataset_train.get_demo(args.demo_num)
            acc = eval(args, model, dataloader_val, tokenizer, device)
            acc_list.append(acc)
            print(acc)

        print("{} average accuracy: {}".format(dataset, torch.Tensor(acc_list).mean().item()))
        print("All accuracy: {}".format(acc_list))
        print(args)


if __name__ == "__main__":
    main()
