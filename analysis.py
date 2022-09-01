import argparse
import os
import json

import torch


from models import OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM, GPT2Tokenizer

from dataset import get_dataset, dataset_dict
from coreset import AlignFeature, RandomSelector, SimpleAlignFeature, BalanceAlignFeature, LossPartition, LossSampling
from utils import validation

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
    # Parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_train_num', type=int, default=512)
    parser.add_argument('--max_val_num', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--coreset_size', type=int, default=48)
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
        selector = AlignFeature(args, model, tokenizer, device, dataset_train, dataset_val)
        dataset_train_core = []
        for index in selector.indices:
            dataset_train_core.append(dataset_train.examples[index])

        acc_whole = validation(model, dataset_train, tokenizer, device)
        acc_core = validation(model, dataset_train_core, tokenizer, device)
        print(acc_whole, acc_core)
 
        log_dict = {
            "acc_whole": acc_whole,
            "acc_core": acc_core
        }
        print(args)
        print(log_dict)
        with open(args.log_path, 'w') as fp:
            fp.write(json.dumps(log_dict, indent=1))


if __name__ == "__main__":
    main()
