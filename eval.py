import argparse
import os
import json

import torch


from models import OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM, GPT2Tokenizer

from dataset import get_dataset, dataset_dict
from coreset import AlignFeature, RandomSelector, SimpleAlignFeature, BalanceAlignFeature, LossPartition, LossSampling, AlignEmbedding
from utils import validate

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
    parser.add_argument('--repeat_num', type=int, default=10)
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
        if args.select_method == "align_feature":
            selector = AlignFeature(args, model, tokenizer, device, dataset_train, dataset_val)
        elif args.select_method == "align_feature_validation":
            selector = AlignFeature(args, model, tokenizer, device, dataset_train, dataset_val, val=True)
        elif args.select_method == "align_embedding":
            selector = AlignEmbedding(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "align_embedding_validation":
            selector = AlignEmbedding(args, model, tokenizer, device, dataset_train, val=True)
        elif args.select_method == "balance_align_feature":
            selector = BalanceAlignFeature(args, model, tokenizer, device, dataset_train, dataset_val)
        elif args.select_method == "simple_align_feature":
            selector = SimpleAlignFeature(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "loss_partition":
            selector = LossPartition(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "loss_sampling":
            selector = LossSampling(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "random_validation":
            selector = RandomSelector(dataset_train, model, tokenizer, device, val=True)
        elif args.select_method == "random":
            selector = RandomSelector(dataset_train)
        else:
            raise NotImplementedError()

        acc_list = []
        for _ in range(args.repeat_num):
            indices = selector.get_demo_indices(args.demo_num)
            dataset_val.demo = dataset_train.get_demo_from_indices(indices)
            print(indices, dataset_val.demo)
            acc = validate(model, dataset_val, tokenizer, device)
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
            fp.write(json.dumps(log_dict, indent=1))


if __name__ == "__main__":
    main()
