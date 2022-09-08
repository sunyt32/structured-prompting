import argparse
import os
import json

import torch


from models import OPTForCausalLM, BloomForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import get_dataset, dataset_dict
from coreset import AlignFeature, RandomSelector, SimpleAlignFeature, BalanceAlignFeature, LossPartition, LossSampling, AlignEmbedding
from utils.functional import select_past_key_value


@torch.no_grad()
def validate(model, dataset, tokenizer, device, past_key_values, chunk_num):
    model.eval()
    correct = 0
    total = 0
    for input_str, output_str, answer in dataset:
        input_encoding = tokenizer(
            [input_str + candidate_str for candidate_str in output_str],
            padding=True,
            return_tensors='pt',
        ).to(device)
        answer_encoding = tokenizer(
            output_str,
            padding=True,
            return_tensors='pt',
        ).to(device)
        answer = torch.LongTensor([answer]).to(device)
        answer_shape = answer_encoding.input_ids.shape

        output = model(
            input_ids=input_encoding.input_ids,
            past_key_values=past_key_values,
            multiply_tgt=chunk_num
            )
        logits = output.logits
        logits = logits[:, -(answer_shape[1] + 1): -1].log_softmax(dim=-1)
        # select answer
        logits = logits.view(answer_shape[0] * answer_shape[1], -1)[torch.arange(answer_shape[0] * answer_shape[1]).to(device), answer_encoding.input_ids.flatten()].view(answer_shape)
        logits = logits.mul(answer_encoding.attention_mask).sum(dim=1).unsqueeze(0)
        preds = logits.argmax(dim=-1)
        correct += preds.eq(answer).sum().item()
        total += len(answer)

    acc = correct / total
    return acc


def main():
    parser = argparse.ArgumentParser()
    # Model setting
    parser.add_argument('--model', type=str, default="bloom-3b")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--parallelize', action='store_true')
    # Data setting
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--log_path', type=str, default="./log/log.json")
    parser.add_argument('--sample_num', type=int, default=512)
    parser.add_argument('--select_method', type=str, default="random")
    # Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--demo_num', type=int, default=16)
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--max_train_num', type=int, default=512)
    parser.add_argument('--max_val_num', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--coreset_size', type=int, default=32)
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, "model", args.model)
    if args.model.startswith('bloom'):
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-350m")
    else:
        raise NotImplementedError()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False)

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    newline_token = tokenizer('\n').input_ids
    
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
            indices = selector.get_demo_indices(args.coreset_size)
            demo_input_ids_batch = []
            for i in range(len(indices) // args.demo_num):
                demo = dataset_train.get_demo_from_indices(indices[i * args.demo_num : (i + 1) * args.demo_num])
                demo_input_ids = tokenizer(demo).input_ids
                demo_input_ids_batch.append(demo_input_ids)

            max_length = max(len(inputs) for inputs in demo_input_ids_batch)
            demo_encoding = []
            demo_attention_mask = []
            for demo_input_ids in demo_input_ids_batch:
                padding_length = max_length - len(demo_input_ids)
                demo_encoding.append(newline_token * padding_length + demo_input_ids)
                demo_attention_mask.append([0] * padding_length + [1] * len(demo_input_ids))

            with torch.no_grad():
                past_key_values = model(
                    input_ids=torch.LongTensor(demo_encoding).to(device), 
                    use_cache=True
                    ).past_key_values

            past_key_values = select_past_key_value(past_key_values, dataset_train.class_num, torch.LongTensor(demo_attention_mask).to(device))
            acc = validate(model, dataset_val, tokenizer, device, past_key_values, len(demo_encoding))
            acc_list.append(acc)
            print(acc)
 
        log_dict = {
            "acc": torch.Tensor(acc_list).mean().item(),
            "details": acc_list
        }
        print(args)
        print(log_dict)
        with open(args.log_path, 'w') as fp:
            fp.write(json.dumps(log_dict, indent=1))


if __name__ == "__main__":
    main()
