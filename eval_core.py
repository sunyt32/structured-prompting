import argparse
import os
import json

import torch

from models.bloom.modeling_bloom import BloomForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import get_dataset, dataset_dict
from coreset import AlignFeature, RandomSelector, LossPartition, LossSampling, AlignEmbedding, VoteK
from utils.functional import select_past_key_value


@torch.no_grad()
def validate(model, dataset, tokenizer, device, past_key_values, chunk_num):
    correct = 0
    total = 0
    for input_str, output_str, answer in dataset:
        input_encoding = tokenizer(
            input_str,
            return_tensors='pt',
        ).input_ids.to(device)
        answer_encoding = tokenizer(
            output_str,
            padding=True,
            return_tensors='pt',
        ).to(device)
        if answer_encoding.input_ids.shape[1] == 1: # classification
            with torch.autocast(device_type="cuda"):
                logits = model(
                    input_ids=input_encoding,
                    past_key_values=past_key_values,
                    prefix_parallel=chunk_num
                    ).logits
                
            logits = logits[0][-1]
            all_logits = logits[answer_encoding.input_ids.flatten()]
        else: # multi-choice
            all_logits = torch.empty(0).to(device)
            for candidate_encoding, candidate_mask in zip(answer_encoding.input_ids, answer_encoding.attention_mask):
                candidate_encoding = candidate_encoding[torch.where(candidate_mask)].unsqueeze(0)
                with torch.autocast(device_type="cuda"):
                    logits = model(
                        input_ids=torch.cat((input_encoding, candidate_encoding), dim=1),
                        past_key_values=past_key_values,
                        prefix_parallel=chunk_num
                        ).logits

                logits = torch.log_softmax(logits[0, (input_encoding.shape[1] - 1): -1], dim=-1)
                # select answer
                logits = logits[torch.arange(logits.shape[0]).to(device), candidate_encoding.flatten()].mean()
                all_logits = torch.cat((all_logits, logits.unsqueeze(0)), dim=0)

        preds = all_logits.argmax(dim=-1)
        correct += int(preds.item() == answer)
        total += 1
        
    acc = correct / total
    return acc


def main():
    parser = argparse.ArgumentParser()
    # Model setting
    parser.add_argument('--model', type=str, default="bloom-560m")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--parallelize', action='store_true')
    # Data setting
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--sample_num', type=int, default=256)
    parser.add_argument('--select_method', type=str, default="random")
    # Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_train_num', type=int, default=4096)
    parser.add_argument('--max_val_num', type=int, default=4096)
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--coreset_size', type=int, default=1024)
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, "model", args.model)   
    if args.model.startswith('bloom'):
        model = BloomForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False)
    
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
        elif args.select_method == "align_embedding":
            selector = AlignEmbedding(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "loss_partition":
            selector = LossPartition(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "loss_sampling":
            selector = LossSampling(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "votek":
            selector = VoteK(device, dataset_train)
        elif args.select_method == "random":
            selector = RandomSelector(dataset_train)
        else:
            raise NotImplementedError()

        acc_list = []
        demo_max_length = args.max_length - dataset_val.get_max_length(tokenizer)
        model.eval()
        for _ in range(args.repeat_num):
            indices = selector.get_demo_indices(args.coreset_size)
            demo_encoding_batch = dataset_train.get_chunk(tokenizer, demo_max_length, indices=indices, chunk_num=None if args.chunk else 1)
            demo_encoding_batch = torch.LongTensor(demo_encoding_batch)
            print(demo_encoding_batch.shape)
            all_past_key_values = []
            for demo_encoding in demo_encoding_batch:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda"):
                        past_key_values = model(
                            input_ids=demo_encoding.unsqueeze(0).to(device), 
                            use_cache=True
                        ).past_key_values

                past_key_values_cpu = ()
                for layer_past in past_key_values:
                    layer_past = tuple(past_state.cpu() for past_state in layer_past)
                    past_key_values_cpu = past_key_values_cpu + (layer_past, )

                all_past_key_values.append(past_key_values_cpu)

            past_key_values = select_past_key_value(all_past_key_values)
            acc = validate(model, dataset_val, tokenizer, device, past_key_values, len(demo_encoding_batch))
            acc_list.append(acc)
            print(acc)
 
        log_dict = {
            "acc": torch.Tensor(acc_list).mean().item(),
            "details": acc_list
        }
        print(args)
        print(log_dict)
        if args.log_path:
            with open(args.log_path, 'w') as fp:
                fp.write(json.dumps(log_dict, indent=1))


if __name__ == "__main__":
    main()
