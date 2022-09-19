import argparse
import os
import json

import torch

from models import BloomForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import get_dataset, dataset_dict

from utils.functional import select_past_key_value


@torch.no_grad()
def validate(model, dataset, tokenizer, device, past_key_values, chunk_num, int8):
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
            with torch.autocast(device_type="cuda", enabled=not int8):
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
                with torch.autocast(device_type="cuda", enabled=not int8):
                    logits = model(
                        input_ids=torch.cat((input_encoding, candidate_encoding), dim=1),
                        past_key_values=past_key_values,
                        prefix_parallel=chunk_num
                        ).logits

                logits = logits[0, (input_encoding.shape[1] - 1): -1]
                logits = torch.log_softmax(logits, dim=-1)
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
    parser.add_argument('--dtype', type=str, default="float16")
    parser.add_argument('--parallel', action='store_true')
    # Data setting
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--log_path', type=str)
    # Parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--coreset_size', type=int, default=4096)
    parser.add_argument('--chunk_num', type=int, default=1)
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, "model", args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False)

    device = torch.cuda.current_device()
    if args.dtype == "int8":
        max_memory_mapping = {i: "24000MB" for i in range(8)}
        model = BloomForCausalLM.from_pretrained(model_path, device_map='auto', load_in_8bit=True, max_memory=max_memory_mapping)
    elif args.model == 'bloom':
        max_memory_mapping = {i: "48000MB" for i in range(8)}
        model = BloomForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, max_memory=max_memory_mapping)
    else:
        if args.model.startswith('bloom'):
            model = BloomForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        if args.parallel:
            model.parallelize()
        else:
            model = model.to(device)

    model.eval()
    print("Model initialized.")
    
    if args.task:
        dataset_list = [args.task]
    else:
        dataset_list = dataset_dict.keys()

    for dataset in dataset_list:
        dataset_train = get_dataset(dataset, is_train=True)
        dataset_val = get_dataset(dataset, is_train=False)
        acc_list = []
        demo_max_length = args.max_length - dataset_val.get_max_length(tokenizer)
        for _ in range(args.repeat_num):
            demo_encoding_batch = dataset_train.get_chunk(tokenizer, demo_max_length, chunk_num=args.chunk_num)
            demo_encoding_batch = torch.LongTensor(demo_encoding_batch)
            print(demo_encoding_batch.shape)
            if args.chunk_num is not None and demo_encoding_batch.shape[0] < args.chunk_num:
                print("The dataset's maximal chunk {} < {}!".format(demo_encoding_batch.shape[0], args.chunk_num))
                exit()

            all_past_key_values = []
            for demo_encoding in demo_encoding_batch:
                with torch.autocast(device_type="cuda", enabled=not (args.dtype == "int8")):
                    with torch.no_grad():
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
            acc = validate(model, dataset_val, tokenizer, device, past_key_values, len(demo_encoding_batch), args.dtype == "int8")
            acc_list.append(acc)
            print(acc)
 
        log_dict = {
            "acc": torch.Tensor(acc_list).mean().item(),
            "details": acc_list
        }

        print(log_dict)
        if args.log_path:
            with open(args.log_path, 'w') as fp:
                fp.write(json.dumps(log_dict, indent=1))


if __name__ == "__main__":
    main()
