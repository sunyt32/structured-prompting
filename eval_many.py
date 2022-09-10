import argparse
import os
import json

import torch

from models import BloomForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import get_dataset, dataset_dict
from coreset import AlignFeature, RandomSelector, LossPartition, LossSampling, AlignEmbedding
from utils.functional import select_past_key_value


@torch.no_grad()
def validate(model, dataset, tokenizer, device, past_key_values, chunk_num):
    correct = 0
    total = 0
    for input_str, output_str, answer in dataset:
        input_ids = tokenizer(input_str).input_ids
        answer = torch.LongTensor([answer]).to(device)
        all_logits = torch.empty(0).to(device)
        for candidate_str in output_str:
            candidate_ids = tokenizer(candidate_str).input_ids
            input_encoding = torch.LongTensor(input_ids + candidate_ids).unsqueeze(0).to(device)
            answer_encoding = tokenizer(
                candidate_str,
                return_tensors='pt',
            ).to(device)
            with torch.cuda.amp.autocast():
                logits = model(
                    input_ids=input_encoding,
                    past_key_values=past_key_values,
                    prefix_parallel=chunk_num
                    ).logits

            logits = logits[0, len(input_ids): -1].log_softmax(dim=-1)
            # select answer
            logits = logits[torch.arange(logits.shape[0]).to(device), answer_encoding.input_ids]
            all_logits = torch.cat((all_logits, torch.sum(logits, dim=1)), dim=0)

        preds = all_logits.argmax(dim=-1)
        correct += preds.eq(answer).sum().item()
        total += len(answer)

    acc = correct / total
    return acc


def main():
    parser = argparse.ArgumentParser()
    # Model setting
    parser.add_argument('--model', type=str, default="bloom-560m")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--parallelize', action='store_true')
    # Data setting
    parser.add_argument('--task', type=str)
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--log_path', type=str, default="./log/log.json")
    parser.add_argument('--sample_num', type=int, default=512)
    parser.add_argument('--select_method', type=str, default="random")
    # Parameters
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--repeat_num', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2000)
    parser.add_argument('--coreset_size', type=int, default=1000)
    args = parser.parse_args()

    model_path = os.path.join(args.data_path, "model", args.model)
    if args.model.startswith('bloom'):
        model = BloomForCausalLM.from_pretrained(model_path)
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

    print("Model initialized with cuda memory allocated {} GB.".format(torch.cuda.memory_allocated(device) / (1024 ** 3)))
    
    if args.task:
        dataset_list = [args.task]
    else:
        dataset_list = dataset_dict.keys()

    for dataset in dataset_list:
        dataset_train = get_dataset(dataset, is_train=True)
        dataset_val = get_dataset(dataset, is_train=False)
        if args.select_method == "align_feature":
            selector = AlignFeature(args, model, tokenizer, device, dataset_train, dataset_val)
        elif args.select_method == "align_embedding":
            selector = AlignEmbedding(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "loss_partition":
            selector = LossPartition(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "loss_sampling":
            selector = LossSampling(args, model, tokenizer, device, dataset_train)
        elif args.select_method == "random":
            selector = RandomSelector(dataset_train)
        else:
            raise NotImplementedError()

        acc_list = []
        demo_max_length = args.max_length - dataset_val.get_max_length(tokenizer)
        model.eval()
        for _ in range(args.repeat_num):
            indices = selector.get_demo_indices(args.coreset_size)
            demo_input_ids_list = []
            for index in indices:
                demo = dataset_train.get_demo_from_indices(index)
                demo_input_ids = tokenizer(demo).input_ids
                demo_input_ids_list.append(demo_input_ids)

            demo_encoding_batch = []
            demo_encoding = []
            for demo_input_ids in demo_input_ids_list:
                if len(demo_encoding) + len(demo_input_ids) <= demo_max_length:
                    demo_encoding += demo_input_ids
                else:
                    demo_encoding_batch.append((demo_encoding + demo_input_ids)[-demo_max_length:])
                    demo_encoding = []
                    if not args.chunk:
                        break

            if len(demo_encoding_batch) == 0: # doesn't need chunk!
                demo_encoding_batch.append(demo_encoding)

            demo_encoding_batch = torch.LongTensor(demo_encoding_batch)
            print(demo_encoding_batch.shape)
            all_past_key_values = []
            for demo_encoding in demo_encoding_batch:
                with torch.no_grad():
                    past_key_values = model(
                        input_ids=demo_encoding.unsqueeze(0).to(device), 
                        use_cache=True
                        ).past_key_values

                all_past_key_values.append(past_key_values)

            all_past_key_values = select_past_key_value(all_past_key_values, 1, torch.ones(demo_encoding_batch.shape))
            acc = validate(model, dataset_val, tokenizer, device, all_past_key_values, len(demo_encoding))
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
