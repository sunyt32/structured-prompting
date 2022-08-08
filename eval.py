import argparse
import os

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import get_dataset

from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import LMTokenizerWrapper, load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
            

@torch.no_grad()
def eval(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in data_loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        preds = torch.argmax(logits, dim = -1)
        correct += (preds == batch["label"]).sum().item()
        total += len(batch["label"])

    return correct / total      

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-j-6B")
    parser.add_argument('--dataset', type=str, default="boolq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--data_path', type=str, default="./data")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--demo_num', type=int, default=10)
    args = parser.parse_args() 

    dataset_train = get_dataset(args.dataset, is_train=True)
    dataset_val = get_dataset(args.dataset, is_train=False, demo=dataset_train.get_demo(args.demo_num))
    
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.data_path, "model", args.model), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(os.path.join(args.data_path, "model", args.model))
    prompt_template = ManualTemplate(
        text = '{"placeholder":"text_a"} {"placeholder":"text_b"} {"mask"}',
        tokenizer = tokenizer
    )
    prompt_verbalizer = ManualVerbalizer(
        label_words = dataset_val.label_words(),
        tokenizer = tokenizer
    )
    valid_loader = PromptDataLoader(
        dataset=dataset_val, 
        tokenizer=tokenizer, 
        template=prompt_template,
        tokenizer_wrapper_class=LMTokenizerWrapper,
        batch_size=args.batch_size)
    prompt_model = PromptForClassification(
        template=prompt_template,
        plm=model,
        verbalizer=prompt_verbalizer
    ).to(device)
    print("Model initialized.")
    acc = eval(prompt_model, valid_loader, device)
    print("{} accuracy: {}".format(args.dataset, acc))


if __name__ == "__main__":
    main()
