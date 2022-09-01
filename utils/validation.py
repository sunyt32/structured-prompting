import torch
from torch.utils.data import DataLoader

from .functional import expand_past_key_value

@torch.no_grad()
def validation(model, dataset, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    for input_str, output_str, answer in dataset:
        input_encoding = tokenizer(
            [input_str],
            truncation=True,
            return_tensors='pt',
        ).to(device)
        answer_encoding = tokenizer(
            output_str,
            truncation=True,
            padding=True,
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
