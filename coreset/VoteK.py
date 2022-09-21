from . import CoreSet

from utils.functional import expand_past_key_value

import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer

class VoteK(CoreSet):
    @torch.no_grad()
    def __init__(self, args, model, tokenizer, device, dataset_train):
        sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x)), drop_last=True)
        all_embeddings = None
        for train_input_str, _, _ in dataloader_train:
            embeddings = sentence_model.encode(list(train_input_str), convert_to_tensor=True, device=device)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)

        norm = torch.norm(all_embeddings, dim=1)
        distance = torch.einsum('ik, jk -> ij', all_embeddings, all_embeddings) / norm / norm.unsqueeze(-1)
        values, _ = torch.sort(distance, dim=-1, descending=True)
        threshold = values[:, 100]
        edge = torch.zeros(distance.shape, dtype=torch.long, device=device)
        edge.masked_fill(distance > threshold.unsqueeze(-1), 1)
        edge = edge | edge.transpose(1, 0)
        select_mask = torch.zeros(distance.shape[0], dtype=torch.long, device=device)
        demo_encoding = []
        self.indices = []
        demo_max_length = args.max_length - dataset_train.get_max_length(tokenizer)
        while True:
            penality = 10 ** (-(edge & select_mask).sum(dim=1))
            score = (edge * penality).sum(dim=1)
            score = score * ~select_mask
            index = torch.argmax(score).item()
            demo = dataset_train.get_demo_from_indices(index)
            demo_input_ids = tokenizer(demo).input_ids
            if len(demo_encoding) + len(demo_input_ids) <= demo_max_length:
                self.indices.append(index)
                select_mask[index] = 1
                demo_encoding += demo_input_ids
            else:
                break
        
        past_key_values = model(
            input_ids=torch.LongTensor([demo_encoding]).to(device), 
            use_cache=True
        ).past_key_values
        past_key_values = expand_past_key_value(past_key_values, args.batch_size)
        all_logits = torch.empty(0).to(device)
        for train_input_str, _, _ in dataloader_train:
            train_input_encoding = tokenizer(
                list(train_input_str),
                padding=True,
                return_tensors='pt',
            ).to(device)
            input_ids = train_input_encoding.input_ids
            attention_mask = train_input_encoding.attention_mask
            logits = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=torch.cat((torch.ones(attention_mask.shape[0], past_key_values[0][0].shape[1], device=device), attention_mask), dim=1)
                ).logits
            logits = torch.log_softmax(logits, dim=-1) # batch * seq_len * dim
            logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])[torch.arange(logits.shape[0] * (logits.shape[1] - 1)).to(device), \
                input_ids[:, 1:].flatten()].view(logits.shape[0], (logits.shape[1] - 1))
            logits = torch.mean(logits * attention_mask[:, 1:], dim=-1)
            all_logits = torch.cat((all_logits, logits), dim=0)

        _, indices = torch.sort(all_logits, descending=True)
        select_indices = indices[torch.arange(0, len(indices), len(indices) // (args.coreset_size), device=device)[len(self.indices):]]
        self.indices += select_indices.tolist()
            


