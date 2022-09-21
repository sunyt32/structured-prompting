from . import CoreSet

import random

import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer

class VoteK(CoreSet):
    def __init__(self, device, dataset_train):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        dataloader_train = DataLoader(dataset_train, 1, shuffle=False, collate_fn=lambda x: list(zip(*x)))
        for train_input_str, _, _ in dataloader_train:
            embeddings = model.encode(train_input_str[0])
            embeddings = torch.Tensor([embeddings]).to(device)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)

        norm = torch.norm(all_embeddings, dim=1)
        distance = torch.einsim('ik, jk -> ij', all_embeddings, all_embeddings) / norm / norm.unsqueeze(-1)
        _, indices = torch.sort(distance, dim=-1, descending=True)
        print(indices.shape)
        exit()
            
        
    
    def get_demo_indices(self, demo_num):
        final_indices = []
        for sub_indices in self.indices.chunk(64):
            final_indices += random.sample(sub_indices.tolist(), demo_num // 64)

        return final_indices

