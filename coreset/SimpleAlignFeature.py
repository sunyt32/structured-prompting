from . import CoreSet
from utils.functional import expand_past_key_value

import torch
import torch.nn as nn


class SimpleAlignFeature(CoreSet):
    def __init__(self, args, model, tokenizer, device, dataset_train):
        self.sample_num = args.sample_num
        self.max_length = args.max_length
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dataset_train = dataset_train
        self.bias = torch.tril(torch.ones(args.max_length, args.max_length).view(1, 1, args.max_length, args.max_length))

    def _attn(self, query, key, value, attention_mask=None, head_mask=None,):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool).to(key.device)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32)).to(attn_weights.device)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
    @torch.no_grad()
    def get_demo_indices(self, demo_num):
        metric = torch.zeros(len(self.dataset_train)).to(self.device)
        self.model.eval()
        print("Begin to calculate simple feature metrics...")
        all_past_hidden_states = []
        for index, (train_input_str, train_output_str, train_answer) in enumerate(self.dataset_train):
            train_input_encoding = self.tokenizer(
                [train_input_str + train_output_str[train_answer]],
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(self.device)
            past_key_values = self.model(
                input_ids=train_input_encoding.input_ids, 
                use_cache=True
                ).past_key_values
            
            shape = past_key_values[0][0].shape
            query = nn.init.orthogonal_(torch.empty(self.sample_num, shape[1], 1, shape[3]))
            past_key_values = expand_past_key_value(past_key_values, self.sample_num)
            hidden_states = torch.empty(0)
            for key, value in past_key_values:
                attn_output, _ = self._attn(query.to(key.device), key, value)
                hidden_states = torch.cat((hidden_states, attn_output.flatten().cpu()))

            all_past_hidden_states.append(hidden_states.cpu())

        sum_hidden_states = torch.zeros(all_past_hidden_states[0].shape).to(self.device)
        norm_square = torch.empty(0).to(self.device)
        for hidden_states in all_past_hidden_states:
            hidden_states = hidden_states.to(self.device)
            sum_hidden_states += hidden_states
            norm_square = torch.cat((norm_square, hidden_states.square().sum().unsqueeze(0)))

        metric += norm_square * len(self.dataset_train) + torch.sum(norm_square)
        for index, hidden_states in enumerate(all_past_hidden_states):
            hidden_states = hidden_states.to(self.device)
            metric[index] -= 2 * torch.sum(sum_hidden_states * hidden_states)

        indices = metric.sort(dim=0).indices
        demo_each_label = demo_num / self.dataset_train.class_num
        label_count = [0 for _ in range(self.dataset_train.class_num)]
        final_indices = []
        for index in indices:
            _, _, label = self.dataset_train.examples[index]
            if label_count[label] < demo_each_label:
                final_indices.append(index.item())     
                label_count[label] += 1

        return final_indices
