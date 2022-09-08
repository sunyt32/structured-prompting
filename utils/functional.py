import torch


def select_past_key_value(past_key_value, class_num, attention_mask):
    """
    Input sentence's batch is 1. To use the past_key_value for multiple answers, we need to expand the key and value's batch to the class number.
    """
    present = ()
    select_index = torch.where(attention_mask)
    for layer_past in past_key_value:
        key = layer_past[0] # [batch_size, qk_length, num_heads, head_dim]
        value = layer_past[1]
        present += ((key[select_index].expand(class_num, -1, -1, -1), value[select_index].expand(class_num, -1, -1, -1)), )

    return present

def expand_past_key_value(past_key_value, class_num):
    """
    Input sentence's batch is 1. To use the past_key_value for multiple answers, we need to expand the key and value's batch to the class number.
    """
    present = ()
    for layer_past in past_key_value:
        key = layer_past[0]
        value = layer_past[1]
        present += ((key.expand(class_num, -1, -1, -1), value.expand(class_num, -1, -1, -1)), )

    return present