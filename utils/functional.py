import torch


def select_past_key_value(past_key_value):
    present = ()
    for layer_past in zip(*past_key_value):
        key, value = tuple(zip(*layer_past))
        key = torch.cat(key, dim=1)
        value = torch.cat(value, dim=1)
        present += ((key, value), )

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