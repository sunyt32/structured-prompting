import torch


def expand_past_key_value(past_key_value, class_num):
    """
    Input sentence's batch is 1. To use the past_key_value for multiple answers, we need to expand the key and value's batch to the class number.
    """
    present = ()
    for layer_past in past_key_value:
        key = layer_past[0]
        value = layer_past[1]
        present += ((key.repeat(class_num, 1, 1, 1), value.repeat(class_num, 1, 1, 1)), )

    return present
    