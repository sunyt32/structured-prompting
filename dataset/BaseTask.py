import random

from torch.utils.data import Dataset


class BaseTask(Dataset):
    def __init__(self, max_data_num=500, temp_index=0, demo=""):
        super().__init__()
        self.temp_index = temp_index
        self.examples = []
        self.max_data_num = max_data_num
        self.demo = demo
        self.templates = self.templates_set_without_newline()
        
    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def preprocess_dataset(self):
        for index, example in enumerate(self.dataset):
            if index >= self.max_data_num:
                break
            
            self.examples.append(self.preprocess_example(example))

    def get_demo_from_indices(self, indices):
        demo_str = ""
        order_indices = [[] for _ in range(self.class_num)]
        for index in indices:
            _, _, label = self.examples[index]
            order_indices[label].append(index)

        for rank in range(len(order_indices[0])):
            for sub_indices in order_indices:
                if len(sub_indices) > rank:
                    index = sub_indices[rank]
                    input_str, output_str, label = self.examples[index]
                    demo_str += input_str + output_str[label] + " \n "

        return demo_str


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_str, output_str, label = self.examples[index]
        return self.demo + input_str, output_str, label

    def __iter__(self):
        for input_str, output_str, label in self.examples:
            yield self.demo + input_str, output_str, label


