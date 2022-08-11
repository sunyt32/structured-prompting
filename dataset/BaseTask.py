import random

from torch.utils.data import Dataset

class BaseTask(Dataset):
    def __init__(self, temp_index=0, demo=""):
        super().__init__()
        self.temp_index = temp_index
        self.examples = []
        self.demo = demo
        self.templates = self.templates_set_without_newline()
        
    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def preprocess_dataset(self):
        for example in self.dataset:
            self.examples.append(self.preprocess_example(example))

    def get_demo(self, demo_num):
        random.shuffle(self.examples)
        demo_str = ""
        demo_each_label = demo_num / self.class_num
        label_count = [0 for _ in range(self.class_num)]
        example_str_list = []
        for input_str, output_str, label in self.examples:
            if label_count[label] < demo_each_label:
                example_str_list.append(input_str + output_str[label] + " ")
                label_count[label] += 1

        random.shuffle(example_str_list)
        for example_str in example_str_list:
            demo_str += example_str

        return demo_str

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_str, output_str, label = self.examples[index]
        return self.demo + input_str, output_str, label

    def __iter__(self):
        for input_str, output_str, label in self.examples:
            yield self.demo + input_str, output_str, label


