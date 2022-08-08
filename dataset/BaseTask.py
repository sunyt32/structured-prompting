import random

from torch.utils.data import Dataset

from openprompt.data_utils import InputExample


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
            input_str, _, label = self.preprocess_example(example)
            input_example = InputExample(
                text_a = self.demo,
                text_b = input_str,
                label = label
            )
            self.examples.append(input_example)

    def get_demo(self, demo_num):
        demo_examples = random.choices(self.dataset, k=demo_num)
        demo_str = ""
        for example in demo_examples:
            input_str, answer_str, _ = self.preprocess_example(example)
            demo_str += input_str + answer_str + " "

        return demo_str

    def label_words(self):
        return self.templates[self.temp_index][2]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def __iter__(self):
        for example in self.examples:
            yield example


