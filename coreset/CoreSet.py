import random


class CoreSet:
    def get_demo_indices(self, demo_num):
        if demo_num < len(self.indices):
            return random.sample(self.indices, demo_num)
        else:
            random.shuffle(self.indices)
            return self.indices


