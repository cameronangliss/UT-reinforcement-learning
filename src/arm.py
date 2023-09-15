import random


class Arm:
    def __init__(self):
        self.mean = 0

    def pull(self) -> float:
        return random.normalvariate(self.mean, 1)

    def update(self):
        self.mean += random.normalvariate(0, 0.01)
