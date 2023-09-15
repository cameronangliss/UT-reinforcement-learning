import numpy as np
import random

from arm import Arm


class SampleAverageAgent:
    def __init__(self):
        self.arms = [Arm() for _ in range(10)]
        self.samples = [[0]] * 10
        self.epsilon = 0.1

    def get_optimal(self):
        return np.argmax([arm.mean for arm in self.arms])

    def choose(self):
        if random.random() < self.epsilon:
            action = random.choice(range(10))
        else:
            action = np.argmax([np.mean(sample) for sample in self.samples])
        reward = self.arms[action].pull()
        self.samples[action] += [reward]
        for arm in self.arms:
            arm.update()
        return action, reward
