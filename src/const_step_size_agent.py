import numpy as np
import random

from arm import Arm


class ConstStepSizeAgent:
    def __init__(self):
        self.arms = [Arm() for _ in range(10)]
        self.q_estimates = [0] * 10
        self.alpha = 0.1
        self.epsilon = 0.1

    def get_optimal(self):
        return np.argmax([arm.mean for arm in self.arms])

    def choose(self):
        if random.random() < self.epsilon:
            action = random.choice(range(10))
        else:
            action = np.argmax(self.q_estimates)
        reward = self.arms[action].pull()
        self.q_estimate[action] += self.alpha * (reward - self.q_estimate[action])
        for arm in self.arms:
            arm.update()
        return action, reward
