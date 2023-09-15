import numpy as np
import random

from arm import Arm


# NOTE: We assume that n begins at 0
def incremental_average(current_average: float, new_value: float, n: int) -> float:
    return (n / (n + 1)) * current_average + new_value / (n + 1)


class SampleAverageAgent:
    def __init__(self):
        self.arms = [Arm() for _ in range(10)]
        self.avg_rewards = [0] * 10
        self.epsilon = 0.1
        self.n = 0

    def get_optimal(self):
        return np.argmax([arm.mean for arm in self.arms])

    def choose(self):
        if random.random() < self.epsilon:
            action = random.choice(range(10))
        else:
            action = np.argmax(self.avg_rewards)
        reward = self.arms[action].pull()
        self.avg_rewards[action] = incremental_average(self.avg_rewards[action], reward, self.n)
        for arm in self.arms:
            arm.update()
        self.n += 1
        return action, reward
