import argparse

import numpy as np


class Arm:
    def __init__(self):
        self.mean = 0

    def pull(self) -> float:
        return np.random.normal(self.mean, 1)

    def update(self):
        self.mean += np.random.normal(0, 0.01)


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
        if np.random.random() < self.epsilon:
            action = np.random.choice(range(10))
        else:
            action = np.argmax(self.avg_rewards)
        reward = self.arms[action].pull()
        self.avg_rewards[action] = incremental_average(self.avg_rewards[action], reward, self.n)
        for arm in self.arms:
            arm.update()
        self.n += 1
        return action, reward


class ConstStepSizeAgent:
    def __init__(self):
        self.arms = [Arm() for _ in range(10)]
        self.q_estimates = [0] * 10
        self.alpha = 0.1
        self.epsilon = 0.1

    def get_optimal(self):
        return np.argmax([arm.mean for arm in self.arms])

    def choose(self):
        if np.random.random() < self.epsilon:
            action = np.random.choice(range(10))
        else:
            action = np.argmax(self.q_estimates)
        reward = self.arms[action].pull()
        self.q_estimates[action] += self.alpha * (reward - self.q_estimates[action])
        for arm in self.arms:
            arm.update()
        return action, reward


def train(agent_class):
    print(f"Training {agent_class.__name__}...")
    reward_history = [0] * 10**4
    optimal_action_ratio_history = [0] * 10**4
    for run_num in range(300):
        print(f"Progress: {round(100 * run_num / 300, ndigits=2)}%", end="\r")
        agent = agent_class()
        for step_num in range(10**4):
            optimal_action = agent.get_optimal()
            action, reward = agent.choose()
            reward_history[step_num] = incremental_average(reward_history[step_num], reward, run_num)
            action_was_optimal = action == optimal_action
            optimal_action_ratio_history[step_num] = incremental_average(optimal_action_ratio_history[step_num], float(action_was_optimal), run_num)
    print("Done!")
    return reward_history, optimal_action_ratio_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    # using sample averaging
    reward_history, optimal_action_ratio_history = train(SampleAverageAgent)
    with open(args.filename, mode="a") as f:
        np.savetxt(f, reward_history, fmt="%.3f", newline=" ")
        f.write("\n")
        np.savetxt(f, optimal_action_ratio_history, fmt="%.3f", newline=" ")
        f.write("\n")
    # using constant step-size
    reward_history, optimal_action_ratio_history = train(ConstStepSizeAgent)
    with open(args.filename, mode="a") as f:
        np.savetxt(f, reward_history, fmt="%.3f", newline=" ")
        f.write("\n")
        np.savetxt(f, optimal_action_ratio_history, fmt="%.3f", newline=" ")


if __name__ == "__main__":
    main()
