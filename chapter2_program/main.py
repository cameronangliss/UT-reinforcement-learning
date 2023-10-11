import argparse

import numpy as np


class Arm:
    def __init__(self):
        self.mean = 0

    def pull(self) -> float:
        return np.random.normal(self.mean, 1)

    def update(self):
        self.mean += np.random.normal(0, 0.01)


class Agent:
    def __init__(self):
        self.arms = [Arm() for _ in range(10)]
        self.q_estimates = [0] * 10
        self.epsilon = 0.1

    def get_optimal(self):
        return np.argmax([arm.mean for arm in self.arms])


class SampleAverageAgent(Agent):
    def __init__(self):
        super().__init__()
        self.ns = [0] * 10

    def choose(self):
        if np.random.random() < self.epsilon:
            action = np.random.choice(range(10))
        else:
            action = np.argmax(self.q_estimates)
        reward = self.arms[action].pull()
        self.ns[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.ns[
            action
        ]
        for arm in self.arms:
            arm.update()
        return action, reward


class ConstStepSizeAgent(Agent):
    def __init__(self):
        super().__init__()
        self.alpha = 0.1

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
            reward_history[step_num] += (reward - reward_history[step_num]) / (
                run_num + 1
            )
            optimal_action_ratio_history[step_num] += (
                float(action == optimal_action) - optimal_action_ratio_history[step_num]
            ) / (run_num + 1)
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
