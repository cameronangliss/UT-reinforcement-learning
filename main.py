import argparse

from const_step_size_agent import ConstStepSizeAgent
from sample_average_agent import SampleAverageAgent, incremental_average


def train(agent_class):
    print(f"Training {agent_class.__name__}...")
    reward_history = [0] * 10**4
    optimal_action_ratio_history = [0] * 10**4
    for run_num in range(300):
        agent = agent_class()
        for step_num in range(10**4):
            print(f"Progress: {round(100 * ((10**4 * run_num) + step_num) / (300 * 10**4), ndigits=2)}%", end="\r")
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
    with open(args.filename, "w") as f:
        f.write(f"{' '.join([str(reward) for reward in reward_history])}\n{' '.join([str(ratio) for ratio in optimal_action_ratio_history])}\n")
    # using constant step-size
    reward_history, optimal_action_ratio_history = train(ConstStepSizeAgent)
    with open(args.filename, "a") as f:
        f.write(f"{' '.join([str(reward) for reward in reward_history])}\n{' '.join([str(ratio) for ratio in optimal_action_ratio_history])}")


if __name__ == "__main__":
    main()
