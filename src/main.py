from const_step_size_agent import ConstStepSizeAgent
from sample_average_agent import SampleAverageAgent


# NOTE: We assume that n begins at 0
def incremental_average(current_average: float, new_value: float, n: int) -> float:
    return (n / (n + 1)) * current_average + new_value / (n + 1)


def train(agent):
    print(f"Training {agent.__class__.__name__}...")
    reward_history = [0] * 10**4
    optimal_action_ratio_history = [0] * 10**4
    for run_num in range(300):
        for step_num in range(10**4):
            optimal_action = agent.get_optimal()
            action, reward = agent.choose()
            reward_history[step_num] = incremental_average(reward_history[step_num], reward, run_num)
            action_was_optimal = action == optimal_action
            optimal_action_ratio_history[step_num] = incremental_average(optimal_action_ratio_history[step_num], float(action_was_optimal), run_num)
        print(f"Progress: {round((run_num + 1) / 300, ndigits=3)}%", end="\r")
    return reward_history, optimal_action_ratio_history


def main():
    # using sample averaging
    agent = SampleAverageAgent()
    reward_history, optimal_action_ratio_history = train(agent)
    with open("result.out", "w") as f:
        f.write(f"{reward_history}\n{optimal_action_ratio_history}\n")
    # using constant step-size
    agent = ConstStepSizeAgent()
    reward_history, optimal_action_ratio_history = train(agent)
    with open("result.out", "a") as f:
        f.write(f"{reward_history}\n{optimal_action_ratio_history}")


if __name__ == "__main__":
    main()
