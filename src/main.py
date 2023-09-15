from const_step_size_agent import ConstStepSizeAgent
from sample_average_agent import SampleAverageAgent


# NOTE: We assume that n begins at 0
def incremental_average(current_average: float, new_value: float, n: int) -> float:
    return (n / (n + 1)) * current_average + new_value / (n + 1)


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
    # using sample averaging
    reward_history, optimal_action_ratio_history = train(SampleAverageAgent)
    with open("result.out", "w") as f:
        f.write(f"{reward_history}\n{optimal_action_ratio_history}\n")
    # using constant step-size
    reward_history, optimal_action_ratio_history = train(ConstStepSizeAgent)
    with open("result.out", "a") as f:
        f.write(f"{reward_history}\n{optimal_action_ratio_history}")


if __name__ == "__main__":
    main()
