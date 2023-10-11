from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy


def get_expected_reward(env: EnvWithModel, V: np.array, state: int, action: int):
    expected_reward = 0
    for next_state in range(env.spec.nS):
        transition_prob = env.TD()[state, action, next_state]
        reward = env.R()[state, action, next_state]
        expected_reward += transition_prob * (reward + env.spec.gamma * V[next_state])
    return expected_reward


def value_prediction(
    env: EnvWithModel, pi: Policy, initV: np.array, theta: float
) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    V = initV
    Q = np.full(shape=[env.spec.nS, env.spec.nA], fill_value=[])
    while True:
        delta = 0
        for state in range(env.spec.nS):
            old_value = V[state]
            Q[state] = [get_expected_reward(env, V, state, action) for action in env.spec.nA]
            V[state] = sum(
                [
                    pi.action_prob(state, action)
                    * Q[state][action]
                    for action in env.spec.nA
                ]
            )
            delta = max(delta, abs(old_value - V[state]))
        if delta < theta:
            break

    return V, Q


def value_iteration(
    env: EnvWithModel, initV: np.array, theta: float
) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    V = initV
    Q = np.full(shape=[env.spec.nS, env.spec.nA], fill_value=[])
    while True:
        delta = 0
        for state in range(env.spec.nS):
            old_value = initV[state]
            Q[state] = [get_expected_reward(env, V, state, action) for action in env.spec.nA]
            V[state] = max(Q[state])
            delta = max(delta, abs(old_value - V[state]))
        if delta < theta:
            break

    class OptimalPolicy(Policy):
        def __init__(self, values: np.array):
            self.values = values

        def action_prob(self, state, action):
            return float(action == np.argmax(Q[state]))

        def action(self, state):
            np.argmax(Q[state])

    pi = OptimalPolicy()

    return V, pi
