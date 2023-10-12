from functools import reduce
from typing import Iterable, Tuple

import numpy as np

from env import EnvSpec
from policy import Policy


def product(lst):
    if len(lst) == 0:
        return 1
    else:
        return reduce(lambda x, y: x * y, lst)


def on_policy_n_step_td(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    n: int,
    alpha: float,
    initV: np.array,
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    V = initV
    for episode in trajs:
        S = [episode[0][0]]
        R = [0]
        T = float("inf")
        for t in range(len(episode)):
            if t < T:
                R += [episode[t][2]]
                S += [episode[t][3]]
                if t == len(episode) - 1:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = sum([env_spec.gamma**(i - tau - 1) * R[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += env_spec.gamma**n * V[S[tau + n]]
                V[S[tau]] += alpha * (G - V[S[tau]])
            if tau == T - 1:
                break
    return V


def off_policy_n_step_sarsa(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    n: int,
    alpha: float,
    initQ: np.array,
) -> Tuple[np.array, Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    class CurrentPolicy(Policy):
        def __init__(self, Q: np.array):
            self.Q = Q

        def action_prob(self, state, action):
            return float(action == np.argmax(self.Q[state]))

        def action(self, state):
            return np.argmax(self.Q[state])
    
    pi = CurrentPolicy(initQ)

    Q = initQ
    for episode in trajs:
        S = [episode[0][0]]
        A = [episode[0][1]]
        R = [0]
        T = float("inf")
        for t in range(len(episode)):
            if t < T:
                R += [episode[t][2]]
                S += [episode[t][3]]
                if t == len(episode) - 1:
                    T = t + 1
                else:
                    A += [episode[t + 1][1]]
            tau = t - n + 1
            if tau >= 0:
                rho = product([pi.action_prob(S[i], A[i]) / bpi.action_prob(S[i], A[i]) for i in range(tau + 1, min(tau + n, T - 1) + 1)])
                G = sum([env_spec.gamma**(i - tau - 1) * R[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += env_spec.gamma**n * Q[S[tau + n]][A[tau + n]]
                Q[S[tau]][A[tau]] += alpha * rho * (G - Q[S[tau]][A[tau]])
            if tau == T - 1:
                break
    return Q, pi
