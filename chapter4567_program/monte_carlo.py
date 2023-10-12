from typing import Iterable, Tuple

import numpy as np

from env import EnvSpec
from policy import Policy


def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.array,
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = initQ
    for episode in trajs:
        G = 0
        W = 1
        ns = np.zeros([env_spec.nS])
        for t in range(len(episode)):
            if W == 0:
                break
            S = episode[t][0]
            A = episode[t][1]
            ns[S] += 1
            G = env_spec.gamma * G + episode[t][2]
            Q[S][A] += W / ns[S] * (G - Q[S][A])
            W *= pi.action_prob(S, A) / bpi.action_prob(S, A)
    return Q


def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec: EnvSpec,
    trajs: Iterable[Iterable[Tuple[int, int, int, int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.array,
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = initQ
    C = np.zeros([env_spec.nS, env_spec.nA])
    for episode in trajs:
        G = 0
        W = 1
        for t in range(len(episode)):
            if W == 0:
                break
            S = episode[t][0]
            A = episode[t][1]
            G = env_spec.gamma * G + episode[t][2]
            C[S][A] += W
            Q[S][A] += W / C[S][A] * (G - Q[S][A])
            W *= pi.action_prob(S, A) / bpi.action_prob(S, A)
    return Q
