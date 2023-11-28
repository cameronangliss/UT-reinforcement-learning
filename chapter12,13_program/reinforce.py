from typing import Iterable
import numpy as np
import torch


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        self.network.eval()
        input_tensor = torch.tensor(s, dtype=torch.float)
        return torch.argmax(self.network(input_tensor)).item()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.network.train()
        output_tensor = self.network(torch.tensor([s], dtype=torch.float))
        loss = self.loss_fn(output_tensor, torch.tensor([gamma_t * delta], dtype=torch.float))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        
        self.loss_fn = torch.nn.MSELoss()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        self.network.eval()
        input_tensor = torch.tensor(s, dtype=torch.float)
        return float(self.network(input_tensor)[0].detach().item())

    def update(self,s,G):
        self.network.train()
        loss = self.loss_fn(self.network(torch.tensor(s, dtype=torch.float)), torch.tensor([G], dtype=torch.float))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    
    G_list = []
    for _ in range(num_episodes):
        # generate a new episode
        S = []
        A = []
        R = [0]
        state = env.reset()
        S += [state]
        done = False
        while not done:
            action = pi(state)
            A += [action]
            state, reward, done, _ = env.step(action)
            S += [state]
            R += [reward]

        # use episode to update policy
        T = len(S)
        for t in range(T):
            G = sum([gamma**(k - t - 1) * R[k] for k in range(t + 1, T - 1)])
            delta = G - V(S[t])
            V.update(S[t], G)
            pi.update(S[t], A[t], gamma**t, delta)
            if t == 0:
                G_list += [G]
    return G_list
