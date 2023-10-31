import numpy as np
from algo import ValueFunctionWithApproximation

import torch

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """

        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def __call__(self,s):
        # print("state:", s)
        return float(self.network(torch.tensor(s))[0].item())

    def update(self,alpha,G,s_tau):
        # print(s_tau)
        loss = G - self(s_tau)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
