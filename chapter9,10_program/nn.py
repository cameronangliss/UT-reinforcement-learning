import numpy as np
from algo import ValueFunctionWithApproximation

import torch

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """

        self.loss_fn = torch.nn.MSELoss()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def __call__(self,s):
        self.network.eval()
        input_tensor = torch.tensor(s, dtype=torch.float)
        return float(self.network(input_tensor)[0].detach().item())

    def update(self,alpha,G,s_tau):
        self.network.train()
        loss = self.loss_fn(self.network(torch.tensor(s_tau, dtype=torch.float)), torch.tensor([G]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
