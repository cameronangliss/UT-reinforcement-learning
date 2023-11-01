import numpy as np
from algo import ValueFunctionWithApproximation

import torch

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """

        self.loss = torch.nn.MSELoss()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(device=self.device)

    def __call__(self,s):
        self.network.eval()
        input_tensor = torch.tensor(s, dtype=torch.float).to(self.device)
        return float(self.network(input_tensor)[0].item())

    def update(self,alpha,G,s_tau):
        self.network.train()
        loss = self.loss(torch.tensor([self(s_tau)], requires_grad=True), torch.tensor([G], requires_grad=True))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
