import numpy as np
from algo import ValueFunctionWithApproximation
from typing import NamedTuple, List, Tuple


class Tiling(NamedTuple):
    x_ticks: List[float]
    y_ticks: List[float]
    values: np.array

    def get_tile_indices(self, s) -> Tuple[int, int]:
        x = 0
        while s[0] > self.x_ticks[x]:
            x += 1
        y = 0
        while s[1] > self.y_ticks[y]:
            y += 1
        return x, y

    def get_value(self, s) -> float:
        x, y = self.get_tile_indices(s)
        return self.values[x][y]
    
    def update(self, alpha, G, s_tau):
        x, y = self.get_tile_indices(s_tau)
        self.values[x][y] += alpha * (G - self.get_value(s_tau))


class ValueFunctionWithTile(ValueFunctionWithApproximation):
    tilings: List[Tiling]

    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """

        self.tilings = []
        for i in range(num_tilings):
            pos = state_low - i * (tile_width / num_tilings)
            x_ticks = []
            y_ticks = []
            while pos[0] <= state_high[0] and pos[1] <= state_high[1]:
                pos += tile_width
                x_ticks += [pos[0]]
                y_ticks += [pos[1]]
            self.tilings += [Tiling(x_ticks, y_ticks, np.zeros([len(x_ticks), len(y_ticks)]))]

    def __call__(self,s):
        return np.sum([tiling.get_value(s) for tiling in self.tilings]) / len(self.tilings)

    def update(self,alpha,G,s_tau):
        for tiling in self.tilings:
            tiling.update(alpha, G, s_tau)
