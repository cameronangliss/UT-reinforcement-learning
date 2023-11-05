import numpy as np
from algo import ValueFunctionWithApproximation
from typing import List, Tuple


class ValueFunctionWithTile(ValueFunctionWithApproximation):
    tilings: List[Tuple[np.ndarray, np.ndarray]]

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

        self.tile_width = tile_width
        self.tilings = []
        num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        for i in range(num_tilings):
            start_pos = state_low - i * (tile_width / num_tilings)
            weights = np.zeros(num_tiles)
            self.tilings += [(start_pos, weights)]

    def __call__(self,s):
        indices = [np.floor((s - start_pos) / self.tile_width).astype(int) for start_pos, _ in self.tilings]
        weights_of_tilings = [weights for _, weights in self.tilings]
        return sum([weights[index[0]][index[1]] for weights, index in zip(weights_of_tilings, indices)])

    def update(self,alpha,G,s_tau):
        for start_pos, weights in self.tilings:
            index = np.floor((s_tau - start_pos) / self.tile_width).astype(int)
            weights[index[0]][index[1]] += alpha * (G - self(s_tau))
