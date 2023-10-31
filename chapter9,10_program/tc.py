import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
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
            pos = state_low - (i / num_tilings) * tile_width
            vert_borders = [pos[0]]
            horiz_borders = [pos[1]]
            while pos[0] <= state_high[0] and pos[1] <= state_high[1]:
                pos += tile_width
                vert_borders += [pos[0]]
                horiz_borders += [pos[1]]
            self.tilings += [(vert_borders, horiz_borders, [0] * len(vert_borders)**2)]

    def __call__(self,s):
        score = 0
        for vert_borders, horiz_borders, values in self.tilings:
            x = 0
            while s[0] < vert_borders[x]:
                x += 1
            y = 0
            while s[1] < horiz_borders[y]:
                y += 1
            score += values[x * len(vert_borders) + y]
        return score / len(self.tilings)


    def update(self,alpha,G,s_tau):
        current_val = self(s_tau)
        for vert_borders, horiz_borders, values in self.tilings:
            x = 0
            while s_tau[0] < vert_borders[x]:
                x += 1
            y = 0
            while s_tau[1] < horiz_borders[y]:
                y += 1
            values[x * len(vert_borders) + y] += alpha * (G - current_val)
