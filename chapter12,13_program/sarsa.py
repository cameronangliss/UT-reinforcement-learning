import numpy as np


class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_actions = num_actions
        self.tile_width = tile_width
        self.num_tilings = num_tilings
        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.start_coords = [state_low - i * (tile_width / num_tilings) for i in range(num_tilings)]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * self.num_tiles[0] * self.num_tiles[1]

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        x = np.zeros((self.feature_vector_len()), dtype=np.float64)
        if done:
            return x
        else:
            tile_indices = [np.floor((s - coord) / self.tile_width).astype(int) for coord in self.start_coords]
            indices = [
                i * self.num_actions * self.num_tiles[0] * self.num_tiles[1] +
                a * self.num_tiles[0] * self.num_tiles[1] +
                tile_index[1] * self.num_tiles[1] +
                tile_index[0]
                for i, tile_index in enumerate(tile_indices)
            ]
            for index in indices:
                x[index] = 1.0
            return x

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for _ in range(num_episode):
        S = []
        A = []
        R = [0]
        state, _ = env.reset()
        S += [state]
        action = epsilon_greedy_policy(state, False, w)
        A += [action]
        x = X(state, False, action)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0
        t = 1
        done = False
        while not done:
            state, reward, done, _, _ = env.step(action)
            S += [state]
            R += [reward]
            action_prime = epsilon_greedy_policy(state, done, w)
            x_prime = X(state, done, action_prime)
            Q = np.dot(w, x)
            Q_prime = np.dot(w, x_prime)
            delta = R[t] + gamma * Q_prime - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
            action = action_prime
            A += [action]
            t += 1
    return w
