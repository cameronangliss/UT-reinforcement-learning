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
        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.tilings = []
        for i in range(num_tilings):
            start_pos = state_low - i * (tile_width / num_tilings)
            weights = np.zeros([*self.num_tiles, self.num_actions])
            self.tilings += [(start_pos, weights)]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * len(self.tilings) * self.num_tiles

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        
        indices = [np.floor((s - start_pos) / self.tile_width).astype(int) for start_pos, _ in self.tilings]
        weights_of_tilings = [weights for _, weights in self.tilings]
        if done:
            return np.zeros(self.feature_vector_len())
        else:
            return sum([weights[index[0]][index[1]][a] for weights, index in zip(weights_of_tilings, indices)])

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
        # generate a new episode
        S = []
        A = []
        R = [0]
        state, _ = env.reset()
        S += [state]
        done = False
        t = 0
        action = epsilon_greedy_policy(state, done, w)
        A += [action]
        x = X(S[t], done, A[t])
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0
        while not done:
            action = epsilon_greedy_policy(state, done, w)
            A += [action]
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
            t += 1
