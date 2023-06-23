import numpy as np
from scipy.special import logsumexp

class PoissonHMM():
    def __init__(self, X, y, n_states, trial_id, pi) -> None: 
        """Poisson GLM-HMM in which Poisson GLM weights are state dependent

        Args:
            X (_type_): Design Matrix (n_samples, n_features)
            y (_type_): emissions/observations (n_samples, n_neurons)
            n_states (_type_): number of HMM states
            trial_id (_type_): array identifying the trial ID of each sample/row of X
            pi (_type_): initial distribution (n_states,)
        """
        self.X, self.y = X, y
        self.n_trials = np.unique(trial_id).size
        self.n_states = n_states
        self.pi = pi
        self.T = None
        self.scale = np.empty(self.n_trials)

    def forward(self, ll):
        alpha = np.empty((self.n_states, self.n_trials))
        prior = np.empty(alpha.shape)

        for t in range(0, self.n_trials):
            if t == 0:
                prior[:, t] = self.pi
            else:
                prior[:, t] = logsumexp(self.T.T + alpha[:, t-1], axis=1)

        update = prior[:, t] + ll[:, t]
        self.scale[t] = logsumexp(update)
        alpha[:, t] = update - self.scale[t]

        return alpha, prior
    
    def backward(self):
        beta = np.empty((self.n_states, self.n_trials))
        beta[:, -1] = 0
        for t in range(self.n_trials-2, -1, -1):
            beta[:, t] = logsumexp(self.T + beta[:, t+1], axis=1) - self.scale[t+1]
        
        return beta
    
    def smooth(self, alpha, beta, ll):
        gamma = alpha + beta
        gamma -= logsumexp(gamma, axis=0)
        xi = np.empty((self.n_trials, self.n_states, self.n_states))
        for t in range(1, self.n_trials):
            xi[t] = alpha[:, t-1].reshape(-1, 1) + (ll[:, t] + beta[:, t])[None, :] + self.T
        return gamma
    


