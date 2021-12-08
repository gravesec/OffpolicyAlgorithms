from Algorithms.BaseTD import BaseTD
import numpy as np


class TDRCEETD(BaseTD):
    """
    An implementation of ETD using a TDRC-like algorithm to estimate emphasis.
    """
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.beta = kwargs.get('beta')  # Chance of not terminating.
        self.tdrce_alpha = kwargs['gem_alpha']  # Step size for main emphasis weights.
        self.eta = kwargs['eta']  # Parameter to determine step size for auxiliary weights.
        self.tdrc_beta = kwargs['tdrc_beta']  # Regularization parameter for auxiliary emphasis weights.

        self.u = np.zeros(self.task.num_features)  # Main emphasis weights.
        self.h = np.zeros(self.task.num_features)  # Auxiliary emphasis weights.

    @staticmethod
    def related_parameters():
        return ['alpha', 'beta', 'gem_alpha', 'eta', 'tdrc_beta']

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)

        # ETD(0) update:
        m = np.dot(self.u, x)  # Use parametric estimate of expected emphasis.
        rho = self.get_isr(s)
        delta = self.get_delta(r, x, x_p)
        self.w += self.alpha * m * rho * delta * x

        # TDRC(0) for estimating emphasis:
        delta_bar = (1 - self.beta) + self.beta * np.dot(self.u, x) - np.dot(self.u, x_p)
        self.u += self.tdrce_alpha * rho * (delta_bar * x_p - self.beta * np.dot(self.h, x_p) * x)
        self.h += self.eta * self.tdrce_alpha * rho * (delta_bar - np.dot(self.h, x_p)) * x_p - self.eta * self.tdrce_alpha * self.tdrc_beta * self.h

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        raise NotImplementedError
