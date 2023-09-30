import numpy as np

# Ornstein Uhlenbeck noise
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        """
        Construct OU noise based on the provided parameters
        :param size: The action dimension
        :param mu: Long-term mean
        :param theta: Mean reversion rate
        :param sigma: Volatility
        """
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the noise
        """
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """
        Randomly sampling the noise
        :return: The sampled noise
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state