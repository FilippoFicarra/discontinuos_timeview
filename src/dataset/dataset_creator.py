import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    def __init__(self, max_discontinuities, num_samples, num_features, end_time=10, num_t_points=100):
        self.max_discontinuities = max_discontinuities
        self.num_samples = num_samples
        self.num_features = num_features
        self.end_time = end_time
        self.num_t_points = num_t_points

    def T_k(self, x: np.ndarray, k: int, w: np.ndarray) -> float:
        """
        Function that defines the time discontinuity for each k, with weight applied to x.
        """
        weighted_x = x * w  # Apply weight to x
        x_norm = np.linalg.norm(weighted_x, 2) / np.sqrt(len(weighted_x))  # Normalize Euclidean norm
        return k * x_norm + k**2 * np.max(weighted_x)

    def Delta_k(self, x: np.ndarray, k: int, w: np.ndarray) -> float:
        """
        Function that defines the change at each discontinuity, with weight applied to x.
        """
        weighted_x = x * w  # Apply weight to x
        x_norm = np.linalg.norm(weighted_x, 1) / len(weighted_x)  # Normalize L1 norm
        return np.exp(x_norm) / k

    def h(self, x: np.ndarray, w: np.ndarray) -> float:
        """
        A custom function based on the Euclidean norm of weighted x.
        """
        weighted_x = x * w
        return np.cos(np.linalg.norm(weighted_x, 2) / np.sqrt(len(weighted_x)))

    def g(self, t: np.ndarray) -> np.ndarray:
        """
        A simple sine function for time-dependent behavior.
        """
        return np.sin(t)

    def f(self, t: np.ndarray, x: np.ndarray, K: int, w: np.ndarray) -> np.ndarray:
        """
        Function that combines the smooth components and discontinuities, with weight applied to x.
        """
        y = self.g(t) + self.h(x, w)
        for k in range(1, K + 1):
            if t >= self.T_k(x, k, w):
                y += self.Delta_k(x, k, w)
        return y

    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic data for multiple samples with discontinuities and weights.
        """
        x_data = np.random.uniform(-1, 1, size=(self.num_samples, self.num_features))
        weights = np.random.uniform(0.1, 2, size=(self.num_samples, self.num_features))  # Random weights for each sample
        t_values = np.linspace(0, self.end_time, self.num_t_points)
        
        dataset = []
        ks = []
        for idx, x in enumerate(x_data):
            w = weights[idx]  # Get corresponding weight for each x vector
            k = np.random.randint(0, self.max_discontinuities + 1)

            for t in t_values:
                y = self.f(t, x, k, w)
                ks.append(k)
                dataset.append({
                    'k': k,
                    't': t,
                    **{f'x{i+1}': x[i] for i in range(self.num_features)},
                    **{f'w{i+1}': w[i] for i in range(self.num_features)},  # Include weights in the dataset
                    'y': y
                })

        return pd.DataFrame(dataset)