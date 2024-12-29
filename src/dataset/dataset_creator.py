import numpy as np
import pandas as pd
from tqdm import tqdm


class SyntheticDataGenerator:
    def __init__(self, max_discontinuities, num_samples, num_features, end_time=10, num_t_points=100):
        self.max_discontinuities = max_discontinuities
        self.num_samples = num_samples
        self.num_features = num_features
        self.end_time = end_time
        self.num_t_points = num_t_points

    def T_k(self, x: np.ndarray, k: int) -> float:
        """
        Function that defines the time discontinuity for each k, with weight applied to x.
        """
        x_norm = np.linalg.norm(x, 2) / np.sqrt(len(x))
        return k * x_norm + k**2 * np.max(x)

    def Delta_k(self, x: np.ndarray, k: int) -> float:
        """
        Function that defines the change at each discontinuity, with weight applied to x.
        """
        x_norm = np.linalg.norm(x, 1) / len(x)
        return np.exp(x_norm) / k

    def h(self, x: np.ndarray) -> float:
        """
        A custom function based on the Euclidean norm of weighted x.
        """
        return np.cos(np.linalg.norm(x, 2) / np.sqrt(len(x)))

    def g(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        A simple sine function for time-dependent behavior.
        """
        return np.sin(np.linalg.norm(x, 2) * t)

    def f(self, t: np.ndarray, x: np.ndarray, K: int) -> np.ndarray:
        """
        Function that combines the smooth components and discontinuities, with weight applied to x.
        """
        y = self.g(x, t) + self.h(x)
        for k in range(1, K + 1):
            if t >= self.T_k(x, k):
                y += self.Delta_k(x, k)
        return y

    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic data for multiple samples with discontinuities and weights.
        """
        x_data = np.random.uniform(-1, 1, size=(self.num_samples, self.num_features))
        t_values = np.linspace(0, self.end_time, self.num_t_points)
        
        dataset = []
        for idx, x in tqdm(enumerate(x_data), total=self.num_samples):
            
            den = np.linalg.norm(x, 2) / np.sqrt(len(x)) + 2 * np.max(x)
            
            k = np.random.randint(0, max(1, min(self.max_discontinuities, (self.end_time // den))))

            for t in t_values:
                y = self.f(t, x, k)
                dataset.append({
                    'function': idx,
                    'num_discontinuities': k,
                    't': t,
                    **{f'x{i+1}': x[i] for i in range(self.num_features)},
                    'y': y
                })

        return pd.DataFrame(dataset)