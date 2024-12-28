import numpy as np
import matplotlib.pyplot as plt

class JumpDiscontinuityGenerator:
    def __init__(self, num_features, num_space_jumps=2, num_time_jumps=2, noise_level=0.1):
        """
        Initialize the generator for functions with both space-only and space-time jumps.
        
        Args:
            num_features (int): Number of input features (M)
            num_space_jumps (int): Number of space-only discontinuities
            num_time_jumps (int): Number of space-time discontinuities
            noise_level (float): Amount of noise to add to the function
        """
        self.num_features = num_features
        self.num_space_jumps = num_space_jumps
        self.num_time_jumps = num_time_jumps
        self.noise_level = noise_level
        
        # Generate parameters for space-only jumps (based only on X)
        self.space_jump_normals = np.random.randn(num_space_jumps, num_features)
        self.space_jump_thresholds = np.random.randn(num_space_jumps)
        self.space_jump_magnitudes = np.random.uniform(0.5, 2.0, num_space_jumps)
        
        # Generate parameters for space-time jumps
        self.time_jump_normals = np.random.randn(num_time_jumps, num_features)
        self.time_jump_thresholds = np.random.randn(num_time_jumps)
        self.time_frequencies = np.random.uniform(1.0, 5.0, num_time_jumps)
        self.time_jump_magnitudes = np.random.uniform(0.5, 2.0, num_time_jumps)
        
        # Generate base function coefficients
        self.base_coeffs = np.random.randn(num_features)
        
    def evaluate(self, X, t):
        """
        Evaluate the function with both space-only and space-time jumps.
        
        Args:
            X (numpy.ndarray): Input points of shape (N, M)
            t (float): Time parameter
            
        Returns:
            numpy.ndarray: Function values at the input points
        """
        # Start with base linear function
        y = np.dot(X, self.base_coeffs)
        
        
        # Add space-time jumps
        for normal, threshold, freq, magnitude in zip(
            self.time_jump_normals,
            self.time_jump_thresholds,
            self.time_frequencies,
            self.time_jump_magnitudes
        ):
            # Combined space-time condition
            space_condition = np.dot(X, normal) - threshold
            time_condition = np.sin(2 * np.pi * freq * t)
            
            # Jump occurs when both space and time conditions are met
            y += magnitude * ((space_condition > 0) & (time_condition > 0)).astype(float)
            
        # Add noise
        if self.noise_level > 0:
            y += np.random.normal(0, self.noise_level, size=y.shape)
            
        return y
    
    def get_space_jump_labels(self, X):
        """
        Get binary labels indicating whether each point is above any space-only jump hyperplane.
        
        Args:
            X (numpy.ndarray): Input points of shape (N, M)
            
        Returns:
            numpy.ndarray: Binary labels of shape (N, num_space_jumps)
        """
        labels = np.zeros((X.shape[0], self.num_space_jumps))
        for i, (normal, threshold) in enumerate(zip(
            self.space_jump_normals, 
            self.space_jump_thresholds
        )):
            side = np.dot(X, normal) - threshold
            labels[:, i] = (side > 0).astype(int)
        return labels
    
    def generate_dataset(self, num_points, num_times):
        """
        Generate a dataset of points with their function values at different times.
        
        Args:
            num_points (int): Number of points to generate
            num_times (int): Number of time steps
            
        Returns:
            tuple: (X, t, y) where:
                X: Input points of shape (num_points, num_features)
                t: Time points of shape (num_times,)
                y: Function values of shape (num_points, num_times)
        """
        # Generate random points
        X = np.random.randn(num_points, self.num_features)
        
        # Generate time points
        t = np.linspace(0, 1, num_times)
        
        # Evaluate function at all points and times
        y = np.zeros((num_points, num_times))
        for i, ti in enumerate(t):
            y[:, i] = self.evaluate(X, ti)
            
        return X, t, y
    
if __name__ == "__main__":


    # Create an instance of the JumpDiscontinuityGenerator
    num_features = 10
    num_space_jumps = 0
    num_time_jumps = 3
    noise_level = 0.1

    generator = JumpDiscontinuityGenerator(num_features, num_space_jumps, num_time_jumps, noise_level)

    # Generate a dataset
    num_points = 1  # Number of points to visualize
    num_times = 1000  # Number of time steps
    X, t, y = generator.generate_dataset(num_points, num_times)

    # Plot y vs t for each point
    plt.figure(figsize=(12, 8))
    for i in range(num_points):
        plt.plot(t, y[i, :], label=f"Point {i + 1}")

    plt.xlabel("Time (t)")
    plt.ylabel("Function Value (y)")
    plt.title("Function Values (y) vs Time (t) for Random Points")
    plt.legend()
    plt.grid(True)
    plt.show()
