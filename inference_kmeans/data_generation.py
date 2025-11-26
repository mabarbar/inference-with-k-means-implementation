"""
Data Generation Approaches
Based on arXiv:2410.17256 - Inference with K-means
"""
import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_blobs, make_moons


class DataGenerator:
    """
    Data generator for testing inference methods with k-means.
    
    Generates data where the last feature/column can be inferred from
    the preceding features.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    # Uniform Distribution Data
    
    def generate_uniform(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data entirely from uniform distribution U(-1, 1).
        
        Paper Section 3.3.1:
        f(x) = 1/2 where -1 <= x <= 1
        """
        return np.random.uniform(-1, 1, size=(n_samples, n_features))
    
    def generate_uniform_squared(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from uniform distribution with last feature
        being sum of squares of preceding features.
        
        Paper Section 3.4.1 (b):
        Inputs: x_0, ..., x_{d-2} ~ U(-1, 1)
        Target: x_{d-1} = Σ(x_i^2)
        """
        X = np.random.uniform(-1, 1, size=(n_samples, n_features - 1))
        last_col = np.sum(X ** 2, axis=1, keepdims=True)
        return np.hstack([X, last_col])
    
    def generate_uniform_cube(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from uniform distribution with last feature
        being sum of cubes of preceding features.
        
        Paper Section 3.4.1 (c):
        Inputs: x_0, ..., x_{d-2} ~ U(-1, 1)
        Target: x_{d-1} = Σ(x_i^3)
        """
        X = np.random.uniform(-1, 1, size=(n_samples, n_features - 1))
        last_col = np.sum(X ** 3, axis=1, keepdims=True)
        return np.hstack([X, last_col])
    
    # Normal Distribution Data
    
    def generate_normal(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data entirely from normal distribution N(0, 1).
        
        Paper Section 3.3.2:
        f(x) = (1/√(2π)) * e^(-x^2/2)
        """
        return np.random.normal(0, 1, size=(n_samples, n_features))
    
    def generate_normal_squared(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from normal distribution with last feature
        being sum of squares of preceding features.
        
        Paper Section 3.4.2 (b):
        Inputs: x_0, ..., x_{d-2} ~ N(0, 1)
        Target: x_{d-1} = Σ(x_i^2)
        """
        X = np.random.normal(0, 1, size=(n_samples, n_features - 1))
        last_col = np.sum(X ** 2, axis=1, keepdims=True)
        return np.hstack([X, last_col])
    
    def generate_normal_cube(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from normal distribution with last feature
        being sum of cubes of preceding features.
        
        Paper Section 3.4.2 (c):
        Inputs: x_0, ..., x_{d-2} ~ N(0, 1)
        Target: x_{d-1} = Σ(x_i^3)
        """
        X = np.random.normal(0, 1, size=(n_samples, n_features - 1))
        last_col = np.sum(X ** 3, axis=1, keepdims=True)
        return np.hstack([X, last_col])
    
    # Gamma Distribution Data
    
    def generate_gamma(
        self, 
        n_samples: int, 
        n_features: int,
        shape: float = 1.0,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Generate data entirely from gamma distribution Γ(shape, scale).
        
        Paper Section 3.3.3:
        f(x) = e^(-x) where x > 0 (for shape=1, scale=1)
        """
        return np.random.gamma(shape, scale, size=(n_samples, n_features))
    
    def generate_gamma_squared(
        self, 
        n_samples: int, 
        n_features: int,
        shape: float = 1.0,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Generate data from gamma distribution with last feature
        being sum of squares of preceding features.
        
        Paper Section 3.4.3 (b):
        Inputs: x_0, ..., x_{d-2} ~ Γ(1, 1)
        Target: x_{d-1} = Σ(x_i^2)
        """
        X = np.random.gamma(shape, scale, size=(n_samples, n_features - 1))
        last_col = np.sum(X ** 2, axis=1, keepdims=True)
        return np.hstack([X, last_col])
    
    def generate_gamma_cube(
        self, 
        n_samples: int, 
        n_features: int,
        shape: float = 1.0,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Generate data from gamma distribution with last feature
        being sum of cubes of preceding features.
        
        Paper Section 3.4.3 (c):
        Inputs: x_0, ..., x_{d-2} ~ Γ(1, 1)
        Target: x_{d-1} = Σ(x_i^3)
        """
        X = np.random.gamma(shape, scale, size=(n_samples, n_features - 1))
        last_col = np.sum(X ** 3, axis=1, keepdims=True)
        return np.hstack([X, last_col])
    
    # Multi-Cluster Data
    
    def generate_blobs(
        self, 
        n_samples: int, 
        n_features: int,
        n_centers: int = 3,
        cluster_std: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate blob-shaped clusters using sklearn.
        """
        X, labels = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        return X, labels
    
    def generate_moons(
        self, 
        n_samples: int,
        noise: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate moon-shaped clusters.
        """
        X, labels = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=self.random_state
        )
        return X, labels
    
    def generate_three_uniform_clusters(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from three different uniform distributions.
        """
        samples_per_cluster = n_samples // 3
        
        # Cluster 1: U(-3, -1)
        X1 = np.random.uniform(-3, -1, size=(samples_per_cluster, n_features))
        
        # Cluster 2: U(-1, 1)
        X2 = np.random.uniform(-1, 1, size=(samples_per_cluster, n_features))
        
        # Cluster 3: U(1, 3)
        X3 = np.random.uniform(1, 3, size=(n_samples - 2 * samples_per_cluster, n_features))
        
        return np.vstack([X1, X2, X3])
    
    def generate_three_normal_clusters(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from three different normal distributions.
        """
        samples_per_cluster = n_samples // 3
        
        # Cluster 1: N(-3, 0.5)
        X1 = np.random.normal(-3, 0.5, size=(samples_per_cluster, n_features))
        
        # Cluster 2: N(0, 0.5)
        X2 = np.random.normal(0, 0.5, size=(samples_per_cluster, n_features))
        
        # Cluster 3: N(3, 0.5)
        X3 = np.random.normal(3, 0.5, size=(n_samples - 2 * samples_per_cluster, n_features))
        
        return np.vstack([X1, X2, X3])
    
    def generate_mixed_clusters(
        self, 
        n_samples: int, 
        n_features: int
    ) -> np.ndarray:
        """
        Generate data from two gamma distributions and one normal distribution.
        """
        samples_per_cluster = n_samples // 3
        
        # Cluster 1: Gamma(2, 1)
        X1 = np.random.gamma(2, 1, size=(samples_per_cluster, n_features))
        
        # Cluster 2: Gamma(5, 0.5) shifted
        X2 = np.random.gamma(5, 0.5, size=(samples_per_cluster, n_features)) + 5
        
        # Cluster 3: Normal(10, 1)
        X3 = np.random.normal(10, 1, size=(n_samples - 2 * samples_per_cluster, n_features))
        
        return np.vstack([X1, X2, X3])
    
    def get_all_generators(self) -> dict:
        """Return dictionary of all data generation methods."""
        return {
            # Single cluster data
            'uniform': self.generate_uniform,
            'uniform_squared': self.generate_uniform_squared,
            'uniform_cube': self.generate_uniform_cube,
            'normal': self.generate_normal,
            'normal_squared': self.generate_normal_squared,
            'normal_cube': self.generate_normal_cube,
            'gamma': self.generate_gamma,
            'gamma_squared': self.generate_gamma_squared,
            'gamma_cube': self.generate_gamma_cube,
            # Multi-cluster data
            'three_uniform': self.generate_three_uniform_clusters,
            'three_normal': self.generate_three_normal_clusters,
            'mixed_clusters': self.generate_mixed_clusters,
        }
