"""
Online Balanced K-means Implementation
Based on arXiv:2410.17256 - Inference with K-means
"""
import numpy as np
from typing import Optional, Tuple, List


class OnlineBalancedKMeans:
    """
    Online Balanced K-means clustering algorithm.
    
    Combines online k-means (sequential point processing) with balanced k-means
    (weight adjustment to balance cluster sizes).
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters (k)
    alpha : float
        Learning rate for centroid updates (default: 0.1)
    beta : float
        Weight adjustment parameter for balancing (default: 0.5)
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self, 
        n_clusters: int, 
        alpha: float = 0.1, 
        beta: float = 0.5,
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        
        # Initialize attributes
        self.centroids: Optional[np.ndarray] = None
        self.cluster_counts: Optional[np.ndarray] = None
        self.cluster_weights: Optional[np.ndarray] = None
        self.n_features: Optional[int] = None
        self._is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_centroids(self, X: np.ndarray) -> None:
        """Initialize centroids randomly from data points."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()
        self.n_features = X.shape[1]
        self.cluster_counts = np.ones(self.n_clusters)  # Start with 1 to avoid division by zero
        self.cluster_weights = np.zeros(self.n_clusters)
    
    def _compute_distance(self, x: np.ndarray, centroid: np.ndarray) -> float:
        """Compute Euclidean distance between point and centroid."""
        return np.sqrt(np.sum((x - centroid) ** 2))
    
    def _assign_point(self, x: np.ndarray) -> int:
        """
        Assign point to cluster based on distance minus weight.
        
        The assignment rule in Online Balanced K-means differs from standard K-means.
        It incorporates a weight term (w_i) to penalize or encourage assignment to specific clusters
        based on their size, promoting balance.
        
        Formula: x ∈ c_i ⟺ ∀j≠i: D(x, μ_i) - w_i ≤ D(x, μ_j) - w_j
        
        Where:
        - D(x, μ_i) is the Euclidean distance between point x and centroid μ_i
        - w_i is the weight of cluster i
        """
        adjusted_distances = np.array([
            self._compute_distance(x, self.centroids[i]) - self.cluster_weights[i]
            for i in range(self.n_clusters)
        ])
        return int(np.argmin(adjusted_distances))
    
    def _update_centroid(self, cluster_idx: int, x: np.ndarray) -> None:
        """
        Update centroid using online learning rule.
        
        Instead of recomputing the mean of all points in a batch, the centroid is
        updated incrementally as each new point arrives. This allows the algorithm
        to adapt to data streams.
        
        Formula: μ_i ← α * x + (1 - α) * μ_i
        
        Where:
        - α (alpha) is the learning rate (0 < α ≤ 1). Higher α means faster adaptation but more noise.
        - x is the new data point assigned to cluster i.
        """
        self.centroids[cluster_idx] = (
            self.alpha * x + (1 - self.alpha) * self.centroids[cluster_idx]
        )
    
    def _update_weights(self) -> None:
        """
        Update cluster weights for balancing.
        
        The weights are updated to penalize clusters that are becoming too large relative
        to the average cluster size. This mechanism forces the algorithm to distribute
        points more evenly across clusters.
        
        Formula: w_i = β * (n_i - E[n]) / V[n]
        
        Where:
        - β (beta) is the balancing parameter.
        - n_i is the number of points currently in cluster i.
        - E[n] is the expected (mean) number of points per cluster.
        - V[n] is the variance of the cluster counts (used for normalization).
        
        Note: If variance is zero (all clusters equal), weights become zero.
        """
        mean_count = np.mean(self.cluster_counts)
        var_count = np.var(self.cluster_counts)
        
        if var_count > 0:
            self.cluster_weights = self.beta * (self.cluster_counts - mean_count) / np.sqrt(var_count)
        else:
            self.cluster_weights = np.zeros(self.n_clusters)
    
    def fit_point(self, x: np.ndarray) -> int:
        """
        Process a single point in online fashion.
        
        Parameters:
        -----------
        x : np.ndarray
            Single data point of shape (n_features,)
            
        Returns:
        --------
        int : Assigned cluster index
        """
        # Assign point to cluster
        cluster_idx = self._assign_point(x)
        
        # Update cluster count
        self.cluster_counts[cluster_idx] += 1
        
        # Update centroid
        self._update_centroid(cluster_idx, x)
        
        # Update weights
        self._update_weights()
        
        return cluster_idx
    
    def fit(self, X: np.ndarray, n_iterations: int = 1) -> 'OnlineBalancedKMeans':
        """
        Fit the model on data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        n_iterations : int
            Number of passes through the data
            
        Returns:
        --------
        self
        """
        if not self._is_fitted:
            self._initialize_centroids(X)
            self._is_fitted = True
        
        for _ in range(n_iterations):
            # Shuffle data for each iteration
            indices = np.random.permutation(len(X))
            for idx in indices:
                self.fit_point(X[idx])
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for data points.
        
        Parameters:
        -----------
        X : np.ndarray
            Data of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Cluster labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        labels = np.array([self._assign_point(x) for x in X])
        return labels
    
    def get_closest_centroids(
        self, 
        x: np.ndarray, 
        n_closest: int = 5,
        use_partial: bool = False,
        n_features_partial: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the n closest centroids to a point.
        
        Parameters:
        -----------
        x : np.ndarray
            Query point
        n_closest : int
            Number of closest centroids to return
        use_partial : bool
            If True, use only first n_features_partial features for distance
        n_features_partial : int, optional
            Number of features to use for partial distance
            
        Returns:
        --------
        Tuple of (centroid_indices, distances, centroids)
        """
        if use_partial and n_features_partial is not None:
            x_partial = x[:n_features_partial]
            distances = np.array([
                self._compute_distance(x_partial, self.centroids[i][:n_features_partial])
                for i in range(self.n_clusters)
            ])
        else:
            distances = np.array([
                self._compute_distance(x, self.centroids[i])
                for i in range(self.n_clusters)
            ])
        
        closest_indices = np.argsort(distances)[:n_closest]
        return closest_indices, distances[closest_indices], self.centroids[closest_indices]
    
    def compute_loss(self, X: np.ndarray) -> float:
        """
        Compute the k-means loss (within-cluster sum of squares).
        
        Parameters:
        -----------
        X : np.ndarray
            Data points
            
        Returns:
        --------
        float : Total loss
        """
        labels = self.predict(X)
        loss = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                loss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return loss
