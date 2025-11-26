import numpy as np
from typing import Optional
from .online_balanced_kmeans import OnlineBalancedKMeans


class KMeansInference:
    """
    Inference methods for predicting the last component of data points using fitted Online Balanced K-means model.
    
    Given x_i = (x_0, x_1, ..., x_{d-2}, x_{d-1}), the goal is to
    estimate x_{d-1} using the first d-1 components.
    
    Parameters:
    -----------
    model : OnlineBalancedKMeans
        Fitted k-means model
    n_closest : int
        Number of closest centroids to use for inference (default: 5)
    """
    
    def __init__(
        self, 
        model: OnlineBalancedKMeans,
        n_closest: int = 5
    ):
        self.model = model
        self.n_closest = min(n_closest, model.n_clusters)
    
    def _get_partial_distances(
        self, 
        x_partial: np.ndarray
    ) -> tuple:
        """
        Get distances from partial point (all but last feature) to centroids.
        
        Returns indices of closest centroids and their distances.
        """
        n_features_partial = len(x_partial)
        centroids_partial = self.model.centroids[:, :n_features_partial]
        
        distances = np.sqrt(np.sum((centroids_partial - x_partial) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:self.n_closest]
        
        return closest_indices, distances
    
    # Method 1: Euclidean Distance Approach
    
    def euclidean_distance_approach(self, x_partial: np.ndarray) -> float:
        """
        Method 3.6.1: The Euclidean distance approach
        
        Estimate last component based on the closest centroid.
        
        Logic:
        1. Identify the single closest centroid μ_k to the partial input x_partial.
        2. Predict x_{d-1} = μ_k, {d-1} (the last component of that centroid).
        
        This is the simplest form of inference, assuming the point belongs strictly
        to the nearest cluster prototype.
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        closest_indices, _ = self._get_partial_distances(x_partial)
        closest_centroid = self.model.centroids[closest_indices[0]]
        return closest_centroid[-1]
    
    # Method 2: Normalized Weights Approach
    
    def normalized_weights_approach(self, x_partial: np.ndarray) -> float:
        """
        Method 3.6.2: The normalized weights approach
        
        Estimate last component using distance-based probability weights.
        
        Logic:
        1. Find the n closest centroids.
        2. Calculate weights inversely proportional to the Euclidean distance.
           Weight w_j = (1 / d(x, μ_j)) / Σ(1 / d(x, μ_k))
        3. Predict x_{d-1} = Σ(w_j * μ_j, {d-1})
        
        This method gives more influence to centroids that are closer to the point.
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        closest_indices, distances = self._get_partial_distances(x_partial)
        closest_distances = distances[closest_indices]
        
        # Compute inverse distance weights (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        inv_distances = 1.0 / (closest_distances + epsilon)
        weights = inv_distances / np.sum(inv_distances)
        
        # Weighted average of last components
        last_components = self.model.centroids[closest_indices, -1]
        return np.sum(weights * last_components)
    
    # Method 3: Cluster Size Approach
    
    def cluster_size_approach(self, x_partial: np.ndarray) -> float:
        """
        Method 3.6.3: The cluster size approach
        
        Estimate last component using cluster sizes as weights.
        
        Logic:
        1. Find the n closest centroids.
        2. Retrieve the size (number of points) of each corresponding cluster.
        3. Calculate weights based on relative cluster sizes:
           Weight w_j = size(C_j) / Σ(size(C_k))
        4. Predict x_{d-1} = Σ(w_j * μ_j, {d-1})
        
        This method assumes that larger clusters are more likely to contain the point.
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        closest_indices, _ = self._get_partial_distances(x_partial)
        
        # Get cluster sizes for closest centroids
        cluster_sizes = self.model.cluster_counts[closest_indices]
        weights = cluster_sizes / np.sum(cluster_sizes)
        
        # Weighted average of last components
        last_components = self.model.centroids[closest_indices, -1]
        return np.sum(weights * last_components)
    
    # Method 4: Overall Mean + Normalized Weights
    
    def overall_mean_normalized_weights(
        self, 
        x_partial: np.ndarray,
        overall_mean: float,
        lambda_param: float = 0.5
    ) -> float:
        """
        Method 3.6.4: The overall mean and the normalized weights approach
        
        Combine overall mean with normalized weights approach.
        
        Logic:
        1. Compute estimate using Method 2 (Normalized Weights).
        2. Combine with the global mean of the last component across all centroids.
        3. Prediction = λ * Overall_Mean + (1-λ) * Method2_Estimate
        
        This acts as a regularization term, pulling estimates towards the global average.
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
        overall_mean : float
            Mean of the last component in training data
        lambda_param : float
            Mixing parameter (default: 0.5)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        nw_estimate = self.normalized_weights_approach(x_partial)
        return lambda_param * overall_mean + (1 - lambda_param) * nw_estimate
    
    # Method 5: Normalized Weights + Cluster Size
    
    def merge_normalized_weights_cluster_size(
        self, 
        x_partial: np.ndarray,
        lambda_param: float = 0.5
    ) -> float:
        """
        Method 3.6.5: Merging the normalized weights and the cluster size approach
        
        Merge normalized weights and cluster size approaches.
        
        Logic:
        1. Compute estimate using Method 2 (Normalized Weights).
        2. Compute estimate using Method 3 (Cluster Size).
        3. Prediction = λ * Method2_Estimate + (1-λ) * Method3_Estimate
        
        This balances proximity (Method 2) with cluster popularity (Method 3).
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
        lambda_param : float
            Mixing parameter (default: 0.5)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        nw_estimate = self.normalized_weights_approach(x_partial)
        cs_estimate = self.cluster_size_approach(x_partial)
        return lambda_param * nw_estimate + (1 - lambda_param) * cs_estimate
    
    # Method 6: Normalized Weights + Euclidean Distance
    
    def merge_normalized_weights_euclidean(
        self, 
        x_partial: np.ndarray,
        lambda_param: float = 0.5
    ) -> float:
        """
        Method 3.6.6: Merging the normalized weights and the Euclidean distance approach
        
        Merge normalized weights and Euclidean distance approaches.
        
        Logic:
        1. Compute estimate using Method 2 (Normalized Weights).
        2. Compute estimate using Method 1 (Euclidean Distance / Nearest Neighbor).
        3. Prediction = λ * Method2_Estimate + (1-λ) * Method1_Estimate
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
        lambda_param : float
            Mixing parameter (default: 0.5)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        nw_estimate = self.normalized_weights_approach(x_partial)
        ed_estimate = self.euclidean_distance_approach(x_partial)
        return lambda_param * nw_estimate + (1 - lambda_param) * ed_estimate
    
    # Method 7: Cluster Size with Exponential Weights
    
    def cluster_size_exponential_weights(
        self, 
        x_partial: np.ndarray,
        beta: float = 1.0
    ) -> float:
        """
        Method 3.6.7: Cluster size with exponential weights approach
        
        Cluster size approach with exponential distance weights.
        
        Logic:
        1. Introduce exponential decay based on distance.
        2. Weight w_j = Size(C_j) * exp(-β * d(x, μ_j))
        3. Normalize weights and compute weighted average.
        
        This refines the cluster size approach by penalizing far-away clusters exponentially.
        
        Parameters:
        -----------
        x_partial : np.ndarray
            Known components (all but last) of shape (d-1,)
        beta : float
            Exponential decay parameter (default: 1.0)
            
        Returns:
        --------
        float : Estimated value for the last component
        """
        closest_indices, distances = self._get_partial_distances(x_partial)
        closest_distances = distances[closest_indices]
        
        # Get cluster sizes
        cluster_sizes = self.model.cluster_counts[closest_indices]
        
        # Compute exponential weights
        exp_weights = np.exp(-beta * closest_distances)
        combined_weights = cluster_sizes * exp_weights
        weights = combined_weights / np.sum(combined_weights)
        
        # Weighted average of last components
        last_components = self.model.centroids[closest_indices, -1]
        return np.sum(weights * last_components)
    
    # ==================== Batch Inference ====================
    
    def infer_batch(
        self, 
        X_partial: np.ndarray, 
        method: str = 'euclidean',
        overall_mean: Optional[float] = None,
        lambda_param: float = 0.5,
        beta: float = 1.0
    ) -> np.ndarray:
        """
        Perform inference on a batch of partial data points.
        
        Parameters:
        -----------
        X_partial : np.ndarray
            Partial data points of shape (n_samples, d-1)
        method : str
            Inference method to use:
            - 'euclidean': Euclidean distance approach
            - 'normalized_weights': Normalized weights approach
            - 'cluster_size': Cluster size approach
            - 'mean_normalized': Overall mean + normalized weights
            - 'merge_nw_cs': Merge normalized weights + cluster size
            - 'merge_nw_ed': Merge normalized weights + Euclidean
            - 'exponential': Cluster size with exponential weights
        overall_mean : float, optional
            Required for 'mean_normalized' method
        lambda_param : float
            Mixing parameter for merge methods
        beta : float
            Exponential decay parameter
            
        Returns:
        --------
        np.ndarray : Estimated last components
        """
        method_map = {
            'euclidean': lambda x: self.euclidean_distance_approach(x),
            'normalized_weights': lambda x: self.normalized_weights_approach(x),
            'cluster_size': lambda x: self.cluster_size_approach(x),
            'mean_normalized': lambda x: self.overall_mean_normalized_weights(
                x, overall_mean, lambda_param
            ),
            'merge_nw_cs': lambda x: self.merge_normalized_weights_cluster_size(
                x, lambda_param
            ),
            'merge_nw_ed': lambda x: self.merge_normalized_weights_euclidean(
                x, lambda_param
            ),
            'exponential': lambda x: self.cluster_size_exponential_weights(x, beta),
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available: {list(method_map.keys())}")
        
        inference_func = method_map[method]
        return np.array([inference_func(x) for x in X_partial])
    
    def get_all_methods(self) -> list:
        return [
            'euclidean',
            'normalized_weights', 
            'cluster_size',
            'mean_normalized',
            'merge_nw_cs',
            'merge_nw_ed',
            'exponential'
        ]
