"""
Evaluation Metrics for Inference Methods
Based on arXiv:2410.17256 - Inference with K-means
"""
import numpy as np
from typing import Dict
from .online_balanced_kmeans import OnlineBalancedKMeans
from .inference_methods import KMeansInference


class Evaluator:
    """
    Evaluation metrics for k-means inference methods.
    
    Section 3.7: Errors and Losses
    The performance is determined by how well the inference method estimates x_{d-1}.
    This class implements the standard error metrics used in the paper.
    """
    
    @staticmethod
    def squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute total squared error.
        
        Paper Section 3.7.1:
        "Hence we use the squared distance approach to evaluate the total errors..."
        Error = Σ(y_true - y_pred)²
        """
        return np.sum((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute mean squared error.
        
        Standard metric for average squared deviation.
        MSE = (1/n) * Σ(y_true - y_pred)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute root mean squared error.
        
        RMSE puts the error back into the same units as the target variable.
        RMSE = sqrt(MSE)
        """
        return np.sqrt(Evaluator.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute mean absolute error.
        
        MAE measures average magnitude of errors without penalizing large errors as heavily as MSE.
        MAE = (1/n) * Σ|y_true - y_pred|
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def evaluate_all_methods(
        model: OnlineBalancedKMeans,
        X_test: np.ndarray,
        n_closest: int = 5,
        lambda_param: float = 0.5,
        beta: float = 1.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all inference methods on test data.
        
        Parameters:
        -----------
        model : OnlineBalancedKMeans
            Fitted k-means model
        X_test : np.ndarray
            Test data of shape (n_samples, n_features)
        n_closest : int
            Number of closest centroids to use
        lambda_param : float
            Mixing parameter for merge methods
        beta : float
            Exponential decay parameter
            
        Returns:
        --------
        Dict : Results for each method containing MSE, RMSE, MAE
        """
        inference = KMeansInference(model, n_closest)
        
        # Separate partial (input) and target (last component)
        X_partial = X_test[:, :-1]
        y_true = X_test[:, -1]
        
        # Compute overall mean for mean_normalized method
        overall_mean = np.mean(model.centroids[:, -1])
        
        results = {}
        methods = inference.get_all_methods()
        
        for method in methods:
            y_pred = inference.infer_batch(
                X_partial, 
                method=method,
                overall_mean=overall_mean,
                lambda_param=lambda_param,
                beta=beta
            )
            
            results[method] = {
                'squared_error': Evaluator.squared_error(y_true, y_pred),
                'mse': Evaluator.mean_squared_error(y_true, y_pred),
                'rmse': Evaluator.root_mean_squared_error(y_true, y_pred),
                'mae': Evaluator.mean_absolute_error(y_true, y_pred)
            }
        
        return results
    
    @staticmethod
    def compute_kmeans_loss(
        model: OnlineBalancedKMeans, 
        X: np.ndarray
    ) -> float:
        """
        Compute the k-means clustering loss (within-cluster sum of squares).
        """
        return model.compute_loss(X)
    
    @staticmethod
    def print_results(results: Dict[str, Dict[str, float]]) -> None:
        """Pretty print evaluation results."""
        print("\n" + "=" * 70)
        print("INFERENCE METHODS EVALUATION RESULTS")
        print("=" * 70)
        print(f"{'Method':<25} {'MSE':>12} {'RMSE':>12} {'MAE':>12}")
        print("-" * 70)
        
        # Sort by MSE
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['mse'])
        
        for method, metrics in sorted_methods:
            print(f"{method:<25} {metrics['mse']:>12.6f} {metrics['rmse']:>12.6f} {metrics['mae']:>12.6f}")
        
        print("=" * 70)
        print(f"Best method: {sorted_methods[0][0]} (MSE: {sorted_methods[0][1]['mse']:.6f})")
        print("=" * 70)
