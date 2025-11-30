"""
Main script demonstrating Inference with K-means
Based on arXiv:2410.17256

This script demonstrates:
1. Online Balanced K-means clustering
2. Various data generation approaches
3. Seven inference methods for predicting last component
4. Evaluation of inference methods
"""
import numpy as np
import matplotlib.pyplot as plt
from inference_kmeans import (
    OnlineBalancedKMeans,
    DataGenerator,
    KMeansInference,
    Evaluator
)


def run_experiment(
    data_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_clusters: int = 100,
    alpha: float = 0.6,
    beta: float = 0.7,
    n_iterations: int = 100
):
    """
    Run a complete experiment with given data.
    
    Workflow:
    1. Initialize Online Balanced K-means with specific hyperparameters (α, β).
    2. Train (fit) the model on the training dataset using the online update rules.
    3. Compute the standard K-means training loss (WCSS) to verify clustering quality.
    4. Apply all 7 inference methods to predict the last component of the test set.
    5. Evaluate and print error metrics (MSE, RMSE, MAE) for each method.
    
    Parameters:
    -----------
    data_name : str
        Name of the dataset for logging
    X_train : np.ndarray
        Training data
    X_test : np.ndarray
        Test data
    n_clusters : int
        Number of clusters for k-means
    alpha : float
        Learning rate
    beta : float
        Weight adjustment parameter
    n_iterations : int
        Number of training iterations
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {data_name}")
    print(f"{'='*70}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}, Clusters: {n_clusters}")
    print(f"Hyperparameters: alpha={alpha}, beta={beta}, iterations={n_iterations}")
    
    # Initialize and fit model
    model = OnlineBalancedKMeans(
        n_clusters=n_clusters,
        alpha=alpha,
        beta=beta,
        random_state=42
    )
    
    print("\nTraining Online Balanced K-means...")
    model.fit(X_train, n_iterations=n_iterations)
    
    # Compute clustering loss
    train_loss = Evaluator.compute_kmeans_loss(model, X_train)
    print(f"Training loss (WCSS): {train_loss:.4f}")
    
    # Evaluate all inference methods
    print("\nEvaluating inference methods...")
    results = Evaluator.evaluate_all_methods(
        model, 
        X_test,
        n_closest=5,
        lambda_param=0.5,
        beta=1.0
    )
    
    Evaluator.print_results(results)
    
    return model, results


def visualize_clustering(
    model: OnlineBalancedKMeans,
    X: np.ndarray,
    title: str = "Online Balanced K-means Clustering"
):
    """
    Visualize clustering results (for 2D data).
    """
    if X.shape[1] != 2:
        print("Visualization only available for 2D data")
        return
    
    labels = model.predict(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=20)
    plt.scatter(
        model.centroids[:, 0], 
        model.centroids[:, 1], 
        c='red', 
        marker='x', 
        s=200, 
        linewidths=3,
        label='Centroids'
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=150)
    plt.show()


def main():
    """Main function demonstrating the inference with k-means approach."""
    
    # Initialize data generator
    generator = DataGenerator(random_state=42)
    
    # Configuration - ADJUSTED TO MATCH PAPER (arXiv:2410.17256)
    # Paper uses: 10,000 training samples, 1,000 test samples
    # Best hyperparameters: k=300, alpha=0.6, beta=0.07
    n_train = 10000      # Paper value: 10,000
    n_test = 1000        # Paper value: 1,000
    n_features = 5
    n_clusters = 300     # Paper's best value: 300
    
    print("=" * 70)
    print("INFERENCE WITH K-MEANS")
    print("Implementation of arXiv:2410.17256")
    print("=" * 70)
    
    # ==================== Experiment 1: Uniform Data ====================
    X_train = generator.generate_uniform_squared(n_train, n_features)
    X_test = generator.generate_uniform_squared(n_test, n_features)
    run_experiment("Uniform Squared", X_train, X_test, n_clusters, alpha=0.6, beta=0.07)
    
    # ==================== Experiment 2: Normal Data ====================
    X_train = generator.generate_normal_squared(n_train, n_features)
    X_test = generator.generate_normal_squared(n_test, n_features)
    run_experiment("Normal Squared", X_train, X_test, n_clusters, alpha=0.6, beta=0.07)
    
    # ==================== Experiment 3: Gamma Data ====================
    X_train = generator.generate_gamma_squared(n_train, n_features)
    X_test = generator.generate_gamma_squared(n_test, n_features)
    run_experiment("Gamma Squared", X_train, X_test, n_clusters, alpha=0.6, beta=0.07)
    
    # ==================== Experiment 4: Three Clusters ====================
    X_train = generator.generate_three_normal_clusters(n_train, n_features)
    X_test = generator.generate_three_normal_clusters(n_test, n_features)
    run_experiment("Three Normal Clusters", X_train, X_test, n_clusters, alpha=0.6, beta=0.07)
    
    # ==================== Experiment 5: Blob Data (2D for visualization) ====================
    X_train, _ = generator.generate_blobs(n_train, 2, n_centers=3)
    X_test, _ = generator.generate_blobs(n_test, 2, n_centers=3)
    model, _ = run_experiment("Blobs (2D)", X_train, X_test, n_clusters=50, alpha=0.6, beta=0.07)
    
    # Visualize the 2D clustering
    visualize_clustering(model, X_train, "Online Balanced K-means - Blobs")
    
    # ==================== Hyperparameter Analysis ====================
    print("\n" + "=" * 70)
    print("HYPERPARAMETER ANALYSIS: Effect of number of clusters")
    print("=" * 70)
    
    X_train = generator.generate_normal_squared(n_train, n_features)
    X_test = generator.generate_normal_squared(n_test, n_features)
    
    cluster_results = {}
    # Paper tested k from 100 to 1000, with k=300 being optimal
    for k in [100, 150, 200, 250, 300, 400, 500]:
        model = OnlineBalancedKMeans(n_clusters=k, alpha=0.6, beta=0.07, random_state=42)
        model.fit(X_train, n_iterations=10)
        results = Evaluator.evaluate_all_methods(model, X_test)
        
        # Get best method MSE
        best_mse = min(r['mse'] for r in results.values())
        cluster_results[k] = best_mse
        print(f"Clusters: {k:3d} | Best MSE: {best_mse:.6f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
