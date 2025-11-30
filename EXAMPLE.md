======================================================================
EXPERIMENT: Uniform Squared
======================================================================
Train samples: 10000, Test samples: 1000
Features: 5, Clusters: 300
Hyperparameters: alpha=0.6, beta=0.07, iterations=100

Training Online Balanced K-means...
Training loss (WCSS): 10404.8572

Evaluating inference methods...

======================================================================
INFERENCE METHODS EVALUATION RESULTS
======================================================================
Method MSE RMSE MAE

---

merge_nw_ed 0.067055 0.258950 0.209891
normalized_weights 0.069927 0.264438 0.217519
merge_nw_cs 0.076351 0.276317 0.227144
exponential 0.080722 0.284117 0.232127
cluster_size 0.084845 0.291281 0.238108
euclidean 0.104861 0.323823 0.260329
mean_normalized 0.202152 0.449613 0.376427
======================================================================
Best method: merge_nw_ed (MSE: 0.067055)
======================================================================

======================================================================
EXPERIMENT: Normal Squared
======================================================================
Train samples: 10000, Test samples: 1000
Features: 5, Clusters: 300
Hyperparameters: alpha=0.6, beta=0.07, iterations=100

Training Online Balanced K-means...
Training loss (WCSS): 12765.3118

Evaluating inference methods...

======================================================================
INFERENCE METHODS EVALUATION RESULTS
======================================================================
Method MSE RMSE MAE

---

normalized_weights 1.030498 1.015134 0.777171
merge_nw_cs 1.164627 1.079179 0.801527
merge_nw_ed 1.270534 1.127180 0.839525
exponential 1.278302 1.130620 0.815969
cluster_size 1.393531 1.180479 0.853507
euclidean 2.392633 1.546814 1.138073
mean_normalized 7.327817 2.706994 2.455527
======================================================================
Best method: normalized_weights (MSE: 1.030498)
======================================================================

======================================================================
EXPERIMENT: Gamma Squared
======================================================================
Train samples: 10000, Test samples: 1000
Features: 5, Clusters: 300
Hyperparameters: alpha=0.6, beta=0.07, iterations=100

Training Online Balanced K-means...
Training loss (WCSS): 10464.8130

Evaluating inference methods...

======================================================================
INFERENCE METHODS EVALUATION RESULTS
======================================================================
Method MSE RMSE MAE

---

merge_nw_ed 4.283992 2.069781 0.989163
normalized_weights 4.870595 2.206943 0.866364
exponential 5.317383 2.305945 1.030740
euclidean 5.876922 2.424236 1.398584
merge_nw_cs 6.198662 2.489711 0.965494
cluster_size 8.060158 2.839042 1.136188
mean_normalized 80.201862 8.955549 8.092899
======================================================================
Best method: merge_nw_ed (MSE: 4.283992)
======================================================================

======================================================================
EXPERIMENT: Three Normal Clusters
======================================================================
Train samples: 10000, Test samples: 1000
Features: 5, Clusters: 300
Hyperparameters: alpha=0.6, beta=0.07, iterations=100

Training Online Balanced K-means...
Training loss (WCSS): 6066.2406

Evaluating inference methods...

======================================================================
INFERENCE METHODS EVALUATION RESULTS
======================================================================
Method MSE RMSE MAE

---

exponential 0.313123 0.559574 0.439990
cluster_size 0.314316 0.560639 0.441810
merge_nw_cs 0.317870 0.563799 0.443951
normalized_weights 0.331476 0.575740 0.452933
merge_nw_ed 0.453552 0.673463 0.529246
euclidean 0.759656 0.871583 0.691658
mean_normalized 1.860324 1.363937 1.173260
======================================================================
Best method: exponential (MSE: 0.313123)
======================================================================

======================================================================
EXPERIMENT: Blobs (2D)
======================================================================
Train samples: 10000, Test samples: 1000
Features: 2, Clusters: 50
Hyperparameters: alpha=0.6, beta=0.07, iterations=100

Training Online Balanced K-means...
Training loss (WCSS): 2963.3788

Evaluating inference methods...

======================================================================
INFERENCE METHODS EVALUATION RESULTS
======================================================================
Method MSE RMSE MAE

---

merge_nw_cs 3.655781 1.912010 1.163366
exponential 3.736021 1.932879 1.174211
normalized_weights 3.979480 1.994863 1.205814
cluster_size 4.050920 2.012690 1.212519
merge_nw_ed 4.742665 2.177766 1.342959
euclidean 6.063143 2.462345 1.589370
mean_normalized 12.642241 3.555593 3.005187
======================================================================
Best method: merge_nw_cs (MSE: 3.655781)
======================================================================

======================================================================
HYPERPARAMETER ANALYSIS: Effect of number of clusters
======================================================================
Clusters: 100 | Best MSE: 1.835444
Clusters: 150 | Best MSE: 1.430744
Clusters: 200 | Best MSE: 1.499134
Clusters: 250 | Best MSE: 1.185524
Clusters: 300 | Best MSE: 0.999059
Clusters: 400 | Best MSE: 1.006490
Clusters: 500 | Best MSE: 1.018633

======================================================================
EXPERIMENTS COMPLETED
======================================================================
