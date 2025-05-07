import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Generate synthetic data (classification data that we can treat for clustering)
X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, random_state=42)

# Split dataset into train and test sets (used only for validation purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Agglomerative Hierarchical Clustering ===
# Use AgglomerativeClustering for clustering
agg_clust = AgglomerativeClustering(n_clusters=2, linkage='ward')

# Fit the model to the data
agg_clust.fit(X_train)

# Predict cluster labels for the test data
y_pred_train = agg_clust.labels_  # For training data

# Evaluate using Silhouette Score
silhouette_train = silhouette_score(X_train, y_pred_train)

# Output clustering quality metrics
print("=== Hierarchical Clustering (Agglomerative) Evaluation ===")
print(f"Silhouette Score (Train)   : {silhouette_train:.4f}")

# === Clustering Results Visualization ===
# Use PCA to reduce the dimensions to 2D for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters formed by AgglomerativeClustering
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_clust.fit_predict(X_pca), cmap='viridis', marker='.')
plt.title("Agglomerative Clustering Visualization (2D PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster ID")
plt.show()

# === Dendrogram Visualization ===
# Perform linkage on the entire dataset
Z = linkage(X_train, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# === K-Fold Cross-Validation (Optional) ===
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

silhouette_cv = []

for train_index, test_index in kf.split(X):
    # Split data based on the K-Fold indices
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    
    # Fit the model on each fold
    agg_clust.fit(X_train_cv)
    y_pred_train_cv = agg_clust.labels_

    # Silhouette Score for each fold
    silhouette_cv.append(silhouette_score(X_train_cv, y_pred_train_cv))

# Output average silhouette score from cross-validation
print("\n=== 5-Fold Cross-Validation ===")
print(f"Avg Silhouette Score : {np.mean(silhouette_cv):.4f}")
