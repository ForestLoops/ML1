import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA  # For dimensionality reduction

# Generate synthetic data (classification data that we can treat for clustering)
X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, random_state=42)

# Split dataset into train and test sets (used only for validation purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the KMeans model
kmeans = KMeans(n_clusters=2, random_state=42)

# Train the model
kmeans.fit(X_train)

# Predict the cluster labels for the test data
y_pred_train = kmeans.predict(X_train)
y_pred_test = kmeans.predict(X_test)

# Evaluate using silhouette score (higher is better)
silhouette_train = silhouette_score(X_train, y_pred_train)
silhouette_test = silhouette_score(X_test, y_pred_test)

# Evaluate using Adjusted Rand Index (ARI) score (compares with true labels if available)
ari_score = adjusted_rand_score(y_test, y_pred_test)

# Output clustering quality metrics
print("=== K-Means Clustering Evaluation ===")
print(f"Silhouette Score (Train)   : {silhouette_train:.4f}")
print(f"Silhouette Score (Test)    : {silhouette_test:.4f}")
print(f"Adjusted Rand Index (ARI) : {ari_score:.4f}")

# === K-Fold Cross-Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

silhouette_cv = []
ari_cv = []

for train_index, test_index in kf.split(X):
    # Ensure correct indexing with train_index and test_index
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]

    # Fit KMeans on each fold
    kmeans.fit(X_train_cv)
    
    # Predict cluster labels
    y_pred_train_cv = kmeans.predict(X_train_cv)
    y_pred_test_cv = kmeans.predict(X_test_cv)
    
    # Silhouette Score for each fold
    silhouette_cv.append(silhouette_score(X_test_cv, y_pred_test_cv))
    
    # ARI for each fold (if available)
    ari_cv.append(adjusted_rand_score(y_test_cv, y_pred_test_cv))

print("\n=== 5-Fold Cross-Validation ===")
print(f"Avg Silhouette Score : {np.mean(silhouette_cv):.4f}")
print(f"Avg ARI Score        : {np.mean(ari_cv):.4f}")

# === Visualization (2D Plot) ===
# Use PCA to reduce the dimensions to 2D for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a new KMeans model for the visualization (fit on the entire dataset)
kmeans_final = KMeans(n_clusters=2, random_state=42)
kmeans_final.fit(X_pca)

# Predict cluster labels for the entire dataset
labels = kmeans_final.labels_

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='.')
plt.title("K-Means Clustering (2D PCA Visualization)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster ID")
plt.show()

# Elbow Method to find optimal k
distortions = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    distortions.append(kmeans.inertia_)

plt.plot(K_range, distortions, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()