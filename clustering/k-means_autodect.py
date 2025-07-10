from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
data_sparse = sparse.load_npz("data_prepared_final.npz")

# 2. Dimensionality Reduction
svd = TruncatedSVD(n_components=14, random_state=42)
data_reduced = svd.fit_transform(data_sparse)

# 3. Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reduced)

# 4. Auto-select best k using silhouette score
k_range = range(2, 15)
best_k = None
best_score = -1
scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, labels)
    scores.append(score)

    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal number of clusters (k): {best_k}")
print(f"Best silhouette score: {best_score:.4f}")

# 5. Final clustering with optimal k
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
labels_final = kmeans_final.fit_predict(data_scaled)
