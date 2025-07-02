import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# Load the saved sparse matrix
data_prepared = sparse.load_npz('data_prepared__.npz')

# Optional: convert to dense if data is small enough
data_prepared_dense = data_prepared.toarray()

# Optional: Reduce dimensionality (recommended for high-dim datasets before HDBSCAN)
svd = TruncatedSVD(n_components=50, random_state=42)
data_reduced = svd.fit_transform(data_prepared_dense)

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean')
labels = clusterer.fit_predict(data_reduced)

# Number of clusters (excluding noise points labeled -1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters found: {n_clusters}")

# Optional: Silhouette score (only if more than 1 cluster and no/all noise)
if n_clusters > 1 and len(set(labels)) < len(data_reduced):
    score = silhouette_score(data_reduced, labels)
    print(f"Silhouette Score: {score:.4f}")

