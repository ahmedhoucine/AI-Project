import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# Load the saved sparse matrix
data_prepared = sparse.load_npz('data_prepared_final2.npz')

# Convert to dense if memory allows
data_prepared_dense = data_prepared.toarray()

# Range of n_components to test for SVD
n_components_list = [10,11,12,13,14,15,16,17,18,19]  # try 10, 20, ..., 100

min_cluster_size = 30
best_score = -1
best_n_components = None
best_labels = None

for n_components in n_components_list:
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    data_reduced = svd.fit_transform(data_prepared_dense)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(data_reduced)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters > 1 and len(set(labels)) < len(data_reduced):
        score = silhouette_score(data_reduced, labels)
        print(f"n_components={n_components}, clusters={n_clusters}, Silhouette Score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_n_components = n_components
            best_labels = labels
    else:
        print(f"n_components={n_components}, clusters={n_clusters}, Silhouette Score=N/A")

print("\nBest configuration:")
print(f"n_components = {best_n_components}")
print(f"min_cluster_size = {min_cluster_size}")
print(f"Best Silhouette Score = {best_score:.4f}")
