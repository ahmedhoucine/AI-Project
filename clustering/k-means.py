from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
data_sparse = sparse.load_npz("data_prepared_final.npz")

# 2. SVD
svd = TruncatedSVD(n_components=14, random_state=42)
data_reduced = svd.fit_transform(data_sparse)

# 3. Scale
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reduced)

# 4. KMeans
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
kmeans.fit(data_scaled)
labels = kmeans.labels_

# 5. Evaluate
score = silhouette_score(data_scaled, labels)
print("Silhouette Score:", score)

# Optional: Save
np.save("kmeans_labels.npy", labels)
