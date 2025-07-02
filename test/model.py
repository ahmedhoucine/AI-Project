import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load your saved sparse matrix
data_prepared = sparse.load_npz('data_prepared.npz')

# Convert to dense array (required for standard KMeans)
data_prepared_dense = data_prepared.toarray()

# Define range of k values to test
possible_k_values = range(2, 11)

# Method 1: Elbow Method (Inertia)
inertias = []
for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_prepared_dense)
    inertias.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(10, 5))
plt.plot(possible_k_values, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Method 2: Silhouette Scores
silhouette_scores = []
for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_prepared_dense)
    score = silhouette_score(data_prepared_dense, labels)
    silhouette_scores.append(score)

# Plot Silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(possible_k_values, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)
plt.show()
