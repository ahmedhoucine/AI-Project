import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load data
with open("sentences.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_vectors = model.encode(sentences).astype('float32')

d = sentence_vectors.shape[1]
query = "I like water"
query_vector = model.encode([query]).astype('float32')

id_to_sentence = {i: s for i, s in enumerate(sentences)}

# -------------------- Flat Index --------------------
flat_index = faiss.IndexFlatL2(d)
flat_index.add(sentence_vectors)

start_time = time.perf_counter()
D_flat, I_flat = flat_index.search(query_vector, k=5)
flat_time = time.perf_counter() - start_time

print("🔍 Flat Index Results:")
for i, dist in zip(I_flat[0], D_flat[0]):
    print(f"- {id_to_sentence[i]}  (distance: {dist:.4f})")
print(f"⏱️ Flat index search time: {flat_time:.6f} seconds\n")

# -------------------- IVFPQ Index --------------------
nlist = 50
m = 8
nbits = 8

quantizer = faiss.IndexFlatL2(d)
ivfpq_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
ivfpq_index.train(sentence_vectors)
ivfpq_index.add(sentence_vectors)
ivfpq_index.nprobe = 10

start_time = time.perf_counter()
D_ivf, I_ivf = ivfpq_index.search(query_vector, k=5)
ivf_time = time.perf_counter() - start_time

print("⚡ IVFPQ Index Results:")
for i, dist in zip(I_ivf[0], D_ivf[0]):
    print(f"- {id_to_sentence[i]}  (distance: {dist:.4f})")
print(f"⏱️ IVFPQ index search time: {ivf_time:.6f} seconds\n")

# -------------------- HNSW Index --------------------
hnsw_index = faiss.IndexHNSWFlat(d, 32)  # M = 32
hnsw_index.hnsw.efSearch = 64
hnsw_index.hnsw.efConstruction = 200

hnsw_index.add(sentence_vectors)

start_time = time.perf_counter()
D_hnsw, I_hnsw = hnsw_index.search(query_vector, k=5)
hnsw_time = time.perf_counter() - start_time

print("🧭 HNSW Index Results:")
for i, dist in zip(I_hnsw[0], D_hnsw[0]):
    print(f"- {id_to_sentence[i]}  (distance: {dist:.4f})")
print(f"⏱️ HNSW index search time: {hnsw_time:.6f} seconds\n")

# -------------------- Evaluation --------------------
def recall_at_k(true_indices, pred_indices, k):
    true_set = set(true_indices[0][:k])
    pred_set = set(pred_indices[0][:k])
    intersection = true_set.intersection(pred_set)
    return len(intersection) / k

recall_ivf = recall_at_k(I_flat, I_ivf, k=5)
recall_hnsw = recall_at_k(I_flat, I_hnsw, k=5)



# -------------------- Summary --------------------
print("📊 Comparison Summary:")
print(f"Flat Search Time   : {flat_time:.6f} s")
print(f"IVFPQ Search Time  : {ivf_time:.6f} s")
print(f"HNSW Search Time   : {hnsw_time:.6f} s")

# Avoid division by zero
ivf_speedup = flat_time / ivf_time if ivf_time > 0 else float('inf')
hnsw_speedup = flat_time / hnsw_time if hnsw_time > 0 else float('inf')

print(f"IVFPQ Speedup      : {ivf_speedup:.2f}x")
print(f"HNSW Speedup       : {hnsw_speedup:.2f}x")

print("📈 Evaluation Summary (Recall@5):")
print(f"IVFPQ Recall@5     : {recall_ivf:.2f}")
print(f"HNSW  Recall@5     : {recall_hnsw:.2f}")
