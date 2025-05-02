# Synthetic sanity check for FLINNG that ensures near duplicate detection works well
import numpy as np
import random
import flinng
from scipy.spatial.distance import cdist
import time

data_dim = 100
dataset_size = 10000
queries_size = 100
dataset_std = 1
queries_std = 0.1
flinng_num_rows = 15
flinngs_cells_per_row = dataset_size // 100
flinng_hashes_per_table = 16
flinng_num_hash_tables = 20
k = 1

np.random.seed(42)
random.seed(42)

print("Generating dataset...")
dataset = np.random.normal(size=(dataset_size, data_dim), scale=dataset_std)

print("Generating query points...")
queries = []
gts = []
for i in range(queries_size):
    gt = random.randrange(dataset_size)
    query = dataset[gt] + np.random.normal(size=(data_dim), scale=queries_std)
    queries.append(query)
    gts.append(gt)
queries = np.array(queries)

print("Building FLINNG index...")
start_time = time.time()
index = flinng.dense_32_bit(num_rows=flinng_num_rows,
                          cells_per_row=flinngs_cells_per_row,
                          data_dimension=data_dim,
                          num_hash_tables=flinng_num_hash_tables,
                          hashes_per_table=flinng_hashes_per_table)
index.add_points(dataset)
index.prepare_for_queries()
build_time = time.time() - start_time
print(f"FLINNG index built in {build_time:.2f} seconds")

print("Querying FLINNG index...")
start_time = time.time()
flinng_results = index.query(queries, k)
query_time = time.time() - start_time
print(f"FLINNG queries completed in {query_time:.2f} seconds")

print("Calculating ground truth (exact KNN)...")
exact_results = []

# Process queries in batches to avoid memory issues
batch_size = 100
num_batches = (queries_size + batch_size - 1) // batch_size
start_time = time.time()

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, queries_size)
    batch_queries = queries[start:end]
    
    # Calculate distances between batch queries and all dataset points
    distances = cdist(batch_queries, dataset, 'euclidean')
    
    # For each query in the batch, find k nearest neighbors
    for i in range(len(batch_queries)):
        # Get indices of k nearest neighbors
        nn_indices = np.argsort(distances[i])[:k]
        exact_results.append(nn_indices.tolist())
    
    if (batch_idx + 1) % 10 == 0:
        print(f"Processed {batch_idx + 1}/{num_batches} batches...")

exact_time = time.time() - start_time
print(f"Exact KNN calculation completed in {exact_time:.2f} seconds")

# Calculate precision and recall for each query
print("Calculating precision and recall metrics...")
precisions = []
recalls = []

for i in range(queries_size):
    flinng_set = set(flinng_results[i])
    exact_set = set(exact_results[i])
    
    # Calculate precision: |flinng ∩ exact| / |flinng|
    precision = len(flinng_set.intersection(exact_set)) / len(flinng_set)
    
    # Calculate recall: |flinng ∩ exact| / |exact|
    recall = len(flinng_set.intersection(exact_set)) / len(exact_set)
    
    precisions.append(precision)
    recalls.append(recall)

# Calculate overall metrics
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

# Print results
print("\n----- Results -----")
print(f"FLINNG Index Build Time: {build_time:.2f} seconds")
print(f"FLINNG Query Time: {query_time:.2f} seconds")
print(f"Exact KNN Time: {exact_time:.2f} seconds")
print(f"Speed-up Factor: {exact_time/query_time:.2f}x")
print("\n----- Performance Metrics -----")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")