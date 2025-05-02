import numpy as np
import random
import flinng
from scipy.spatial.distance import cdist
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Fixed parameters
dataset_size = 10000
queries_size = 100
dataset_std = 1
queries_std = 0.5
flinng_hashes_per_table = 16
flinng_num_hash_tables = 20
k = 1

# Grid settings
num_rows_list = [5, 10, 15, 20, 25]
data_dims = [25, 50, 100, 150, 200]

# Result storage
query_times = np.zeros((len(num_rows_list), len(data_dims)))
precisions = np.zeros_like(query_times)
recalls = np.zeros_like(query_times)
f1_scores = np.zeros_like(query_times)

np.random.seed(42)
random.seed(42)

for i, num_rows in enumerate(num_rows_list):
    for j, data_dim in enumerate(data_dims):
        print(f"\n[Running] num_rows={num_rows}, data_dim={data_dim}")
        
        # Generate dataset and queries
        dataset = np.random.normal(size=(dataset_size, data_dim), scale=dataset_std)
        queries = []
        gts = []
        for _ in range(queries_size):
            gt = random.randrange(dataset_size)
            query = dataset[gt] + np.random.normal(size=(data_dim,), scale=queries_std)
            queries.append(query)
            gts.append(gt)
        queries = np.array(queries)

        # Create FLINNG index
        index = flinng.dense_32_bit(
            num_rows=num_rows,
            cells_per_row=dataset_size // 100,
            data_dimension=data_dim,
            num_hash_tables=flinng_num_hash_tables,
            hashes_per_table=flinng_hashes_per_table
        )
        index.add_points(dataset)
        index.prepare_for_queries()

        # FLINNG query
        start_time = time.time()
        flinng_results = index.query(queries, k)
        query_time = time.time() - start_time

        # Ground truth
        exact_results = []
        distances = cdist(queries, dataset, 'euclidean')
        for q in distances:
            nn_indices = np.argsort(q)[:k]
            exact_results.append(nn_indices.tolist())

        # Precision, recall
        precision_vals, recall_vals = [], []
        for x, y in zip(flinng_results, exact_results):
            f_set = set(x)
            e_set = set(y)
            common = f_set.intersection(e_set)
            precision_vals.append(len(common) / len(f_set))
            recall_vals.append(len(common) / len(e_set))

        avg_precision = np.mean(precision_vals)
        avg_recall = np.mean(recall_vals)
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        # Store metrics
        query_times[i][j] = query_time
        precisions[i][j] = avg_precision
        recalls[i][j] = avg_recall
        f1_scores[i][j] = f1

# Plot heatmaps
def plot_heatmap(data, title, filename, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt=".3f", xticklabels=data_dims, yticklabels=num_rows_list, cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_heatmap(query_times, "Query Time (s)", "query_time_heatmap.png", "Data Dimension", "Num Rows")
plot_heatmap(precisions, "Average Precision", "precision_heatmap.png", "Data Dimension", "Num Rows")
plot_heatmap(recalls, "Average Recall", "recall_heatmap.png", "Data Dimension", "Num Rows")
plot_heatmap(f1_scores, "F1 Score", "f1_score_heatmap.png", "Data Dimension", "Num Rows")

print("✅ All 4 heatmaps (query time, precision, recall, F1 score) saved.")
