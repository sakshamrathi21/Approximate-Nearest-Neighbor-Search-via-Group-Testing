import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import psutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

class FLINNGMetrics:
    def __init__(self, flinng, data, k=10):
        """
        Initialize the metrics evaluator for FLINNG.
        
        Args:
            flinng: The FLINNG object to evaluate
            data: The dataset used for building the index
            k: Number of nearest neighbors to consider
        """
        self.flinng = flinng
        self.data = data
        self.k = k
        self.results = {}
        
    def compute_ground_truth(self, n_samples=100):
        """Compute ground truth nearest neighbors for evaluation"""
        print("[METRICS] Computing ground truth nearest neighbors...")
        flat_data = np.array([x.flatten() for x in self.data])
        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='brute').fit(flat_data)
        
        # Select random samples for evaluation
        if n_samples < len(self.data):
            indices = np.random.choice(len(self.data), n_samples, replace=False)
        else:
            indices = np.arange(len(self.data))
            
        self.eval_indices = indices
        query_data = flat_data[indices]
        
        # Compute exact nearest neighbors
        start_time = time.time()
        distances, indices = nbrs.kneighbors(query_data)
        end_time = time.time()
        
        # Remove self-matches
        self.ground_truth = indices[:, 1:self.k+1]
        self.exact_query_time = (end_time - start_time) / len(indices)
        
        print(f"[METRICS] Ground truth computation complete. Avg brute force time: {self.exact_query_time:.6f}s per query")
        return self.ground_truth
        
    def evaluate_recall(self, n_samples=100):
        """Evaluate recall of FLINNG against ground truth"""
        if not hasattr(self, 'ground_truth'):
            self.compute_ground_truth(n_samples)
            
        print("[METRICS] Evaluating FLINNG recall...")
        recall_rates = []
        precision_rates = []
        f1_rates = []
        query_times = []
        
        for i, idx in enumerate(tqdm(self.eval_indices)):
            query_vector = self.data[idx]
            
            # Time the FLINNG query
            start_time = time.time()
            approx_neighbors = self.flinng.query_using_threshold_relaxation(query_vector)
            end_time = time.time()
            query_times.append(end_time - start_time)
            
            # Calculate recall (what fraction of true nearest neighbors were found)
            gt_neighbors = set(self.ground_truth[i])
            found_neighbors = set(approx_neighbors)
            if idx in found_neighbors:  # Remove self if present
                found_neighbors.remove(idx)
                
            common = gt_neighbors.intersection(found_neighbors)
            
            # Calculate precision, recall and F1 score
            precision = len(common) / len(found_neighbors) if found_neighbors else 0
            recall = len(common) / len(gt_neighbors) if gt_neighbors else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_rates.append(precision)
            recall_rates.append(recall)
            f1_rates.append(f1)
        
        avg_precision = np.mean(precision_rates)
        avg_recall = np.mean(recall_rates)
        avg_f1 = np.mean(f1_rates)
        avg_query_time = np.mean(query_times)
        speedup = self.exact_query_time / avg_query_time if avg_query_time > 0 else 0
        
        self.results['precision'] = avg_precision
        self.results['recall'] = avg_recall
        self.results['f1_score'] = avg_f1
        self.results['query_time'] = avg_query_time
        self.results['speedup'] = speedup
        
        print(f"[METRICS] Average precision: {avg_precision:.4f}")
        print(f"[METRICS] Average recall: {avg_recall:.4f}")
        print(f"[METRICS] Average F1 score: {avg_f1:.4f}")
        print(f"[METRICS] Average query time: {avg_query_time:.6f}s (vs {self.exact_query_time:.6f}s for brute force)")
        print(f"[METRICS] Speedup factor: {speedup:.2f}x")
        
        return avg_precision, avg_recall, avg_f1, avg_query_time
    
    def measure_index_building(self):
        """Measure time and memory for building the index"""
        print("[METRICS] Measuring index building performance...")
        
        # Start memory tracking
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Time index building
        start_time = time.time()
        self.flinng.buildIndex(self.data)
        end_time = time.time()
        
        # Calculate memory usage
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = mem_after - mem_before
        
        build_time = end_time - start_time
        
        self.results['build_time'] = build_time
        self.results['memory_usage'] = memory_used
        
        print(f"[METRICS] Index building time: {build_time:.2f} seconds")
        print(f"[METRICS] Memory used: {memory_used:.2f} MB")
        
        return build_time, memory_used
    
    def evaluate_bloom_filter_accuracy(self, n_tests=100):
        """Evaluate false positive/negative rates of the Bloom filter component"""
        print("[METRICS] Evaluating Bloom filter accuracy...")
        
        false_positives = 0
        false_negatives = 0
        
        # Select some random vectors for testing
        indices = np.random.choice(len(self.data), n_tests, replace=False)
        
        for repeat in range(self.flinng.R):
            for bucket in range(self.flinng.B):
                dsbf = self.flinng.filters[repeat][bucket]
                bucket_indices = set(self.flinng.groups[repeat][bucket])
                
                for idx in indices:
                    vector = self.data[idx].flatten()
                    combinedHash = ''
                    for tableIndex in range(self.flinng.lsh.numTables):
                        hashValue = self.flinng.lsh._hash(vector, self.flinng.lsh.hyperplanes[tableIndex])
                        combinedHash += str(hashValue)
                    
                    # Check if the query result matches the ground truth
                    query_result = dsbf.query(combinedHash)
                    should_be_in = idx in bucket_indices
                    
                    if query_result and not should_be_in:
                        false_positives += 1
                    elif not query_result and should_be_in:
                        false_negatives += 1
        
        total_tests = n_tests * self.flinng.R * self.flinng.B
        fp_rate = false_positives / total_tests if total_tests > 0 else 0
        fn_rate = false_negatives / total_tests if total_tests > 0 else 0
        
        self.results['bloom_fp_rate'] = fp_rate
        self.results['bloom_fn_rate'] = fn_rate
        
        print(f"[METRICS] Bloom filter false positive rate: {fp_rate:.6f}")
        print(f"[METRICS] Bloom filter false negative rate: {fn_rate:.6f}")
        
        return fp_rate, fn_rate
    
    def run_parameter_sweep(self, param_name, param_values, n_samples=50):
        """Run a parameter sweep and evaluate performance"""
        print(f"[METRICS] Running parameter sweep for {param_name}...")
        
        recalls = []
        precisions = []
        f1_scores = []
        query_times = []
        build_times = []
        memory_usages = []
        
        original_value = getattr(self.flinng, param_name)
        
        for value in param_values:
            print(f"[METRICS] Testing {param_name}={value}")
            setattr(self.flinng, param_name, value)
            
            # Rebuild index with new parameter
            build_time, memory = self.measure_index_building()
            precision, recall, f1, query_time = self.evaluate_recall(n_samples)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            query_times.append(query_time)
            build_times.append(build_time)
            memory_usages.append(memory)
        
        # Restore original value
        setattr(self.flinng, param_name, original_value)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(param_values, precisions)
        plt.title(f'Precision vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Precision')
        
        plt.subplot(2, 3, 2)
        plt.plot(param_values, recalls)
        plt.title(f'Recall vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Recall')
        
        plt.subplot(2, 3, 3)
        plt.plot(param_values, f1_scores)
        plt.title(f'F1 Score vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        
        plt.subplot(2, 3, 4)
        plt.plot(param_values, query_times)
        plt.title(f'Query Time vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Query Time (s)')
        
        plt.subplot(2, 3, 5)
        plt.plot(param_values, build_times)
        plt.title(f'Build Time vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Build Time (s)')
        
        plt.subplot(2, 3, 6)
        plt.plot(param_values, memory_usages)
        plt.title(f'Memory Usage vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(f'parameter_sweep_{param_name}.png')
        plt.show()
        
        return {
            'param_values': param_values,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores,
            'query_times': query_times,
            'build_times': build_times,
            'memory_usages': memory_usages
        }
    
    def run_heatmap_evaluation(self, r_values, dim_values, n_samples=50):
        """
        Run evaluations across different R values and input dimensions to create heatmaps
        
        Args:
            r_values: List of R values to test
            dim_values: List of input dimension values to test
            n_samples: Number of samples to use for evaluation
        """
        print(f"[METRICS] Running heatmap evaluation for R vs Input Dimension...")
        
        # Initialize result matrices
        precision_matrix = np.zeros((len(r_values), len(dim_values)))
        recall_matrix = np.zeros((len(r_values), len(dim_values)))
        f1_matrix = np.zeros((len(r_values), len(dim_values)))
        query_time_matrix = np.zeros((len(r_values), len(dim_values)))
        
        # Original values
        original_r = self.flinng.R
        original_dim = self.flinng.inputDim
        
        # Create the heatmaps
        for i, r in enumerate(r_values):
            for j, dim in enumerate(dim_values):
                print(f"[METRICS] Testing R={r}, InputDim={dim}")
                
                # Generate new data with the specified dimension if needed
                if dim != self.flinng.inputDim:
                    from sklearn.datasets import make_blobs
                    new_data, _ = make_blobs(n_samples=len(self.data), centers=5, n_features=dim, random_state=42)
                    new_data = [x.reshape(-1, 1) for x in new_data]
                    
                    # Create a new FLINNG object with the new dimension
                    from flinng import FLINNG
                    new_flinng = FLINNG(
                        inputDim=dim, 
                        numHashes=min(48, dim * 2),  # Adjust hash count based on dimension
                        numTables=self.flinng.lsh.numTables, 
                        bitArraySize=self.flinng.dsbfParameters[0], 
                        hashCount=self.flinng.dsbfParameters[1], 
                        vectorLength=self.flinng.dsbfParameters[2], 
                        maxDistance=self.flinng.dsbfParameters[3], 
                        threshold=self.flinng.threshold, 
                        B=self.flinng.B, 
                        R=r
                    )
                    
                    # Create a temporary metrics object
                    temp_metrics = FLINNGMetrics(new_flinng, new_data, k=self.k)
                    temp_metrics.measure_index_building()
                    precision, recall, f1, query_time = temp_metrics.evaluate_recall(n_samples)
                    
                else:
                    # Just update R on the existing object
                    self.flinng.R = r
                    self.measure_index_building()
                    precision, recall, f1, query_time = self.evaluate_recall(n_samples)
                
                # Store results
                precision_matrix[i, j] = precision
                recall_matrix[i, j] = recall
                f1_matrix[i, j] = f1
                query_time_matrix[i, j] = query_time
        
        # Restore original values
        self.flinng.R = original_r
        self.flinng.inputDim = original_dim
        
        # Plot heatmaps
        plt.figure(figsize=(20, 15))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(precision_matrix, annot=True, fmt=".3f", 
                    xticklabels=dim_values, yticklabels=r_values, cmap="YlGnBu")
        plt.title('Precision Heatmap')
        plt.xlabel('Input Dimension')
        plt.ylabel('R Value')
        
        plt.subplot(2, 2, 2)
        sns.heatmap(recall_matrix, annot=True, fmt=".3f", 
                    xticklabels=dim_values, yticklabels=r_values, cmap="YlGnBu")
        plt.title('Recall Heatmap')
        plt.xlabel('Input Dimension')
        plt.ylabel('R Value')
        
        plt.subplot(2, 2, 3)
        sns.heatmap(f1_matrix, annot=True, fmt=".3f", 
                    xticklabels=dim_values, yticklabels=r_values, cmap="YlGnBu")
        plt.title('F1 Score Heatmap')
        plt.xlabel('Input Dimension')
        plt.ylabel('R Value')
        
        plt.subplot(2, 2, 4)
        sns.heatmap(query_time_matrix, annot=True, fmt=".6f", 
                    xticklabels=dim_values, yticklabels=r_values, cmap="YlOrRd")
        plt.title('Query Time Heatmap (seconds)')
        plt.xlabel('Input Dimension')
        plt.ylabel('R Value')
        
        plt.tight_layout()
        plt.savefig('flinng_heatmap_evaluation.png')
        plt.show()
        
        # Return the result matrices for further analysis
        return {
            'r_values': r_values,
            'dim_values': dim_values,
            'precision_matrix': precision_matrix,
            'recall_matrix': recall_matrix,
            'f1_matrix': f1_matrix,
            'query_time_matrix': query_time_matrix
        }
    
    def get_summary(self):
        """Get a summary of all metrics"""
        return self.results
    
    def plot_results(self):
        """Plot the key metrics"""
        if not self.results:
            print("[METRICS] No results to plot. Run evaluations first.")
            return
            
        metrics = list(self.results.keys())
        values = list(self.results.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values)
        plt.title('FLINNG Performance Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('flinng_metrics.png')
        plt.show()


# Example usage:
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from flinng import FLINNG
    
    # Generate sample data
    data, labels = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)
    data = [x.reshape(-1, 1) for x in data]
    
    # Create FLINNG object
    flinng = FLINNG(
        inputDim=10, 
        numHashes=48, 
        numTables=3, 
        bitArraySize=10000, 
        hashCount=7, 
        vectorLength=48, 
        maxDistance=2, 
        threshold=5, 
        B=50, 
        R=7
    )
    
    # Initialize metrics
    metrics = FLINNGMetrics(flinng, data, k=10)
    
    # Run the heatmap evaluation
    r_values = [3, 5, 7, 9, 11]  # Different values of R to test
    dim_values = [5, 10, 20, 30]  # Different input dimensions to test
    
    heatmap_results = metrics.run_heatmap_evaluation(r_values, dim_values, n_samples=50)
    
    print("\n[METRICS] Heatmap Evaluation Complete:")
    print("  Best precision:", np.max(heatmap_results['precision_matrix']), 
          "at R =", r_values[np.argmax(heatmap_results['precision_matrix']) // len(dim_values)],
          "dim =", dim_values[np.argmax(heatmap_results['precision_matrix']) % len(dim_values)])
    
    print("  Best recall:", np.max(heatmap_results['recall_matrix']), 
          "at R =", r_values[np.argmax(heatmap_results['recall_matrix']) // len(dim_values)],
          "dim =", dim_values[np.argmax(heatmap_results['recall_matrix']) % len(dim_values)])
    
    print("  Best F1 score:", np.max(heatmap_results['f1_matrix']), 
          "at R =", r_values[np.argmax(heatmap_results['f1_matrix']) // len(dim_values)],
          "dim =", dim_values[np.argmax(heatmap_results['f1_matrix']) % len(dim_values)])
    
    print("  Best query time:", np.min(heatmap_results['query_time_matrix']), 
          "at R =", r_values[np.argmin(heatmap_results['query_time_matrix']) // len(dim_values)],
          "dim =", dim_values[np.argmin(heatmap_results['query_time_matrix']) % len(dim_values)])