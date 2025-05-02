import numpy as np
from lsh import LSH
from dsbf import DistanceSensitiveBloomFilter
from typing import List
from sklearn.datasets import make_blobs


class FLINNG:
    def __init__(self, inputDim: int, numHashes: int, numTables: int, bitArraySize: int, hashCount: int, vectorLength: int, maxDistance: int, threshold: int = None, k: int = 10, B: int = 10, R: int = 3):
        '''
        Initialize the FLINNG object with the given parameters.
        Arguments:
            inputDim : int : The dimension of the input data.
            numHashes : int : The number of hash functions.
            numTables : int : The number of hash tables for the LSH component.
            bitArraySize : int : The size of the bit array for the DSBF component.
            hashCount : int : The number of hash functions for the DSBF component.
            vectorLength : int : The length of the vectors for the DSBF component.
            maxDistance : int : The maximum distance for the DSBF component.
            threshold : int : (Optional) The distance threshold for nearest neighbors (used in original implementation).
            k : int : The number of nearest neighbors to return (used for threshold relaxation).
            B : int : The number of buckets per repetition.
            R : int : The number of repetitions.
        '''
        self.inputDim = inputDim
        self.B = B
        self.R = R
        self.k = k
        self.threshold = threshold

        self.lsh = LSH(inputDim, numHashes, numTables)
        self.dsbfParameters = (bitArraySize, hashCount, self.lsh.numTables * vectorLength, maxDistance)
        self.filters = [[None for _ in range(B)] for _ in range(R)]
        self.groups = [[[] for _ in range(B)] for _ in range(R)]
        # Store collision count for each bucket to sort during query
        self.cell_collisions = []
    
    def _hashToBitArray(self, combinedHash: str) -> np.ndarray:
        '''
        Convert the binary string array in a numpy uint8 bit array
        '''
        return np.fromiter((int(b) for b in combinedHash), dtype=np.uint8)
    
    def buildIndex(self, data: List[np.ndarray]):
        '''
        Build the FLINNG Index
        '''
        print("[DEBUG] Building FLINNG Index ... ")
        self.data = data
        self.N = len(data)
        # Start by hashing the data points
        self.lsh._createHashTable(data)
        # Create the groups and the DSBF Filters
        for repeat in range(self.R):
            indexPermutation = np.random.permutation(self.N)
            for bucket in range(self.B):
                indices = indexPermutation[bucket::self.B]
                self.groups[repeat][bucket] = indices.tolist()
                # Create the DSBF for this group
                dsbf = DistanceSensitiveBloomFilter(*self.dsbfParameters)
                for index in indices:
                    vector = data[index].flatten()
                    combinedHash = ''
                    for tableIndex in range(self.lsh.numTables):
                        hashValue = self.lsh._hash(vector, self.lsh.hyperplanes[tableIndex])
                        combinedHash += str(hashValue)
                    dsbf.insert(combinedHash)
                self.filters[repeat][bucket] = dsbf
        print("[DEBUG] FLINNG Index Built ... ")
    
    def query_original(self, queryVector: np.ndarray) -> List[int]:
        '''
        Original implementation of query - returns neighbors within threshold distance
        '''
        queryVector = queryVector.flatten()
        # Start with all the points
        candidates = None

        for repeat in range(self.R):
            Y = set()
            for bucket in range(self.B):
                dsbf = self.filters[repeat][bucket]
                combinedHash = ''
                for tableIndex in range(self.lsh.numTables):
                    hashValue = self.lsh._hash(queryVector, self.lsh.hyperplanes[tableIndex])
                    combinedHash += str(hashValue)
                
                if dsbf.query(combinedHash):
                    Y.update(self.groups[repeat][bucket])
            
            if candidates is None:
                candidates = Y
            else:
                candidates &= Y
            if not candidates:
                break
        
        results = []
        print(f"[DEBUG] Number of Candidates Found: {len(candidates) if candidates else 0}...")
        if candidates and self.threshold:
            for index in candidates:
                distance = np.linalg.norm(self.data[index].flatten() - queryVector)
                if distance <= self.threshold:
                    results.append(index)
        print(f"[DEBUG] Number of Approximate Neighbours Found: {len(results)}...")
        return results

    def query(self, queryVector: np.ndarray) -> List[int]:
        '''
        Implementation of the Threshold Relaxation Algorithm (Algorithm 3)
        Returns the k nearest neighbors for the query vector
        Arguments:
            queryVector: np.ndarray: The data point that will be used to query the Search Space
        Returns:
            List[int]: The indices of the k nearest neighbors in the dataset
        '''
        queryVector = queryVector.flatten()
        
        # Generate candidate cells with query point
        cell_matches = []
        for repeat in range(self.R):
            for bucket in range(self.B):
                dsbf = self.filters[repeat][bucket]
                combinedHash = ''
                for tableIndex in range(self.lsh.numTables):
                    hashValue = self.lsh._hash(queryVector, self.lsh.hyperplanes[tableIndex])
                    combinedHash += str(hashValue)
                
                if dsbf.query(combinedHash):
                    # Store the repeat, bucket, and group for matching cells
                    cell_matches.append((repeat, bucket))
        
        # Count collisions for each cell to sort them
        cell_collision_counts = []
        for repeat, bucket in cell_matches:
            # Count how many points are in this cell
            collision_count = len(self.groups[repeat][bucket])
            cell_collision_counts.append((repeat, bucket, collision_count))
        
        # Sort cells in descending order by hash collisions
        sorted_cells = sorted(cell_collision_counts, key=lambda x: x[2], reverse=True)
        
        # Initialize counts for each point
        counts = np.zeros(self.N, dtype=int)
        result = []
        
        # Process cells in sorted order (Algorithm 3)
        for repeat, bucket, _ in sorted_cells:
            # Process each point in the cell
            for point_idx in self.groups[repeat][bucket]:
                counts[point_idx] += 1
                
                # If point appears in R cells, add it to results
                if counts[point_idx] == self.R:
                    result.append(point_idx)
                    
                    # Return if we have k neighbors
                    if len(result) == self.k:
                        print(f"[DEBUG] Found {len(result)} neighbors using threshold relaxation")
                        return result
        
        # If we couldn't find exactly k neighbors that appeared in all R repetitions,
        # we can relax the threshold and return points with highest counts
        if len(result) < self.k:
            # Get indices sorted by their counts in descending order
            remaining_candidates = sorted(range(self.N), key=lambda i: counts[i], reverse=True)
            
            # Add more points until we have k or run out of candidates
            for idx in remaining_candidates:
                if idx not in result and counts[idx] > 0:  # Only consider points with non-zero counts
                    result.append(idx)
                    if len(result) == self.k:
                        break
        
        print(f"[DEBUG] Found {len(result)} neighbors using threshold relaxation")
        return result
    
    def info(self):
        print("[INFO] FLINNG Index Information ... ")
        print(f"    Input Dimension: {self.inputDim}")
        print(f"    Number of Hashes: {self.lsh.numHashes}")
        print(f"    Number of Buckets: {self.B}")
        print(f"    Number of Repetitions: {self.R}")
        print(f"    Total Data Points: {self.N}")
        print(f"    DSBF Parameters: {self.dsbfParameters}")
        if self.threshold:
            print(f"    Distance Threshold: {self.threshold}")
        print(f"    k Nearest Neighbors: {self.k}")
    

# Example usage
if __name__ == '__main__':
    # Generate the dataset using scikit-learn
    print("[TEST] Generating Dataset ...")
    data, labels = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)
    data = [x.reshape(-1, 1) for x in data]
    labels = labels.tolist()
    
    # Create the FLINNG Object
    print("[TEST] Initializing FLINNG ...")
    # Set k=5 to return 5 nearest neighbors
    flinng = FLINNG(inputDim=10, numHashes=48, numTables=3, bitArraySize=10000, 
                   hashCount=7, vectorLength=48, maxDistance=2, threshold=None, 
                   k=5, B=50, R=7)
    
    # Build the index
    print("[TEST] Building FLINNG Index ...")
    flinng.buildIndex(data)
    
    # Print the details of the FLINNG index
    print("[TEST] Printing FLINNG Index Details ...")
    flinng.info()

    # Create a query vector
    print("[TEST] Creating Query Vector ...")
    queryVector = data[42] + np.random.normal(0, 0.1, (10, 1))
    
    # Query using threshold relaxation algorithm
    print("[TEST] Querying with Threshold Relaxation Algorithm ...")
    neighbors = flinng.query(queryVector)
    
    print("[TEST] Query Results ...")
    print(f"[TEST] Approximate Neighbours: {neighbors}")
    print(f"[INFO] Ground Truth Label: {labels[42]}")
    print(f"[INFO] Approximate Neighbours Labels: {[labels[i] for i in neighbors]}")
    print("[TEST] Querying Complete ...")