# This file includes the functions required to perform locality sensitive hashing on the input dataset
# and to creat the hash table for the LSH algorithm
import numpy as np
# import pandas as pd
from collections import defaultdict
from typing import List, Tuple
# from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class LSH:
    def __init__(self, inputDim: int, numHashes: int, numTables: int):
        self.inputDim = inputDim
        self.numHashes = numHashes
        self.numTables = numTables
        # Define the hyperplanes that will be used to create the hash functions
        self.hyperplanes = [np.random.randn(numHashes, inputDim) for _ in range(numTables)]
        # Initialize the hash tables
        self.hashTables = [defaultdict(list) for _ in range(numTables)]
        self.data = None
    
    def _hash(self, vector: np.ndarray, planes: np.ndarray) -> str:
        """
        Hash a vector into a binary string using the given hyperplanes.
        Input :
            vector : np.ndarray : The input vector to hash.
            planes : np.ndarray : The hyperplanes used for hashing.
        Returns:
            str : The binary string representing the hash of the vector.
        """
        # Compute the dot product of the vector with the hyperplanes
        hashValues = np.dot(planes, vector)
        # Create a binary string based on the sign of the dot products
        return ''.join(['1' if value > 0 else '0' for value in hashValues])

    def _createHashTable(self, data: List[np.ndarray]) -> None:
        '''
        Insert the data into all the hash tables
        Input :
            data : List[np.ndarray] : The input data to hash.
        '''
        print("[DEBUG] Creating Hash Tables for LSH ... ")
        self.data = data
        for index, vector in enumerate(data):
            for tableIndex in range(self.numTables):
                # Hash the vector using the hyperplanes for the current table
                hashValue = self._hash(vector, self.hyperplanes[tableIndex])
                # Insert the vector into the hash table
                self.hashTables[tableIndex][hashValue].append(index)
        print("[DEBUG] Hashing of the Dataset Complete ...")

    def query(self, vector: np.ndarray, numResults: int = 3) -> List[Tuple[int, float]]:
        '''
        Query the hash tables for similar vector to the given input vector
        Input :
            vector : np.ndarray : The input vector to query.
            numResults : int : The number of similar vectors to return.
        Returns:
            List[Tuple[int, float]] : A list of tuples containing the index and distance of the similar vectors.
        '''
        print("[DEBUG] Querying Hash Tables for similar vectors ... ")
        candidates = set()
        for tableIndex in range(self.numTables):
            hashValue = self._hash(vector, self.hyperplanes[tableIndex])
            bucket = self.hashTables[tableIndex].get(hashValue, [])
            for index in bucket:
                candidates.add(index)
        
        # Compute the cosine similarity and return the top K results
        results = []
        for index in candidates:
            candidateVector = self.data[index]
            similarity = float(np.dot(vector.T, candidateVector) / (np.linalg.norm(vector) * np.linalg.norm(candidateVector)))
            results.append((index, similarity))
        # print(results[0])
        results.sort(key = lambda x:-x[1])
        return results[:numResults]
    
    def info(self):
        print("[INFO] LSH Class Details:")
        print("1. Input Dimension:", self.inputDim)
        print("2. Number of Hashes:", self.numHashes)
        print("3. Number of Tables:", self.numTables)
        print("4. Hyperplanes Count:", len(self.hyperplanes))
        for i, table in enumerate(self.hashTables):
            print(f"    Table {i} Size:", len(table))



if __name__ == '__main__':
    # Ensuring results can be reproduced
    np.random.seed(42)
    # Generating 100 data points in a 10D space randomly    
    data = [np.random.randn(10).reshape(-1, 1) for _ in range(100)]
    # print(f"[DEBUG] Sample Data {data[0]} ... ")
    # print(f"[DEBUG] Data Shape : {data[0].shape}")
    # Define the lsh object using 5 Hash Tables
    lsh = LSH(inputDim=10, numHashes=8, numTables=5)
    lsh._createHashTable(data)
    lsh.info()

    # Query with a known vector and add a slight perturbation to the vector
    queryVector = data[5] + np.random.normal(0, 0.1, (10, 1))
    print(f"[DEBUG] Query Vector Shape : {queryVector.shape}")
    nearestNeighbours = lsh.query(queryVector, numResults = 5)
    
    print("Nearest neighbours (index, similarity):")
    for index, similarity in nearestNeighbours:
        print(f"Index : {index}, Similarity : {similarity:.4f}")