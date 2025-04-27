import numpy as np
from lsh import LSH
from dsbf import DistanceSensitiveBloomFilter
from typing import List
from sklearn.datasets import make_blobs


class FLINNG:
    def __init__(self, inputDim: int, numHashes: int, numTables: int, bitArraySize: int, hashCount: int, vectorLength: int, maxDistance: int, threshold : int, B: int = 10, R: int = 3):
        '''
        Initialize the FLINNG object with the given parameters.
        Arguments :
            inputDim : int : The dimension of the input data.
            numHashes : int : The number of hash functions.
            numTables : int : The number of hash tables for the LSH component.
            bitArraySize : int : The size of the bit array for the DSBF component.
            hashCount : int : The number of hash functions for the DSBF component.
            vectorLength : int : The length of the vectors for the DSBF component.
            maxDistance : int : The maximum distance for the DSBF component.
            B : int : The number of buckets per repitition.
            R : int : The number of repititions.
        '''
        self.inputDim = inputDim
        self.B = B
        self.R = R

        self.lsh = LSH(inputDim, numHashes, numTables)
        self.dsbfParameters = (bitArraySize, hashCount, self.lsh.numTables * vectorLength, maxDistance)
        self.threshold = threshold
        self.filters = [[None for _ in range(B)] for _ in range(R)]
        self.groups = [[[] for _ in range(B)] for _ in range(R)]
    
    def _hashToBitArray(self, combinedHash : str) -> np.ndarray:
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
                    # print(combinedHash)
                    dsbf.insert(combinedHash)
                self.filters[repeat][bucket] = dsbf
        print("[DEBUG] FLINNG Index Built ... ")

    
    def query(self, queryVector : np.ndarray) -> List[int]:
        '''
        Return the nearest neighbours for the FLINNG index
        Arguments:
            queryVector : np.ndarray : The data point that will be used to query the Search Space
        Returns:
            List[int] : The indices of the nearest neighbours in the dataset
        '''
        queryVector = queryVector.flatten()
        # Start with all the points
        candidates = None

        for repeat in range(self.R):
            Y = set()
            for bucket in range(self.B):
                dsbf = self.filters[repeat][bucket]
                # Check if the query vector passes the DSBF
                # isCandidate = False
                combinedHash = ''
                for tableIndex in range(self.lsh.numTables):
                    hashValue = self.lsh._hash(queryVector, self.lsh.hyperplanes[tableIndex])
                    combinedHash += str(hashValue)
                if dsbf.query(combinedHash):
                    Y.update(self.groups[repeat][bucket])
            # approxNeighbours = approxNeighbours.intersection(Y)
            if candidates is None:
                candidates = Y
            else:
                candidates &= Y
            if not candidates:
                break     
        results = []
        print(f"[DEBUG] Number of Candidates Found : {len(candidates)}... ")
        for index in candidates:
            distance = np.linalg.norm(self.data[index].flatten() - queryVector)
            if distance <= self.threshold:
                results.append(index)
        print(f"[DEBUG] Number of Approximate Neighbours Found : {len(results)}... ")
        return results    

    def info(self):
        print("[INFO] FLINNG Index Information ... ")
        print(f"    Input Dimension : {self.inputDim}")
        print(f"    Number of Hashes : {self.lsh.numHashes}")
        print(f"    Number of Buckets : {self.B}")
        print(f"    Number of Repetitions : {self.R}")
        print(f"    Total Data Points : {self.N}")
        print(f"    DSBF Parameters : {self.dsbfParameters}")
    

# Example usage
if __name__ == '__main__':
    # Generate the dataset using scikit-learn
    print("[TEST] Generating Dataset ...")
    data, labels = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)
    data = [x.reshape(-1, 1) for x in data]
    labels = labels.tolist()
    # Create the FLINNG Object
    print("[TEST] Initializing FLINNG ...")
    flinng = FLINNG(inputDim=10, numHashes=48, numTables=3, bitArraySize=10000, hashCount=7, vectorLength=48, maxDistance=1, threshold=2, B=50, R=7)
    # Build the index
    print("[TEST] Building FLINNG Index ...")
    flinng.buildIndex(data)
    # Print the details of the FLINNG index
    print("[TEST] Printing FLINNG Index Details ...")
    flinng.info()

    # Create a query vector
    print("[TEST] Creating Query Vector ...")
    queryVector = data[42] + np.random.normal(0, 0.1, (10, 1))
    neighbours = flinng.query(queryVector)
    print("[TEST] Query Results ...")
    print(f"[TEST] Approximate Neighbours : {neighbours}")
    print(f"[INFO] Ground Truth Label : {labels[42]}")
    print(f"[INFO] Approximate Neighbours Labels : {[labels[i] for i in neighbours]}")
    print("[TEST] Querying Complete ...")

# list(approxNeighbours)    