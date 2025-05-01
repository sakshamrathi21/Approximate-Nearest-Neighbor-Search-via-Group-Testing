from dsbf import DistanceSensitiveBloomFilter
from lsh import LSH
import numpy as np

def main():
    # Generate the dataset
    print("[TEST] Generating Dataset ...")
    np.random.seed(42)
    data = [np.random.randn(10).reshape(-1, 1) for _ in range(1000)]

    # Initialize the LSH object and create the hash tables
    print("[TEST] Initializing LSH and Making Hash Tables ...")
    lsh = LSH(inputDim=10, numHashes=8, numTables=1)
    lsh._createHashTable(data)

    # Initialize the Distance Sensitive Bloom Filter
    print("[TEST] Initializing Distance Sensitive Bloom Filter ...")
    dsbf = DistanceSensitiveBloomFilter(bitArraySize=100, hashCount=3, vectorLength=8, maxDistance=1)

    # Convert each data point into the binary string via LSH and insert them into the DSBF
    print("[TEST] Inserting Data into DSBF ...")
    for vector in data:
        hashString = lsh._hash(vector, lsh.hyperplanes[0])
        dsbf.insert(hashString)
    
    # Print the details of the LSH and the DSBF
    print("[TEST] Printing Details of LSH and DSBF ...")
    lsh.info()
    dsbf.info()

    # Create a query vector and convert it into a binary string
    print("[TEST] Creating Query Vector ...")

    # True case: Query vector is in the dataset
    queryVector = data[42] + np.random.normal(0, 0.1, (10, 1))

    # False case: Query vector is not in the dataset (Attempted)
    # queryVector = np.random.uniform(0, 1, 10).reshape(-1, 1)

    queryHashString = lsh._hash(queryVector, lsh.hyperplanes[0])

    # Query the DSBF with the binary string
    print(f"[TEST] Querying DSBF with Query String {queryHashString} ...")
    result = dsbf.query(queryHashString)
    if result:
        print("[TEST] Match found in the DSBF ... Proceed with full LSH Search ...")
        results = lsh.query(queryVector, numResults=5)
        print("[TEST] Top 5 Similar Vectors:")
        for index, similarity in results:
            print(f"Index: {index}, Similarity: {similarity}")
    else:
        print("[TEST] No match found in the DSBF ... No furthur processing ...")

if __name__ == "__main__":
    main()