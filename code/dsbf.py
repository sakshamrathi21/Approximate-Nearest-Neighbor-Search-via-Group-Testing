from hashlib import sha256
import itertools
import numpy as np

class DistanceSensitiveBloomFilter:
    def __init__(self, bitArraySize = 1000, hashCount = 3, vectorLength = 8, maxDistance = 1):
        self.M = bitArraySize
        self.K = hashCount
        self.N = vectorLength
        self.D = maxDistance
        # Initialize the bit array 
        self.bitArray = np.zeros(self.M, dtype=bool) 
    
    def _hashes(self, binaryString : str) -> list:
        """
        Generate K hash values for the given binary string.
        Input :
            binaryString : str : The binary string to hash.
        Returns:
            list : A list of K hash values.
        """
        # Use a simple hash function to generate K different hash values
        hashes = []
        for i in range(self.K):
            combined = binaryString + str(i)
            hashValue = int(sha256(combined.encode()).hexdigest(), 16) % self.M
            hashes.append(hashValue)
        return hashes
    
    def _generateNeighbours(self, binaryString : str) -> list:
        """
        Generate all possible neighbours of the given binary string within the distance D.
        Input :
            binaryString : str : The binary string to generate neighbours for.
        Returns:
            list : A list of all possible neighbours within distance D.
        """
        neighbours = set()
        bits = list(binaryString)
        # Generate all possible combinations of bits within distance D
        for distance in range(self.D + 1):
            for indices in itertools.combinations(range(self.N), distance):
                flipped = bits.copy()
                for i in indices:
                    flipped[i] = '1' if bits[i] == '0' else '0'
                neighbours.add(''.join(flipped))
        return neighbours
    
    def insert(self, binaryString : str) -> None:
        """
        Insert a binary string into the Bloom filter.
        Input :
            binaryString : str : The binary string to insert.
        """
        # Check if the binary string is of the correct length
        if len(binaryString) != self.N:
            raise ValueError(f"Binary string must be of length {self.N}")
        # Check if the binary string contains only '0's and '1's
        if not all(bit in '01' for bit in binaryString):
            raise ValueError("Binary string must contain only '0's and '1's")
        # print("[DEBUG] Inserting binary string into Bloom filter ... ")
        # # Generate K hash values for the binary string
        # hashes = self._hashes(binaryString)
        # # Set the bits at the hash indices to 1
        # for hashValue in hashes:
        #     self.bitArray[hashValue] = True
        # print("[DEBUG] Inserting neighbours into Bloom filter ... ")
        # Generate neighbours and insert them into the Bloom filter
        neighbours = self._generateNeighbours(binaryString)
        for neighbour in neighbours:
            neighbourHashes = self._hashes(neighbour)
            for hashValue in neighbourHashes:
                self.bitArray[hashValue] = True
    
    def query(self, binaryString : str) -> bool:
        """
        Query the Bloom filter for a binary string.
        Input :
            binaryString : str : The binary string to query.
        Returns:
            bool : True if the binary string or its neighbours are in the Bloom filter, False otherwise.
        """
        # Check if the binary string is of the correct length
        if len(binaryString) != self.N:
            raise ValueError(f"Binary string must be of length {self.N}")
        # Check if the binary string contains only '0's and '1's
        if not all(bit in '01' for bit in binaryString):
            raise ValueError("Binary string must contain only '0's and '1's")
        print("[DEBUG] Querying Bloom filter ... ")
        neighbours = self._generateNeighbours(binaryString)
        for neighbour in neighbours:
            neighbourHashes = self._hashes(neighbour)
            if all(self.bitArray[hashValue] for hashValue in neighbourHashes):
                return True
        return False
    
    def info(self):
        """
        Print the details of the Bloom filter.
        """
        print("[INFO] Bloom Filter Details:")
        print("1. Bit Array Size:", self.M)
        print("2. Number of Hashes:", self.K)
        print("3. Vector Length:", self.N)
        print("4. Maximum Distance:", self.D)
        # print("5. Bit Array:", self.bitArray)
        # print("6. Number of bits set:", np.sum(self.bitArray))


# Example usage
if __name__ == '__main__':
    # Create a Distance Sensitive Bloom Filter
    dsbf = DistanceSensitiveBloomFilter(bitArraySize=500, hashCount=3, vectorLength=8, maxDistance=1)
    
    # Insert a binary string into the Bloom filter
    dsbf.insert("11001010")
    dsbf.insert("11110000")
    
    # Query the Bloom filter for a binary string
    result = dsbf.query("11001011")
    print("Query result for '11001011':", result)  # Should return True due to distance 1
    
    # Query the Bloom filter for a binary string not in the filter
    result = dsbf.query("00000000")
    print("Query result for '00000000':", result)  # Should return False
