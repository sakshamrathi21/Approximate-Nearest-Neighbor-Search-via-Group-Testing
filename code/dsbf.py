from hashlib import sha256
import itertools
import numpy as np

class DistanceSensitiveBloomFilter:
    def __init__(self, bitArraySize = 1000, hashCount = 3, vectorLength = 8, maxDistance = 1):
        self.M = bitArraySize
        self.K = hashCount
        self.N = vectorLength
        self.D = maxDistance
        self.bitArray = np.zeros(self.M, dtype=bool) 
    
    def _hashes(self, binaryString : str) -> list:
        """
        Generate K hash values for the given binary string.
        Input :
            binaryString : str : The binary string to hash.
        Returns:
            list : A list of K hash values.
        """
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
        if len(binaryString) != self.N:
            print(len(binaryString), self.N)    
            raise ValueError(f"Binary string must be of length {self.N}")
        if not all(bit in '01' for bit in binaryString):
            raise ValueError("Binary string must contain only '0's and '1's")
        hashes = self._hashes(binaryString)
        for hashValue in hashes:
            self.bitArray[hashValue] = True
    
    def query(self, binaryString : str) -> bool:
        """
        Query the Bloom filter for a binary string.
        Input :
            binaryString : str : The binary string to query.
        Returns:
            bool : True if the binary string or its neighbours are in the Bloom filter, False otherwise.
        """
        if len(binaryString) != self.N:
            raise ValueError(f"Binary string must be of length {self.N}")
        if not all(bit in '01' for bit in binaryString):
            raise ValueError("Binary string must contain only '0's and '1's")
        # print("[DEBUG] Querying Bloom filter ... ", flush=True)
        neighbours = self._generateNeighbours(binaryString)
        for neighbour in neighbours:
            neighbourHashes = self._hashes(neighbour)
            count = 0
            for hashValue in neighbourHashes:
                if self.bitArray[hashValue]:
                    count += 1
            if count >= len(neighbourHashes)/3:
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

if __name__ == '__main__':
    dsbf = DistanceSensitiveBloomFilter(bitArraySize=500, hashCount=3, vectorLength=8, maxDistance=1)
    dsbf.insert("11001010")
    dsbf.insert("11110000")
    result = dsbf.query("11001011")
    print("Query result for '11001011':", result)
    result = dsbf.query("00000000")
    print("Query result for '00000000':", result)