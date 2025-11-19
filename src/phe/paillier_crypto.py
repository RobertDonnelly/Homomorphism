"""
Paillier Cryptosystem Implementation using python-paillier (phe)
Homomorphic encryption for privacy-preserving federated learning
"""

import numpy as np
import time
from typing import List, Tuple, Union
from phe import paillier
import pickle
import json


class PaillierCrypto:
    """
    Wrapper class for Paillier Homomorphic Encryption operations.
    Supports encryption, decryption, and homomorphic operations on encrypted data.
    """
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize Paillier cryptosystem.
        
        Args:
            key_size: Bit length of the modulus (1024, 2048, 3072, or 4096)
                     Higher = more secure but slower
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        
    def generate_keypair(self) -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
        """
        Generate new Paillier key pair.
        
        Returns:
            Tuple of (public_key, private_key)
        """
        print(f"Generating {self.key_size}-bit Paillier keypair...")
        start_time = time.time()
        
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=self.key_size
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Keypair generated in {elapsed:.3f}s")
        print(f"  Public key modulus (n): {self.public_key.n}")
        print(f"  Bit length: {self.public_key.n.bit_length()}")
        
        return self.public_key, self.private_key
    
    def encrypt(self, plaintext: Union[int, float, np.ndarray]) -> Union[paillier.EncryptedNumber, List]:
        """
        Encrypt plaintext data.
        
        Args:
            plaintext: Single value or numpy array to encrypt
            
        Returns:
            Encrypted value(s)
        """
        if self.public_key is None:
            raise ValueError("Generate keypair first using generate_keypair()")
        
        if isinstance(plaintext, np.ndarray):
            return [self.public_key.encrypt(float(x)) for x in plaintext.flatten()]
        else:
            return self.public_key.encrypt(float(plaintext))
    
    def decrypt(self, ciphertext: Union[paillier.EncryptedNumber, List]) -> Union[float, np.ndarray]:
        """
        Decrypt ciphertext data.
        
        Args:
            ciphertext: Encrypted value(s)
            
        Returns:
            Decrypted value(s)
        """
        if self.private_key is None:
            raise ValueError("Private key not available")
        
        if isinstance(ciphertext, list):
            return np.array([self.private_key.decrypt(c) for c in ciphertext])
        else:
            return self.private_key.decrypt(ciphertext)
    
    def encrypt_vector(self, vector: np.ndarray) -> List[paillier.EncryptedNumber]:
        """
        Encrypt a vector (1D array).
        
        Args:
            vector: numpy array to encrypt
            
        Returns:
            List of encrypted values
        """
        return [self.public_key.encrypt(float(x)) for x in vector]
    
    def decrypt_vector(self, encrypted_vector: List[paillier.EncryptedNumber]) -> np.ndarray:
        """
        Decrypt a vector of encrypted values.
        
        Args:
            encrypted_vector: List of encrypted values
            
        Returns:
            numpy array of decrypted values
        """
        return np.array([self.private_key.decrypt(c) for c in encrypted_vector])
    
    def add_encrypted(self, enc1: paillier.EncryptedNumber, 
                     enc2: paillier.EncryptedNumber) -> paillier.EncryptedNumber:
        """
        Add two encrypted values homomorphically: E(a) + E(b) = E(a+b)
        
        Args:
            enc1: First encrypted value
            enc2: Second encrypted value
            
        Returns:
            Encrypted sum
        """
        return enc1 + enc2
    
    def add_encrypted_vectors(self, enc_vec1: List, enc_vec2: List) -> List:
        """
        Add two encrypted vectors element-wise.
        
        Args:
            enc_vec1: First encrypted vector
            enc_vec2: Second encrypted vector
            
        Returns:
            Encrypted sum vector
        """
        if len(enc_vec1) != len(enc_vec2):
            raise ValueError("Vectors must have same length")
        
        return [e1 + e2 for e1, e2 in zip(enc_vec1, enc_vec2)]
    
    def multiply_encrypted_by_scalar(self, ciphertext: paillier.EncryptedNumber, 
                                    scalar: float) -> paillier.EncryptedNumber:
        """
        Multiply encrypted value by plaintext scalar: E(a) * k = E(a*k)
        
        Args:
            ciphertext: Encrypted value
            scalar: Plaintext scalar
            
        Returns:
            Encrypted product
        """
        return ciphertext * scalar
    
    def multiply_encrypted_vector_by_scalar(self, enc_vector: List, 
                                          scalar: float) -> List:
        """
        Multiply encrypted vector by plaintext scalar.
        
        Args:
            enc_vector: Encrypted vector
            scalar: Plaintext scalar
            
        Returns:
            Encrypted product vector
        """
        return [c * scalar for c in enc_vector]
    
    def weighted_average_encrypted(self, enc_vectors: List[List], 
                                  weights: List[float]) -> List:
        """
        Compute weighted average of encrypted vectors.
        Useful for federated averaging of model parameters.
        
        Args:
            enc_vectors: List of encrypted vectors
            weights: Weights for each vector (should sum to 1)
            
        Returns:
            Encrypted weighted average vector
        """
        if len(enc_vectors) != len(weights):
            raise ValueError("Number of vectors must match number of weights")
        
        # Initialize with first weighted vector
        result = self.multiply_encrypted_vector_by_scalar(enc_vectors[0], weights[0])
        
        # Add remaining weighted vectors
        for enc_vec, weight in zip(enc_vectors[1:], weights[1:]):
            weighted_vec = self.multiply_encrypted_vector_by_scalar(enc_vec, weight)
            result = self.add_encrypted_vectors(result, weighted_vec)
        
        return result
    
    def save_public_key(self, filepath: str):
        """Save public key to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.public_key, f)
        print(f"✓ Public key saved to {filepath}")
    
    def load_public_key(self, filepath: str):
        """Load public key from file."""
        with open(filepath, 'rb') as f:
            self.public_key = pickle.load(f)
        print(f"✓ Public key loaded from {filepath}")
    
    def save_private_key(self, filepath: str):
        """Save private key to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.private_key, f)
        print(f"✓ Private key saved to {filepath}")
    
    def load_private_key(self, filepath: str):
        """Load private key from file."""
        with open(filepath, 'rb') as f:
            self.private_key = pickle.load(f)
        print(f"✓ Private key loaded from {filepath}")


# Example usage and comprehensive tests
if __name__ == "__main__":
    print("PAILLIER HOMOMORPHIC ENCRYPTION DEMO")
    print("=" * 60)
    
    # Initialize system
    paillier_sys = PaillierCrypto(key_size=2048)
    paillier_sys.generate_keypair()
    
    # Test 1: Basic encryption/decryption
    print("\n--- Test 1: Basic Encryption/Decryption ---")
    plaintext = 42.5
    ciphertext = paillier_sys.encrypt(plaintext)
    decrypted = paillier_sys.decrypt(ciphertext)
    print(f"Original:   {plaintext}")
    print(f"Decrypted:  {decrypted}")
    print(f"✓ Match: {abs(plaintext - decrypted) < 1e-6}")
    
    # Test 2: Homomorphic addition
    print("\n--- Test 2: Homomorphic Addition ---")
    a, b = 15.5, 27.3
    enc_a = paillier_sys.encrypt(a)
    enc_b = paillier_sys.encrypt(b)
    enc_sum = paillier_sys.add_encrypted(enc_a, enc_b)
    dec_sum = paillier_sys.decrypt(enc_sum)
    print(f"{a} + {b} = {a + b}")
    print(f"Encrypted sum: {dec_sum}")
    print(f"✓ Match: {abs(dec_sum - (a + b)) < 1e-6}")
    
    # Test 3: Scalar multiplication
    print("\n--- Test 3: Scalar Multiplication ---")
    value = 10.5
    scalar = 3.2
    enc_value = paillier_sys.encrypt(value)
    enc_product = paillier_sys.multiply_encrypted_by_scalar(enc_value, scalar)
    dec_product = paillier_sys.decrypt(enc_product)
    print(f"{value} × {scalar} = {value * scalar}")
    print(f"Encrypted product: {dec_product}")
    print(f"✓ Match: {abs(dec_product - (value * scalar)) < 1e-6}")
    
    # Test 4: Vector operations
    print("\n--- Test 4: Vector Operations ---")
    vec1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vec2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    
    enc_vec1 = paillier_sys.encrypt_vector(vec1)
    enc_vec2 = paillier_sys.encrypt_vector(vec2)
    enc_sum_vec = paillier_sys.add_encrypted_vectors(enc_vec1, enc_vec2)
    dec_sum_vec = paillier_sys.decrypt_vector(enc_sum_vec)
    
    print(f"Vec1: {vec1}")
    print(f"Vec2: {vec2}")
    print(f"Expected sum: {vec1 + vec2}")
    print(f"Encrypted sum: {dec_sum_vec}")
    print(f"✓ Match: {np.allclose(dec_sum_vec, vec1 + vec2)}")
    
    # Test 5: Weighted average (federated learning scenario)
    print("\n--- Test 5: Weighted Average (FL Aggregation) ---")
    # Simulate 3 clients with different model parameters
    client1_params = np.array([1.0, 2.0, 3.0])
    client2_params = np.array([2.0, 3.0, 4.0])
    client3_params = np.array([3.0, 4.0, 5.0])
    
    # Encrypt client parameters
    enc_client1 = paillier_sys.encrypt_vector(client1_params)
    enc_client2 = paillier_sys.encrypt_vector(client2_params)
    enc_client3 = paillier_sys.encrypt_vector(client3_params)
    
    # Weights based on data size (sum to 1)
    weights = [0.3, 0.5, 0.2]
    
    # Compute encrypted weighted average
    enc_avg = paillier_sys.weighted_average_encrypted(
        [enc_client1, enc_client2, enc_client3],
        weights
    )
    dec_avg = paillier_sys.decrypt_vector(enc_avg)
    
    # Expected result
    expected_avg = (client1_params * weights[0] + 
                   client2_params * weights[1] + 
                   client3_params * weights[2])
    
    print(f"Client 1 params: {client1_params} (weight: {weights[0]})")
    print(f"Client 2 params: {client2_params} (weight: {weights[1]})")
    print(f"Client 3 params: {client3_params} (weight: {weights[2]})")
    print(f"Expected average: {expected_avg}")
    print(f"Encrypted average: {dec_avg}")
    print(f"✓ Match: {np.allclose(dec_avg, expected_avg)}")
    
    # Run benchmarks
    benchmark_operations(key_size=2048, num_iterations=10)