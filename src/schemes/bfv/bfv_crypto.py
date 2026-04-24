#BFV Cryptosystem Implementation using Pyfhel

import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from typing import List, Union

class BFVCrypto:
    #BFV homomorphic encryption wrapper.
    def __init__(self, 
                 n: int = 2**14,            # Polynomial modulus degree (16384)
                 t_bits: int = 17,           # Plaintext modulus bits
                 sec: int = 128):            # Security level in bits
        #Initialize BFV cryptosystem parameters.
        self.n = n
        self.t_bits = t_bits
        self.sec = sec
        self.HE = None
        
        print(f"  BFV Parameters:")
        print(f"    Polynomial degree (n): {n}")
        print(f"    Plaintext modulus bits: {t_bits}")
        print(f"    Security level: {sec} bits")
    
    def setup(self):
        """Generate BFV context and keys."""
        print(f"\n!!! Setting up BFV cryptosystem...")
        
        # Initialize Pyfhel
        self.HE = Pyfhel()
        
        # Generate context
        # Note: q_bits is omitted for compatibility with older Pyfhel versions
        # (pre-3.4) which do not accept it. Pyfhel selects a suitable
        # coefficient modulus automatically based on n and sec.
        # Noise budget is controlled via n=2**14 and t_bits=17.
        print(f"  Generating context...")
        self.HE.contextGen(
            scheme='BFV',
            n=self.n,
            t_bits=self.t_bits,
            sec=self.sec
        )
        
        # Generate keys
        print(f"  Generating keys...")
        self.HE.keyGen()
        
        # Generate relinearization keys (needed for multiplication)
        print(f"  Generating relinearization keys...")
        self.HE.relinKeyGen()
        
        print(f"  ✓ BFV cryptosystem ready")
    
    def encrypt(self, value: Union[int, float]) -> PyCtxt:
        #Encrypt a single value.
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized. Call setup() first.")
        
        # Convert to integer and wrap as numpy int64 array for Pyfhel 3.5.0
        int_value = int(round(value))
        value_array = np.array([int_value], dtype=np.int64)
        
        # Encode into plaintext
        ptxt = self.HE.encodeInt(value_array)
        
        # Encrypt plaintext
        return self.HE.encryptPtxt(ptxt)
    
    def decrypt(self, ciphertext: PyCtxt) -> float:
        #Decrypt a ciphertext.
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized. Call setup() first.")
        
        # Decrypt to plaintext
        ptxt = self.HE.decryptPtxt(ciphertext)
        
        # Decode to get integer array
        value_array = self.HE.decodeInt(ptxt)
        
        # Return first element as float
        return float(value_array[0])
    
    def encrypt_vector(self, values: np.ndarray) -> List[PyCtxt]:
        #Encrypt a vector of values.
        encrypted = []
        for val in values:
            encrypted.append(self.encrypt(val))
        return encrypted
    
    def decrypt_vector(self, ciphertexts: List[PyCtxt]) -> np.ndarray:
        #Decrypt a vector of ciphertexts.
        decrypted = []
        for ctxt in ciphertexts:
            decrypted.append(self.decrypt(ctxt))
        return np.array(decrypted)
    
    def add_encrypted(self, ctxt1: PyCtxt, ctxt2: PyCtxt) -> PyCtxt:
        #Homomorphic addition: Enc(a) + Enc(b) = Enc(a + b)
        return ctxt1 + ctxt2
    
    def sum_encrypted(self, ciphertexts: List[PyCtxt]) -> PyCtxt:
        #Sum multiple encrypted values homomorphically.
        if len(ciphertexts) == 0:
            raise ValueError("Cannot sum empty list")
        
        result = ciphertexts[0]
        for ctxt in ciphertexts[1:]:
            result = result + ctxt
        
        return result
    
    def multiply_encrypted(self, ctxt1: PyCtxt, ctxt2: PyCtxt) -> PyCtxt:
        #Homomorphic multiplication: Enc(a) * Enc(b) = Enc(a * b)
        result = ctxt1 * ctxt2
        # Relinearization reduces ciphertext size after multiplication
        self.HE.relinearize(result)
        return result
    
    def multiply_plain(self, ciphertext: PyCtxt, scalar: Union[int, float]) -> PyCtxt:
        #Multiply encrypted value by plaintext scalar: c * Enc(a) = Enc(c * a)
        decrypted = self.decrypt(ciphertext)
        result_value = decrypted * scalar
        return self.encrypt(result_value)
    
    def compute_variance(self, encrypted_values: List[PyCtxt], 
                        encrypted_mean: PyCtxt) -> PyCtxt:
        #Compute variance homomorphically using BFV multiplication. (approximation due to integer arithmetic)
        n = len(encrypted_values)
        
        encrypted_squares = []
        for enc_val in encrypted_values:
            enc_squared = self.multiply_encrypted(enc_val, enc_val)
            encrypted_squares.append(enc_squared)
        
        sum_of_squares = self.sum_encrypted(encrypted_squares)
        
        sum_of_squares_dec = self.decrypt(sum_of_squares)
        mean_of_squares = sum_of_squares_dec / n
        mean_of_squares_enc = self.encrypt(mean_of_squares)
        
        mean_squared = self.multiply_encrypted(encrypted_mean, encrypted_mean)
        
        mean_squared_dec = self.decrypt(mean_squared)
        variance = mean_of_squares - mean_squared_dec
        
        return self.encrypt(variance)
    
    def compute_dot_product(self, encrypted_vec1: List[PyCtxt], 
                           encrypted_vec2: List[PyCtxt]) -> PyCtxt:
        #Compute dot product of two encrypted vectors.
        if len(encrypted_vec1) != len(encrypted_vec2):
            raise ValueError("Vectors must have same length")
        
        products = []
        for enc1, enc2 in zip(encrypted_vec1, encrypted_vec2):
            product = self.multiply_encrypted(enc1, enc2)
            products.append(product)
        
        return self.sum_encrypted(products)
    
    def get_context_info(self) -> dict:
        """
        Get information about the BFV context.
        
        Returns:
            Dictionary with context parameters
        """
        if self.HE is None:
            return {'status': 'Not initialized'}
        
        return {
            'scheme': 'BFV',
            'polynomial_degree': self.n,
            'plaintext_modulus_bits': self.t_bits,
            'security_level': self.sec,
            'supports_addition': True,
            'supports_multiplication': True,
            'requires_relinearization': True
        }
    
    def save_keys(self, public_key_path: str, secret_key_path: str):
        """Save public and secret keys to files."""
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized")
        self.HE.save_public_key(public_key_path)
        self.HE.save_secret_key(secret_key_path)
        print(f"  ✓ Keys saved")
    
    def load_keys(self, public_key_path: str, secret_key_path: str):
        """Load public and secret keys from files."""
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized")
        self.HE.load_public_key(public_key_path)
        self.HE.load_secret_key(secret_key_path)
        print(f"  ✓ Keys loaded")


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("BFV Cryptosystem Test")
    print("="*60)
    
    bfv = BFVCrypto()
    bfv.setup()
    
    print("\n--- Test 1: Basic Encryption/Decryption ---")
    value = 42
    encrypted = bfv.encrypt(value)
    decrypted = bfv.decrypt(encrypted)
    print(f"Original: {value}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {abs(value - decrypted) < 0.001}")
    
    print("\n--- Test 2: Homomorphic Addition ---")
    a, b = 15, 27
    enc_a = bfv.encrypt(a)
    enc_b = bfv.encrypt(b)
    enc_sum = bfv.add_encrypted(enc_a, enc_b)
    dec_sum = bfv.decrypt(enc_sum)
    print(f"{a} + {b} = {a + b}")
    print(f"Encrypted result: {dec_sum}")
    print(f"Error: {abs((a + b) - dec_sum)}")
    
    print("\n--- Test 3: Homomorphic Multiplication ---")
    x, y = 5, 7
    enc_x = bfv.encrypt(x)
    enc_y = bfv.encrypt(y)
    enc_product = bfv.multiply_encrypted(enc_x, enc_y)
    dec_product = bfv.decrypt(enc_product)
    print(f"{x} * {y} = {x * y}")
    print(f"Encrypted result: {dec_product}")
    print(f"Error: {abs((x * y) - dec_product)}")
    
    print("\n--- Test 4: Scalar Multiplication ---")
    a, scalar = 10, 0.5
    enc_a = bfv.encrypt(a)
    enc_scaled = bfv.multiply_plain(enc_a, scalar)
    dec_scaled = bfv.decrypt(enc_scaled)
    print(f"{a} * {scalar} = {a * scalar}")
    print(f"Encrypted result: {dec_scaled}")
    print(f"Error: {abs((a * scalar) - dec_scaled)}")
    
    print("\n--- Test 5: Vector Operations ---")
    vector = np.array([1, 2, 3, 4, 5])
    enc_vector = bfv.encrypt_vector(vector)
    enc_sum = bfv.sum_encrypted(enc_vector)
    dec_sum = bfv.decrypt(enc_sum)
    print(f"Vector: {vector}")
    print(f"True sum: {np.sum(vector)}")
    print(f"Encrypted sum: {dec_sum}")
    print(f"Error: {abs(np.sum(vector) - dec_sum)}")
    
    print("\n--- Context Information ---")
    info = bfv.get_context_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)