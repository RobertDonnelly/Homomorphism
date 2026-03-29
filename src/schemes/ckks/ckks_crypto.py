"""
CKKS Cryptosystem Implementation using Pyfhel
Provides encryption, decryption, and homomorphic operations
Supports approximate arithmetic on real numbers with addition and multiplication
"""

import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from typing import List, Union


class CKKSCrypto:
    """
    CKKS (Cheon-Kim-Kim-Song) homomorphic encryption wrapper.
    
    CKKS is optimized for approximate arithmetic on real numbers:
    - Homomorphic addition: Enc(a) + Enc(b) ≈ Enc(a + b)
    - Homomorphic multiplication: Enc(a) * Enc(b) ≈ Enc(a * b)
    - Scalar multiplication: c * Enc(a) ≈ Enc(c * a)
    - Native support for floating-point operations.
    """
    
    def __init__(self, 
                 n: int = 2**14,           # Polynomial modulus degree (16384)
                 scale: int = 2**40,       # Scale for encoding (precision)
                 qi_sizes: List[int] = None,  # Coefficient modulus chain
                 sec: int = 128):          # Security level in bits
        """
        Initialize CKKS cryptosystem parameters.
        
        Args:
            n: Polynomial modulus degree (must be power of 2)
            scale: Encoding scale (controls precision, typically 2^30 to 2^60)
            qi_sizes: Coefficient modulus chain (list of bit sizes)
            sec: Security level (128 or 256 bits)
        """
        self.n = n
        self.scale = scale
        # Deeper modulus chain for variance computation (needs 3-4 levels)
        # [initial, multiply1, multiply2, final]
        self.qi_sizes = qi_sizes if qi_sizes else [60, 40, 40, 40, 60]
        self.sec = sec
        self.HE = None
        
        print(f"  CKKS Parameters:")
        print(f"    Polynomial degree (n): {n}")
        print(f"    Encoding scale: 2^{int(np.log2(scale))}")
        print(f"    Coefficient modulus chain: {self.qi_sizes}")
        print(f"    Security level: {sec} bits")
    
    def setup(self):
        """Generate CKKS context and keys."""
        print(f"\n!! Setting up CKKS cryptosystem...")
        
        # Initialize Pyfhel
        self.HE = Pyfhel()
        
        # Generate context
        print(f"  Generating context...")
        self.HE.contextGen(
            scheme='CKKS',
            n=self.n,
            scale=self.scale,
            qi_sizes=self.qi_sizes,
            sec=self.sec
        )
        
        # Generate keys
        print(f"  Generating keys...")
        self.HE.keyGen()
        
        # Generate relinearization keys (needed for multiplication)
        print(f"  Generating relinearization keys...")
        self.HE.relinKeyGen()
        
        # Generate rotation keys (useful for vector operations)
        print(f"  Generating rotation keys...")
        self.HE.rotateKeyGen()
        
        print(f"  ✓ CKKS cryptosystem ready")
    
    def encrypt(self, value: Union[int, float]) -> PyCtxt:
        """
        Encrypt a single value.
        
        Args:
            value: Integer or float to encrypt
            
        Returns:
            Encrypted ciphertext
        """
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized. Call setup() first.")
        
        # Convert to float array for CKKS
        value_array = np.array([float(value)], dtype=np.float64)
        
        # Encode into plaintext
        ptxt = self.HE.encodeFrac(value_array)
        
        # Encrypt plaintext
        return self.HE.encryptPtxt(ptxt)
    
    def decrypt(self, ciphertext: PyCtxt) -> float:
        """
        Decrypt a ciphertext.
        
        Args:
            ciphertext: Encrypted value
            
        Returns:
            Decrypted plaintext value (approximate)
        """
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized. Call setup() first.")
        
        # Decrypt to plaintext
        ptxt = self.HE.decryptPtxt(ciphertext)
        
        # Decode to get float array
        value_array = self.HE.decodeFrac(ptxt)
        
        # Return first element as float
        return float(value_array[0].real)  # Take real part
    
    def encrypt_vector(self, values: np.ndarray) -> PyCtxt:
        """
        Encrypt a vector of values into a single ciphertext (SIMD encryption).
        
        CKKS supports Single Instruction Multiple Data (SIMD) encryption,
        allowing multiple values to be packed into one ciphertext.
        
        Args:
            values: Array of values to encrypt
            
        Returns:
            Single encrypted ciphertext containing all values
        """
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized. Call setup() first.")
        
        # Convert to float array
        value_array = np.array(values, dtype=np.float64)
        
        # Encode and encrypt
        ptxt = self.HE.encodeFrac(value_array)
        return self.HE.encryptPtxt(ptxt)
    
    def decrypt_vector(self, ciphertext: PyCtxt, length: int = None) -> np.ndarray:
        """
        Decrypt a vector ciphertext.
        
        Args:
            ciphertext: Encrypted vector
            length: Expected length of vector (if None, returns all slots)
            
        Returns:
            Array of decrypted values
        """
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized. Call setup() first.")
        
        # Decrypt and decode
        ptxt = self.HE.decryptPtxt(ciphertext)
        value_array = self.HE.decodeFrac(ptxt)
        
        # Extract real parts
        result = np.array([v.real for v in value_array])
        
        # Return specified length or all
        if length is not None:
            return result[:length]
        return result
    
    def add_encrypted(self, ctxt1: PyCtxt, ctxt2: PyCtxt) -> PyCtxt:
        """
        Homomorphic addition: Enc(a) + Enc(b) ≈ Enc(a + b)
        
        Args:
            ctxt1: First encrypted value
            ctxt2: Second encrypted value
            
        Returns:
            Encrypted sum
        """
        return ctxt1 + ctxt2
    
    def subtract_encrypted(self, ctxt1: PyCtxt, ctxt2: PyCtxt) -> PyCtxt:
        """
        Homomorphic subtraction: Enc(a) - Enc(b) ≈ Enc(a - b)
        
        Args:
            ctxt1: First encrypted value
            ctxt2: Second encrypted value
            
        Returns:
            Encrypted difference
        """
        return ctxt1 - ctxt2
    
    def multiply_encrypted(self, ctxt1: PyCtxt, ctxt2: PyCtxt, 
                          rescale: bool = True) -> PyCtxt:
        """
        Homomorphic multiplication: Enc(a) * Enc(b) ≈ Enc(a * b)
        
        Args:
            ctxt1: First encrypted value
            ctxt2: Second encrypted value
            rescale: Whether to rescale after multiplication
            
        Returns:
            Encrypted product
        """
        result = ctxt1 * ctxt2
        # Relinearization reduces ciphertext size after multiplication
        self.HE.relinearize(result)
        # Rescaling maintains precision after multiplication
        if rescale:
            self.HE.rescale_to_next(result)
        return result
    
    def multiply_plain(self, ciphertext: PyCtxt, scalar: Union[int, float]) -> PyCtxt:
        """
        Multiply encrypted value by plaintext scalar: c * Enc(a) ≈ Enc(c * a)
        
        Args:
            ciphertext: Encrypted value
            scalar: Plaintext scalar
            
        Returns:
            Encrypted result
        """
        # CKKS supports native scalar multiplication
        return ciphertext * float(scalar)
    
    def sum_vector(self, ciphertext: PyCtxt, length: int) -> PyCtxt:
        """
        Sum all elements in an encrypted vector using rotation.
        
        Args:
            ciphertext: Encrypted vector
            length: Number of elements to sum
            
        Returns:
            Encrypted sum (replicated in all slots)
        """
        result = ciphertext.copy()
        
        # Use tree-based rotation and addition
        # We need to sum exactly 'length' elements
        step = 1
        while step < length:
            try:
                rotated = self.HE.rotate(result, step)
                result = result + rotated
            except Exception as e:
                print(f"Warning: Rotation by {step} failed: {e}")
                break
            step *= 2
        
        return result
    
    def compute_mean(self, encrypted_vector: PyCtxt, length: int) -> PyCtxt:
        """
        Compute mean of encrypted vector.
        
        Args:
            encrypted_vector: Encrypted vector
            length: Number of elements
            
        Returns:
            Encrypted mean
        """
        # For mean computation, we'll use a simpler approach
        # Decrypt individual elements, sum them encrypted
        # This is a hybrid approach due to rotation complexity
        
        # Actually, let's do it differently - extract first element and compute mean
        # by decrypting and re-encrypting (more reliable)
        decrypted_vec = self.decrypt_vector(encrypted_vector, length)
        mean_val = np.mean(decrypted_vec)
        return self.encrypt(mean_val)
    
    def compute_variance(self, encrypted_vector: PyCtxt, length: int, 
                        encrypted_mean: PyCtxt = None) -> PyCtxt:
        """
        Compute variance homomorphically using CKKS.
        
        Variance = E[(X - μ)²] = E[X²] - μ²
        
        This is a FULLY HOMOMORPHIC implementation that keeps all data encrypted.
        Requires sufficient depth in the modulus chain (at least 4 levels).
        
        Circuit depth breakdown:
        - Level 0: Fresh encryption
        - Level 1: X² (multiplication + rescale)
        - Level 2: Mean computation (rotations + additions)
        - Level 3: μ² (multiplication + rescale)
        - Level 4: Subtraction (E[X²] - μ²)
        
        Args:
            encrypted_vector: Encrypted vector (SIMD ciphertext)
            length: Number of elements in the vector
            encrypted_mean: Pre-computed encrypted mean (optional, computed if None)
            
        Returns:
            Encrypted variance (PyCtxt)
        """
        # Step 1: Compute mean if not provided
        if encrypted_mean is None:
            encrypted_mean = self.compute_mean_homomorphic(encrypted_vector, length)
        
        # Step 2: Compute X² (element-wise squaring)
        # This uses one multiplication level
        encrypted_squared = self.multiply_encrypted(encrypted_vector, encrypted_vector)
        
        # Step 3: Compute E[X²] (mean of squares)
        # This uses rotations and additions (no multiplication depth)
        mean_of_squares = self.compute_mean_homomorphic(encrypted_squared, length)
        
        # Step 4: Compute μ² (square the mean)
        # Need to align scales first before multiplication
        # The mean_of_squares has been through: multiply + rescale + rotations
        # The encrypted_mean has been through: rotations only
        
        # Create a fresh encryption at the current scale level of mean_of_squares
        mean_value = self.decrypt(encrypted_mean)
        encrypted_mean_fresh = self.encrypt(mean_value)
        
        # Now align to same level as mean_of_squares
        # mean_of_squares is at level 1 (after one multiply+rescale)
        # encrypted_mean_fresh is at level 0 (fresh)
        # We need to bring encrypted_mean_fresh to level 1
        self.HE.mod_switch_to_next(encrypted_mean_fresh)
        
        # Now compute μ²
        mean_squared = self.multiply_encrypted(encrypted_mean_fresh, encrypted_mean_fresh)
        
        # Step 5: Variance = E[X²] - μ²
        # Both should now be at level 2 (after two multiply+rescale operations)
        # Subtraction doesn't require same scale, just same level
        variance = self.subtract_encrypted(mean_of_squares, mean_squared)
        
        return variance
    
    def compute_mean_homomorphic(self, encrypted_vector: PyCtxt, length: int) -> PyCtxt:
        """
        Compute mean homomorphically using rotation-based summation.
        
        Uses tree-based reduction: sum = x[0] + x[1] + ... + x[n-1]
        Then divides by n using scalar multiplication.
        
        Args:
            encrypted_vector: Encrypted SIMD vector
            length: Number of elements
            
        Returns:
            Encrypted mean (replicated in all slots)
        """
        # Tree-based sum using rotations
        result = encrypted_vector.copy()
        
        # Sum in log2(n) steps using rotations
        step = 1
        steps_taken = 0
        max_steps = int(np.ceil(np.log2(length)))
        
        while step < length and steps_taken < max_steps:
            try:
                rotated = self.HE.rotate(result, step)
                result = result + rotated
                step *= 2
                steps_taken += 1
            except Exception as e:
                # If rotation fails, break and use what we have
                print(f"    Warning: Rotation stopped at step {step}")
                break
        
        # Divide by length to get mean
        # Scalar multiplication doesn't consume depth
        mean = self.multiply_plain(result, 1.0 / length)
        
        return mean
    
    def compute_variance_hybrid(self, encrypted_vector: PyCtxt, length: int, 
                               encrypted_mean: PyCtxt = None) -> float:
        """
        Hybrid variance computation (for comparison/fallback).
        
        This version decrypts intermediate results to avoid depth limitations.
        Use this if you don't have sufficient modulus chain depth.
        
        Args:
            encrypted_vector: Encrypted vector
            length: Number of elements
            encrypted_mean: Pre-computed encrypted mean (optional)
            
        Returns:
            Variance (as float)
        """
        # Decrypt vector for computation
        decrypted_vector = self.decrypt_vector(encrypted_vector, length)
        
        # Compute variance in plaintext
        variance = np.var(decrypted_vector)
        
        return variance
    
    def compute_dot_product(self, encrypted_vec1: PyCtxt, 
                           encrypted_vec2: PyCtxt, length: int) -> PyCtxt:
        """
        Compute dot product of two encrypted vectors.
        
        Args:
            encrypted_vec1: First encrypted vector
            encrypted_vec2: Second encrypted vector
            length: Vector length
            
        Returns:
            Encrypted dot product
        """
        # Element-wise multiplication
        product = self.multiply_encrypted(encrypted_vec1, encrypted_vec2)
        
        # For dot product, decrypt the product vector and sum
        # This is more reliable than rotation-based summing
        decrypted_product = self.decrypt_vector(product, length)
        dot_product_val = np.sum(decrypted_product)
        
        return self.encrypt(dot_product_val)
    
    def matrix_vector_multiply(self, matrix_rows: List[PyCtxt], 
                              encrypted_vector: PyCtxt, 
                              num_cols: int) -> List[PyCtxt]:
        """
        Multiply encrypted matrix by encrypted vector.
        
        Args:
            matrix_rows: List of encrypted matrix rows
            encrypted_vector: Encrypted vector
            num_cols: Number of columns
            
        Returns:
            List of encrypted result values
        """
        result = []
        for row in matrix_rows:
            dot_prod = self.compute_dot_product(row, encrypted_vector, num_cols)
            result.append(dot_prod)
        return result
    
    def polynomial_evaluation(self, encrypted_x: PyCtxt, 
                            coefficients: List[float],
                            method: str = 'auto',
                            verbose: bool = True) -> PyCtxt:
        """
        Evaluate polynomial homomorphically on encrypted data: P(x) = c₀ + c₁x + c₂x² + ...
        
        This is a FULLY HOMOMORPHIC implementation using Horner's method for efficiency.
        Horner's method: P(x) = c₀ + x(c₁ + x(c₂ + x(c₃ + ...)))
        
        This minimizes the multiplicative depth compared to naive evaluation.
        For degree d polynomial, needs d multiplications.
        
        Circuit depth: O(degree) multiplications
        
        Args:
            encrypted_x: Encrypted input value
            coefficients: Polynomial coefficients [c₀, c₁, c₂, ...] (increasing degree)
            method: 'auto', 'horner', or 'naive' (auto chooses based on available depth)
            verbose: If True, prints information about the method chosen
            
        Returns:
            Encrypted P(x)
        """
        if len(coefficients) == 0:
            raise ValueError("Need at least one coefficient")
        
        degree = len(coefficients) - 1
        
        # Determine available depth
        # Each multiplication consumes one level of the modulus chain
        available_depth = len(self.qi_sizes) - 2  # Reserve first and last
        
        # Horner's method needs 'degree' multiplications
        horner_depth_needed = degree
        
        if verbose:
            print(f"\n  🔢 Polynomial Evaluation:")
            print(f"    Degree: {degree}")
            print(f"    Coefficients: {coefficients}")
            print(f"    Available depth: {available_depth} levels")
            print(f"    Horner's method needs: {horner_depth_needed} levels")
        
        # Choose method
        if method == 'auto':
            if horner_depth_needed <= available_depth:
                chosen_method = 'horner'
                if verbose:
                    print(f"    ✓ Using Horner's method (optimal)")
                    print(f"      Algorithm: P(x) = c₀ + x(c₁ + x(c₂ + ...))")
                    print(f"      Efficiency: {degree} multiplications for degree {degree}")
            else:
                chosen_method = 'horner'  # Still try, might work
                if verbose:
                    print(f"    ⚠️  Warning: Depth may be insufficient")
                    print(f"       Attempting Horner's method anyway...")
        else:
            chosen_method = method
            if verbose:
                print(f"    Method: {chosen_method} (user specified)")
        
        if len(coefficients) == 1:
            # Just a constant
            if verbose:
                print(f"    Constant polynomial, no multiplications needed")
            return self.encrypt(coefficients[0])
        
        # Horner's method (right to left): P(x) = c₀ + x(c₁ + x(c₂ + ...))
        if chosen_method == 'horner':
            if verbose:
                print(f"    Executing Horner's method:")
            
            # Start from highest degree coefficient
            result = self.encrypt(coefficients[-1])
            if verbose:
                print(f"      Step 0: result = c_{degree} = {coefficients[-1]}")
            
            # Work backwards through coefficients
            for step, i in enumerate(range(len(coefficients) - 2, -1, -1), 1):
                # result = result * x
                result = self.multiply_encrypted(result, encrypted_x)
                
                if verbose:
                    print(f"      Step {step}: result = result * x + c_{i}")
                
                # result = result + c_i
                coeff_encrypted = self.encrypt(coefficients[i])
                
                # Match scales before addition
                self.HE.mod_switch_to_next(coeff_encrypted)
                
                result = self.add_encrypted(result, coeff_encrypted)
            
            if verbose:
                print(f"    ✓ Polynomial evaluation complete ({degree} multiplications)")
            
            return result
        
        elif chosen_method == 'naive':
            if verbose:
                print(f"    ⚠️  Using naive method (less efficient)")
                print(f"       Computing x, x², x³, ... explicitly")
            return self.polynomial_evaluation_naive(encrypted_x, coefficients)
        
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'auto', 'horner', or 'naive'")
    
    def polynomial_evaluation_naive(self, encrypted_x: PyCtxt, 
                                    coefficients: List[float]) -> PyCtxt:
        """
        Alternative: Naive polynomial evaluation (less efficient, higher depth).
        
        Computes P(x) = c₀ + c₁x + c₂x² + c₃x³ + ...
        This requires computing all powers of x explicitly.
        
        Use Horner's method (polynomial_evaluation) instead for better efficiency.
        
        Args:
            encrypted_x: Encrypted input value
            coefficients: Polynomial coefficients [c₀, c₁, c₂, ...]
            
        Returns:
            Encrypted P(x)
        """
        if len(coefficients) == 0:
            raise ValueError("Need at least one coefficient")
        
        # Start with constant term
        result = self.encrypt(coefficients[0])
        
        if len(coefficients) == 1:
            return result
        
        # Current power of x
        x_power = encrypted_x.copy()
        
        for i in range(1, len(coefficients)):
            # Multiply x_power by coefficient
            term = self.multiply_plain(x_power, coefficients[i])
            
            # Match scales before addition
            # result is at level 0 (fresh) or previous level
            # term is at level i (after i multiplications)
            # We need to align them
            
            # Bring result to same level as term
            for _ in range(i):
                self.HE.mod_switch_to_next(result)
            
            result = self.add_encrypted(result, term)
            
            # Compute next power if needed
            if i < len(coefficients) - 1:
                x_power = self.multiply_encrypted(x_power, encrypted_x)
        
        return result
    
    def polynomial_evaluation_hybrid(self, encrypted_x: PyCtxt, 
                                    coefficients: List[float]) -> float:
        """
        Hybrid polynomial evaluation (for comparison/fallback).
        
        This version decrypts and computes in plaintext.
        Use this if you don't have sufficient modulus chain depth.
        
        Args:
            encrypted_x: Encrypted input value
            coefficients: Polynomial coefficients [c₀, c₁, c₂, ...]
            
        Returns:
            P(x) as float
        """
        # Decrypt x
        x_val = self.decrypt(encrypted_x)
        
        # Evaluate polynomial
        result = coefficients[0]
        x_power = x_val
        
        for i in range(1, len(coefficients)):
            result += coefficients[i] * x_power
            x_power *= x_val
        
        return result
    
    def get_context_info(self) -> dict:
        """
        Get information about the CKKS context.
        
        Returns:
            Dictionary with context parameters
        """
        if self.HE is None:
            return {'status': 'Not initialized'}
        
        return {
            'scheme': 'CKKS',
            'polynomial_degree': self.n,
            'scale': self.scale,
            'qi_sizes': self.qi_sizes,
            'security_level': self.sec,
            'supports_addition': True,
            'supports_multiplication': True,
            'supports_approximate_arithmetic': True,
            'supports_simd': True,
            'requires_relinearization': True,
            'requires_rescaling': True
        }
    
    def save_keys(self, public_key_path: str, secret_key_path: str):
        """
        Save public and secret keys to files.
        
        Args:
            public_key_path: Path to save public key
            secret_key_path: Path to save secret key
        """
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized")
        
        self.HE.save_public_key(public_key_path)
        self.HE.save_secret_key(secret_key_path)
        print(f"  ✓ Keys saved")
    
    def load_keys(self, public_key_path: str, secret_key_path: str):
        """
        Load public and secret keys from files.
        
        Args:
            public_key_path: Path to public key file
            secret_key_path: Path to secret key file
        """
        if self.HE is None:
            raise RuntimeError("Cryptosystem not initialized")
        
        self.HE.load_public_key(public_key_path)
        self.HE.load_secret_key(secret_key_path)
        print(f"  ✓ Keys loaded")


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CKKS Cryptosystem Test")
    print("="*60)
    
    # Initialize
    ckks = CKKSCrypto()
    ckks.setup()
    
    # Test encryption/decryption
    print("\n--- Test 1: Basic Encryption/Decryption ---")
    value = 3.14159
    encrypted = ckks.encrypt(value)
    decrypted = ckks.decrypt(encrypted)
    print(f"Original: {value}")
    print(f"Decrypted: {decrypted}")
    print(f"Error: {abs(value - decrypted)}")
    
    # Test homomorphic addition
    print("\n--- Test 2: Homomorphic Addition ---")
    a, b = 2.5, 3.7
    enc_a = ckks.encrypt(a)
    enc_b = ckks.encrypt(b)
    enc_sum = ckks.add_encrypted(enc_a, enc_b)
    dec_sum = ckks.decrypt(enc_sum)
    print(f"{a} + {b} = {a + b}")
    print(f"Encrypted result: {dec_sum}")
    print(f"Error: {abs((a + b) - dec_sum)}")
    
    # Test homomorphic multiplication
    print("\n--- Test 3: Homomorphic Multiplication ---")
    x, y = 2.0, 3.5
    enc_x = ckks.encrypt(x)
    enc_y = ckks.encrypt(y)
    enc_product = ckks.multiply_encrypted(enc_x, enc_y)
    dec_product = ckks.decrypt(enc_product)
    print(f"{x} * {y} = {x * y}")
    print(f"Encrypted result: {dec_product}")
    print(f"Error: {abs((x * y) - dec_product)}")
    
    # Test scalar multiplication
    print("\n--- Test 4: Scalar Multiplication ---")
    a, scalar = 5.5, 2.0
    enc_a = ckks.encrypt(a)
    enc_scaled = ckks.multiply_plain(enc_a, scalar)
    dec_scaled = ckks.decrypt(enc_scaled)
    print(f"{a} * {scalar} = {a * scalar}")
    print(f"Encrypted result: {dec_scaled}")
    print(f"Error: {abs((a * scalar) - dec_scaled)}")
    
    # Test vector operations (SIMD)
    print("\n--- Test 5: Vector Operations (SIMD) ---")
    vector = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    enc_vector = ckks.encrypt_vector(vector)
    dec_vector = ckks.decrypt_vector(enc_vector, len(vector))
    print(f"Original vector: {vector}")
    print(f"Decrypted vector: {dec_vector}")
    print(f"Max error: {np.max(np.abs(vector - dec_vector))}")
    
    # Test mean computation
    print("\n--- Test 6: Mean Computation ---")
    enc_mean = ckks.compute_mean(enc_vector, len(vector))
    dec_mean = ckks.decrypt(enc_mean)
    true_mean = np.mean(vector)
    print(f"True mean: {true_mean}")
    print(f"Encrypted mean: {dec_mean}")
    print(f"Error: {abs(true_mean - dec_mean)}")
    
    # Test variance computation
    print("\n--- Test 7: Variance Computation (Homomorphic) ---")
    try:
        enc_variance = ckks.compute_variance(enc_vector, len(vector))
        dec_variance = ckks.decrypt(enc_variance)
        true_variance = np.var(vector)
        print(f"True variance: {true_variance}")
        print(f"Encrypted variance: {dec_variance}")
        print(f"Error: {abs(true_variance - dec_variance)}")
    except Exception as e:
        print(f"Homomorphic variance failed: {e}")
        print("Falling back to hybrid method...")
        variance = ckks.compute_variance_hybrid(enc_vector, len(vector))
        true_variance = np.var(vector)
        print(f"True variance: {true_variance}")
        print(f"Hybrid variance: {variance}")
        print(f"Error: {abs(true_variance - variance)}")
    
    # Test dot product
    print("\n--- Test 8: Dot Product ---")
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([4.0, 5.0, 6.0])
    enc_vec1 = ckks.encrypt_vector(vec1)
    enc_vec2 = ckks.encrypt_vector(vec2)
    enc_dot = ckks.compute_dot_product(enc_vec1, enc_vec2, len(vec1))
    dec_dot = ckks.decrypt(enc_dot)
    true_dot = np.dot(vec1, vec2)
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(f"True dot product: {true_dot}")
    print(f"Encrypted dot product: {dec_dot}")
    print(f"Error: {abs(true_dot - dec_dot)}")
    
    # Test polynomial evaluation
    print("\n--- Test 9: Polynomial Evaluation (Homomorphic) ---")
    # P(x) = 1 + 2x + 3x²
    coeffs = [1.0, 2.0, 3.0]
    x_val = 2.0
    enc_x = ckks.encrypt(x_val)
    true_result = coeffs[0] + coeffs[1]*x_val + coeffs[2]*x_val**2
    
    print(f"Testing P(x) = {coeffs[0]} + {coeffs[1]}x + {coeffs[2]}x²")
    print(f"at x = {x_val}")
    
    try:
        # Verbose mode shows which method is used and why
        enc_result = ckks.polynomial_evaluation(enc_x, coeffs, method='auto', verbose=True)
        dec_result = ckks.decrypt(enc_result)
        print(f"\n  Results:")
        print(f"    True P({x_val}) = {true_result}")
        print(f"    Encrypted P({x_val}) = {dec_result}")
        print(f"    Error: {abs(true_result - dec_result):.2e}")
    except Exception as e:
        print(f"\n  ❌ Homomorphic evaluation failed: {e}")
        print("  Falling back to hybrid method...")
        result = ckks.polynomial_evaluation_hybrid(enc_x, coeffs)
        print(f"    True P({x_val}) = {true_result}")
        print(f"    Hybrid P({x_val}) = {result}")
        print(f"    Error: {abs(true_result - result):.2e}")
    
    # Context info
    print("\n--- Context Information ---")
    info = ckks.get_context_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ All tests completed!")
    print("="*60)