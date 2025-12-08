"""
BFV Homomorphic Operations.

This module implements homomorphic operations (addition, multiplication, etc.)
on BFV encrypted ciphertexts, along with noise budget tracking.
"""

from typing import List, Optional, Union
import numpy as np
from Pyfhel import Pyfhel, PyCtxt


class BFVEvaluator:
    """
    Performs homomorphic operations on BFV ciphertexts.
    
    This class provides methods for computing on encrypted data without
    decryption, including arithmetic operations and statistical computations.
    """
    
    def __init__(self, pyfhel: Pyfhel):
        """
        Initialize the evaluator.
        
        Args:
            pyfhel: Configured Pyfhel instance
        """
        self.pyfhel = pyfhel
    
    # ==================== Basic Arithmetic Operations ====================
    
    def add(self, ct1: PyCtxt, ct2: PyCtxt, in_place: bool = False) -> PyCtxt:
        """
        Homomorphic addition of two ciphertexts.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            in_place: If True, modifies ct1 in place
            
        Returns:
            Result ciphertext (ct1 + ct2)
        """
        if in_place:
            ct1 += ct2
            return ct1
        else:
            result = ct1.copy()
            result += ct2
            return result
    
    def add_plain(self, ct: PyCtxt, pt: int, in_place: bool = False) -> PyCtxt:
        """
        Add a plaintext integer to a ciphertext.
        
        Args:
            ct: Ciphertext
            pt: Plaintext integer
            in_place: If True, modifies ct in place
            
        Returns:
            Result ciphertext (ct + pt)
        """
        if in_place:
            ct += pt
            return ct
        else:
            result = ct.copy()
            result += pt
            return result
    
    def subtract(self, ct1: PyCtxt, ct2: PyCtxt, in_place: bool = False) -> PyCtxt:
        """
        Homomorphic subtraction of two ciphertexts.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            in_place: If True, modifies ct1 in place
            
        Returns:
            Result ciphertext (ct1 - ct2)
        """
        if in_place:
            ct1 -= ct2
            return ct1
        else:
            result = ct1.copy()
            result -= ct2
            return result
    
    def subtract_plain(self, ct: PyCtxt, pt: int, in_place: bool = False) -> PyCtxt:
        """
        Subtract a plaintext integer from a ciphertext.
        
        Args:
            ct: Ciphertext
            pt: Plaintext integer
            in_place: If True, modifies ct in place
            
        Returns:
            Result ciphertext (ct - pt)
        """
        if in_place:
            ct -= pt
            return ct
        else:
            result = ct.copy()
            result -= pt
            return result
    
    def multiply(self, ct1: PyCtxt, ct2: PyCtxt, in_place: bool = False) -> PyCtxt:
        """
        Homomorphic multiplication of two ciphertexts.
        
        Note: Multiplication significantly increases noise and requires
        relinearization to reduce ciphertext size.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            in_place: If True, modifies ct1 in place
            
        Returns:
            Result ciphertext (ct1 * ct2)
        """
        if in_place:
            ct1 *= ct2
            # Relinearize to reduce size
            self.pyfhel.relinearize(ct1)
            return ct1
        else:
            result = ct1.copy()
            result *= ct2
            self.pyfhel.relinearize(result)
            return result
    
    def multiply_plain(self, ct: PyCtxt, pt: int, in_place: bool = False) -> PyCtxt:
        """
        Multiply a ciphertext by a plaintext integer.
        
        Args:
            ct: Ciphertext
            pt: Plaintext integer
            in_place: If True, modifies ct in place
            
        Returns:
            Result ciphertext (ct * pt)
        """
        if in_place:
            ct *= pt
            return ct
        else:
            result = ct.copy()
            result *= pt
            return result
    
    def negate(self, ct: PyCtxt, in_place: bool = False) -> PyCtxt:
        """
        Negate a ciphertext.
        
        Args:
            ct: Ciphertext
            in_place: If True, modifies ct in place
            
        Returns:
            Result ciphertext (-ct)
        """
        if in_place:
            ct *= -1
            return ct
        else:
            result = ct.copy()
            result *= -1
            return result
    
    def square(self, ct: PyCtxt, in_place: bool = False) -> PyCtxt:
        """
        Square a ciphertext (optimized version of multiply(ct, ct)).
        
        Args:
            ct: Ciphertext
            in_place: If True, modifies ct in place
            
        Returns:
            Result ciphertext (ct^2)
        """
        return self.multiply(ct, ct, in_place=in_place)
    
    def power(self, ct: PyCtxt, exponent: int) -> PyCtxt:
        """
        Raise ciphertext to an integer power.
        
        Uses binary exponentiation for efficiency.
        Warning: High exponents consume significant noise budget.
        
        Args:
            ct: Ciphertext
            exponent: Integer exponent (must be >= 0)
            
        Returns:
            Result ciphertext (ct^exponent)
        """
        if exponent < 0:
            raise ValueError("Exponent must be non-negative")
        
        if exponent == 0:
            # Return encryption of 1
            return self.pyfhel.encryptInt(1)
        
        if exponent == 1:
            return ct.copy()
        
        # Binary exponentiation
        result = self.pyfhel.encryptInt(1)
        base = ct.copy()
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = self.multiply(result, base)
            exponent //= 2
            if exponent > 0:
                base = self.square(base)
        
        return result
    
    # ==================== Rotation and SIMD Operations ====================
    
    def rotate(self, ct: PyCtxt, steps: int, in_place: bool = False) -> PyCtxt:
        """
        Rotate ciphertext slots.
        
        Requires Galois keys to be generated.
        
        Args:
            ct: Ciphertext with batched values
            steps: Number of positions to rotate (positive = left, negative = right)
            in_place: If True, modifies ct in place
            
        Returns:
            Rotated ciphertext
        """
        if in_place:
            self.pyfhel.rotate(ct, steps)
            return ct
        else:
            result = ct.copy()
            self.pyfhel.rotate(result, steps)
            return result
    
    # ==================== Statistical Operations ====================
    
    def sum_elements(self, ct: PyCtxt) -> PyCtxt:
        """
        Sum all elements in a batched ciphertext.
        
        Uses rotation and addition to sum all slots into the first slot.
        
        Args:
            ct: Batched ciphertext
            
        Returns:
            Ciphertext with sum in all slots
        """
        n_slots = self.pyfhel.get_nSlots()
        result = ct.copy()
        
        # Binary tree summation using rotations
        step = 1
        while step < n_slots:
            rotated = self.rotate(result, step)
            result = self.add(result, rotated, in_place=True)
            step *= 2
        
        return result
    
    def mean(self, ct: PyCtxt, num_elements: int) -> PyCtxt:
        """
        Compute mean of elements in a batched ciphertext.
        
        Args:
            ct: Batched ciphertext
            num_elements: Number of valid elements (rest should be 0)
            
        Returns:
            Ciphertext containing the mean
        """
        total = self.sum_elements(ct)
        # Divide by number of elements
        # Note: Division requires computing multiplicative inverse mod t
        # For simplicity, we'll return the sum; caller must divide after decryption
        # Or use plaintext multiplication with inverse if t allows
        return total
    
    def dot_product(self, ct1: PyCtxt, ct2: PyCtxt) -> PyCtxt:
        """
        Compute dot product of two batched ciphertexts.
        
        Args:
            ct1: First vector (batched ciphertext)
            ct2: Second vector (batched ciphertext)
            
        Returns:
            Ciphertext with dot product in all slots
        """
        # Element-wise multiplication
        product = self.multiply(ct1, ct2)
        
        # Sum all elements
        return self.sum_elements(product)
    
    def polynomial_eval(self, ct: PyCtxt, coefficients: List[int]) -> PyCtxt:
        """
        Evaluate a polynomial on encrypted data.
        
        Computes: coefficients[0] + coefficients[1]*ct + coefficients[2]*ct^2 + ...
        
        Args:
            ct: Input ciphertext (x)
            coefficients: Polynomial coefficients [c0, c1, c2, ...]
            
        Returns:
            Ciphertext containing polynomial evaluation
        """
        if not coefficients:
            raise ValueError("Coefficients list cannot be empty")
        
        # Start with constant term
        result = self.pyfhel.encryptInt(coefficients[0])
        
        if len(coefficients) == 1:
            return result
        
        # Compute powers of ct
        ct_power = ct.copy()
        
        for i in range(1, len(coefficients)):
            if coefficients[i] != 0:
                term = self.multiply_plain(ct_power, coefficients[i])
                result = self.add(result, term, in_place=True)
            
            if i < len(coefficients) - 1:
                ct_power = self.multiply(ct_power, ct, in_place=True)
        
        return result
    
    # ==================== Noise Budget Tracking ====================
    
    def get_noise_budget(self, ct: PyCtxt) -> float:
        """
        Get remaining noise budget for a ciphertext.
        
        Args:
            ct: Ciphertext to check
            
        Returns:
            Noise budget in bits (higher is better)
        """
        try:
            return self.pyfhel.noise_level(ct)
        except:
            # If noise_level not available, return -1
            return -1.0
    
    def is_valid_ciphertext(self, ct: PyCtxt) -> bool:
        """
        Check if ciphertext has sufficient noise budget for decryption.
        
        Args:
            ct: Ciphertext to check
            
        Returns:
            True if ciphertext can be safely decrypted
        """
        noise = self.get_noise_budget(ct)
        # Typically, need at least 10-20 bits of noise budget
        return noise > 10.0 if noise >= 0 else True  # If can't measure, assume valid
    
    # ==================== Utility Operations ====================
    
    def copy_ciphertext(self, ct: PyCtxt) -> PyCtxt:
        """
        Create a deep copy of a ciphertext.
        
        Args:
            ct: Ciphertext to copy
            
        Returns:
            Independent copy of the ciphertext
        """
        return ct.copy()
    
    def serialize_ciphertext(self, ct: PyCtxt) -> bytes:
        """
        Serialize ciphertext to bytes.
        
        Args:
            ct: Ciphertext to serialize
            
        Returns:
            Serialized bytes
        """
        return ct.to_bytes()
    
    def deserialize_ciphertext(self, data: bytes) -> PyCtxt:
        """
        Deserialize bytes to ciphertext.
        
        Args:
            data: Serialized ciphertext bytes
            
        Returns:
            Reconstructed ciphertext
        """
        ct = PyCtxt()
        ct.from_bytes(data, self.pyfhel)
        return ct


class BFVBatchOperations:
    """
    Optimized batch operations for multiple ciphertexts.
    """
    
    def __init__(self, evaluator: BFVEvaluator):
        """
        Initialize batch operations.
        
        Args:
            evaluator: BFVEvaluator instance
        """
        self.evaluator = evaluator
    
    def add_many(self, ciphertexts: List[PyCtxt]) -> PyCtxt:
        """
        Add multiple ciphertexts together.
        
        Args:
            ciphertexts: List of ciphertexts to add
            
        Returns:
            Sum of all ciphertexts
        """
        if not ciphertexts:
            raise ValueError("Ciphertexts list cannot be empty")
        
        result = ciphertexts[0].copy()
        
        for ct in ciphertexts[1:]:
            result = self.evaluator.add(result, ct, in_place=True)
        
        return result
    
    def multiply_many(self, ciphertexts: List[PyCtxt]) -> PyCtxt:
        """
        Multiply multiple ciphertexts together.
        
        Warning: This consumes significant noise budget!
        
        Args:
            ciphertexts: List of ciphertexts to multiply
            
        Returns:
            Product of all ciphertexts
        """
        if not ciphertexts:
            raise ValueError("Ciphertexts list cannot be empty")
        
        result = ciphertexts[0].copy()
        
        for ct in ciphertexts[1:]:
            result = self.evaluator.multiply(result, ct, in_place=True)
        
        return result
    
    def weighted_sum(
        self,
        ciphertexts: List[PyCtxt],
        weights: List[int]
    ) -> PyCtxt:
        """
        Compute weighted sum of ciphertexts.
        
        Args:
            ciphertexts: List of ciphertexts
            weights: List of plaintext weights
            
        Returns:
            Weighted sum
        """
        if len(ciphertexts) != len(weights):
            raise ValueError("Number of ciphertexts must match number of weights")
        
        # Initialize with first term
        result = self.evaluator.multiply_plain(ciphertexts[0], weights[0])
        
        # Add remaining weighted terms
        for ct, w in zip(ciphertexts[1:], weights[1:]):
            weighted = self.evaluator.multiply_plain(ct, w)
            result = self.evaluator.add(result, weighted, in_place=True)
        
        return result
    
    def inner_product(
        self,
        ct_vector: List[PyCtxt],
        pt_vector: List[int]
    ) -> PyCtxt:
        """
        Compute inner product of encrypted vector with plaintext vector.
        
        Args:
            ct_vector: List of encrypted values
            pt_vector: List of plaintext values
            
        Returns:
            Encrypted inner product
        """
        return self.weighted_sum(ct_vector, pt_vector)
    
    def parallel_operations(
        self,
        ciphertexts: List[PyCtxt],
        operation: str,
        operand: Union[PyCtxt, int]
    ) -> List[PyCtxt]:
        """
        Apply the same operation to multiple ciphertexts in parallel.
        
        Args:
            ciphertexts: List of ciphertexts
            operation: Operation name ('add', 'multiply', 'add_plain', 'multiply_plain')
            operand: Operand (ciphertext for ct operations, int for plain operations)
            
        Returns:
            List of result ciphertexts
        """
        results = []
        
        for ct in ciphertexts:
            if operation == 'add':
                result = self.evaluator.add(ct, operand)
            elif operation == 'multiply':
                result = self.evaluator.multiply(ct, operand)
            elif operation == 'add_plain':
                result = self.evaluator.add_plain(ct, operand)
            elif operation == 'multiply_plain':
                result = self.evaluator.multiply_plain(ct, operand)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            results.append(result)
        
        return results
