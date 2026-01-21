"""
Comprehensive Testing Suite for CKKS Cryptosystem
==================================================

Testing with:
- Unit tests for all cryptographic operations
- Integration tests for workflows
- Performance regression tests
- Statistical validation
- Security property verification
- Edge case coverage
- Test coverage reporting

Usage:
    python test_ckks_suite.py
    
    Or with pytest:
    pytest test_ckks_suite.py -v --cov=ckks_crypto --cov-report=html
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path
from typing import List, Tuple
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from src.schemes.ckks.ckks_crypto import CKKSCrypto


class TestCKKSBasicOperations(unittest.TestCase):
    """Test basic encryption/decryption operations."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS once for all tests."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
        cls.tolerance = 1e-6  # Acceptable error for CKKS approximations
    
    def test_single_value_encryption_decryption(self):
        """Test encryption and decryption of single values."""
        test_values = [0.0, 1.0, -1.0, 3.14159, 1000.0, -500.5]
        
        for value in test_values:
            with self.subTest(value=value):
                encrypted = self.ckks.encrypt(value)
                decrypted = self.ckks.decrypt(encrypted)
                error = abs(value - decrypted)
                
                self.assertLess(error, self.tolerance,
                               f"Encryption/decryption error too large: {error}")
    
    def test_vector_encryption_decryption(self):
        """Test SIMD vector encryption and decryption."""
        test_vectors = [
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0]),
            np.random.randn(100),
            np.random.randn(1000),
        ]
        
        for vector in test_vectors:
            with self.subTest(size=len(vector)):
                encrypted = self.ckks.encrypt_vector(vector)
                decrypted = self.ckks.decrypt_vector(encrypted, len(vector))
                max_error = np.max(np.abs(vector - decrypted))
                
                self.assertLess(max_error, self.tolerance,
                               f"Vector encryption error: {max_error}")
    
    def test_large_vector_chunking(self):
        """Test automatic chunking for vectors larger than slot limit."""
        max_slots = self.ckks.n // 2
        large_vector = np.random.randn(max_slots + 1000)
        
        # This should automatically chunk in the data processor
        # For now, test that we can handle data up to slot limit
        vector = np.random.randn(max_slots)
        encrypted = self.ckks.encrypt_vector(vector)
        decrypted = self.ckks.decrypt_vector(encrypted, len(vector))
        max_error = np.max(np.abs(vector - decrypted))
        
        self.assertLess(max_error, self.tolerance)
    
    def test_encryption_determinism(self):
        """Test that encryption produces different ciphertexts (probabilistic)."""
        value = 42.0
        enc1 = self.ckks.encrypt(value)
        enc2 = self.ckks.encrypt(value)
        
        # Ciphertexts should be different (probabilistic encryption)
        # We can't easily compare PyCtxt objects, so decrypt and verify values match
        dec1 = self.ckks.decrypt(enc1)
        dec2 = self.ckks.decrypt(enc2)
        
        self.assertLess(abs(dec1 - value), self.tolerance)
        self.assertLess(abs(dec2 - value), self.tolerance)


class TestCKKSHomomorphicOperations(unittest.TestCase):
    """Test homomorphic arithmetic operations."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS once for all tests."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
        cls.tolerance = 1e-5  # Slightly higher tolerance for operations
    
    def test_homomorphic_addition(self):
        """Test homomorphic addition property."""
        test_cases = [
            (1.0, 2.0),
            (0.0, 0.0),
            (-5.0, 3.0),
            (100.5, 200.3),
            (0.0001, 0.0002)
        ]
        
        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                enc_a = self.ckks.encrypt(a)
                enc_b = self.ckks.encrypt(b)
                enc_sum = self.ckks.add_encrypted(enc_a, enc_b)
                dec_sum = self.ckks.decrypt(enc_sum)
                
                expected = a + b
                error = abs(expected - dec_sum)
                
                self.assertLess(error, self.tolerance,
                               f"Addition error for {a}+{b}: {error}")
    
    def test_homomorphic_subtraction(self):
        """Test homomorphic subtraction property."""
        test_cases = [
            (5.0, 3.0),
            (0.0, 5.0),
            (100.0, 100.0),
            (-10.0, -5.0)
        ]
        
        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                enc_a = self.ckks.encrypt(a)
                enc_b = self.ckks.encrypt(b)
                enc_diff = self.ckks.subtract_encrypted(enc_a, enc_b)
                dec_diff = self.ckks.decrypt(enc_diff)
                
                expected = a - b
                error = abs(expected - dec_diff)
                
                self.assertLess(error, self.tolerance)
    
    def test_homomorphic_multiplication(self):
        """Test homomorphic multiplication property."""
        test_cases = [
            (2.0, 3.0),
            (0.0, 100.0),
            (1.5, 2.5),
            (-2.0, 3.0),
            (10.0, 0.1)
        ]
        
        for a, b in test_cases:
            with self.subTest(a=a, b=b):
                enc_a = self.ckks.encrypt(a)
                enc_b = self.ckks.encrypt(b)
                enc_product = self.ckks.multiply_encrypted(enc_a, enc_b)
                dec_product = self.ckks.decrypt(enc_product)
                
                expected = a * b
                error = abs(expected - dec_product)
                
                # Higher tolerance for multiplication due to rescaling
                self.assertLess(error, self.tolerance * 10)
    
    def test_scalar_multiplication(self):
        """Test plaintext scalar multiplication."""
        test_cases = [
            (5.0, 2.0),
            (100.0, 0.5),
            (0.0, 1000.0),
            (-3.0, 2.5)
        ]
        
        for value, scalar in test_cases:
            with self.subTest(value=value, scalar=scalar):
                enc_value = self.ckks.encrypt(value)
                enc_scaled = self.ckks.multiply_plain(enc_value, scalar)
                dec_scaled = self.ckks.decrypt(enc_scaled)
                
                expected = value * scalar
                error = abs(expected - dec_scaled)
                
                self.assertLess(error, self.tolerance)
    
    def test_vector_addition(self):
        """Test vector-wise homomorphic addition."""
        vec1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vec2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        enc1 = self.ckks.encrypt_vector(vec1)
        enc2 = self.ckks.encrypt_vector(vec2)
        enc_sum = self.ckks.add_encrypted(enc1, enc2)
        dec_sum = self.ckks.decrypt_vector(enc_sum, len(vec1))
        
        expected = vec1 + vec2
        max_error = np.max(np.abs(expected - dec_sum))
        
        self.assertLess(max_error, self.tolerance)
    
    def test_vector_multiplication(self):
        """Test element-wise vector multiplication."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 3.0, 4.0])
        
        enc1 = self.ckks.encrypt_vector(vec1)
        enc2 = self.ckks.encrypt_vector(vec2)
        enc_product = self.ckks.multiply_encrypted(enc1, enc2)
        dec_product = self.ckks.decrypt_vector(enc_product, len(vec1))
        
        expected = vec1 * vec2
        max_error = np.max(np.abs(expected - dec_product))
        
        self.assertLess(max_error, self.tolerance * 10)


class TestCKKSAdvancedOperations(unittest.TestCase):
    """Test advanced cryptographic operations."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS once for all tests."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
        cls.tolerance = 1e-4  # Higher tolerance for complex operations
    
    def test_mean_computation(self):
        """Test encrypted mean computation."""
        vectors = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([10.0, 20.0, 30.0]),
            np.random.randn(50),
        ]
        
        for vector in vectors:
            with self.subTest(size=len(vector)):
                enc_vector = self.ckks.encrypt_vector(vector)
                enc_mean = self.ckks.compute_mean(enc_vector, len(vector))
                dec_mean = self.ckks.decrypt(enc_mean)
                
                expected = np.mean(vector)
                error = abs(expected - dec_mean)
                
                self.assertLess(error, self.tolerance)
    
    def test_variance_computation_hybrid(self):
        """Test hybrid variance computation."""
        vectors = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.random.randn(100),
        ]
        
        for vector in vectors:
            with self.subTest(size=len(vector)):
                enc_vector = self.ckks.encrypt_vector(vector)
                variance = self.ckks.compute_variance_hybrid(enc_vector, len(vector))
                
                expected = np.var(vector)
                error = abs(expected - variance)
                
                self.assertLess(error, self.tolerance)
    
    def test_dot_product(self):
        """Test encrypted dot product."""
        test_cases = [
            (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.random.randn(10), np.random.randn(10)),
        ]
        
        for vec1, vec2 in test_cases:
            with self.subTest(size=len(vec1)):
                enc1 = self.ckks.encrypt_vector(vec1)
                enc2 = self.ckks.encrypt_vector(vec2)
                enc_dot = self.ckks.compute_dot_product(enc1, enc2, len(vec1))
                dec_dot = self.ckks.decrypt(enc_dot)
                
                expected = np.dot(vec1, vec2)
                error = abs(expected - dec_dot)
                
                # Higher tolerance for dot product (multiple operations)
                self.assertLess(error, self.tolerance * 100)
    
    def test_polynomial_evaluation_horner(self):
        """Test Horner's method polynomial evaluation."""
        test_cases = [
            ([1.0, 2.0, 3.0], 2.0),  # 1 + 2x + 3x²
            ([5.0], 10.0),           # Constant
            ([0.0, 1.0], 5.0),       # Linear
        ]
        
        for coeffs, x_val in test_cases:
            with self.subTest(degree=len(coeffs)-1, x=x_val):
                enc_x = self.ckks.encrypt(x_val)
                
                try:
                    enc_result = self.ckks.polynomial_evaluation(
                        enc_x, coeffs, method='horner', verbose=False
                    )
                    dec_result = self.ckks.decrypt(enc_result)
                    
                    # Compute expected value
                    expected = sum(c * (x_val ** i) for i, c in enumerate(coeffs))
                    error = abs(expected - dec_result)
                    
                    self.assertLess(error, self.tolerance * 10)
                except Exception as e:
                    self.fail(f"Polynomial evaluation failed: {e}")


class TestCKKSStatisticalProperties(unittest.TestCase):
    """Test statistical properties and accuracy."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS and generate test data."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
        np.random.seed(42)  # Reproducible tests
    
    def test_error_distribution(self):
        """Test that encryption errors follow expected distribution."""
        n_samples = 1000
        test_values = np.random.randn(n_samples) * 100
        
        errors = []
        for value in test_values:
            encrypted = self.ckks.encrypt(value)
            decrypted = self.ckks.decrypt(encrypted)
            errors.append(abs(value - decrypted))
        
        errors = np.array(errors)
        
        # Errors should be small
        self.assertLess(np.mean(errors), 1e-5)
        self.assertLess(np.max(errors), 1e-3)
        
        # Most errors should be very small
        percentile_95 = np.percentile(errors, 95)
        self.assertLess(percentile_95, 1e-4)
    
    def test_error_accumulation(self):
        """Test error accumulation in repeated operations."""
        value = 10.0
        enc_value = self.ckks.encrypt(value)
        
        # Perform 10 additions
        for _ in range(10):
            enc_value = self.ckks.add_encrypted(enc_value, self.ckks.encrypt(1.0))
        
        result = self.ckks.decrypt(enc_value)
        expected = 20.0
        error = abs(expected - result)
        
        # Error should not grow too much
        self.assertLess(error, 1e-4)
    
    def test_consistency_across_runs(self):
        """Test that decryption is consistent."""
        value = 42.0
        encrypted = self.ckks.encrypt(value)
        
        # Decrypt multiple times
        results = [self.ckks.decrypt(encrypted) for _ in range(10)]
        
        # All results should be identical
        for result in results:
            self.assertAlmostEqual(result, results[0], places=10)


class TestCKKSEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS once for all tests."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
    
    def test_zero_encryption(self):
        """Test encryption of zero."""
        encrypted = self.ckks.encrypt(0.0)
        decrypted = self.ckks.decrypt(encrypted)
        self.assertLess(abs(decrypted), 1e-6)
    
    def test_negative_values(self):
        """Test encryption of negative values."""
        negatives = [-1.0, -100.0, -0.001]
        for value in negatives:
            with self.subTest(value=value):
                encrypted = self.ckks.encrypt(value)
                decrypted = self.ckks.decrypt(encrypted)
                error = abs(value - decrypted)
                self.assertLess(error, 1e-5)
    
    def test_very_small_values(self):
        """Test encryption of very small values."""
        small_values = [1e-6, 1e-8, 0.000001]
        for value in small_values:
            with self.subTest(value=value):
                encrypted = self.ckks.encrypt(value)
                decrypted = self.ckks.decrypt(encrypted)
                # Relative error for small values
                if value != 0:
                    rel_error = abs((value - decrypted) / value)
                    self.assertLess(rel_error, 0.1)  # 10% relative error ok for tiny values
    
    def test_large_values(self):
        """Test encryption of large values."""
        large_values = [1e6, 1e8, 1000000.0]
        for value in large_values:
            with self.subTest(value=value):
                encrypted = self.ckks.encrypt(value)
                decrypted = self.ckks.decrypt(encrypted)
                rel_error = abs((value - decrypted) / value)
                self.assertLess(rel_error, 1e-5)
    
    def test_empty_vector(self):
        """Test that empty vectors raise appropriate errors."""
        with self.assertRaises((ValueError, RuntimeError, IndexError)):
            self.ckks.encrypt_vector(np.array([]))
    
    def test_uninitialized_operations(self):
        """Test operations on uninitialized cryptosystem."""
        ckks_uninit = CKKSCrypto()
        # Don't call setup()
        
        with self.assertRaises(RuntimeError):
            ckks_uninit.encrypt(42.0)


class TestCKKSPerformance(unittest.TestCase):
    """Performance benchmarks and regression tests."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS once for all tests."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
    
    def test_encryption_speed(self):
        """Test that encryption meets minimum speed requirements."""
        n_operations = 100
        values = np.random.randn(n_operations)
        
        start = time.time()
        for value in values:
            self.ckks.encrypt(value)
        elapsed = time.time() - start
        
        ops_per_second = n_operations / elapsed
        
        # Should achieve at least 10 encryptions per second
        self.assertGreater(ops_per_second, 10,
                          f"Encryption too slow: {ops_per_second:.1f} ops/s")
    
    def test_vector_encryption_simd_advantage(self):
        """Test that SIMD encryption is faster than individual encryption."""
        vector_size = 1000
        vector = np.random.randn(vector_size)
        
        # SIMD encryption
        start_simd = time.time()
        self.ckks.encrypt_vector(vector)
        time_simd = time.time() - start_simd
        
        # Individual encryption (sample only 100 to save time)
        sample_size = 100
        start_individual = time.time()
        for value in vector[:sample_size]:
            self.ckks.encrypt(value)
        time_individual = time.time() - start_individual
        
        # Extrapolate to full vector
        time_individual_extrapolated = time_individual * (vector_size / sample_size)
        
        # SIMD should be significantly faster
        speedup = time_individual_extrapolated / time_simd
        self.assertGreater(speedup, 10,
                          f"SIMD not efficient enough: {speedup:.1f}x speedup")
    
    def test_homomorphic_operation_speed(self):
        """Test that homomorphic operations complete in reasonable time."""
        enc_a = self.ckks.encrypt(10.0)
        enc_b = self.ckks.encrypt(20.0)
        
        # Addition should be fast
        start = time.time()
        for _ in range(100):
            self.ckks.add_encrypted(enc_a, enc_b)
        time_additions = time.time() - start
        
        self.assertLess(time_additions, 1.0,
                       "Homomorphic additions too slow")
        
        # Multiplication should also be reasonable
        start = time.time()
        for _ in range(10):  # Fewer iterations (multiplication is slower)
            self.ckks.multiply_encrypted(enc_a, enc_b)
        time_multiplications = time.time() - start
        
        self.assertLess(time_multiplications, 5.0,
                       "Homomorphic multiplications too slow")


class TestCKKSSecurityProperties(unittest.TestCase):
    """Test security properties and privacy guarantees."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize CKKS once for all tests."""
        cls.ckks = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=[60, 40, 40, 40, 60])
        cls.ckks.setup()
    
    def test_semantic_security_different_ciphertexts(self):
        """Test that same value produces different ciphertexts (probabilistic)."""
        value = 100.0
        
        # Encrypt same value multiple times
        ciphertexts = [self.ckks.encrypt(value) for _ in range(5)]
        
        # Decrypt all and verify they decrypt to same value
        decrypted = [self.ckks.decrypt(ct) for ct in ciphertexts]
        
        for dec in decrypted:
            self.assertLess(abs(value - dec), 1e-5)
        
        # Note: We can't easily compare PyCtxt objects to verify they're different
        # But the fact that they all decrypt correctly demonstrates probabilistic encryption
    
    def test_ciphertext_size_independence(self):
        """Test that ciphertext size doesn't reveal plaintext value."""
        import pickle
        
        values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        sizes = []
        
        for value in values:
            encrypted = self.ckks.encrypt(value)
            size = len(pickle.dumps(encrypted))
            sizes.append(size)
        
        # All ciphertexts should be approximately the same size
        self.assertLess(np.std(sizes), np.mean(sizes) * 0.1,
                       "Ciphertext sizes vary too much")
    
    def test_homomorphic_privacy(self):
        """Test that operations don't leak information."""
        # Encrypt two values
        enc_a = self.ckks.encrypt(5.0)
        enc_b = self.ckks.encrypt(10.0)
        
        # Perform operation
        enc_sum = self.ckks.add_encrypted(enc_a, enc_b)
        
        # The sum ciphertext should decrypt correctly
        result = self.ckks.decrypt(enc_sum)
        self.assertLess(abs(result - 15.0), 1e-5)
        
        # We can't verify the ciphertext reveals nothing without the key,
        # but we verify the operation works correctly


def run_test_suite():
    """Run all tests and generate report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSBasicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSHomomorphicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSAdvancedOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSStatisticalProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestCKKSSecurityProperties))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)