"""
Comprehensive BFV Testing Suite for Masters Research

This module provides systematic testing including:
- Correctness verification
- Performance benchmarking  
- Accuracy analysis
- Parameter sensitivity
- Scalability testing
- Statistical validation
- Edge case handling
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
from typing import Dict, List, Tuple
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.schemes.bfv.bfv_crypto import BFVCrypto


class BFVTestSuite:
    """Comprehensive testing suite for BFV implementation."""
    
    def __init__(self, results_dir: str = "results/tests/bfv"):
        """Initialize test suite."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {
            'correctness': {},
            'performance': {},
            'accuracy': {},
            'scalability': {},
            'statistical': {}
        }
        
    # ==================== CORRECTNESS TESTS ====================
    
    def test_correctness_basic_operations(self) -> Dict:
        """Test basic encryption operations for correctness."""
        print("\n" + "="*70)
        print("TEST 1: Basic Operations Correctness")
        print("="*70)
        
        bfv = BFVCrypto()
        bfv.setup()
        
        results = {
            'test_name': 'basic_operations',
            'tests_passed': 0,
            'tests_failed': 0,
            'details': []
        }
        
        # Test 1.1: Encryption/Decryption Identity
        print("\n1.1 Testing encryption/decryption identity...")
        test_values = [0, 1, -1, 42, -42, 100, -100, 1000, -1000]
        
        for val in test_values:
            enc = bfv.encrypt(val)
            dec = bfv.decrypt(enc)
            
            passed = abs(dec - val) < 1e-6
            results['tests_passed' if passed else 'tests_failed'] += 1
            
            if not passed:
                results['details'].append({
                    'test': 'encrypt_decrypt',
                    'input': val,
                    'expected': val,
                    'actual': dec,
                    'error': abs(dec - val),
                    'passed': False
                })
            
            print(f"  Value={val:5d}, Decrypted={dec:8.2f}, "
                  f"Error={abs(dec-val):.2e}, {'✓' if passed else '✗'}")
        
        # Test 1.2: Homomorphic Addition
        print("\n1.2 Testing homomorphic addition...")
        test_pairs = [(5, 10), (100, 200), (-50, 30), (0, 42)]
        
        for a, b in test_pairs:
            enc_a = bfv.encrypt(a)
            enc_b = bfv.encrypt(b)
            enc_sum = bfv.add_encrypted(enc_a, enc_b)
            dec_sum = bfv.decrypt(enc_sum)
            expected = a + b
            
            passed = abs(dec_sum - expected) < 1e-6
            results['tests_passed' if passed else 'tests_failed'] += 1
            
            print(f"  {a} + {b} = {expected}, "
                  f"Encrypted={dec_sum:.2f}, "
                  f"Error={abs(dec_sum-expected):.2e}, {'✓' if passed else '✗'}")
        
        # Test 1.3: Homomorphic Multiplication  
        print("\n1.3 Testing homomorphic multiplication...")
        test_pairs = [(3, 5), (10, 10), (-5, 4), (7, -3)]
        
        for a, b in test_pairs:
            enc_a = bfv.encrypt(a)
            enc_b = bfv.encrypt(b)
            enc_prod = bfv.multiply_encrypted(enc_a, enc_b)
            dec_prod = bfv.decrypt(enc_prod)
            expected = a * b
            
            passed = abs(dec_prod - expected) < 1e-6
            results['tests_passed' if passed else 'tests_failed'] += 1
            
            print(f"  {a} × {b} = {expected}, "
                  f"Encrypted={dec_prod:.2f}, "
                  f"Error={abs(dec_prod-expected):.2e}, {'✓' if passed else '✗'}")
        
        # Test 1.4: Commutativity
        print("\n1.4 Testing commutativity...")
        a, b = 12, 15
        enc_a = bfv.encrypt(a)
        enc_b = bfv.encrypt(b)
        
        sum_ab = bfv.decrypt(bfv.add_encrypted(enc_a, enc_b))
        sum_ba = bfv.decrypt(bfv.add_encrypted(enc_b, enc_a))
        
        passed = abs(sum_ab - sum_ba) < 1e-6
        results['tests_passed' if passed else 'tests_failed'] += 1
        print(f"  Add commutative: {passed} ({'✓' if passed else '✗'})")
        
        prod_ab = bfv.decrypt(bfv.multiply_encrypted(enc_a, enc_b))
        prod_ba = bfv.decrypt(bfv.multiply_encrypted(enc_b, enc_a))
        
        passed = abs(prod_ab - prod_ba) < 1e-6
        results['tests_passed' if passed else 'tests_failed'] += 1
        print(f"  Multiply commutative: {passed} ({'✓' if passed else '✗'})")
        
        # Test 1.5: Associativity
        print("\n1.5 Testing associativity...")
        a, b, c = 5, 7, 11
        enc_a, enc_b, enc_c = bfv.encrypt(a), bfv.encrypt(b), bfv.encrypt(c)
        
        # (a + b) + c = a + (b + c)
        left = bfv.decrypt(bfv.add_encrypted(
            bfv.add_encrypted(enc_a, enc_b), enc_c))
        right = bfv.decrypt(bfv.add_encrypted(
            enc_a, bfv.add_encrypted(enc_b, enc_c)))
        
        passed = abs(left - right) < 1e-6
        results['tests_passed' if passed else 'tests_failed'] += 1
        print(f"  Add associative: {passed} ({'✓' if passed else '✗'})")
        
        results['pass_rate'] = results['tests_passed'] / (
            results['tests_passed'] + results['tests_failed']) * 100
        
        self.test_results['correctness']['basic_operations'] = results
        
        print(f"\n✓ Correctness Tests: {results['tests_passed']}/{results['tests_passed'] + results['tests_failed']} passed "
              f"({results['pass_rate']:.1f}%)")
        
        return results
    
    # ==================== PERFORMANCE BENCHMARKS ====================
    
    def test_performance_operations(self, num_trials: int = 100) -> Dict:
        """Benchmark performance of BFV operations."""
        print("\n" + "="*70)
        print("TEST 2: Performance Benchmarking")
        print("="*70)
        
        bfv = BFVCrypto()
        bfv.setup()
        
        results = {
            'test_name': 'performance_benchmark',
            'num_trials': num_trials,
            'operations': {}
        }
        
        # Test 2.1: Encryption Performance
        print(f"\n2.1 Benchmarking encryption ({num_trials} trials)...")
        test_values = np.random.randint(-1000, 1000, num_trials)
        
        times = []
        for val in test_values:
            start = time.perf_counter()
            _ = bfv.encrypt(val)
            times.append((time.perf_counter() - start) * 1000)  # ms
        
        results['operations']['encryption'] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }
        
        print(f"  Mean: {results['operations']['encryption']['mean_ms']:.3f} ms")
        print(f"  Std:  {results['operations']['encryption']['std_ms']:.3f} ms")
        
        # Test 2.2: Decryption Performance
        print(f"\n2.2 Benchmarking decryption ({num_trials} trials)...")
        encrypted_values = [bfv.encrypt(val) for val in test_values[:num_trials]]
        
        times = []
        for enc_val in encrypted_values:
            start = time.perf_counter()
            _ = bfv.decrypt(enc_val)
            times.append((time.perf_counter() - start) * 1000)
        
        results['operations']['decryption'] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times))
        }
        
        print(f"  Mean: {results['operations']['decryption']['mean_ms']:.3f} ms")
        
        # Test 2.3: Addition Performance
        print(f"\n2.3 Benchmarking addition ({num_trials} trials)...")
        pairs = [(encrypted_values[i], encrypted_values[i+1]) 
                 for i in range(0, len(encrypted_values)-1, 2)]
        
        times = []
        for enc_a, enc_b in pairs[:num_trials//2]:
            start = time.perf_counter()
            _ = bfv.add_encrypted(enc_a, enc_b)
            times.append((time.perf_counter() - start) * 1000)
        
        results['operations']['addition'] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'operations_per_sec': float(1000 / np.mean(times))
        }
        
        print(f"  Mean: {results['operations']['addition']['mean_ms']:.3f} ms")
        print(f"  Ops/sec: {results['operations']['addition']['operations_per_sec']:.0f}")
        
        # Test 2.4: Multiplication Performance
        print(f"\n2.4 Benchmarking multiplication ({num_trials//2} trials)...")
        
        times = []
        for enc_a, enc_b in pairs[:num_trials//4]:
            start = time.perf_counter()
            _ = bfv.multiply_encrypted(enc_a, enc_b)
            times.append((time.perf_counter() - start) * 1000)
        
        results['operations']['multiplication'] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'operations_per_sec': float(1000 / np.mean(times))
        }
        
        print(f"  Mean: {results['operations']['multiplication']['mean_ms']:.3f} ms")
        
        # Performance ratios
        print(f"\n2.5 Performance ratios:")
        mult_add_ratio = (results['operations']['multiplication']['mean_ms'] / 
                         results['operations']['addition']['mean_ms'])
        print(f"  Multiplication/Addition ratio: {mult_add_ratio:.2f}x")
        
        results['performance_ratios'] = {
            'multiplication_to_addition': mult_add_ratio,
            'encryption_to_addition': (results['operations']['encryption']['mean_ms'] /
                                       results['operations']['addition']['mean_ms'])
        }
        
        self.test_results['performance']['operations'] = results
        
        return results
    
    # ==================== ACCURACY TESTS ====================
    
    def test_accuracy_statistical_operations(self, data_size: int = 100) -> Dict:
        """Test accuracy of statistical computations."""
        print("\n" + "="*70)
        print("TEST 3: Statistical Accuracy Testing")
        print("="*70)
        
        bfv = BFVCrypto()
        bfv.setup()
        
        results = {
            'test_name': 'statistical_accuracy',
            'data_size': data_size,
            'tests': {}
        }
        
        # Generate test data
        np.random.seed(42)  # Reproducibility
        test_data = np.random.normal(100, 20, data_size).round().astype(int)
        
        # Test 3.1: Mean Accuracy
        print(f"\n3.1 Testing mean accuracy (n={data_size})...")
        
        # Plaintext baseline
        true_mean = float(np.mean(test_data))
        
        # Encrypted computation
        enc_values = bfv.encrypt_vector(test_data)
        enc_sum = bfv.sum_encrypted(enc_values)
        enc_mean = bfv.multiply_plain(enc_sum, 1.0/data_size)
        computed_mean = bfv.decrypt(enc_mean)
        
        abs_error = abs(computed_mean - true_mean)
        rel_error = abs_error / abs(true_mean) if true_mean != 0 else 0
        
        results['tests']['mean'] = {
            'true_value': true_mean,
            'computed_value': computed_mean,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'acceptable': rel_error < 0.01  # 1% tolerance
        }
        
        print(f"  True mean: {true_mean:.4f}")
        print(f"  Computed mean: {computed_mean:.4f}")
        print(f"  Absolute error: {abs_error:.6f}")
        print(f"  Relative error: {rel_error*100:.4f}%")
        print(f"  {'✓ PASS' if results['tests']['mean']['acceptable'] else '✗ FAIL'}")
        
        # Test 3.2: Sum Accuracy
        print(f"\n3.2 Testing sum accuracy...")
        
        true_sum = float(np.sum(test_data))
        computed_sum = bfv.decrypt(enc_sum)
        
        abs_error = abs(computed_sum - true_sum)
        rel_error = abs_error / abs(true_sum) if true_sum != 0 else 0
        
        results['tests']['sum'] = {
            'true_value': true_sum,
            'computed_value': computed_sum,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'acceptable': rel_error < 0.01
        }
        
        print(f"  True sum: {true_sum:.2f}")
        print(f"  Computed sum: {computed_sum:.2f}")
        print(f"  Relative error: {rel_error*100:.6f}%")
        
        # Test 3.3: Variance Accuracy
        print(f"\n3.3 Testing variance accuracy...")
        
        true_var = float(np.var(test_data))
        enc_var = bfv.compute_variance(enc_values, enc_mean)
        computed_var = bfv.decrypt(enc_var)
        
        abs_error = abs(computed_var - true_var)
        rel_error = abs_error / abs(true_var) if true_var != 0 else 0
        
        results['tests']['variance'] = {
            'true_value': true_var,
            'computed_value': computed_var,
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'acceptable': rel_error < 0.05  # 5% tolerance (looser due to squaring)
        }
        
        print(f"  True variance: {true_var:.4f}")
        print(f"  Computed variance: {computed_var:.4f}")
        print(f"  Relative error: {rel_error*100:.4f}%")
        
        # Overall accuracy assessment
        all_acceptable = all(test['acceptable'] for test in results['tests'].values())
        results['all_tests_passed'] = all_acceptable
        
        print(f"\n{'✓ All accuracy tests PASSED' if all_acceptable else '✗ Some tests FAILED'}")
        
        self.test_results['accuracy']['statistical'] = results
        
        return results
    
    # ==================== SCALABILITY TESTS ====================
    
    def test_scalability_data_size(self, sizes: List[int] = None) -> Dict:
        """Test how performance scales with data size."""
        print("\n" + "="*70)
        print("TEST 4: Scalability Analysis")
        print("="*70)
        
        if sizes is None:
            sizes = [10, 50, 100, 250, 500, 1000]
        
        bfv = BFVCrypto()
        bfv.setup()
        
        results = {
            'test_name': 'scalability',
            'data_sizes': sizes,
            'measurements': []
        }
        
        for size in sizes:
            print(f"\n4.Testing with data size = {size}...")
            
            # Generate data
            data = np.random.randint(-100, 100, size)
            
            # Measure encryption time
            start = time.perf_counter()
            enc_values = bfv.encrypt_vector(data)
            encryption_time = time.perf_counter() - start
            
            # Measure sum computation time
            start = time.perf_counter()
            enc_sum = bfv.sum_encrypted(enc_values)
            sum_time = time.perf_counter() - start
            
            # Measure decryption time
            start = time.perf_counter()
            _ = bfv.decrypt_vector(enc_values)
            decryption_time = time.perf_counter() - start
            
            total_time = encryption_time + sum_time + decryption_time
            
            measurement = {
                'size': size,
                'encryption_time_s': encryption_time,
                'sum_time_s': sum_time,
                'decryption_time_s': decryption_time,
                'total_time_s': total_time,
                'throughput_ops_per_sec': size / total_time
            }
            
            results['measurements'].append(measurement)
            
            print(f"  Encryption: {encryption_time:.3f}s")
            print(f"  Sum: {sum_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
            print(f"  Throughput: {measurement['throughput_ops_per_sec']:.1f} ops/sec")
        
        # Analyze scaling behavior
        sizes_arr = np.array([m['size'] for m in results['measurements']])
        times_arr = np.array([m['total_time_s'] for m in results['measurements']])
        
        # Fit linear model: time = a * size + b (manual linear regression)
        # Using least squares: slope = cov(x,y) / var(x)
        mean_x = np.mean(sizes_arr)
        mean_y = np.mean(times_arr)
        
        numerator = np.sum((sizes_arr - mean_x) * (times_arr - mean_y))
        denominator = np.sum((sizes_arr - mean_x) ** 2)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = mean_y - slope * mean_x
        
        # Calculate R-squared
        y_pred = slope * sizes_arr + intercept
        ss_res = np.sum((times_arr - y_pred) ** 2)
        ss_tot = np.sum((times_arr - mean_y) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results['scaling_analysis'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'scaling_type': 'linear' if r_squared > 0.95 else 'non-linear'
        }
        
        print(f"\nScaling analysis:")
        print(f"  R² = {results['scaling_analysis']['r_squared']:.4f}")
        print(f"  Scaling: {results['scaling_analysis']['scaling_type']}")
        
        self.test_results['scalability']['data_size'] = results
        
        return results
    
    # ==================== EDGE CASE TESTS ====================
    
    def test_edge_cases(self) -> Dict:
        """Test edge cases and boundary conditions."""
        print("\n" + "="*70)
        print("TEST 5: Edge Case Testing")
        print("="*70)
        
        bfv = BFVCrypto()
        bfv.setup()
        
        results = {
            'test_name': 'edge_cases',
            'tests': []
        }
        
        # Test 5.1: Zero values
        print("\n5.1 Testing zero handling...")
        enc_zero = bfv.encrypt(0)
        dec_zero = bfv.decrypt(enc_zero)
        passed = abs(dec_zero) < 1e-6
        results['tests'].append({'test': 'zero_value', 'passed': passed})
        print(f"  Zero encryption/decryption: {'✓' if passed else '✗'}")
        
        # Test 5.2: Large values
        print("\n5.2 Testing large values...")
        large_val = 1000000
        enc_large = bfv.encrypt(large_val)
        dec_large = bfv.decrypt(enc_large)
        passed = abs(dec_large - large_val) / large_val < 0.01
        results['tests'].append({'test': 'large_value', 'passed': passed})
        print(f"  Large value (1M): {'✓' if passed else '✗'}, Error: {abs(dec_large - large_val)}")
        
        # Test 5.3: Negative values
        print("\n5.3 Testing negative values...")
        neg_val = -500
        enc_neg = bfv.encrypt(neg_val)
        dec_neg = bfv.decrypt(enc_neg)
        passed = abs(dec_neg - neg_val) < 1e-6
        results['tests'].append({'test': 'negative_value', 'passed': passed})
        print(f"  Negative value: {'✓' if passed else '✗'}")
        
        # Test 5.4: Chain of operations
        print("\n5.4 Testing operation chains...")
        a, b, c = 5, 10, 15
        enc_a, enc_b, enc_c = bfv.encrypt(a), bfv.encrypt(b), bfv.encrypt(c)
        
        # ((a + b) * c) + a
        result = bfv.add_encrypted(
            bfv.multiply_encrypted(
                bfv.add_encrypted(enc_a, enc_b), 
                enc_c
            ),
            enc_a
        )
        expected = ((a + b) * c) + a
        dec_result = bfv.decrypt(result)
        passed = abs(dec_result - expected) < 1e-6
        results['tests'].append({'test': 'operation_chain', 'passed': passed})
        print(f"  Chain result: {dec_result:.2f}, Expected: {expected}, {'✓' if passed else '✗'}")
        
        results['pass_rate'] = sum(1 for t in results['tests'] if t['passed']) / len(results['tests']) * 100
        
        print(f"\n✓ Edge cases: {results['pass_rate']:.0f}% passed")
        
        self.test_results['correctness']['edge_cases'] = results
        
        return results
    
    # ==================== STATISTICAL VALIDATION ====================
    
    def test_statistical_properties(self, num_samples: int = 1000) -> Dict:
        """Test statistical properties of encrypted operations."""
        print("\n" + "="*70)
        print("TEST 6: Statistical Properties Validation")
        print("="*70)
        
        bfv = BFVCrypto()
        bfv.setup()
        
        results = {
            'test_name': 'statistical_properties',
            'num_samples': num_samples,
            'tests': {}
        }
        
        # Test 6.1: Error distribution
        print(f"\n6.1 Analyzing encryption error distribution ({num_samples} samples)...")
        
        test_values = np.random.randint(-1000, 1000, num_samples)
        errors = []
        
        for val in test_values:
            enc = bfv.encrypt(val)
            dec = bfv.decrypt(enc)
            errors.append(dec - val)
        
        errors = np.array(errors)
        
        # Statistical tests (without scipy)
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        
        # Simple normality check using skewness and kurtosis
        # Normal distribution has skewness ≈ 0 and kurtosis ≈ 3
        # Calculate manually without scipy
        n = len(errors)
        m2 = np.mean((errors - mean_error)**2)
        m3 = np.mean((errors - mean_error)**3)
        m4 = np.mean((errors - mean_error)**4)
        
        skewness = m3 / (m2**1.5) if m2 > 0 else 0
        kurtosis = m4 / (m2**2) if m2 > 0 else 0
        
        # Approximate normality: skewness close to 0, kurtosis close to 3
        is_approximately_normal = (abs(skewness) < 0.5 and abs(kurtosis - 3) < 1.0)
        
        results['tests']['error_distribution'] = {
            'mean_error': mean_error,
            'std_error': std_error,
            'max_abs_error': float(np.max(np.abs(errors))),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'is_approximately_normal': is_approximately_normal,
            'is_unbiased': abs(mean_error) < 0.1  # Mean should be near zero
        }
        
        print(f"  Mean error: {mean_error:.6f}")
        print(f"  Std error: {std_error:.6f}")
        print(f"  Max abs error: {results['tests']['error_distribution']['max_abs_error']:.6f}")
        print(f"  Skewness: {skewness:.4f} (normal ≈ 0)")
        print(f"  Kurtosis: {kurtosis:.4f} (normal ≈ 3)")
        print(f"  Approximately normal: {'✓' if is_approximately_normal else '✗'}")
        print(f"  Unbiased: {'✓' if results['tests']['error_distribution']['is_unbiased'] else '✗'}")
        
        self.test_results['statistical']['properties'] = results
        
        return results
    
    # ==================== REPORTING ====================
    
    def save_results(self, filename: str = "bfv_test_results.json"):
        """Save all test results to JSON."""
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {filepath}")
        
    def generate_report(self):
        """Generate comprehensive test report."""
        report_path = self.results_dir / "BFV_TEST_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# BFV Implementation - Comprehensive Test Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Correctness summary
            if 'basic_operations' in self.test_results.get('correctness', {}):
                res = self.test_results['correctness']['basic_operations']
                f.write(f"- **Correctness Tests**: {res['pass_rate']:.1f}% passed ")
                f.write(f"({res['tests_passed']}/{res['tests_passed'] + res['tests_failed']})\n")
            
            # Performance summary
            if 'operations' in self.test_results.get('performance', {}):
                res = self.test_results['performance']['operations']
                f.write(f"- **Encryption Performance**: ")
                f.write(f"{res['operations']['encryption']['mean_ms']:.3f} ms avg\n")
                f.write(f"- **Addition Performance**: ")
                f.write(f"{res['operations']['addition']['operations_per_sec']:.0f} ops/sec\n")
            
            # Accuracy summary
            if 'statistical' in self.test_results.get('accuracy', {}):
                res = self.test_results['accuracy']['statistical']
                f.write(f"- **Mean Accuracy**: ")
                f.write(f"{res['tests']['mean']['relative_error']*100:.4f}% relative error\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("See `bfv_test_results.json` for complete numerical results.\n")
        
        print(f"📄 Test report saved to: {report_path}")
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "="*70)
        print("BFV COMPREHENSIVE TEST SUITE")
        print("Masters-Level Evaluation")
        print("="*70)
        
        # Run all tests
        self.test_correctness_basic_operations()
        self.test_performance_operations(num_trials=100)
        self.test_accuracy_statistical_operations(data_size=100)
        self.test_scalability_data_size()
        self.test_edge_cases()
        self.test_statistical_properties(num_samples=500)
        
        # Save results
        self.save_results()
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)


if __name__ == "__main__":
    suite = BFVTestSuite()
    suite.run_all_tests()
