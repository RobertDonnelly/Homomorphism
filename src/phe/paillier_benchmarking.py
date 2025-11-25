"""
Performance Benchmarking Suite for Paillier Homomorphic Encryption
Measures encryption, decryption, and homomorphic operation performance
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
import json
from datetime import datetime
from memory_profiler import memory_usage
import psutil


from paillier_crypto import PaillierCrypto

class PaillierBenchmark:
    """Comprehensive benchmarking suite for Paillier operations."""
    
    def __init__(self, key_sizes: List[int] = [1024, 2048, 3072]):
        """
        Initialize benchmark suite.
        
        Args:
            key_sizes: List of key sizes to benchmark
        """
        self.key_sizes = key_sizes
        self.results = {}
        
    def benchmark_keygen(self, key_size: int, iterations: int = 3) -> Dict:
        """Benchmark key generation."""
        print(f"\n📊 Benchmarking key generation ({key_size}-bit)...")
        
        times = []
        for i in range(iterations):
            paillier_sys = PaillierCrypto(key_size=key_size)
            start = time.time()
            paillier_sys.generate_keypair()
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}/{iterations}: {elapsed:.3f}s")
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'times': times
        }
    
    def benchmark_single_operations(self, key_size: int, 
                                   iterations: int = 100) -> Dict:
        """Benchmark single value operations."""
        print(f"\n📊 Benchmarking single operations ({key_size}-bit)...")
        
        paillier_sys = PaillierCrypto(key_size=key_size)
        paillier_sys.generate_keypair()
        
        results = {}
        test_value = 42.5
        
        # Encryption
        print("  - Encryption...")
        times = []
        for _ in range(iterations):
            start = time.time()
            enc = paillier_sys.encrypt(test_value)
            times.append(time.time() - start)
        results['encryption'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'throughput': iterations / sum(times)
        }
        
        # Decryption
        print("  - Decryption...")
        enc_value = paillier_sys.encrypt(test_value)
        times = []
        for _ in range(iterations):
            start = time.time()
            dec = paillier_sys.decrypt(enc_value)
            times.append(time.time() - start)
        results['decryption'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'throughput': iterations / sum(times)
        }
        
        # Homomorphic addition
        print("  - Homomorphic addition...")
        enc1 = paillier_sys.encrypt(test_value)
        enc2 = paillier_sys.encrypt(test_value * 2)
        times = []
        for _ in range(iterations):
            start = time.time()
            result = paillier_sys.add_encrypted(enc1, enc2)
            times.append(time.time() - start)
        results['addition'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'throughput': iterations / sum(times)
        }
        
        # Scalar multiplication
        print("  - Scalar multiplication...")
        times = []
        for _ in range(iterations):
            start = time.time()
            result = paillier_sys.multiply_encrypted_by_scalar(enc1, 2.5)
            times.append(time.time() - start)
        results['scalar_mult'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'throughput': iterations / sum(times)
        }
        
        return results
    
    def benchmark_vector_operations(self, key_size: int, 
                                   vector_sizes: List[int] = [10, 100, 1000],
                                   iterations: int = 10) -> Dict:
        """Benchmark vector operations for different sizes."""
        print(f"\n📊 Benchmarking vector operations ({key_size}-bit)...")
        
        paillier_sys = PaillierCrypto(key_size=key_size)
        paillier_sys.generate_keypair()
        
        results = {}
        
        for vec_size in vector_sizes:
            print(f"  - Vector size: {vec_size}")
            test_vector = np.random.randn(vec_size)
            
            # Vector encryption
            times = []
            for _ in range(iterations):
                start = time.time()
                enc_vec = paillier_sys.encrypt_vector(test_vector)
                times.append(time.time() - start)
            
            enc_time_mean = np.mean(times)
            
            # Vector decryption
            enc_vec = paillier_sys.encrypt_vector(test_vector)
            times = []
            for _ in range(iterations):
                start = time.time()
                dec_vec = paillier_sys.decrypt_vector(enc_vec)
                times.append(time.time() - start)
            
            dec_time_mean = np.mean(times)
            
            # Vector addition
            enc_vec1 = paillier_sys.encrypt_vector(test_vector)
            enc_vec2 = paillier_sys.encrypt_vector(test_vector * 2)
            times = []
            for _ in range(iterations):
                start = time.time()
                sum_vec = paillier_sys.add_encrypted_vectors(enc_vec1, enc_vec2)
                times.append(time.time() - start)
            
            add_time_mean = np.mean(times)
            
            results[vec_size] = {
                'encryption': enc_time_mean,
                'decryption': dec_time_mean,
                'addition': add_time_mean,
                'enc_per_element': enc_time_mean / vec_size,
                'dec_per_element': dec_time_mean / vec_size
            }
        
        return results
    
    def benchmark_federated_aggregation(self, key_size: int, 
                                       num_clients: List[int] = [5, 10, 20],
                                       param_size: int = 100,
                                       iterations: int = 5) -> Dict:
        """Benchmark federated learning aggregation scenario."""
        print(f"\n📊 Benchmarking FL aggregation ({key_size}-bit)...")
        
        paillier_sys = PaillierCrypto(key_size=key_size)
        paillier_sys.generate_keypair()
        
        results = {}
        
        for n_clients in num_clients:
            print(f"  - {n_clients} clients, {param_size} parameters")
            
            # Generate client parameters
            client_params = [np.random.randn(param_size) for _ in range(n_clients)]
            weights = np.random.dirichlet(np.ones(n_clients))  # Random weights that sum to 1
            
            # Encrypt client parameters
            enc_start = time.time()
            enc_params = [paillier_sys.encrypt_vector(params) for params in client_params]
            enc_time = time.time() - enc_start
            
            # Aggregate
            agg_times = []
            for _ in range(iterations):
                start = time.time()
                enc_avg = paillier_sys.weighted_average_encrypted(enc_params, weights)
                agg_times.append(time.time() - start)
            
            agg_time_mean = np.mean(agg_times)
            
            # Decrypt result
            dec_start = time.time()
            dec_avg = paillier_sys.decrypt_vector(enc_avg)
            dec_time = time.time() - dec_start
            
            total_time = enc_time + agg_time_mean + dec_time
            
            results[n_clients] = {
                'encryption_time': enc_time,
                'aggregation_time': agg_time_mean,
                'decryption_time': dec_time,
                'total_time': total_time,
                'time_per_client': total_time / n_clients
            }
        
        return results
    
    def benchmark_memory_usage(self, key_size: int, vector_size: int = 1000):
        """Benchmark memory usage."""
        print(f"\n📊 Benchmarking memory usage ({key_size}-bit)...")
        
        def encrypt_vector():
            paillier_sys = PaillierCrypto(key_size=key_size)
            paillier_sys.generate_keypair()
            test_vector = np.random.randn(vector_size)
            enc_vec = paillier_sys.encrypt_vector(test_vector)
            return enc_vec
        
        mem_usage = memory_usage(encrypt_vector, interval=0.1, timeout=None)
        
        return {
            'peak_memory_mb': max(mem_usage),
            'avg_memory_mb': np.mean(mem_usage),
            'memory_increase_mb': max(mem_usage) - min(mem_usage)
        }
    
    def run_comprehensive_benchmark(self, save_results: bool = True):
        """Run all benchmarks and compile results."""
        print("\n" + "="*70)
        print("COMPREHENSIVE PAILLIER PERFORMANCE BENCHMARK")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for key_size in self.key_sizes:
            print(f"\n{'='*70}")
            print(f"KEY SIZE: {key_size} bits")
            print(f"{'='*70}")
            
            self.results[key_size] = {}
            
            # Key generation
            self.results[key_size]['keygen'] = self.benchmark_keygen(key_size)
            
            # Single operations
            self.results[key_size]['single_ops'] = self.benchmark_single_operations(key_size)
            
            # Vector operations
            self.results[key_size]['vector_ops'] = self.benchmark_vector_operations(key_size)
            
            # FL aggregation
            self.results[key_size]['fl_aggregation'] = self.benchmark_federated_aggregation(key_size)
            
            # Memory usage
            self.results[key_size]['memory'] = self.benchmark_memory_usage(key_size)
        
        if save_results:
            filename = f"benchmark_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n✓ Results saved to {filename}")
        
        self.generate_visualizations(timestamp)
        self.print_summary()
        
        return self.results
    
    def generate_visualizations(self, timestamp: str):
        """Generate benchmark visualization plots."""
        print("\n📈 Generating visualizations...")
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Paillier Cryptosystem Performance Benchmarks', fontsize=16, fontweight='bold')
        
        # Plot 1: Single operations by key size
        ax1 = axes[0, 0]
        ops = ['encryption', 'decryption', 'addition', 'scalar_mult']
        op_names = ['Encryption', 'Decryption', 'Addition', 'Scalar Mult']
        
        x = np.arange(len(ops))
        width = 0.25
        
        for i, key_size in enumerate(self.key_sizes):
            times = [self.results[key_size]['single_ops'][op]['mean'] * 1000 
                    for op in ops]
            ax1.bar(x + i*width, times, width, label=f'{key_size}-bit')
        
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Single Operation Performance')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(op_names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot 2: Vector encryption time vs size
        ax2 = axes[0, 1]
        for key_size in self.key_sizes:
            vec_ops = self.results[key_size]['vector_ops']
            sizes = list(vec_ops.keys())
            times = [vec_ops[size]['encryption'] for size in sizes]
            ax2.plot(sizes, times, marker='o', label=f'{key_size}-bit')
        
        ax2.set_xlabel('Vector Size')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Vector Encryption Time')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # Plot 3: FL Aggregation time vs number of clients
        ax3 = axes[1, 0]
        for key_size in self.key_sizes:
            fl_ops = self.results[key_size]['fl_aggregation']
            n_clients = list(fl_ops.keys())
            times = [fl_ops[n]['total_time'] for n in n_clients]
            ax3.plot(n_clients, times, marker='s', label=f'{key_size}-bit')
        
        ax3.set_xlabel('Number of Clients')
        ax3.set_ylabel('Total Time (s)')
        ax3.set_title('Federated Learning Aggregation')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Memory usage
        ax4 = axes[1, 1]
        key_sizes_list = list(self.results.keys())
        peak_mem = [self.results[ks]['memory']['peak_memory_mb'] for ks in key_sizes_list]
        mem_increase = [self.results[ks]['memory']['memory_increase_mb'] for ks in key_sizes_list]
        
        x = np.arange(len(key_sizes_list))
        width = 0.35
        ax4.bar(x - width/2, peak_mem, width, label='Peak Memory')
        ax4.bar(x + width/2, mem_increase, width, label='Memory Increase')
        
        ax4.set_ylabel('Memory (MB)')
        ax4.set_title('Memory Usage')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{ks}-bit' for ks in key_sizes_list])
        ax4.legend()
        
        plt.tight_layout()
        filename = f"benchmark_plots_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {filename}")
        plt.close()
    
    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        for key_size in self.key_sizes:
            print(f"\n{key_size}-bit Key:")
            print("-" * 50)
            
            # Key generation
            keygen = self.results[key_size]['keygen']
            print(f"  Key Generation: {keygen['mean']:.3f}s (±{keygen['std']:.3f}s)")
            
            # Single operations
            single = self.results[key_size]['single_ops']
            print(f"  Encryption:     {single['encryption']['mean']*1000:.2f}ms")
            print(f"  Decryption:     {single['decryption']['mean']*1000:.2f}ms")
            print(f"  Addition:       {single['addition']['mean']*1000:.2f}ms")
            print(f"  Scalar Mult:    {single['scalar_mult']['mean']*1000:.2f}ms")
            
            # Vector ops (size 100)
            if 100 in self.results[key_size]['vector_ops']:
                vec = self.results[key_size]['vector_ops'][100]
                print(f"  Vector (100):   {vec['encryption']:.3f}s encrypt, "
                      f"{vec['decryption']:.3f}s decrypt")
            
            # FL aggregation (10 clients)
            if 10 in self.results[key_size]['fl_aggregation']:
                fl = self.results[key_size]['fl_aggregation'][10]
                print(f"  FL (10 clients): {fl['total_time']:.3f}s total")
            
            # Memory
            mem = self.results[key_size]['memory']
            print(f"  Peak Memory:    {mem['peak_memory_mb']:.2f}MB")


def quick_benchmark():
    """Run a quick benchmark with default settings."""
    print("Running quick benchmark (this may take a few minutes)...")
    benchmark = PaillierBenchmark(key_sizes=[2048])
    results = benchmark.run_comprehensive_benchmark()
    return results


def full_benchmark():
    """Run comprehensive benchmark with multiple key sizes."""
    print("Running comprehensive benchmark (this will take some time)...")
    benchmark = PaillierBenchmark(key_sizes=[1024, 2048, 3072])
    results = benchmark.run_comprehensive_benchmark()
    return results


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("PAILLIER HOMOMORPHIC ENCRYPTION - BENCHMARKING SUITE")
    print("="*70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        full_benchmark()
    else:
        print("\nRunning quick benchmark with 2048-bit keys...")
        print("(Use '--full' flag for comprehensive benchmark with multiple key sizes)")
        quick_benchmark()
    
    print("\n✓ Benchmarking complete!")