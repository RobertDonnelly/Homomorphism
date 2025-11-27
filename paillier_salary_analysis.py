"""
Employee Salary Analysis with Paillier Encryption
Analyze 'Monthly_Salary' column from employee_salary_dataset.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.phe.paillier_crypto import PaillierCrypto


class SalaryAnalyzer:
    """Analyze salary data with Paillier homomorphic encryption."""
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize analyzer with Paillier cryptosystem.
        
        Args:
            key_size: Bit length for Paillier keys
        """
        print("="*70)
        print("EMPLOYEE SALARY ANALYSIS - PAILLIER ENCRYPTION")
        print("="*70)
        
        self.paillier = PaillierCrypto(key_size=key_size)
        self.paillier.generate_keypair()
        
        self.data_dir = project_root / 'data'
        self.results_dir = project_root / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.analysis_results = {}
        
    def load_dataset(self, filename: str = 'employee_salary_dataset.csv'):
        """Load employee salary dataset."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}\n"
                f"Please place '{filename}' in the 'data/' directory"
            )
        
        print(f"\n📂 Loading dataset: {filename}")
        self.df = pd.read_csv(filepath)
        
        print(f"✓ Loaded successfully!")
        print(f"  Rows: {len(self.df)}")
        print(f"  Columns: {list(self.df.columns)}")
        
        # Check if Monthly_Salary exists
        if 'Monthly_Salary' not in self.df.columns:
            print(f"\n⚠️  Warning: 'Monthly_Salary' column not found!")
            print(f"  Available columns: {list(self.df.columns)}")
            return False
        
        # Basic statistics
        print(f"\n📊 Monthly_Salary Statistics:")
        print(f"  Count: {self.df['Monthly_Salary'].count()}")
        print(f"  Mean:  ${self.df['Monthly_Salary'].mean():,.2f}")
        print(f"  Std:   ${self.df['Monthly_Salary'].std():,.2f}")
        print(f"  Min:   ${self.df['Monthly_Salary'].min():,.2f}")
        print(f"  Max:   ${self.df['Monthly_Salary'].max():,.2f}")
        
        return True
    
    def encrypt_salaries(self):
        """Encrypt Monthly_Salary column."""
        print("\n" + "="*70)
        print("PHASE 1: ENCRYPTION")
        print("="*70)
        
        salaries = self.df['Monthly_Salary'].fillna(0).values
        
        print(f"\n🔒 Encrypting {len(salaries)} salary values...")
        start_time = time.time()
        
        self.encrypted_salaries = self.paillier.encrypt_vector(salaries)
        
        elapsed = time.time() - start_time
        
        print(f"✓ Encryption complete!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {len(salaries)/elapsed:.1f} values/sec")
        print(f"  Avg per value: {(elapsed/len(salaries))*1000:.2f}ms")
        
        self.analysis_results['encryption'] = {
            'num_values': len(salaries),
            'time_seconds': elapsed,
            'rate_per_second': len(salaries)/elapsed,
            'avg_time_ms': (elapsed/len(salaries))*1000
        }
        
        return self.encrypted_salaries
    
    def compute_encrypted_statistics(self):
        """Compute statistics on encrypted salary data."""
        print("\n" + "="*70)
        print("PHASE 2: ENCRYPTED COMPUTATIONS")
        print("="*70)
        
        n = len(self.encrypted_salaries)
        
        # 1. Encrypted Sum
        print("\n📊 Computing encrypted sum...")
        start_time = time.time()
        encrypted_sum = self.encrypted_salaries[0]
        for enc_val in self.encrypted_salaries[1:]:
            encrypted_sum = self.paillier.add_encrypted(encrypted_sum, enc_val)
        sum_time = time.time() - start_time
        
        print(f"  ✓ Sum computed in {sum_time:.3f}s")
        
        # 2. Encrypted Mean (sum / n)
        print("\n📊 Computing encrypted mean...")
        start_time = time.time()
        encrypted_mean = self.paillier.multiply_encrypted_by_scalar(
            encrypted_sum, 
            1.0/n
        )
        mean_time = time.time() - start_time
        print(f"  ✓ Mean computed in {mean_time:.3f}s")
        
        # 3. Decrypt results
        print("\n🔓 Decrypting results...")
        start_time = time.time()
        decrypted_sum = self.paillier.decrypt(encrypted_sum)
        decrypted_mean = self.paillier.decrypt(encrypted_mean)
        decrypt_time = time.time() - start_time
        
        # Compare with true values
        true_sum = self.df['Monthly_Salary'].sum()
        true_mean = self.df['Monthly_Salary'].mean()
        
        print(f"\n✓ Results:")
        print(f"  Encrypted Sum:  ${decrypted_sum:,.2f}")
        print(f"  True Sum:       ${true_sum:,.2f}")
        print(f"  Error:          ${abs(decrypted_sum - true_sum):,.2f}")
        print(f"\n  Encrypted Mean: ${decrypted_mean:,.2f}")
        print(f"  True Mean:      ${true_mean:,.2f}")
        print(f"  Error:          ${abs(decrypted_mean - true_mean):,.2f}")
        
        self.analysis_results['statistics'] = {
            'encrypted_sum': float(decrypted_sum),
            'true_sum': float(true_sum),
            'sum_error': float(abs(decrypted_sum - true_sum)),
            'encrypted_mean': float(decrypted_mean),
            'true_mean': float(true_mean),
            'mean_error': float(abs(decrypted_mean - true_mean)),
            'sum_computation_time': sum_time,
            'mean_computation_time': mean_time,
            'decryption_time': decrypt_time
        }
        
        return decrypted_sum, decrypted_mean
    
    def simulate_department_aggregation(self, num_departments: int = 5):
        """
        Simulate multi-department salary aggregation.
        Each department encrypts locally, server aggregates securely.
        """
        print("\n" + "="*70)
        print(f"PHASE 3: FEDERATED DEPARTMENT AGGREGATION ({num_departments} depts)")
        print("="*70)
        
        salaries = self.df['Monthly_Salary'].fillna(0).values
        
        # Split salaries into departments
        dept_splits = np.array_split(salaries, num_departments)
        
        print(f"\n🏢 Department data distribution:")
        for i, dept_data in enumerate(dept_splits):
            print(f"  Department {i+1}: {len(dept_data)} employees, "
                  f"avg salary: ${np.mean(dept_data):,.2f}")
        
        # Phase 1: Each department encrypts locally
        print(f"\n🔒 Phase 1: Local encryption by each department")
        encrypted_depts = []
        dept_enc_times = []
        
        for i, dept_data in enumerate(dept_splits):
            start = time.time()
            enc_data = self.paillier.encrypt_vector(dept_data)
            elapsed = time.time() - start
            encrypted_depts.append(enc_data)
            dept_enc_times.append(elapsed)
            print(f"  Dept {i+1}: {len(dept_data)} values encrypted in {elapsed:.3f}s")
        
        # Phase 2: Server aggregates without seeing raw data
        print(f"\n🔐 Phase 2: Secure aggregation (server cannot see raw salaries)")
        start_time = time.time()
        
        # Compute weights (proportional to department size)
        weights = [len(dept) / len(salaries) for dept in dept_splits]
        
        # Weighted average of encrypted salaries
        encrypted_global_avg = self.paillier.weighted_average_encrypted(
            encrypted_depts,
            weights
        )
        
        agg_time = time.time() - start_time
        print(f"  ✓ Aggregation completed in {agg_time:.3f}s")
        
        # Phase 3: Decrypt final result
        print(f"\n🔓 Phase 3: Decrypt aggregated result")
        start_time = time.time()
        decrypted_avg = self.paillier.decrypt_vector(encrypted_global_avg)
        decrypt_time = time.time() - start_time
        
        global_avg = np.mean(decrypted_avg)
        true_avg = np.mean(salaries)
        
        print(f"  ✓ Decrypted in {decrypt_time:.3f}s")
        print(f"\n📊 Results:")
        print(f"  Federated Average: ${global_avg:,.2f}")
        print(f"  True Average:      ${true_avg:,.2f}")
        print(f"  Error:             ${abs(global_avg - true_avg):,.2f}")
        
        # Total time
        total_time = sum(dept_enc_times) + agg_time + decrypt_time
        print(f"\n⏱️  Total Time Breakdown:")
        print(f"  Encryption:  {sum(dept_enc_times):.3f}s ({sum(dept_enc_times)/total_time*100:.1f}%)")
        print(f"  Aggregation: {agg_time:.3f}s ({agg_time/total_time*100:.1f}%)")
        print(f"  Decryption:  {decrypt_time:.3f}s ({decrypt_time/total_time*100:.1f}%)")
        print(f"  Total:       {total_time:.3f}s")
        
        self.analysis_results['federated_aggregation'] = {
            'num_departments': num_departments,
            'dept_sizes': [len(d) for d in dept_splits],
            'dept_encryption_times': dept_enc_times,
            'aggregation_time': agg_time,
            'decryption_time': decrypt_time,
            'total_time': total_time,
            'federated_average': float(global_avg),
            'true_average': float(true_avg),
            'error': float(abs(global_avg - true_avg))
        }
        
        return global_avg
    
    def compare_salary_ranges(self):
        """Compare encrypted operations on different salary ranges."""
        print("\n" + "="*70)
        print("PHASE 4: SALARY RANGE COMPARISON")
        print("="*70)
        
        # Define salary ranges
        ranges = [
            (0, 5000, 'Low'),
            (5000, 10000, 'Medium'),
            (10000, float('inf'), 'High')
        ]
        
        range_results = {}
        
        for low, high, label in ranges:
            mask = (self.df['Monthly_Salary'] >= low) & (self.df['Monthly_Salary'] < high)
            range_salaries = self.df[mask]['Monthly_Salary'].values
            
            if len(range_salaries) == 0:
                continue
            
            print(f"\n💼 {label} Salary Range (${low:,} - ${high:,})")
            print(f"  Employees: {len(range_salaries)}")
            print(f"  True Mean: ${np.mean(range_salaries):,.2f}")
            
            # Encrypt and compute mean
            start = time.time()
            enc_range = self.paillier.encrypt_vector(range_salaries)
            enc_time = time.time() - start
            
            # Encrypted sum
            enc_sum = enc_range[0]
            for enc_val in enc_range[1:]:
                enc_sum = self.paillier.add_encrypted(enc_sum, enc_val)
            
            # Encrypted mean
            enc_mean = self.paillier.multiply_encrypted_by_scalar(
                enc_sum, 
                1.0/len(range_salaries)
            )
            
            dec_mean = self.paillier.decrypt(enc_mean)
            total_time = time.time() - start
            
            print(f"  Encrypted Mean: ${dec_mean:,.2f}")
            print(f"  Time: {total_time:.3f}s")
            
            range_results[label] = {
                'count': len(range_salaries),
                'true_mean': float(np.mean(range_salaries)),
                'encrypted_mean': float(dec_mean),
                'computation_time': total_time
            }
        
        self.analysis_results['salary_ranges'] = range_results
        
        return range_results
    
    def generate_visualizations(self):
        """Generate visualization plots."""
        print("\n" + "="*70)
        print("PHASE 5: GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Employee Salary Analysis with Paillier Encryption', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Salary Distribution
        ax1 = axes[0, 0]
        self.df['Monthly_Salary'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black')
        ax1.axvline(self.df['Monthly_Salary'].mean(), color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('Monthly Salary ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Salary Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Encryption Performance
        ax2 = axes[0, 1]
        metrics = ['Encryption', 'Sum', 'Mean', 'Decrypt']
        times = [
            self.analysis_results['encryption']['time_seconds'],
            self.analysis_results['statistics']['sum_computation_time'],
            self.analysis_results['statistics']['mean_computation_time'],
            self.analysis_results['statistics']['decryption_time']
        ]
        bars = ax2.bar(metrics, times, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Operation Performance')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom')
        
        # Plot 3: Federated Aggregation Breakdown
        ax3 = axes[1, 0]
        if 'federated_aggregation' in self.analysis_results:
            fl_data = self.analysis_results['federated_aggregation']
            phases = ['Encryption', 'Aggregation', 'Decryption']
            phase_times = [
                sum(fl_data['dept_encryption_times']),
                fl_data['aggregation_time'],
                fl_data['decryption_time']
            ]
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            wedges, texts, autotexts = ax3.pie(phase_times, labels=phases, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
            ax3.set_title('Federated Aggregation Time Breakdown')
        
        # Plot 4: Accuracy Comparison
        ax4 = axes[1, 1]
        metrics = ['Sum', 'Mean', 'FL Average']
        true_vals = [
            self.analysis_results['statistics']['true_sum'],
            self.analysis_results['statistics']['true_mean'],
            self.analysis_results['federated_aggregation']['true_average']
        ]
        encrypted_vals = [
            self.analysis_results['statistics']['encrypted_sum'],
            self.analysis_results['statistics']['encrypted_mean'],
            self.analysis_results['federated_aggregation']['federated_average']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, true_vals, width, label='True Value', color='steelblue')
        ax4.bar(x + width/2, encrypted_vals, width, label='Encrypted Result', color='lightcoral')
        
        ax4.set_ylabel('Value ($)')
        ax4.set_title('Accuracy: True vs Encrypted Computation')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'salary_analysis_{timestamp}.png'
        plot_path = self.results_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {plot_path}")
        
        plt.close()
    
    def save_results(self):
        """Save analysis results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f'salary_analysis_results_{timestamp}.json'
        results_path = self.results_dir / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
    
    def run_complete_analysis(self):
        """Run complete salary analysis pipeline."""
        # Load data
        if not self.load_dataset():
            return
        
        # Encrypt salaries
        self.encrypt_salaries()
        
        # Compute encrypted statistics
        self.compute_encrypted_statistics()
        
        # Simulate federated aggregation
        self.simulate_department_aggregation(num_departments=5)
        
        # Compare salary ranges
        self.compare_salary_ranges()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved in: {self.results_dir}/")
        print("  - JSON results file")
        print("  - Visualization plots")


if __name__ == "__main__":
    # Run the complete salary analysis
    analyzer = SalaryAnalyzer(key_size=2048)
    analyzer.run_complete_analysis()
