import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import json
from typing import Dict, List, Union

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.schemes.bfv.bfv_crypto import BFVCrypto


class BFVDataProcessor:
    """Process datasets with BFV encryption and generate statistical summaries."""
    
    def __init__(self):
        """Initialize processor with BFV cryptosystem."""
        self.bfv = BFVCrypto()
        self.bfv.setup()
        self.data_dir = project_root / 'data/raw'
        self.results_dir = project_root / 'results/bfv'
        self.results_dir.mkdir(exist_ok=True)
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file from data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"\n❌ ERROR: File not found!")
            print(f"   Expected location: {filepath}")
            print(f"\n   Please ensure '{filename}' is in the 'data/raw/' directory")
            print(f"   Current data directory: {self.data_dir.absolute()}")
            sys.exit(1)
        
        print(f"📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
        return df
    
    def compute_encrypted_statistical_summary(self, data: np.ndarray, column_name: str) -> Dict:
        """
        Compute comprehensive statistical summary on encrypted data.
        Includes: count, sum, mean, variance, std dev, min, max, quartiles
        
        Args:
            data: numpy array of values
            column_name: name of the column
            
        Returns:
            Dictionary with encrypted statistics (decrypted for visualization)
        """
        print(f"\n📊 Computing encrypted statistics for '{column_name}'...")
       
        start_time = time.time()
        
        # Encrypt all values
        encrypted_values = self.bfv.encrypt_vector(data)
        
        # Count (public, not sensitive)
        count = len(data)
        
        # Sum (encrypted homomorphic addition)
        print("  Computing encrypted sum...")
        encrypted_sum = self.bfv.sum_encrypted(encrypted_values)
        decrypted_sum = self.bfv.decrypt(encrypted_sum)
        
        # Mean (encrypted)
        print("  Computing encrypted mean...")
        encrypted_mean = self.bfv.multiply_plain(encrypted_sum, 1.0/count)
        decrypted_mean = self.bfv.decrypt(encrypted_mean)
        
        # Variance (using homomorphic operations)
        print("  Computing encrypted variance...")
        encrypted_variance = self.bfv.compute_variance(encrypted_values, encrypted_mean)
        decrypted_variance = self.bfv.decrypt(encrypted_variance)
        std_dev = float(np.sqrt(abs(decrypted_variance)))
        
        # For min/max/quartiles, we decrypt (in real FL, these might be computed locally)
        decrypted_values = self.bfv.decrypt_vector(encrypted_values)
        
        # Quartiles
        q1 = float(np.percentile(decrypted_values, 25))
        q2 = float(np.percentile(decrypted_values, 50))  # median
        q3 = float(np.percentile(decrypted_values, 75))
        
        # Min/Max
        min_val = float(np.min(decrypted_values))
        max_val = float(np.max(decrypted_values))
        
        # Distribution binning for histogram
        num_bins = 20
        hist, bin_edges = np.histogram(decrypted_values, bins=num_bins)
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        
        computation_time = time.time() - start_time
        
        print(f"  ✓ Statistics computed in {computation_time:.3f}s")
        print(f"    Mean: {decrypted_mean:.2f}")
        print(f"    Median: {q2:.2f}")
        print(f"    Std Dev: {std_dev:.2f}")
        print(f"    Range: [{min_val:.2f}, {max_val:.2f}]")
        
        return {
            'column_name': column_name,
            'count': count,
            'sum': float(decrypted_sum),
            'mean': float(decrypted_mean),
            'median': q2,
            'std_dev': std_dev,
            'min': min_val,
            'max': max_val,
            'q1': q1,
            'q2': q2,
            'q3': q3,
            'iqr': q3 - q1,
            'range': max_val - min_val,
            'variance': float(decrypted_variance),
            'skewness': float(pd.Series(decrypted_values).skew()),
            'kurtosis': float(pd.Series(decrypted_values).kurtosis()),
            'histogram': {
                'counts': [int(c) for c in hist],
                'bin_centers': [float(b) for b in bin_centers],
                'bin_edges': [float(e) for e in bin_edges]
            },
            'computation_time': computation_time
        }
    
    def compute_encrypted_polynomial_stats(self, data: np.ndarray, column_name: str) -> Dict:
        """
        Compute polynomial statistics using BFV's multiplication capability.
        Computes squared values, cubed values, and higher moments.
        """
        print(f"\n🔢 Computing polynomial statistics for '{column_name}'...")
        start_time = time.time()
        
        # Encrypt values
        encrypted_values = self.bfv.encrypt_vector(data)
        
        # Compute sum of squares using homomorphic multiplication
        print("  Computing encrypted sum of squares...")
        encrypted_squares = []
        for enc_val in encrypted_values:
            enc_squared = self.bfv.multiply_encrypted(enc_val, enc_val)
            encrypted_squares.append(enc_squared)
        
        encrypted_sum_squares = self.bfv.sum_encrypted(encrypted_squares)
        decrypted_sum_squares = self.bfv.decrypt(encrypted_sum_squares)
        
        # Mean of squares
        mean_squares = decrypted_sum_squares / len(data)
        
        # Root mean square
        rms = float(np.sqrt(abs(mean_squares)))
        
        # Compare with true values
        true_sum_squares = np.sum(data ** 2)
        true_rms = np.sqrt(np.mean(data ** 2))
        
        computation_time = time.time() - start_time
        
        print(f"  ✓ Polynomial stats computed in {computation_time:.3f}s")
        print(f"    Sum of squares: {decrypted_sum_squares:.2f}")
        print(f"    RMS: {rms:.2f}")
        print(f"    True RMS: {true_rms:.2f}")
        print(f"    Error: {abs(rms - true_rms):.6f}")
        
        return {
            'column_name': column_name,
            'sum_of_squares': float(decrypted_sum_squares),
            'mean_of_squares': float(mean_squares),
            'root_mean_square': rms,
            'true_sum_squares': float(true_sum_squares),
            'true_rms': float(true_rms),
            'error': float(abs(rms - true_rms)),
            'computation_time': computation_time
        }
    
    def compute_range_analysis(self, data: np.ndarray, column_name: str) -> Dict:
        """
        Analyze data by ranges (low, medium, high) with encrypted operations.
        """
        print(f"\n📊 Computing range analysis for '{column_name}'...")
        
        # Define ranges based on quartiles
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        
        ranges = {
            'low': (float(data.min()), float(q1)),
            'medium': (float(q1), float(q3)),
            'high': (float(q3), float(data.max()))
        }
        
        range_stats = {}
        
        for range_name, (low, high) in ranges.items():
            mask = (data >= low) & (data <= high)
            range_data = data[mask]
            
            if len(range_data) > 0:
                # Encrypt and compute stats for this range
                enc_data = self.bfv.encrypt_vector(range_data)
                
                # Sum
                enc_sum = self.bfv.sum_encrypted(enc_data)
                dec_sum = self.bfv.decrypt(enc_sum)
                
                # Mean
                enc_mean = self.bfv.multiply_plain(enc_sum, 1.0/len(range_data))
                dec_mean = self.bfv.decrypt(enc_mean)
                
                range_stats[range_name] = {
                    'range_bounds': (low, high),
                    'count': len(range_data),
                    'percentage': float(len(range_data) / len(data) * 100),
                    'mean': float(dec_mean),
                    'sum': float(dec_sum),
                    'min': float(range_data.min()),
                    'max': float(range_data.max())
                }
                
                print(f"  {range_name.capitalize()}: {len(range_data)} values, mean={dec_mean:.2f}")
        
        return {
            'column_name': column_name,
            'ranges': range_stats,
            'total_count': len(data)
        }
    
    def encrypt_dataframe(self, df: pd.DataFrame, columns: List[str] = None) -> Dict:
        """Encrypt specified columns of a DataFrame."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"📊 Auto-selected numeric columns: {columns}")
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"\n❌ ERROR: The following columns were not found: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
        
        encrypted_data = {}
        total_start = time.time()
        
        for col in columns:
            print(f"\n🔐 Encrypting column: '{col}'")
            print(f"  Data type: {df[col].dtype}")
            print(f"  Size: {len(df[col])} values")
            
            values = df[col].fillna(0).values
            
            start_time = time.time()
            encrypted_values = []
            total = len(values)
            
            for i, v in enumerate(values, 1):
                encrypted_values.append(self.bfv.encrypt(v))
                if i % 10 == 0 or i == total:
                    progress = (i / total) * 100
                    print(f"\r  Encrypting... {progress:.1f}%", end="")
            
            elapsed = time.time() - start_time
            print(f"\n  ✓ Encrypted in {elapsed:.3f}s ({len(df[col])/elapsed:.1f} values/sec)")
            
            encrypted_data[col] = {
                'column_name': col,
                'original_size': len(df[col]),
                'encryption_time': elapsed,
                'data_type': str(df[col].dtype)
            }
        
        total_time = time.time() - total_start
        
        result = {
            'encrypted_columns': encrypted_data,
            'total_encryption_time': total_time,
            'num_columns': len(columns),
            'num_rows': len(df),
            'scheme': 'BFV',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n✓ Total encryption time: {total_time:.3f}s")
        return result
    
    def simulate_federated_aggregation(self, df: pd.DataFrame, 
                                      column: str,
                                      num_clients: int = 3) -> Dict:
        """Simulate federated learning aggregation with BFV encryption."""
        if column not in df.columns:
            print(f"\n❌ ERROR: Column '{column}' not found in dataset")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
        
        print(f"\n🌐 Simulating Federated Learning with {num_clients} clients")
        print(f"  Column: '{column}'")
        print(f"  Using BFV homomorphic encryption")
        
        data = df[column].fillna(0).values
        
        # Split data among clients
        splits = np.array_split(data, num_clients)
        print(f"  Data split: {[len(s) for s in splits]} samples per client")
        
        # Each client computes encrypted local sum and count
        print("\n  Phase 1: Client-side local computation")
        encrypted_sums = []
        client_counts = []
        client_times = []
        client_stats = []
        
        for i, client_data in enumerate(splits):
            start = time.time()
            
            # Client encrypts their data
            enc_data = self.bfv.encrypt_vector(client_data)
            
            # Compute encrypted local sum using homomorphic addition
            local_sum = self.bfv.sum_encrypted(enc_data)
            
            # Local stats (in practice, only aggregated result would be decrypted)
            local_mean = np.mean(client_data)
            
            encrypted_sums.append(local_sum)
            client_counts.append(len(client_data))
            
            elapsed = time.time() - start
            client_times.append(elapsed)
            
            client_stats.append({
                'client_id': i + 1,
                'sample_count': len(client_data),
                'local_mean': float(local_mean),
                'computation_time': elapsed
            })
            
            print(f"    Client {i+1}: {len(client_data)} values, encrypted sum computed in {elapsed:.3f}s")
        
        # Server aggregates encrypted sums
        print("\n  Phase 2: Server-side secure aggregation")
        start = time.time()
        
        # Aggregate all encrypted sums using homomorphic addition
        global_encrypted_sum = self.bfv.sum_encrypted(encrypted_sums)
        
        # Total count (public, not sensitive)
        total_count = sum(client_counts)
        
        # Compute encrypted global average
        global_encrypted_avg = self.bfv.multiply_plain(
            global_encrypted_sum, 
            1.0 / total_count
        )
        
        agg_time = time.time() - start
        print(f"    ✓ Aggregation completed in {agg_time:.3f}s")
        
        # Decrypt final result
        print("\n  Phase 3: Decryption")
        start = time.time()
        decrypted_global_sum = self.bfv.decrypt(global_encrypted_sum)
        decrypted_global_avg = self.bfv.decrypt(global_encrypted_avg)
        dec_time = time.time() - start
        
        # Compute true average for comparison
        true_sum = np.sum(data)
        true_avg = np.mean(data)
        
        print(f"    ✓ Decrypted in {dec_time:.3f}s")
        print(f"    True sum:     {true_sum:.2f}")
        print(f"    FL sum:       {decrypted_global_sum:.2f}")
        print(f"    True average: {true_avg:.6f}")
        print(f"    FL average:   {decrypted_global_avg:.6f}")
        print(f"    Error: {abs(true_avg - decrypted_global_avg):.6e}")
        
        # Additional metrics
        total_time = sum(client_times) + agg_time + dec_time
        print(f"\n  ⏱️  Time Breakdown:")
        print(f"    Client computation: {sum(client_times):.3f}s ({sum(client_times)/total_time*100:.1f}%)")
        print(f"    Server aggregation: {agg_time:.3f}s ({agg_time/total_time*100:.1f}%)")
        print(f"    Decryption:         {dec_time:.3f}s ({dec_time/total_time*100:.1f}%)")
        print(f"    Total:              {total_time:.3f}s")
        
        return {
            'scheme': 'BFV',
            'num_clients': num_clients,
            'data_distribution': [len(s) for s in splits],
            'client_computation_times': client_times,
            'client_statistics': client_stats,
            'aggregation_time': agg_time,
            'decryption_time': dec_time,
            'total_time': total_time,
            'true_sum': float(true_sum),
            'true_average': float(true_avg),
            'fl_sum': float(decrypted_global_sum),
            'fl_average': float(decrypted_global_avg),
            'error': float(abs(true_avg - decrypted_global_avg)),
            'total_samples': total_count
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file in results directory."""
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")


def main():
    """Main processing function with comprehensive BFV operations."""
    
    print("="*70)
    print("ENHANCED BFV DATA PROCESSING (Pyfhel)")
    print("Generates comprehensive encrypted statistical summaries")
    print("Supports addition AND multiplication on encrypted data")
    print("="*70)
    
    # Configuration
    DATASET_FILENAME = 'Final_data copy.csv'
    TARGET_COLUMN = 'Weight (kg)'
    NUM_CLIENTS = 3
    
    # Initialize processor
    print(f"\nInitializing BFV Cryptosystem...")
    processor = BFVDataProcessor()
    
    # Load dataset (exits if not found)
    print(f"\n--- Loading Dataset ---")
    df = processor.load_csv(DATASET_FILENAME)
    
    # Display basic info
    print(f"\nDataset Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    
    # Validate target column
    if TARGET_COLUMN not in df.columns:
        print(f"\n❌ ERROR: Column '{TARGET_COLUMN}' not found!")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # 1. Encrypt target column
    print(f"\n--- Encrypting '{TARGET_COLUMN}' Column ---")
    encrypted_results = processor.encrypt_dataframe(df, columns=[TARGET_COLUMN])
    
    # 2. Compute comprehensive encrypted statistics
    print(f"\n--- Computing Encrypted Statistical Summary ---")
    column_data = df[TARGET_COLUMN].fillna(0).values
    statistical_summary = processor.compute_encrypted_statistical_summary(column_data, TARGET_COLUMN)
    
    # 3. Polynomial statistics (using BFV multiplication)
    print(f"\n--- Computing Polynomial Statistics (BFV Multiplication) ---")
    polynomial_stats = processor.compute_encrypted_polynomial_stats(column_data, TARGET_COLUMN)
    
    # 4. Range analysis
    print(f"\n--- Computing Range Analysis ---")
    range_analysis = processor.compute_range_analysis(column_data, TARGET_COLUMN)
    
    # 5. Federated learning simulation
    print(f"\n--- Federated Learning Simulation ---")
    fl_results = processor.simulate_federated_aggregation(
        df, 
        column=TARGET_COLUMN,
        num_clients=NUM_CLIENTS
    )
    
    # Save all results
    print("\n--- Saving Results ---")
    processor.save_results(encrypted_results, 'bfv_encrypted_data_results.json')
    processor.save_results(statistical_summary, 'bfv_statistical_summary.json')
    processor.save_results(polynomial_stats, 'bfv_polynomial_stats.json')
    processor.save_results(range_analysis, 'bfv_range_analysis.json')
    processor.save_results(fl_results, 'bfv_fl_simulation_results.json')
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {processor.results_dir.absolute()}/")
    print("  - bfv_encrypted_data_results.json (encryption metadata)")
    print("  - bfv_statistical_summary.json (comprehensive stats)")
    print("  - bfv_polynomial_stats.json (multiplication-based stats)")
    print("  - bfv_range_analysis.json (range-based analysis)")
    print("  - bfv_fl_simulation_results.json (federated learning)")
    print("\nThese files can be visualized without accessing raw CSV data!")
    print("\n🔢 BFV Advantages:")
    print("  ✓ Supports both addition AND multiplication")
    print("  ✓ Enables polynomial computations (squares, cubes, etc.)")
    print("  ✓ More versatile than Paillier for complex statistics")


if __name__ == "__main__":
    main()