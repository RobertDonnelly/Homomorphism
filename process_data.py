"""
Enhanced Paillier Data Processing Script
Generates comprehensive encrypted statistical summaries for visualization
"""

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

from src.phe.paillier_crypto import PaillierCrypto


class PaillierDataProcessor:
    """Process datasets with Paillier encryption and generate statistical summaries."""
    
    def __init__(self, key_size: int = 2048):
        """Initialize processor with Paillier cryptosystem."""
        self.paillier = PaillierCrypto(key_size=key_size)
        self.paillier.generate_keypair()
        self.data_dir = project_root / 'data'
        self.results_dir = project_root / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file from data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"\n❌ ERROR: File not found!")
            print(f"   Expected location: {filepath}")
            print(f"\n   Please ensure '{filename}' is in the 'data/' directory")
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
        Includes: count, sum, mean, min, max, quartiles, std dev
        
        Args:
            data: numpy array of values
            column_name: name of the column
            
        Returns:
            Dictionary with encrypted statistics (decrypted for visualization)
        """
        print(f"\n📊 Computing encrypted statistics for '{column_name}'...")
       
        start_time = time.time()
        
        # Encrypt all values
        encrypted_values = self.paillier.encrypt_vector(data)
        
        # Count (public, not sensitive)
        count = len(data)
        
        # Sum (encrypted)
        encrypted_sum = encrypted_values[0]
        for enc_val in encrypted_values[1:]:
            encrypted_sum = self.paillier.add_encrypted(encrypted_sum, enc_val)
        decrypted_sum = self.paillier.decrypt(encrypted_sum)
        
        # Mean (encrypted)
        encrypted_mean = self.paillier.multiply_encrypted_by_scalar(encrypted_sum, 1.0/count)
        decrypted_mean = self.paillier.decrypt(encrypted_mean)
        
        # For min/max/quartiles, we decrypt (in real FL, these might be computed locally)
        decrypted_values = self.paillier.decrypt_vector(encrypted_values)
        
        # Quartiles
        q1 = float(np.percentile(decrypted_values, 25))
        q2 = float(np.percentile(decrypted_values, 50))  # median
        q3 = float(np.percentile(decrypted_values, 75))
        
        # Min/Max
        min_val = float(np.min(decrypted_values))
        max_val = float(np.max(decrypted_values))
        
        # Standard deviation (computed from encrypted mean)
        squared_diffs = [(val - decrypted_mean) ** 2 for val in decrypted_values]
        variance = np.mean(squared_diffs)
        std_dev = float(np.sqrt(variance))
        
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
            'variance': float(variance),
            'skewness': float(pd.Series(decrypted_values).skew()),
            'kurtosis': float(pd.Series(decrypted_values).kurtosis()),
            'histogram': {
                'counts': [int(c) for c in hist],
                'bin_centers': [float(b) for b in bin_centers],
                'bin_edges': [float(e) for e in bin_edges]
            },
            'computation_time': computation_time
        }
    
    def compute_range_analysis(self, data: np.ndarray, column_name: str) -> Dict:
        """
        Analyze data by ranges (low, medium, high).
        
        Args:
            data: numpy array of values
            column_name: name of the column
            
        Returns:
            Dictionary with range-based statistics
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
                enc_data = self.paillier.encrypt_vector(range_data)
                
                # Sum
                enc_sum = enc_data[0]
                for enc_val in enc_data[1:]:
                    enc_sum = self.paillier.add_encrypted(enc_sum, enc_val)
                dec_sum = self.paillier.decrypt(enc_sum)
                
                # Mean
                enc_mean = self.paillier.multiply_encrypted_by_scalar(enc_sum, 1.0/len(range_data))
                dec_mean = self.paillier.decrypt(enc_mean)
                
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
    
    def compute_encrypted_statistics(self, encrypted_data: List) -> Dict:
        """Compute statistics on encrypted data."""
        print(f"\n📈 Computing encrypted statistics...")
        
        # Sum of encrypted values
        start_time = time.time()
        encrypted_sum = encrypted_data[0]
        for enc_val in encrypted_data[1:]:
            encrypted_sum = self.paillier.add_encrypted(encrypted_sum, enc_val)
        sum_time = time.time() - start_time
        
        # Decrypt to verify
        decrypted_sum = self.paillier.decrypt(encrypted_sum)
        
        # Mean (sum / n)
        n = len(encrypted_data)
        encrypted_mean = self.paillier.multiply_encrypted_by_scalar(encrypted_sum, 1.0/n)
        decrypted_mean = self.paillier.decrypt(encrypted_mean)
        
        print(f"  ✓ Encrypted sum computed in {sum_time:.3f}s")
        print(f"  Sum: {decrypted_sum:.4f}")
        print(f"  Mean: {decrypted_mean:.4f}")
        
        return {
            'sum': decrypted_sum,
            'mean': decrypted_mean,
            'count': n,
            'computation_time': sum_time
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
            print(f"\n🔒 Encrypting column: '{col}'")
            print(f"  Data type: {df[col].dtype}")
            print(f"  Size: {len(df[col])} values")
            
            values = df[col].fillna(0).values
            
            
            
            start_time = time.time()
            encrypted_values = []

            total = len(values)
            for i, v in enumerate(values, 1):
                encrypted_values.append(self.paillier.encrypt(v))

            # progress %
            progress = (i / total) * 100
            print(f"\rEncrypting... {progress:.2f}%", end="")

            elapsed = time.time() - start_time
            print(f"\nDone in {elapsed:.2f}s")
            
    
            
           # start_time = time.time()
           # encrypted_values = self.paillier.encrypt_vector(values)
           # elapsed = time.time() - start_time
            
           # print(f"  ✓ Encrypted in {elapsed:.3f}s ({len(df[col])/elapsed:.1f} values/sec)")
            
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
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n✓ Total encryption time: {total_time:.3f}s")
        return result
    
    def simulate_federated_aggregation(self, df: pd.DataFrame, 
                                      column: str,
                                      num_clients: int = 3) -> Dict:
        """Simulate federated learning aggregation on a column."""
        if column not in df.columns:
            print(f"\n❌ ERROR: Column '{column}' not found in dataset")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
        
        print(f"\n🌐 Simulating Federated Learning with {num_clients} clients")
        print(f"  Column: '{column}'")
        
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
            enc_data = self.paillier.encrypt_vector(client_data)
            
            # Compute encrypted local sum
            local_sum = enc_data[0]
            for enc_val in enc_data[1:]:
                local_sum = self.paillier.add_encrypted(local_sum, enc_val)
            
            # Decrypt for local stats (in practice, only aggregated result would be decrypted)
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
            
            print(f"    Client {i+1}: {len(client_data)} values, local sum computed in {elapsed:.3f}s")
        
        # Server aggregates encrypted sums
        print("\n  Phase 2: Server-side secure aggregation")
        start = time.time()
        
        # Aggregate all encrypted sums
        global_encrypted_sum = encrypted_sums[0]
        for enc_sum in encrypted_sums[1:]:
            global_encrypted_sum = self.paillier.add_encrypted(global_encrypted_sum, enc_sum)
        
        # Total count (public, not sensitive)
        total_count = sum(client_counts)
        
        # Compute encrypted global average
        global_encrypted_avg = self.paillier.multiply_encrypted_by_scalar(
            global_encrypted_sum, 
            1.0 / total_count
        )
        
        agg_time = time.time() - start
        print(f"    Aggregation completed in {agg_time:.3f}s")
        
        # Decrypt final result
        print("\n  Phase 3: Decryption")
        start = time.time()
        decrypted_global_avg = self.paillier.decrypt(global_encrypted_avg)
        dec_time = time.time() - start
        
        # Compute true average for comparison
        true_avg = np.mean(data)
        
        print(f"    ✓ Decrypted in {dec_time:.3f}s")
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
            'num_clients': num_clients,
            'data_distribution': [len(s) for s in splits],
            'client_computation_times': client_times,
            'client_statistics': client_stats,
            'aggregation_time': agg_time,
            'decryption_time': dec_time,
            'total_time': total_time,
            'true_average': float(true_avg),
            'fl_average': float(decrypted_global_avg),
            'error': float(abs(true_avg - decrypted_global_avg)),
            'total_samples': total_count
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file in results directory."""
        filepath = self.results_dir / filename
        
        # Convert encrypted data to serializable format
        serializable_results = {}
        for key, value in results.items():
            if key == 'encrypted_columns':
                serializable_results[key] = {
                    col: {
                        'column_name': data['column_name'],
                        'original_size': data['original_size'],
                        'encryption_time': data['encryption_time'],
                        'data_type': data['data_type']
                    }
                    for col, data in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")


def main():
    """Main processing function with enhanced statistical summaries."""
    
    print("="*70)
    print("ENHANCED PAILLIER DATA PROCESSING")
    print("Generates comprehensive encrypted statistical summaries")
    print("="*70)
    
    # Configuration
    DATASET_FILENAME = 'employee_salary_dataset.csv'
    TARGET_COLUMN = 'Monthly_Salary'
    NUM_CLIENTS = 3
    KEY_SIZE = 2048
    
    # Initialize processor
    print(f"\nInitializing Paillier Cryptosystem ({KEY_SIZE}-bit keys)...")
    processor = PaillierDataProcessor(key_size=KEY_SIZE)
    
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
    
    # 3. Range analysis
    print(f"\n--- Computing Range Analysis ---")
    range_analysis = processor.compute_range_analysis(column_data, TARGET_COLUMN)
    
    # 4. Basic encrypted statistics
    print("\n--- Computing Basic Encrypted Statistics ---")
    encrypted_data = processor.paillier.encrypt_vector(column_data)
    basic_stats = processor.compute_encrypted_statistics(encrypted_data)
    
    # 5. Federated learning simulation
    print(f"\n--- Federated Learning Simulation ---")
    fl_results = processor.simulate_federated_aggregation(
        df, 
        column=TARGET_COLUMN,
        num_clients=NUM_CLIENTS
    )
    
    # Save all results
    print("\n--- Saving Results ---")
    processor.save_results(encrypted_results, 'encrypted_data_results.json')
    processor.save_results(statistical_summary, 'statistical_summary.json')
    processor.save_results(range_analysis, 'range_analysis.json')
    processor.save_results(basic_stats, 'encrypted_statistics.json')
    processor.save_results(fl_results, 'fl_simulation_results.json')
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {processor.results_dir.absolute()}/")
    print("  - encrypted_data_results.json (encryption metadata)")
    print("  - statistical_summary.json (comprehensive stats)")
    print("  - range_analysis.json (range-based analysis)")
    print("  - encrypted_statistics.json (basic stats)")
    print("  - fl_simulation_results.json (federated learning)")
    print("\nThese files can be visualized without accessing raw CSV data!")


if __name__ == "__main__":
    main()