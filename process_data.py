"""
Paillier Data Processing Script
Process datasets from the data/ folder with Paillier encryption
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
    """Process datasets with Paillier encryption."""
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize processor with Paillier cryptosystem.
        
        Args:
            key_size: Bit length for Paillier keys
        """
        self.paillier = PaillierCrypto(key_size=key_size)
        self.paillier.generate_keypair()
        self.data_dir = project_root / 'data'
        self.results_dir = project_root / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV file from data directory.
        
        Args:
            filename: Name of CSV file (e.g., 'dataset.csv')
            
        Returns:
            pandas DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            SystemExit: Exits program if file not found
        """
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
    
    def encrypt_column(self, data: pd.Series, column_name: str) -> Dict:
        """
        Encrypt a single column of data.
        
        Args:
            data: pandas Series (column data)
            column_name: Name of the column
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        print(f"\n🔒 Encrypting column: '{column_name}'")
        print(f"  Data type: {data.dtype}")
        print(f"  Size: {len(data)} values")
        
        # Convert to numpy array and handle NaN
        values = data.fillna(0).values
        
        # Encrypt
        start_time = time.time()
        encrypted_values = self.paillier.encrypt_vector(values)
        elapsed = time.time() - start_time
        
        print(f"  ✓ Encrypted in {elapsed:.3f}s ({len(data)/elapsed:.1f} values/sec)")
        
        return {
            'column_name': column_name,
            'encrypted_data': encrypted_values,
            'original_size': len(data),
            'encryption_time': elapsed,
            'data_type': str(data.dtype)
        }
    
    def encrypt_dataframe(self, df: pd.DataFrame, 
                         columns: List[str] = None) -> Dict:
        """
        Encrypt specified columns of a DataFrame.
        
        Args:
            df: pandas DataFrame
            columns: List of column names to encrypt (None = all numeric columns)
            
        Returns:
            Dictionary with all encrypted data and metadata
        """
        if columns is None:
            # Auto-select numeric columns
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
            encrypted_data[col] = self.encrypt_column(df[col], col)
        
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
    
    def compute_encrypted_statistics(self, encrypted_data: List) -> Dict:
        """
        Compute statistics on encrypted data (demonstrations).
        
        Args:
            encrypted_data: List of encrypted values
            
        Returns:
            Dictionary with computed statistics
        """
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
    
    def simulate_federated_aggregation(self, df: pd.DataFrame, 
                                      column: str,
                                      num_clients: int = 3) -> Dict:
        """
        Simulate federated learning aggregation on a column.
        Each client computes local statistics (encrypted), server aggregates.
        
        Args:
            df: DataFrame with data
            column: Column to use for simulation
            num_clients: Number of simulated clients
            
        Returns:
            Dictionary with aggregation results
        """
        # Validate column exists
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
        
        for i, client_data in enumerate(splits):
            start = time.time()
            
            # Client encrypts their data
            enc_data = self.paillier.encrypt_vector(client_data)
            
            # Compute encrypted local sum
            local_sum = enc_data[0]
            for enc_val in enc_data[1:]:
                local_sum = self.paillier.add_encrypted(local_sum, enc_val)
            
            encrypted_sums.append(local_sum)
            client_counts.append(len(client_data))
            
            elapsed = time.time() - start
            client_times.append(elapsed)
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
    """
    Main processing function.
    Requires actual dataset - exits if not found.
    """
    
    print("="*70)
    print("PAILLIER DATA PROCESSING")
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
    
    # Encrypt target column
    print(f"\n--- Encrypting '{TARGET_COLUMN}' Column ---")
    if TARGET_COLUMN not in df.columns:
        print(f"\n❌ ERROR: Column '{TARGET_COLUMN}' not found!")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    encrypted_results = processor.encrypt_dataframe(df, columns=[TARGET_COLUMN])
    
    # Compute encrypted statistics
    print("\n--- Computing Encrypted Statistics ---")
    column_data = df[TARGET_COLUMN].fillna(0).values
    encrypted_data = processor.paillier.encrypt_vector(column_data)
    stats = processor.compute_encrypted_statistics(encrypted_data)
    
    # Federated learning simulation
    print(f"\n--- Federated Learning Simulation ---")
    fl_results = processor.simulate_federated_aggregation(
        df, 
        column=TARGET_COLUMN,
        num_clients=NUM_CLIENTS
    )
    
    # Save results
    print("\n--- Saving Results ---")
    processor.save_results(encrypted_results, 'encrypted_data_results.json')
    processor.save_results(fl_results, 'fl_simulation_results.json')
    processor.save_results(stats, 'encrypted_statistics.json')
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {processor.results_dir.absolute()}/")
    print("  - encrypted_data_results.json")
    print("  - fl_simulation_results.json")
    print("  - encrypted_statistics.json")


if __name__ == "__main__":
    main()