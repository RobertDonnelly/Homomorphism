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
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
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
        
        encrypted_data = {}
        total_start = time.time()
        
        for col in columns:
            if col not in df.columns:
                print(f"⚠️  Warning: Column '{col}' not found, skipping")
                continue
            
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
        Split data among clients, encrypt locally, aggregate securely.
        
        Args:
            df: DataFrame with data
            column: Column to use for simulation
            num_clients: Number of simulated clients
            
        Returns:
            Dictionary with aggregation results
        """
        print(f"\n🌐 Simulating Federated Learning with {num_clients} clients")
        print(f"  Column: '{column}'")
        
        data = df[column].fillna(0).values
        
        # Split data among clients
        splits = np.array_split(data, num_clients)
        print(f"  Data split: {[len(s) for s in splits]} samples per client")
        
        # Each client encrypts their local data
        print("\n  Phase 1: Client-side encryption")
        encrypted_splits = []
        client_times = []
        
        for i, client_data in enumerate(splits):
            start = time.time()
            enc_data = self.paillier.encrypt_vector(client_data)
            elapsed = time.time() - start
            encrypted_splits.append(enc_data)
            client_times.append(elapsed)
            print(f"    Client {i+1}: {len(client_data)} values encrypted in {elapsed:.3f}s")
        
        # Server aggregates encrypted values
        print("\n  Phase 2: Server-side secure aggregation")
        start = time.time()
        
        # Compute weights (proportional to data size)
        weights = [len(s) / len(data) for s in splits]
        
        # Compute weighted average of encrypted data
        encrypted_avg = self.paillier.weighted_average_encrypted(
            encrypted_splits, 
            weights
        )
        
        agg_time = time.time() - start
        print(f"    Aggregation completed in {agg_time:.3f}s")
        
        # Decrypt final result
        print("\n  Phase 3: Decryption")
        start = time.time()
        decrypted_avg = self.paillier.decrypt_vector(encrypted_avg)
        dec_time = time.time() - start
        
        # Compute true average for comparison
        true_avg = np.mean(data)
        global_mean = np.mean(decrypted_avg)
        
        print(f"    ✓ Decrypted in {dec_time:.3f}s")
        print(f"    True average: {true_avg:.6f}")
        print(f"    FL average:   {global_mean:.6f}")
        print(f"    Error: {abs(true_avg - global_mean):.6e}")
        
        return {
            'num_clients': num_clients,
            'data_distribution': [len(s) for s in splits],
            'client_encryption_times': client_times,
            'aggregation_time': agg_time,
            'decryption_time': dec_time,
            'total_time': sum(client_times) + agg_time + dec_time,
            'true_average': float(true_avg),
            'fl_average': float(global_mean),
            'error': float(abs(true_avg - global_mean))
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


def example_usage():
    """Example: Process a dataset with Paillier encryption."""
    
    print("="*70)
    print("PAILLIER DATA PROCESSING EXAMPLE")
    print("="*70)
    
    # Initialize processor
    processor = PaillierDataProcessor(key_size=2048)
    
    # Example 1: Load and encrypt CSV
    print("\n--- Example 1: Encrypt CSV Data ---")
    try:
        # Replace 'your_dataset.csv' with your actual filename
        df = processor.load_csv('employee_salary_dataset.csv')
        
        # Encrypt all numeric columns
        encrypted_results = processor.encrypt_dataframe(df)
        
        # Save results
        processor.save_results(
            encrypted_results, 
            'encrypted_data_results.json'
        )
        
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Please place your CSV file in the 'data/' directory")
        
        # Create sample data for demonstration
        print("\n   Creating sample dataset for demonstration...")
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        
        # Encrypt sample data
        encrypted_results = processor.encrypt_dataframe(
            df, 
            columns=['feature1', 'feature2', 'feature3']
        )
    
    # Example 2: Compute statistics on encrypted data
    print("\n--- Example 2: Encrypted Statistics ---")
    column_data = df['feature1'].fillna(0).values[:20]  # Use first 20 values
    encrypted_data = processor.paillier.encrypt_vector(column_data)
    stats = processor.compute_encrypted_statistics(encrypted_data)
    
    # Example 3: Federated learning simulation
    print("\n--- Example 3: Federated Learning Simulation ---")
    fl_results = processor.simulate_federated_aggregation(
        df, 
        column='feature1',
        num_clients=3
    )
    
    processor.save_results(fl_results, 'fl_simulation_results.json')
    
    print("\n" + "="*70)
    print("✓ Processing complete!")
    print("="*70)


if __name__ == "__main__":
    example_usage()