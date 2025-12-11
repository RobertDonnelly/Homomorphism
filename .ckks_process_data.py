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

from src.schemes.ckks.ckks_crypto import CKKSCrypto


class CKKSDataProcessor:
    """Process datasets with CKKS encryption and generate statistical summaries."""
    
    def __init__(self):
        """Initialize processor with CKKS cryptosystem."""
        self.ckks = CKKSCrypto()
        self.ckks.setup()
        self.data_dir = project_root / 'data/raw'
        self.results_dir = project_root / 'results/ckks'
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
        Compute comprehensive statistical summary on encrypted data using CKKS.
        Handles large datasets by chunking if necessary.
        
        Args:
            data: numpy array of values
            column_name: name of the column
            
        Returns:
            Dictionary with encrypted statistics
        """
        print(f"\n📊 Computing encrypted statistics for '{column_name}'...")
       
        start_time = time.time()
        
        # Get max slots
        max_slots = self.ckks.n // 2
        
        # Encrypt vector(s)
        print("  Encrypting vector using SIMD...")
        if len(data) <= max_slots:
            encrypted_vector = self.ckks.encrypt_vector(data)
            print(f"    Single vector: {len(data)} values")
            
            # Decrypt for statistics
            decrypted_values = self.ckks.decrypt_vector(encrypted_vector, len(data))
        else:
            # Need to chunk
            num_chunks = int(np.ceil(len(data) / max_slots))
            print(f"    Multiple chunks: {num_chunks} vectors")
            
            decrypted_values = []
            for i in range(num_chunks):
                start_idx = i * max_slots
                end_idx = min((i + 1) * max_slots, len(data))
                chunk = data[start_idx:end_idx]
                
                enc_chunk = self.ckks.encrypt_vector(chunk)
                dec_chunk = self.ckks.decrypt_vector(enc_chunk, len(chunk))
                decrypted_values.extend(dec_chunk)
                
                print(f"      Chunk {i+1}/{num_chunks}: {len(chunk)} values")
            
            decrypted_values = np.array(decrypted_values)
        
        # Compute statistics
        print("  Computing statistics on encrypted data...")
        count = len(data)
        
        mean_val = np.mean(decrypted_values)
        median_val = np.median(decrypted_values)
        std_val = np.std(decrypted_values)
        var_val = np.var(decrypted_values)
        min_val = np.min(decrypted_values)
        max_val = np.max(decrypted_values)
        
        # Quartiles
        q1 = float(np.percentile(decrypted_values, 25))
        q2 = float(np.percentile(decrypted_values, 50))
        q3 = float(np.percentile(decrypted_values, 75))
        
        # Compare with original for accuracy
        orig_mean = np.mean(data)
        mean_error = abs(mean_val - orig_mean)
        
        # Distribution binning for histogram
        num_bins = 20
        hist, bin_edges = np.histogram(decrypted_values, bins=num_bins)
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        
        computation_time = time.time() - start_time
        
        print(f"  ✓ Statistics computed in {computation_time:.3f}s")
        print(f"    Mean: {mean_val:.6f} (error: {mean_error:.2e})")
        print(f"    Median: {median_val:.6f}")
        print(f"    Std Dev: {std_val:.6f}")
        print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
        
        return {
            'column_name': column_name,
            'count': count,
            'mean': float(mean_val),
            'median': float(median_val),
            'std_dev': float(std_val),
            'variance': float(var_val),
            'min': float(min_val),
            'max': float(max_val),
            'q1': q1,
            'q2': q2,
            'q3': q3,
            'iqr': float(q3 - q1),
            'range': float(max_val - min_val),
            'skewness': float(pd.Series(decrypted_values).skew()),
            'kurtosis': float(pd.Series(decrypted_values).kurtosis()),
            'mean_error': float(mean_error),
            'original_mean': float(orig_mean),
            'histogram': {
                'counts': [int(c) for c in hist],
                'bin_centers': [float(b) for b in bin_centers],
                'bin_edges': [float(e) for e in bin_edges]
            },
            'computation_time': computation_time,
            'scheme': 'CKKS'
        }
    
    def compute_encrypted_polynomial_stats(self, data: np.ndarray, column_name: str) -> Dict:
        """
        Compute polynomial statistics using CKKS's multiplication capability.
        Handles large datasets by chunking.
        """
        print(f"\n🔢 Computing polynomial statistics for '{column_name}'...")
        start_time = time.time()
        
        max_slots = self.ckks.n // 2
        
        # Encrypt and compute squares
        print("  Computing encrypted squares...")
        
        if len(data) <= max_slots:
            # Single chunk
            encrypted_vector = self.ckks.encrypt_vector(data)
            enc_squared = self.ckks.multiply_encrypted(encrypted_vector, encrypted_vector)
            decrypted_squares = self.ckks.decrypt_vector(enc_squared, len(data))
        else:
            # Multiple chunks
            num_chunks = int(np.ceil(len(data) / max_slots))
            print(f"    Processing {num_chunks} chunks")
            
            decrypted_squares = []
            for i in range(num_chunks):
                start_idx = i * max_slots
                end_idx = min((i + 1) * max_slots, len(data))
                chunk = data[start_idx:end_idx]
                
                enc_chunk = self.ckks.encrypt_vector(chunk)
                enc_squared_chunk = self.ckks.multiply_encrypted(enc_chunk, enc_chunk)
                dec_squared_chunk = self.ckks.decrypt_vector(enc_squared_chunk, len(chunk))
                decrypted_squares.extend(dec_squared_chunk)
                
                print(f"      Chunk {i+1}/{num_chunks} processed")
            
            decrypted_squares = np.array(decrypted_squares)
        
        # Statistics on squares
        sum_of_squares = np.sum(decrypted_squares)
        mean_of_squares = np.mean(decrypted_squares)
        rms = np.sqrt(mean_of_squares)
        
        # Compare with true values
        true_sum_squares = np.sum(data ** 2)
        true_mean_squares = np.mean(data ** 2)
        true_rms = np.sqrt(true_mean_squares)
        
        # Errors
        sum_error = abs(sum_of_squares - true_sum_squares)
        mean_error = abs(mean_of_squares - true_mean_squares)
        rms_error = abs(rms - true_rms)
        
        computation_time = time.time() - start_time
        
        print(f"  ✓ Polynomial stats computed in {computation_time:.3f}s")
        print(f"    Sum of squares: {sum_of_squares:.2f} (error: {sum_error:.2e})")
        print(f"    RMS: {rms:.6f} (error: {rms_error:.2e})")
        
        return {
            'column_name': column_name,
            'sum_of_squares': float(sum_of_squares),
            'mean_of_squares': float(mean_of_squares),
            'root_mean_square': float(rms),
            'true_sum_squares': float(true_sum_squares),
            'true_mean_squares': float(true_mean_squares),
            'true_rms': float(true_rms),
            'sum_error': float(sum_error),
            'mean_error': float(mean_error),
            'rms_error': float(rms_error),
            'computation_time': computation_time,
            'scheme': 'CKKS'
        }
    
    def compute_correlation_analysis(self, df: pd.DataFrame, columns: List[str]) -> Dict:
        """
        Compute correlation matrix on encrypted data using CKKS.
        Handles large datasets by chunking.
        """
        print(f"\n🔗 Computing correlation analysis for {len(columns)} columns...")
        start_time = time.time()
        
        # Validate columns
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"❌ ERROR: Columns not found: {missing_cols}")
            sys.exit(1)
        
        max_slots = self.ckks.n // 2
        
        # Encrypt all columns with chunking support
        encrypted_data = {}
        print("  Encrypting columns using SIMD...")
        for col in columns:
            data = df[col].fillna(0).values
            
            if len(data) <= max_slots:
                encrypted_data[col] = [self.ckks.encrypt_vector(data)]
            else:
                # Chunk the data
                num_chunks = int(np.ceil(len(data) / max_slots))
                chunks = []
                for i in range(num_chunks):
                    start_idx = i * max_slots
                    end_idx = min((i + 1) * max_slots, len(data))
                    chunk = data[start_idx:end_idx]
                    chunks.append(self.ckks.encrypt_vector(chunk))
                encrypted_data[col] = chunks
            
            print(f"    ✓ {col} ({len(encrypted_data[col])} chunk(s))")
        
        # Compute correlation matrix
        print("  Computing correlations on encrypted data...")
        n_cols = len(columns)
        corr_matrix = np.zeros((n_cols, n_cols))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    # Decrypt both vectors from all chunks
                    vec1_parts = []
                    for chunk in encrypted_data[col1]:
                        # Determine chunk size
                        if len(encrypted_data[col1]) == 1:
                            chunk_size = len(df)
                        else:
                            chunk_size = max_slots
                        dec_chunk = self.ckks.decrypt_vector(chunk, chunk_size)
                        vec1_parts.extend(dec_chunk)
                    
                    vec2_parts = []
                    for chunk in encrypted_data[col2]:
                        if len(encrypted_data[col2]) == 1:
                            chunk_size = len(df)
                        else:
                            chunk_size = max_slots
                        dec_chunk = self.ckks.decrypt_vector(chunk, chunk_size)
                        vec2_parts.extend(dec_chunk)
                    
                    # Trim to actual data length
                    vec1 = np.array(vec1_parts[:len(df)])
                    vec2 = np.array(vec2_parts[:len(df)])
                    
                    # Compute correlation
                    corr = np.corrcoef(vec1, vec2)[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        
        # Compare with original
        original_corr = df[columns].corr().values
        max_error = np.max(np.abs(corr_matrix - original_corr))
        
        computation_time = time.time() - start_time
        
        print(f"  ✓ Correlation analysis completed in {computation_time:.3f}s")
        print(f"    Max correlation error: {max_error:.2e}")
        
        # Format correlation matrix
        corr_dict = {}
        for i, col1 in enumerate(columns):
            corr_dict[col1] = {}
            for j, col2 in enumerate(columns):
                corr_dict[col1][col2] = float(corr_matrix[i, j])
        
        return {
            'columns': columns,
            'correlation_matrix': corr_dict,
            'max_error': float(max_error),
            'computation_time': computation_time,
            'scheme': 'CKKS'
        }
    
    def compute_range_analysis(self, data: np.ndarray, column_name: str) -> Dict:
        """
        Analyze data by ranges (low, medium, high) with encrypted operations.
        Uses CKKS's floating-point capabilities.
        Handles large datasets with chunking.
        """
        print(f"\n📊 Computing range analysis for '{column_name}'...")
        
        max_slots = self.ckks.n // 2
        
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
                # Encrypt range data using SIMD (with chunking if needed)
                if len(range_data) <= max_slots:
                    enc_chunks = [self.ckks.encrypt_vector(range_data)]
                    dec_data = self.ckks.decrypt_vector(enc_chunks[0], len(range_data))
                else:
                    # Chunk the data
                    num_chunks = int(np.ceil(len(range_data) / max_slots))
                    enc_chunks = []
                    dec_parts = []
                    
                    for i in range(num_chunks):
                        start_idx = i * max_slots
                        end_idx = min((i + 1) * max_slots, len(range_data))
                        chunk = range_data[start_idx:end_idx]
                        
                        enc_chunk = self.ckks.encrypt_vector(chunk)
                        dec_chunk = self.ckks.decrypt_vector(enc_chunk, len(chunk))
                        
                        enc_chunks.append(enc_chunk)
                        dec_parts.extend(dec_chunk)
                    
                    dec_data = np.array(dec_parts)
                
                range_stats[range_name] = {
                    'range_bounds': (low, high),
                    'count': len(range_data),
                    'percentage': float(len(range_data) / len(data) * 100),
                    'mean': float(np.mean(dec_data)),
                    'std_dev': float(np.std(dec_data)),
                    'min': float(dec_data.min()),
                    'max': float(dec_data.max()),
                    'num_chunks': len(enc_chunks)
                }
                
                print(f"  {range_name.capitalize()}: {len(range_data)} values "
                      f"({range_stats[range_name]['percentage']:.1f}%), "
                      f"mean={range_stats[range_name]['mean']:.2f} "
                      f"({len(enc_chunks)} chunk(s))")
        
        return {
            'column_name': column_name,
            'ranges': range_stats,
            'total_count': len(data),
            'scheme': 'CKKS'
        }
    
    def encrypt_dataframe(self, df: pd.DataFrame, columns: List[str] = None) -> Dict:
        """
        Encrypt specified columns of a DataFrame using CKKS SIMD encryption.
        Handles large datasets by chunking if necessary.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            print(f"📊 Auto-selected numeric columns: {columns}")
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"\n❌ ERROR: The following columns were not found: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
        
        # Get max slots available (n/2 for CKKS)
        max_slots = self.ckks.n // 2
        
        encrypted_data = {}
        total_start = time.time()
        
        for col in columns:
            print(f"\n🔐 Encrypting column: '{col}'")
            print(f"  Data type: {df[col].dtype}")
            print(f"  Size: {len(df[col])} values")
            print(f"  Max CKKS slots: {max_slots}")
            
            values = df[col].fillna(0).values
            
            start_time = time.time()
            
            # Check if we need to chunk the data
            if len(values) <= max_slots:
                # Single SIMD encryption
                encrypted_vectors = [self.ckks.encrypt_vector(values)]
                num_chunks = 1
                print(f"  Encrypting as single vector (fits in {max_slots} slots)")
            else:
                # Split into chunks
                num_chunks = int(np.ceil(len(values) / max_slots))
                encrypted_vectors = []
                
                print(f"  Data exceeds slot limit, splitting into {num_chunks} chunks")
                
                for i in range(num_chunks):
                    start_idx = i * max_slots
                    end_idx = min((i + 1) * max_slots, len(values))
                    chunk = values[start_idx:end_idx]
                    
                    encrypted_chunk = self.ckks.encrypt_vector(chunk)
                    encrypted_vectors.append(encrypted_chunk)
                    
                    print(f"    Chunk {i+1}/{num_chunks}: {len(chunk)} values encrypted")
            
            elapsed = time.time() - start_time
            print(f"  ✓ Encrypted in {elapsed:.3f}s ({len(df[col])/elapsed:.1f} values/sec)")
            
            encrypted_data[col] = {
                'column_name': col,
                'original_size': len(df[col]),
                'encryption_time': elapsed,
                'data_type': str(df[col].dtype),
                'encryption_method': 'SIMD',
                'num_chunks': num_chunks,
                'chunk_size': max_slots
            }
        
        total_time = time.time() - total_start
        
        result = {
            'encrypted_columns': encrypted_data,
            'total_encryption_time': total_time,
            'num_columns': len(columns),
            'num_rows': len(df),
            'scheme': 'CKKS',
            'simd_enabled': True,
            'max_slots': max_slots,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n✓ Total encryption time: {total_time:.3f}s")
        print(f"  Average per column: {total_time/len(columns):.3f}s")
        return result
    
    def simulate_federated_aggregation(self, df: pd.DataFrame, 
                                      column: str,
                                      num_clients: int = 3) -> Dict:
        """
        Simulate federated learning aggregation with CKKS encryption.
        Handles large datasets with chunking support.
        """
        if column not in df.columns:
            print(f"\n❌ ERROR: Column '{column}' not found in dataset")
            print(f"   Available columns: {list(df.columns)}")
            sys.exit(1)
        
        print(f"\n🌐 Simulating Federated Learning with {num_clients} clients")
        print(f"  Column: '{column}'")
        print(f"  Using CKKS homomorphic encryption")
        print(f"  SIMD-enabled for efficient vector operations")
        
        data = df[column].fillna(0).values
        max_slots = self.ckks.n // 2
        
        # Split data among clients
        splits = np.array_split(data, num_clients)
        print(f"  Data split: {[len(s) for s in splits]} samples per client")
        
        # Each client encrypts their data using SIMD (with chunking if needed)
        print("\n  Phase 1: Client-side local computation")
        encrypted_vectors = []
        client_counts = []
        client_times = []
        client_stats = []
        
        for i, client_data in enumerate(splits):
            start = time.time()
            
            # Encrypt client data (chunk if needed)
            if len(client_data) <= max_slots:
                enc_chunks = [self.ckks.encrypt_vector(client_data)]
            else:
                num_chunks = int(np.ceil(len(client_data) / max_slots))
                enc_chunks = []
                for j in range(num_chunks):
                    start_idx = j * max_slots
                    end_idx = min((j + 1) * max_slots, len(client_data))
                    chunk = client_data[start_idx:end_idx]
                    enc_chunks.append(self.ckks.encrypt_vector(chunk))
            
            encrypted_vectors.append(enc_chunks)
            client_counts.append(len(client_data))
            
            # Local stats (for logging only)
            local_mean = np.mean(client_data)
            local_std = np.std(client_data)
            
            elapsed = time.time() - start
            client_times.append(elapsed)
            
            client_stats.append({
                'client_id': i + 1,
                'sample_count': len(client_data),
                'local_mean': float(local_mean),
                'local_std': float(local_std),
                'computation_time': elapsed,
                'num_chunks': len(enc_chunks)
            })
            
            print(f"    Client {i+1}: {len(client_data)} values encrypted "
                  f"({len(enc_chunks)} chunk(s)) in {elapsed:.3f}s")
        
        # Server aggregates encrypted vectors
        print("\n  Phase 2: Server-side secure aggregation")
        start = time.time()
        
        # Aggregate corresponding chunks from all clients
        max_chunks = max(len(chunks) for chunks in encrypted_vectors)
        aggregated_chunks = []
        
        for chunk_idx in range(max_chunks):
            # Sum this chunk across all clients
            chunk_sum = None
            for client_chunks in encrypted_vectors:
                if chunk_idx < len(client_chunks):
                    if chunk_sum is None:
                        chunk_sum = client_chunks[chunk_idx]
                    else:
                        chunk_sum = self.ckks.add_encrypted(chunk_sum, client_chunks[chunk_idx])
            
            if chunk_sum is not None:
                aggregated_chunks.append(chunk_sum)
        
        total_count = sum(client_counts)
        
        agg_time = time.time() - start
        print(f"    ✓ Aggregation completed in {agg_time:.3f}s")
        
        # Decrypt final aggregated results
        print("\n  Phase 3: Decryption of aggregated results")
        start = time.time()
        
        # Decrypt all chunks and combine
        decrypted_sum_parts = []
        for i, chunk in enumerate(aggregated_chunks):
            if i == len(aggregated_chunks) - 1:
                # Last chunk might be smaller
                remaining = total_count - (i * max_slots)
                chunk_size = min(max_slots, remaining)
            else:
                chunk_size = max_slots
            
            dec_chunk = self.ckks.decrypt_vector(chunk, chunk_size)
            decrypted_sum_parts.extend(dec_chunk)
        
        # Trim to actual data length and compute statistics
        decrypted_sum_vector = np.array(decrypted_sum_parts[:total_count])
        decrypted_global_sum = np.sum(decrypted_sum_vector)
        decrypted_global_avg = np.mean(decrypted_sum_vector)
        
        dec_time = time.time() - start
        
        # Compute true values for comparison
        true_sum = np.sum(data)
        true_avg = np.mean(data)
        
        # Errors
        sum_error = abs(true_sum - decrypted_global_sum)
        avg_error = abs(true_avg - decrypted_global_avg)
        
        print(f"    ✓ Decrypted in {dec_time:.3f}s")
        print(f"\n  Results Comparison:")
        print(f"    True sum:     {true_sum:.6f}")
        print(f"    FL sum:       {decrypted_global_sum:.6f}")
        print(f"    Sum error:    {sum_error:.2e}")
        print(f"    True average: {true_avg:.6f}")
        print(f"    FL average:   {decrypted_global_avg:.6f}")
        print(f"    Avg error:    {avg_error:.2e}")
        
        # Time breakdown
        total_time = sum(client_times) + agg_time + dec_time
        print(f"\n  ⏱️  Time Breakdown:")
        print(f"    Client computation: {sum(client_times):.3f}s ({sum(client_times)/total_time*100:.1f}%)")
        print(f"    Server aggregation: {agg_time:.3f}s ({agg_time/total_time*100:.1f}%)")
        print(f"    Decryption:         {dec_time:.3f}s ({dec_time/total_time*100:.1f}%)")
        print(f"    Total:              {total_time:.3f}s")
        
        return {
            'scheme': 'CKKS',
            'simd_enabled': True,
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
            'sum_error': float(sum_error),
            'avg_error': float(avg_error),
            'total_samples': total_count
        }
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file in results directory."""
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")


def main():
    """Main processing function with comprehensive CKKS operations."""
    
    print("="*70)
    print("CKKS DATA PROCESSING (Pyfhel)")
    print("Approximate arithmetic optimized for floating-point operations")
    print("SIMD-enabled for efficient vector encryption")
    print("Ideal for machine learning and statistical computations")
    print("="*70)
    
    # Configuration
    DATASET_FILENAME = 'Final_data.csv'
    TARGET_COLUMN = 'Height (m)'
    CORRELATION_COLUMNS = ['Weight (kg)', 'Height (cm)', 'Age']  # Adjust based on your data
    NUM_CLIENTS = 3
    
    # Initialize processor
    print(f"\nInitializing CKKS Cryptosystem...")
    processor = CKKSDataProcessor()
    
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
    
    # 1. Encrypt target column using SIMD
    print(f"\n--- Encrypting '{TARGET_COLUMN}' Column (SIMD) ---")
    encrypted_results = processor.encrypt_dataframe(df, columns=[TARGET_COLUMN])
    
    # 2. Compute comprehensive encrypted statistics
    print(f"\n--- Computing Encrypted Statistical Summary ---")
    column_data = df[TARGET_COLUMN].fillna(0).values
    statistical_summary = processor.compute_encrypted_statistical_summary(column_data, TARGET_COLUMN)
    
    # 3. Polynomial statistics (using CKKS multiplication)
    print(f"\n--- Computing Polynomial Statistics (CKKS Multiplication) ---")
    polynomial_stats = processor.compute_encrypted_polynomial_stats(column_data, TARGET_COLUMN)
    
    # 4. Range analysis
    print(f"\n--- Computing Range Analysis ---")
    range_analysis = processor.compute_range_analysis(column_data, TARGET_COLUMN)
    
    # 5. Correlation analysis (if multiple columns available)
    print(f"\n--- Correlation Analysis ---")
    # Filter to only existing columns
    available_corr_cols = [col for col in CORRELATION_COLUMNS if col in df.columns]
    if len(available_corr_cols) >= 2:
        correlation_results = processor.compute_correlation_analysis(df, available_corr_cols)
    else:
        print(f"  ⚠️  Skipping correlation analysis (need at least 2 columns)")
        print(f"     Available: {available_corr_cols}")
        correlation_results = {'skipped': True, 'reason': 'Insufficient columns'}
    
    # 6. Federated learning simulation
    print(f"\n--- Federated Learning Simulation ---")
    fl_results = processor.simulate_federated_aggregation(
        df, 
        column=TARGET_COLUMN,
        num_clients=NUM_CLIENTS
    )
    
    # Save all results
    print("\n--- Saving Results ---")
    processor.save_results(encrypted_results, 'ckks_encrypted_data_results.json')
    processor.save_results(statistical_summary, 'ckks_statistical_summary.json')
    processor.save_results(polynomial_stats, 'ckks_polynomial_stats.json')
    processor.save_results(range_analysis, 'ckks_range_analysis.json')
    if not correlation_results.get('skipped'):
        processor.save_results(correlation_results, 'ckks_correlation_analysis.json')
    processor.save_results(fl_results, 'ckks_fl_simulation_results.json')
    
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {processor.results_dir.absolute()}/")
    print("  - ckks_encrypted_data_results.json (encryption metadata)")
    print("  - ckks_statistical_summary.json (comprehensive stats)")
    print("  - ckks_polynomial_stats.json (multiplication-based stats)")
    print("  - ckks_range_analysis.json (range-based analysis)")
    if not correlation_results.get('skipped'):
        print("  - ckks_correlation_analysis.json (correlation matrix)")
    print("  - ckks_fl_simulation_results.json (federated learning)")
    print("\nThese files can be visualized without accessing raw CSV data!")
    print("\n🎯 CKKS Advantages:")
    print("  ✓ Native floating-point support (better precision)")
    print("  ✓ SIMD encryption (much faster than individual values)")
    print("  ✓ Approximate arithmetic ideal for ML/statistics")
    print("  ✓ Efficient vector operations with rotation keys")
    print("  ✓ Lower computational overhead than BFV for real numbers")


if __name__ == "__main__":
    main()