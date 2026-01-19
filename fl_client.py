"""
Federated Learning Client with CKKS Homomorphic Encryption
===========================================================

Client encrypts local data statistics and sends to server for
privacy-preserving aggregation. Only encrypts aggregates, never raw data.

Features:
- Local data loading and preprocessing
- CKKS encryption with public key only
- Computes local statistics (sum, sum of squares)
- Sends encrypted statistics to server
- Receives aggregated results
"""

import json
import pickle
import base64
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from src.schemes.ckks.ckks_crypto import CKKSCrypto



class FederatedClient:
    """
    Client for federated learning with CKKS homomorphic encryption.
    Encrypts local statistics and sends to server.
    """
    
    def __init__(self, 
                 client_id: str,
                 server_url: str = 'http://localhost:5000'):
        """
        Initialize federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            server_url: Server URL
        """
        self.client_id = client_id
        self.server_url = server_url
        self.ckks = None
        self.local_data = None
        self.local_stats = None
        
        print(f"\n{'='*70}")
        print(f"FEDERATED LEARNING CLIENT: {client_id}")
        print(f"{'='*70}")
        print(f"  Server: {server_url}")
    
    def connect_to_server(self):
        """
        Connect to server and retrieve public key bundle.
        """
        print(f"\n🔌 Connecting to server...")
        
        try:
            # Get public key bundle
            response = requests.get(f"{self.server_url}/api/public-key")
            response.raise_for_status()
            
            bundle = response.json()
            
            print(f"  ✓ Connected successfully")
            print(f"  Server ID: {bundle['server_id']}")
            print(f"  Timestamp: {bundle['timestamp']}")
            
            # Initialize CKKS with server's parameters
            params = bundle['context_params']
            self.ckks = CKKSCrypto(
                n=params['n'],
                scale=params['scale'],
                qi_sizes=params['qi_sizes'],
                sec=params['sec']
            )
            self.ckks.setup()
            
            # Save and load public key
            public_key_bytes = base64.b64decode(bundle['public_key'])
            
            keys_dir = Path(f"client_{self.client_id}_keys")
            keys_dir.mkdir(exist_ok=True)
            
            public_key_path = keys_dir / "public_key.bin"
            with open(public_key_path, 'wb') as f:
                f.write(public_key_bytes)
            
            # Note: We need both public and secret key files for load_keys
            # For client, we only use public key for encryption
            # Create a dummy secret key file (won't be used)
            secret_key_path = keys_dir / "secret_key.bin"
            with open(secret_key_path, 'wb') as f:
                f.write(b'')  # Empty file
            
            print(f"\n  🔑 Public key received and saved")
            print(f"  ⚠️  Note: Client only has public key (cannot decrypt)")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Connection failed: {e}")
            return False
    
    def register(self, info: Optional[Dict[str, Any]] = None):
        """
        Register with the server.
        
        Args:
            info: Optional client metadata
        """
        print(f"\n📝 Registering with server...")
        
        if info is None:
            info = {
                'type': 'hospital',
                'location': 'unknown'
            }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/register",
                json={
                    'client_id': self.client_id,
                    'info': info
                }
            )
            response.raise_for_status()
            
            result = response.json()
            print(f"  ✓ Registered successfully")
            print(f"  Current round: {result['current_round']}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Registration failed: {e}")
            return False
    
    def load_local_data(self, data: np.ndarray, column_name: str = 'value'):
        """
        Load local data for this client.
        
        Args:
            data: Local data array
            column_name: Name of the data column
        """
        self.local_data = data
        self.column_name = column_name
        
        print(f"\n📊 Local data loaded:")
        print(f"  Column: {column_name}")
        print(f"  Samples: {len(data)}")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Std: {np.std(data):.2f}")
        print(f"  Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    
    def load_from_csv(self, filepath: str, column: str):
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            column: Column name to use
        """
        print(f"\n📂 Loading data from CSV...")
        print(f"  File: {filepath}")
        print(f"  Column: {column}")
        
        df = pd.read_csv(filepath)
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV")
        
        data = df[column].fillna(0).values
        self.load_local_data(data, column)
    
    def compute_local_statistics(self):
        """
        Compute local statistics (sum, sum of squares, count).
        These will be encrypted before sending to server.
        """
        if self.local_data is None:
            raise ValueError("No local data loaded")
        
        print(f"\n🔢 Computing local statistics...")
        
        start_time = time.time()
        
        # Compute statistics
        count = len(self.local_data)
        local_sum = float(np.sum(self.local_data))
        local_sum_squares = float(np.sum(self.local_data ** 2))
        local_mean = float(np.mean(self.local_data))
        local_std = float(np.std(self.local_data))
        
        computation_time = time.time() - start_time
        
        self.local_stats = {
            'count': count,
            'sum': local_sum,
            'sum_squares': local_sum_squares,
            'mean': local_mean,  # Not sent to server
            'std': local_std      # Not sent to server
        }
        
        print(f"  ✓ Statistics computed in {computation_time:.3f}s")
        print(f"    Count: {count}")
        print(f"    Sum: {local_sum:.2f}")
        print(f"    Sum of squares: {local_sum_squares:.2f}")
        print(f"    Local mean: {local_mean:.2f} (not shared)")
        print(f"    Local std: {local_std:.2f} (not shared)")
    
    def encrypt_and_send(self, round_id: int):
        """
        Encrypt local statistics and send to server.
        
        Args:
            round_id: Current aggregation round ID
        """
        if self.ckks is None:
            raise ValueError("Not connected to server. Call connect_to_server() first.")
        
        if self.local_stats is None:
            raise ValueError("No local statistics computed. Call compute_local_statistics() first.")
        
        print(f"\n🔐 Encrypting local statistics...")
        
        start_time = time.time()
        
        # Encrypt sum and sum of squares
        enc_sum = self.ckks.encrypt(self.local_stats['sum'])
        enc_sum_squares = self.ckks.encrypt(self.local_stats['sum_squares'])
        
        encryption_time = time.time() - start_time
        
        print(f"  ✓ Encryption complete in {encryption_time:.3f}s")
        print(f"    Encrypted: sum, sum_squares")
        print(f"    Count sent in plaintext: {self.local_stats['count']}")
        
        # Serialize encrypted values
        print(f"\n📤 Sending encrypted data to server...")
        
        encrypted_data = {
            'count': self.local_stats['count'],
            'encrypted_sum': base64.b64encode(pickle.dumps(enc_sum)).decode('utf-8'),
            'encrypted_sum_squares': base64.b64encode(pickle.dumps(enc_sum_squares)).decode('utf-8')
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/contribute",
                json={
                    'round_id': round_id,
                    'client_id': self.client_id,
                    'encrypted_data': encrypted_data
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"  ✓ Data sent successfully")
            print(f"    Round: {result['round_id']}")
            print(f"    Status: {result['status']}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to send data: {e}")
            return False
    
    def get_results(self, round_id: int) -> Optional[Dict[str, Any]]:
        """
        Get aggregated results from server.
        
        Args:
            round_id: Round ID to get results for
            
        Returns:
            Results dictionary or None if failed
        """
        print(f"\n📥 Retrieving results for round {round_id}...")
        
        try:
            response = requests.get(
                f"{self.server_url}/api/round/{round_id}/results"
            )
            response.raise_for_status()
            
            results = response.json()
            
            print(f"\n  ✅ Global Results:")
            print(f"    Total samples: {results['total_count']}")
            print(f"    Contributing clients: {results['num_clients']}")
            print(f"    Global mean: {results['global_mean']:.6f}")
            print(f"    Global std: {results['global_std']:.6f}")
            print(f"    Aggregation time: {results['aggregation_time']:.3f}s")
            
            # Compare with local statistics
            if self.local_stats:
                print(f"\n  📊 Local vs Global Comparison:")
                print(f"    Local mean:  {self.local_stats['mean']:.6f}")
                print(f"    Global mean: {results['global_mean']:.6f}")
                diff = self.local_stats['mean'] - results['global_mean']
                print(f"    Difference:  {diff:+.6f}")
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to get results: {e}")
            return None
    
    def participate_in_round(self, round_id: int):
        """
        Complete participation in an aggregation round.
        
        Args:
            round_id: Round ID to participate in
        """
        print(f"\n{'='*70}")
        print(f"🚀 Participating in Round {round_id}")
        print(f"{'='*70}")
        
        # Compute local statistics
        self.compute_local_statistics()
        
        # Encrypt and send
        success = self.encrypt_and_send(round_id)
        
        if success:
            print(f"\n  ✓ Successfully participated in round {round_id}")
        else:
            print(f"\n  ❌ Failed to participate in round {round_id}")
        
        return success


def create_sample_client(client_id: str, 
                        num_samples: int = 500,
                        mean: float = 65000,
                        std: float = 15000,
                        server_url: str = 'http://localhost:5000'):
    """
    Create a sample client with synthetic salary data.
    
    Args:
        client_id: Client identifier
        num_samples: Number of samples
        mean: Mean salary
        std: Standard deviation
        server_url: Server URL
        
    Returns:
        Configured FederatedClient
    """
    client = FederatedClient(client_id, server_url)
    
    # Generate synthetic salary data
    data = np.random.normal(mean, std, num_samples)
    data = np.abs(data)  # Ensure positive salaries
    
    client.load_local_data(data, 'salary')
    
    return client


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("FEDERATED CLIENT - EXAMPLE USAGE")
    print("="*70)
    
    # Create client
    client = create_sample_client(
        client_id='Hospital_A',
        num_samples=500,
        mean=65000,
        std=15000
    )
    
    # Connect to server
    if client.connect_to_server():
        # Register
        client.register({'type': 'hospital', 'location': 'City A'})
        
        # Participate in round 1
        client.participate_in_round(round_id=1)
        
        print("\n✓ Client example completed")
    else:
        print("\n❌ Could not connect to server")