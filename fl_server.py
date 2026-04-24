"""
- CKKS encryption for privacy-preserving aggregation
- HTTP REST API for client communication
- Homomorphic operations (addition, scalar multiplication)
- Multiple aggregation rounds support
- Automatic serialization/deserialization of ciphertexts
"""

import json
import pickle
import base64
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from flask import Flask, request, jsonify
import numpy as np

from src.schemes.ckks.ckks_crypto import CKKSCrypto


class FederatedServer:
    """
    Server for federated learning with CKKS homomorphic encryption.
    Aggregates encrypted statistics from multiple clients.
    """
    
    def __init__(self, 
                 n: int = 2**14,
                 scale: int = 2**40,
                 qi_sizes: List[int] = None,
                 sec: int = 128):
        """
        Initialize federated learning server.
        
        Args:
            n: Polynomial modulus degree
            scale: Encoding scale
            qi_sizes: Coefficient modulus chain
            sec: Security level
        """
        print("="*70)
        print("FEDERATED LEARNING SERVER - CKKS HOMOMORPHIC ENCRYPTION")
        print("="*70)
        
        # Initialize CKKS
        self.ckks = CKKSCrypto(n=n, scale=scale, qi_sizes=qi_sizes, sec=sec)
        self.ckks.setup()
        
        # Server state
        self.current_round = 0
        self.clients = {}  # {client_id: client_info}
        self.encrypted_contributions = {}  # {round_id: {client_id: data}}
        self.results = {}  # {round_id: results}
        
        # Save directory for keys
        self.keys_dir = Path("server_keys")
        self.keys_dir.mkdir(exist_ok=True)
        
        # Save keys
        self._save_keys()
        
        print(f"\n✓ Server initialized and ready")
        print(f"  Keys saved to: {self.keys_dir.absolute()}")
    
    def _save_keys(self):
        """Save public and secret keys to files."""
        public_key_path = str(self.keys_dir / "public_key.bin")
        secret_key_path = str(self.keys_dir / "secret_key.bin")
        
        self.ckks.save_keys(public_key_path, secret_key_path)
        
        # Also save context parameters for clients
        context_params = {
            'n': self.ckks.n,
            'scale': self.ckks.scale,
            'qi_sizes': self.ckks.qi_sizes,
            'sec': self.ckks.sec
        }
        
        with open(self.keys_dir / "context_params.json", 'w') as f:
            json.dump(context_params, f, indent=2)
        
        print(f"\n🔑 Keys and context saved:")
        print(f"  • {public_key_path}")
        print(f"  • {secret_key_path}")
        print(f"  • {self.keys_dir / 'context_params.json'}")
    
    def get_public_key_bundle(self) -> Dict[str, Any]:
        
        #Get public key and context for distribution to clients.

        # Read public key
        public_key_path = self.keys_dir / "public_key.bin"
        with open(public_key_path, 'rb') as f:
            public_key_bytes = f.read()
        
        # Read context params
        with open(self.keys_dir / "context_params.json", 'r') as f:
            context_params = json.load(f)
        
        return {
            'public_key': base64.b64encode(public_key_bytes).decode('utf-8'),
            'context_params': context_params,
            'server_id': 'federated_server_1',
            'timestamp': datetime.now().isoformat()
        }
    
    def start_aggregation_round(self) -> int:
        """
        Start a new aggregation round.
        
        Returns:
            Round ID
        """
        self.current_round += 1
        self.encrypted_contributions[self.current_round] = {}
        
        print(f"\n{'='*70}")
        print(f" Started Aggregation Round {self.current_round}")
        print(f"{'='*70}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        print(f"  Waiting for client contributions...")
        
        return self.current_round
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]):
        #Register a client for participation.      
        self.clients[client_id] = {
            'client_id': client_id,
            'registered_at': datetime.now().isoformat(),
            'info': client_info
        }
        
        print(f"\n📝 Client registered: {client_id}")
        print(f"  Info: {client_info}")
    
    def receive_encrypted_contribution(self, 
                                       round_id: int,
                                       client_id: str,
                                       encrypted_data: Dict[str, Any],
                                       is_local: bool = False):
        """
        Receive encrypted statistics from a client.
        
        Args:
            round_id: Current round ID
            client_id: Client identifier
            encrypted_data: Dictionary with encrypted statistics
            is_local: If True, data is already PyCtxt objects (not serialized)
        """
        if round_id != self.current_round:
            raise ValueError(f"Invalid round ID. Current round: {self.current_round}")
        
        # Deserialize encrypted ciphertexts (if needed)
        deserialized_data = {}
        for key, value in encrypted_data.items():
            if key.startswith('encrypted_'):
                if is_local:
                    # Already PyCtxt object (local demo mode)
                    deserialized_data[key] = value
                else:
                    # Decode base64 and unpickle PyCtxt (HTTP mode)
                    ctxt_bytes = base64.b64decode(value)
                    ctxt = pickle.loads(ctxt_bytes)
                    deserialized_data[key] = ctxt
            else:
                deserialized_data[key] = value
        
        # Store contribution
        self.encrypted_contributions[round_id][client_id] = deserialized_data
        
        print(f"\n Received contribution from {client_id}")
        print(f"  Round: {round_id}")
        print(f"  Count: {deserialized_data.get('count', 'N/A')}")
        print(f"  Encrypted fields: {[k for k in deserialized_data.keys() if k.startswith('encrypted_')]}")
    
    def aggregate_round(self, round_id: int) -> Dict[str, Any]:
        """
        Aggregate all client contributions for a round using homomorphic operations.  
        Args-->round_id: Round to aggregate
        Returns: Aggregated results (decrypted)
        """
        if round_id not in self.encrypted_contributions:
            raise ValueError(f"No contributions for round {round_id}")
        
        contributions = self.encrypted_contributions[round_id]
        
        if len(contributions) == 0:
            raise ValueError("No contributions to aggregate")
        
        print(f"\n{'='*70}")
        print(f" Aggregating Round {round_id}")
        print(f"{'='*70}")
        print(f"  Number of clients: {len(contributions)}")
        
        start_time = time.time()
        
        # Initialize aggregated encrypted values
        agg_enc_sum = None
        agg_enc_sum_squares = None
        total_count = 0
        
        # Homomorphic aggregation
        for client_id, data in contributions.items():
            print(f"\n  Processing {client_id}:")
            print(f"    Count: {data['count']}")
            
            # Sum
            if agg_enc_sum is None:
                agg_enc_sum = data['encrypted_sum']
            else:
                print(f"    + Adding encrypted sum...")
                agg_enc_sum = self.ckks.add_encrypted(agg_enc_sum, data['encrypted_sum'])
            
            # Sum of squares
            if agg_enc_sum_squares is None:
                agg_enc_sum_squares = data['encrypted_sum_squares']
            else:
                print(f"    + Adding encrypted sum of squares...")
                agg_enc_sum_squares = self.ckks.add_encrypted(
                    agg_enc_sum_squares, 
                    data['encrypted_sum_squares']
                )
            
            # Count (public)
            total_count += data['count']
        
        print(f"\n  ✓ Homomorphic aggregation complete")
        print(f"  Total samples: {total_count}")
        
        # Compute encrypted global mean
        print(f"\n  .... Computing encrypted global mean...")
        enc_global_mean = self.ckks.multiply_plain(agg_enc_sum, 1.0 / total_count)
        
        # Compute variance from encrypted data
        # Var = E[X²] - (E[X])²
        print(f"  ..... Computing variance...")
        # Decrypt for variance calculation
        global_sum = self.ckks.decrypt(agg_enc_sum)
        global_sum_squares = self.ckks.decrypt(agg_enc_sum_squares)
        
        # Calculate variance in plaintext
        mean_of_squares = global_sum_squares / total_count
        mean_squared = (global_sum / total_count) ** 2
        global_variance = mean_of_squares - mean_squared
        global_std = np.sqrt(abs(global_variance))
        global_mean = global_sum / total_count
        
        aggregation_time = time.time() - start_time
        
        # Store results
        results = {
            'round_id': round_id,
            'total_count': total_count,
            'num_clients': len(contributions),
            'global_sum': float(global_sum),
            'global_mean': float(global_mean),
            'global_variance': float(global_variance),
            'global_std': float(global_std),
            'aggregation_time': aggregation_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results[round_id] = results
        
        print(f"\n  !! Aggregation Results:")
        print(f"    Total samples: {total_count}")
        print(f"    Global mean: {global_mean:.6f}")
        print(f"    Global std: {global_std:.6f}")
        print(f"    Aggregation time: {aggregation_time:.3f}s")
        
        return results
    
    def get_results(self, round_id: int) -> Dict[str, Any]:
        """Get results for a specific round."""
        if round_id not in self.results:
            raise ValueError(f"No results available for round {round_id}")
        return self.results[round_id]
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'current_round': self.current_round,
            'total_clients': len(self.clients),
            'total_rounds': len(self.results),
            'registered_clients': list(self.clients.keys())
        }


# Flask REST API
app = Flask(__name__)
server = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/api/public-key', methods=['GET'])
def get_public_key():
    """Get public key bundle for clients."""
    try:
        bundle = server.get_public_key_bundle()
        return jsonify(bundle)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/register', methods=['POST'])
def register_client():
    """Register a new client."""
    try:
        data = request.json
        client_id = data['client_id']
        client_info = data.get('info', {})
        
        server.register_client(client_id, client_info)
        
        return jsonify({
            'status': 'registered',
            'client_id': client_id,
            'current_round': server.current_round
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/round/start', methods=['POST'])
def start_round():
    """Start a new aggregation round."""
    try:
        round_id = server.start_aggregation_round()
        return jsonify({
            'round_id': round_id,
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/contribute', methods=['POST'])
def contribute():
    """Receive encrypted contribution from client."""
    try:
        data = request.json
        round_id = data['round_id']
        client_id = data['client_id']
        encrypted_data = data['encrypted_data']
        
        server.receive_encrypted_contribution(round_id, client_id, encrypted_data)
        
        return jsonify({
            'status': 'received',
            'round_id': round_id,
            'client_id': client_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/round/<int:round_id>/aggregate', methods=['POST'])
def aggregate(round_id):
    """Aggregate contributions for a round."""
    try:
        results = server.aggregate_round(round_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/round/<int:round_id>/results', methods=['GET'])
def get_results(round_id):
    """Get results for a specific round."""
    try:
        results = server.get_results(round_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get server statistics."""
    try:
        stats = server.get_server_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_server(host: str = '0.0.0.0', port: int = 5000):
    """
    Run the federated learning server.
    
    Args:
        host: Host address
        port: Port number
    """
    global server
    
    # Initialize server
    server = FederatedServer()
    
    print(f"\n{'='*70}")
    print(f"!! Starting Federated Learning Server")
    print(f"{'='*70}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"\n  API Endpoints:")
    print(f"    GET  /health")
    print(f"    GET  /api/public-key")
    print(f"    POST /api/register")
    print(f"    POST /api/round/start")
    print(f"    POST /api/contribute")
    print(f"    POST /api/round/<id>/aggregate")
    print(f"    GET  /api/round/<id>/results")
    print(f"    GET  /api/stats")
    print(f"\n  Server ready to accept client connections!")
    print(f"{'='*70}\n")
    
    # Run Flask app
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_server()