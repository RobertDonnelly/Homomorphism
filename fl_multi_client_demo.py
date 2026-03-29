"""
Multi-Client Federated Learning Demo
=====================================

Demonstrates multiple clients connecting to a federated learning server
and participating in aggregation rounds.

Usage:
    python multi_client_demo.py

This will:
1. Verify server is running
2. Create multiple clients (hospitals)
3. Each client encrypts their local data
4. Send encrypted data to server
5. Server aggregates without seeing raw data
6. Clients receive global statistics
"""

import time
import requests
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Import the fixed client
try:
    from fl_client import FederatedClient
except ImportError:
    print("Error: fl_client_fixed.py not found!")
    print("Please ensure fl_client_fixed.py is in the same directory")
    exit(1)


class MultiClientOrchestrator:
    """Orchestrates multiple federated learning clients."""
    
    def __init__(self, server_url: str = 'http://localhost:5000'):
        """Initialize orchestrator."""
        self.server_url = server_url
        self.clients = []
        
        print("\n" + "="*80)
        print("MULTI-CLIENT FEDERATED LEARNING ORCHESTRATOR")
        print("="*80)
        print(f"Server URL: {server_url}")
    
    def check_server(self) -> bool:
        """Check if server is running."""
        print("\n🔍 Checking server status...")
        
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("  ✓ Server is running and healthy")
                return True
            else:
                print(f"  X Server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"  X Cannot connect to server at {self.server_url}")
            print(f"\n  Please start the server first:")
            print(f"    python fl_server.py")
            return False
        except Exception as e:
            print(f"  X Error: {e}")
            return False
    
    def create_clients(self, client_configs: List[Dict[str, Any]]) -> bool:
        """
        Create multiple clients with different data distributions.
        
        Args:
            client_configs: List of client configuration dictionaries
                Each dict should have: id, samples, mean, std, location
        """
        print("\n" + "="*80)
        print(f"CREATING {len(client_configs)} CLIENTS")
        print("="*80)
        
        for config in client_configs:
            print(f"\n.. Creating client: {config['id']}")
            print(f"   Location: {config.get('location', 'Unknown')}")
            print(f"   Samples: {config['samples']}")
            print(f"   Mean: ${config['mean']:,.2f}")
            print(f"   Std Dev: ${config['std']:,.2f}")
            
            # Create client
            client = FederatedClient(
                client_id=config['id'],
                server_url=self.server_url
            )
            
            # Generate data for this client
            data = np.random.normal(
                loc=config['mean'],
                scale=config['std'],
                size=config['samples']
            )
            data = np.abs(data)  # Ensure positive values
            
            # Load data
            client.load_local_data(data, column_name=config.get('column', 'salary'))
            
            # Store client info
            client.metadata = {
                'location': config.get('location', 'Unknown'),
                'type': config.get('type', 'hospital')
            }
            
            self.clients.append(client)
        
        print(f"\n✓ Created {len(self.clients)} clients successfully")
        return True
    
    def connect_all_clients(self) -> bool:
        """Connect all clients to the server."""
        print("\n" + "="*80)
        print("CONNECTING CLIENTS TO SERVER")
        print("="*80)
        
        all_connected = True
        
        for i, client in enumerate(self.clients, 1):
            print(f"\n[{i}/{len(self.clients)}] Connecting {client.client_id}...")
            
            if client.connect_to_server():
                print(f"  ✓ {client.client_id} connected")
            else:
                print(f"  X {client.client_id} failed to connect")
                all_connected = False
        
        if all_connected:
            print(f"\n! All {len(self.clients)} clients connected successfully!")
        else:
            print(f"\n!  Some clients failed to connect")
        
        return all_connected
    
    def register_all_clients(self) -> bool:
        """Register all clients with the server."""
        print("\n" + "="*80)
        print("REGISTERING CLIENTS")
        print("="*80)
        
        all_registered = True
        
        for i, client in enumerate(self.clients, 1):
            print(f"\n[{i}/{len(self.clients)}] Registering {client.client_id}...")
            
            if client.register(client.metadata):
                print(f"  ✓ {client.client_id} registered")
            else:
                print(f"  X {client.client_id} failed to register")
                all_registered = False
        
        if all_registered:
            print(f"\n All {len(self.clients)} clients registered")
        else:
            print(f"\n!  Some clients failed to register")
        
        return all_registered
    
    def run_federated_round(self, round_id: int = None) -> Dict[str, Any]:
        """
        Run a complete federated learning round with all clients.
        
        Args:
            round_id: Round ID (if None, server will create new round)
        
        Returns:
            Aggregated results
        """
        print("\n" + "="*80)
        print(f"RUNNING FEDERATED LEARNING ROUND")
        print("="*80)
        
        # Start round on server (first client will do this)
        if round_id is None:
            print("\n🔄 Starting new aggregation round on server...")
            try:
                response = requests.post(f"{self.server_url}/api/round/start")
                if response.status_code == 200:
                    round_id = response.json()['round_id']
                    print(f"  ! Round {round_id} started!")
                else:
                    print(f"  X Failed to start round")
                    return None
            except Exception as e:
                print(f"  X Error starting round: {e}")
                return None
        
        # Each client computes local stats and encrypts
        print(f"\n{'='*80}")
        print(f"PHASE 1: LOCAL COMPUTATION & ENCRYPTION")
        print(f"{'='*80}")
        
        for i, client in enumerate(self.clients, 1):
            print(f"\n[{i}/{len(self.clients)}] {client.client_id}:")
            
            # Compute local statistics
            client.compute_local_statistics()
        
        # Each client sends encrypted data to server
        print(f"\n{'='*80}")
        print(f"PHASE 2: SENDING ENCRYPTED DATA")
        print(f"{'='*80}")
        
        successful_clients = 0
        
        for i, client in enumerate(self.clients, 1):
            print(f"\n[{i}/{len(self.clients)}] {client.client_id}:")
            
            if client.encrypt_and_send(round_id):
                successful_clients += 1
                print(f"  ✓ Successfully contributed")
            else:
                print(f"  ! Failed to contribute")
        
        print(f"\n..Here is your Contribution Summary:")
        print(f"  Total clients: {len(self.clients)}")
        print(f"  Successful: {successful_clients}")
        print(f"  Failed: {len(self.clients) - successful_clients}")
        
        # Trigger server aggregation
        print(f"\n{'='*80}")
        print(f"PHASE 3: SERVER AGGREGATION")
        print(f"{'='*80}")
        
        print(f"\n... Requesting server to aggregate round {round_id}...")
        
        try:
            response = requests.post(
                f"{self.server_url}/api/round/{round_id}/aggregate"
            )
            
            if response.status_code == 200:
                print(f"  ✓ Aggregation completed")
            else:
                print(f"  X Aggregation failed: {response.text}")
                return None
                
        except Exception as e:
            print(f"  X Error: {e}")
            return None
        
        # Get results
        print(f"\n{'='*80}")
        print(f"PHASE 4: RETRIEVING GLOBAL RESULTS")
        print(f"{'='*80}")
        
        time.sleep(1)  # Brief pause
        
        # First client retrieves and displays results
        results = self.clients[0].get_results(round_id)
        
        if results:
            # Show comparison for all clients
            print(f"\n{'='*80}")
            print(f"LOCAL vs GLOBAL COMPARISON")
            print(f"{'='*80}")
            
            print(f"\n{'Client ID':<20} {'Local Mean':>15} {'Global Mean':>15} {'Difference':>15}")
            print("-" * 70)
            
            for client in self.clients:
                if client.local_stats:
                    local_mean = client.local_stats['mean']
                    global_mean = results['global_mean']
                    diff = local_mean - global_mean
                    
                    print(f"{client.client_id:<20} ${local_mean:>14,.2f} ${global_mean:>14,.2f} ${diff:>+14,.2f}")
            
            print("-" * 70)
            print(f"{'GLOBAL (Aggregated)':<20} {' ':>15} ${results['global_mean']:>14,.2f}")
            print(f"{'Global Std Dev:':<20} {' ':>15} ${results['global_std']:>14,.2f}")
            print(f"{'Total Samples:':<20} {' ':>15} {results['total_count']:>15,}")
        
        return results
    
    def run_multiple_rounds(self, num_rounds: int = 3):
        """Run multiple federated learning rounds."""
        print("\n" + "="*80)
        print(f"RUNNING {num_rounds} FEDERATED LEARNING ROUNDS")
        print("="*80)
        
        results_history = []
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'#'*80}")
            print(f"# ROUND {round_num}/{num_rounds}")
            print(f"{'#'*80}")
            
            results = self.run_federated_round()
            
            if results:
                results_history.append(results)
                print(f"\n.. Round {round_num} Has completed successfully!")
            else:
                print(f"\n!! Round {round_num} failed")
            
            # Brief pause between rounds
            if round_num < num_rounds:
                print(f"\nWaiting 2 seconds before next round...")
                time.sleep(2)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"MULTI-ROUND SUMMARY")
        print(f"{'='*80}")
        
        if results_history:
            print(f"\n{'Round':<10} {'Global Mean':>15} {'Global Std':>15} {'Total Samples':>15}")
            print("-" * 60)
            for i, res in enumerate(results_history, 1):
                print(f"{i:<10} ${res['global_mean']:>14,.2f} ${res['global_std']:>14,.2f} {res['total_count']:>15,}")
        
        return results_history


def create_hospital_scenario():
    """
    Create a realistic hospital scenario with different salary distributions.
    """
    return [
        {
            'id': 'Hospital_NYC',
            'location': 'New York City, NY',
            'type': 'hospital',
            'samples': 5000,
            'mean': 85000,  # Higher cost of living
            'std': 22000,
            'column': 'salary'
        },
        {
            'id': 'Hospital_Chicago',
            'location': 'Chicago, IL',
            'type': 'hospital',
            'samples': 4000,
            'mean': 72000,
            'std': 18000,
            'column': 'salary'
        },
        {
            'id': 'Hospital_Austin',
            'location': 'Austin, TX',
            'type': 'hospital',
            'samples': 3500,
            'mean': 68000,
            'std': 16000,
            'column': 'salary'
        },
        {
            'id': 'Hospital_Seattle',
            'location': 'Seattle, WA',
            'type': 'hospital',
            'samples': 3800,
            'mean': 78000,
            'std': 20000,
            'column': 'salary'
        },
        {
            'id': 'Hospital_Atlanta',
            'location': 'Atlanta, GA',
            'type': 'hospital',
            'samples': 3900,
            'mean': 65000,
            'std': 15000,
            'column': 'salary'
        }
    ]


def create_simple_scenario():
    """Create a simple 3-client scenario for testing."""
    return [
        {
            'id': 'Client_A',
            'location': 'Region A',
            'type': 'organization',
            'samples': 400,
            'mean': 60000,
            'std': 12000,
            'column': 'value'
        },
        {
            'id': 'Client_B',
            'location': 'Region B',
            'type': 'organization',
            'samples': 500,
            'mean': 70000,
            'std': 15000,
            'column': 'value'
        },
        {
            'id': 'Client_C',
            'location': 'Region C',
            'type': 'organization',
            'samples': 350,
            'mean': 55000,
            'std': 10000,
            'column': 'value'
        }
    ]


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MULTI-CLIENT FEDERATED LEARNING DEMONSTRATION")
    print("="*80)
    
    # Create orchestrator
    orchestrator = MultiClientOrchestrator(server_url='http://localhost:5000')
    
    # Check server
    if not orchestrator.check_server():
        print("\nX Server check failed. Exiting.")
        print("\nPlease start the server first:")
        print("  python fl_server.py")
        exit(1)
    
    # Choose scenario
    print("\n" + "="*80)
    print("SELECT SCENARIO")
    print("="*80)
    print("1. Simple (3 clients) - Quick test")
    print("2. Hospital (5 clients) - Realistic healthcare scenario")
    
    choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        client_configs = create_hospital_scenario()
        print("\n✓ Selected: Hospital Scenario (5 clients)")
    else:
        client_configs = create_simple_scenario()
        print("\n✓ Selected: Simple Scenario (3 clients)")
    
    # Create clients
    if not orchestrator.create_clients(client_configs):
        print("\n.. Failed to create clients")
        exit(1)
    
    # Connect all clients
    if not orchestrator.connect_all_clients():
        print("\n.. Failed to connect all clients")
        exit(1)
    
    # Register all clients
    if not orchestrator.register_all_clients():
        print("\n.. Failed to register all clients")
        exit(1)
    
    # Run federated learning
    print("\n" + "="*80)
    print("READY TO RUN FEDERATED LEARNING")
    print("="*80)
    print("\nOptions:")
    print("1. Run single round")
    print("2. Run multiple rounds (3)")
    
    rounds_choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    if rounds_choice == "2":
        results = orchestrator.run_multiple_rounds(num_rounds=3)
    else:
        results = orchestrator.run_federated_round()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ MULTI-CLIENT DEMO COMPLETE")
    print("="*80)
    print("\nKey Points:")
    print("  • Multiple clients connected to single server")
    print("  • Each client encrypted their local data")
    print("  • Server aggregated WITHOUT seeing raw data")
    print("  • Global statistics computed on encrypted data")
    print("  • Privacy preserved throughout the process")
    print("\n" + "="*80)
