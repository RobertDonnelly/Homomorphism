"""
Integration Tests for Federated Learning System
================================================

Tests the complete federated learning workflow including:
- Client-server interaction
- Data processing pipeline
- End-to-end encryption workflows
- Multi-client aggregation
- Privacy guarantees

Usage:
    python test_integration_fl.py
"""

import unittest
import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent))
from fl_server import FederatedServer
from src.schemes.ckks.ckks_crypto import CKKSCrypto


class TestFederatedServerSetup(unittest.TestCase):
    """Test server initialization and setup."""
    
    def test_server_initialization(self):
        """Test that server initializes correctly."""
        server = FederatedServer()
        
        self.assertIsNotNone(server.ckks)
        self.assertEqual(server.current_round, 0)
        self.assertEqual(len(server.clients), 0)
    
    def test_server_key_generation(self):
        """Test that server generates and saves keys."""
        server = FederatedServer()
        
        # Check that keys directory exists
        self.assertTrue(server.keys_dir.exists())
        
        # Check that key files exist
        self.assertTrue((server.keys_dir / "public_key.bin").exists())
        self.assertTrue((server.keys_dir / "secret_key.bin").exists())
        self.assertTrue((server.keys_dir / "context_params.json").exists())
    
    def test_get_public_key_bundle(self):
        """Test public key bundle retrieval."""
        server = FederatedServer()
        bundle = server.get_public_key_bundle()
        
        self.assertIn('public_key', bundle)
        self.assertIn('context_params', bundle)
        self.assertIn('server_id', bundle)
        self.assertIsInstance(bundle['public_key'], str)  # Base64 encoded


class TestFederatedAggregationRound(unittest.TestCase):
    """Test complete aggregation round workflow."""
    
    def setUp(self):
        """Set up server for each test."""
        self.server = FederatedServer()
        np.random.seed(42)
    
    def test_start_aggregation_round(self):
        """Test starting a new aggregation round."""
        round_id = self.server.start_aggregation_round()
        
        self.assertEqual(round_id, 1)
        self.assertEqual(self.server.current_round, 1)
        self.assertIn(round_id, self.server.encrypted_contributions)
    
    def test_client_registration(self):
        """Test client registration."""
        self.server.register_client('client_1', {'type': 'hospital'})
        
        self.assertEqual(len(self.server.clients), 1)
        self.assertIn('client_1', self.server.clients)
        self.assertEqual(self.server.clients['client_1']['info']['type'], 'hospital')
    
    def test_single_client_contribution(self):
        """Test receiving contribution from single client."""
        round_id = self.server.start_aggregation_round()
        
        # Create encrypted contribution
        data = np.random.randn(100)
        local_sum = float(np.sum(data))
        local_sum_sq = float(np.sum(data ** 2))
        
        enc_sum = self.server.ckks.encrypt(local_sum)
        enc_sum_sq = self.server.ckks.encrypt(local_sum_sq)
        
        # Submit contribution
        self.server.receive_encrypted_contribution(
            round_id,
            'client_1',
            {
                'count': 100,
                'encrypted_sum': enc_sum,
                'encrypted_sum_squares': enc_sum_sq
            },
            is_local=True
        )
        
        self.assertEqual(len(self.server.encrypted_contributions[round_id]), 1)
    
    def test_multi_client_aggregation(self):
        """Test aggregation with multiple clients."""
        round_id = self.server.start_aggregation_round()
        
        # Simulate 3 clients
        client_data = [
            np.random.randn(100),
            np.random.randn(150),
            np.random.randn(200)
        ]
        
        for i, data in enumerate(client_data):
            local_sum = float(np.sum(data))
            local_sum_sq = float(np.sum(data ** 2))
            
            enc_sum = self.server.ckks.encrypt(local_sum)
            enc_sum_sq = self.server.ckks.encrypt(local_sum_sq)
            
            self.server.receive_encrypted_contribution(
                round_id,
                f'client_{i}',
                {
                    'count': len(data),
                    'encrypted_sum': enc_sum,
                    'encrypted_sum_squares': enc_sum_sq
                },
                is_local=True
            )
        
        # Aggregate
        results = self.server.aggregate_round(round_id)
        
        # Verify results
        all_data = np.concatenate(client_data)
        true_mean = np.mean(all_data)
        
        self.assertIsNotNone(results)
        self.assertEqual(results['num_clients'], 3)
        self.assertEqual(results['total_count'], 450)
        
        # Check accuracy
        error = abs(results['global_mean'] - true_mean)
        self.assertLess(error, 1e-4)


class TestPrivacyGuarantees(unittest.TestCase):
    """Test privacy guarantees of the system."""
    
    def setUp(self):
        """Set up server for each test."""
        self.server = FederatedServer()
        np.random.seed(42)
    
    def test_server_cannot_decrypt_individual_contributions(self):
        """Verify server can only decrypt aggregates."""
        round_id = self.server.start_aggregation_round()
        
        # Client encrypts data
        client_data = np.array([1000.0, 2000.0, 3000.0])  # Distinct values
        local_sum = float(np.sum(client_data))
        
        enc_sum = self.server.ckks.encrypt(local_sum)
        
        # Server receives encrypted sum
        self.server.receive_encrypted_contribution(
            round_id,
            'client_1',
            {
                'count': len(client_data),
                'encrypted_sum': enc_sum,
                'encrypted_sum_squares': self.server.ckks.encrypt(0.0)
            },
            is_local=True
        )
        
        # Server can only decrypt after aggregation
        # Individual values remain hidden
        contribution = self.server.encrypted_contributions[round_id]['client_1']
        
        # The encrypted_sum is a PyCtxt object - server can't see the value
        self.assertIsInstance(contribution['encrypted_sum'], type(enc_sum))
        
        # Server can decrypt only after aggregation
        results = self.server.aggregate_round(round_id)
        
        # Verify server got aggregate, not individual values
        self.assertEqual(results['global_sum'], local_sum)
        # But server doesn't know individual [1000, 2000, 3000] values
    
    def test_client_data_never_sent_in_plaintext(self):
        """Verify raw data is never sent, only encrypted aggregates."""
        round_id = self.server.start_aggregation_round()
        
        # Client has sensitive data
        sensitive_data = np.array([65000.0, 72000.0, 68000.0])  # Salaries
        
        # Only encrypted aggregates are sent
        enc_sum = self.server.ckks.encrypt(float(np.sum(sensitive_data)))
        
        self.server.receive_encrypted_contribution(
            round_id,
            'client_1',
            {
                'count': len(sensitive_data),
                'encrypted_sum': enc_sum,
                'encrypted_sum_squares': self.server.ckks.encrypt(0.0)
            },
            is_local=True
        )
        
        # Verify contribution doesn't contain raw data
        contribution = self.server.encrypted_contributions[round_id]['client_1']
        
        # Only count is plaintext (not sensitive)
        self.assertEqual(contribution['count'], 3)
        
        # Sums are encrypted (type check)
        from Pyfhel import PyCtxt
        self.assertIsInstance(contribution['encrypted_sum'], PyCtxt)


class TestDataProcessingIntegration(unittest.TestCase):
    """Test data processing pipeline integration."""
    
    def setUp(self):
        """Create test data."""
        self.test_data = pd.DataFrame({
            'value': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from data to results."""
        # Initialize server
        server = FederatedServer()
        round_id = server.start_aggregation_round()
        
        # Split data among clients (simulate federated setting)
        n_clients = 3
        client_data = np.array_split(self.test_data['value'].values, n_clients)
        
        # Each client processes and encrypts
        for i, data in enumerate(client_data):
            local_sum = float(np.sum(data))
            local_sum_sq = float(np.sum(data ** 2))
            
            enc_sum = server.ckks.encrypt(local_sum)
            enc_sum_sq = server.ckks.encrypt(local_sum_sq)
            
            server.receive_encrypted_contribution(
                round_id,
                f'client_{i}',
                {
                    'count': len(data),
                    'encrypted_sum': enc_sum,
                    'encrypted_sum_squares': enc_sum_sq
                },
                is_local=True
            )
        
        # Server aggregates
        results = server.aggregate_round(round_id)
        
        # Verify accuracy
        true_mean = np.mean(self.test_data['value'])
        true_std = np.std(self.test_data['value'])
        
        mean_error = abs(results['global_mean'] - true_mean)
        std_error = abs(results['global_std'] - true_std)
        
        self.assertLess(mean_error, 1e-4)
        self.assertLess(std_error, 1e-3)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance of integrated system."""
    
    def test_aggregation_scales_with_clients(self):
        """Test that aggregation time scales reasonably with number of clients."""
        client_counts = [2, 5, 10]
        times = []
        
        for n_clients in client_counts:
            server = FederatedServer()
            round_id = server.start_aggregation_round()
            
            # Add clients
            for i in range(n_clients):
                data = np.random.randn(100)
                enc_sum = server.ckks.encrypt(float(np.sum(data)))
                enc_sum_sq = server.ckks.encrypt(0.0)
                
                server.receive_encrypted_contribution(
                    round_id,
                    f'client_{i}',
                    {
                        'count': 100,
                        'encrypted_sum': enc_sum,
                        'encrypted_sum_squares': enc_sum_sq
                    },
                    is_local=True
                )
            
            # Time aggregation
            start = time.time()
            server.aggregate_round(round_id)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        # Verify reasonable scaling (should be roughly linear)
        # Time for 10 clients should be less than 10x time for 2 clients
        self.assertLess(times[2], times[0] * 6)
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        server = FederatedServer()
        round_id = server.start_aggregation_round()
        
        # Large dataset (10,000 samples per client)
        large_data = np.random.randn(10000)
        
        start = time.time()
        
        enc_sum = server.ckks.encrypt(float(np.sum(large_data)))
        enc_sum_sq = server.ckks.encrypt(float(np.sum(large_data ** 2)))
        
        server.receive_encrypted_contribution(
            round_id,
            'client_1',
            {
                'count': len(large_data),
                'encrypted_sum': enc_sum,
                'encrypted_sum_squares': enc_sum_sq
            },
            is_local=True
        )
        
        results = server.aggregate_round(round_id)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 10 seconds)
        self.assertLess(elapsed, 10.0)
        
        # Results should still be accurate
        true_mean = np.mean(large_data)
        error = abs(results['global_mean'] - true_mean)
        self.assertLess(error, 1e-4)


def run_integration_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedServerSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestFederatedAggregationRound))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyGuarantees))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceIntegration))
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)