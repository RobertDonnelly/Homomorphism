"""
Core benchmarking infrastructure for HE schemes.

This module provides tools for systematic performance measurement including:
- Timing measurements
- Memory profiling
- Metric collection and aggregation
- Result export to CSV
"""

import time
import tracemalloc
import psutil
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import csv
import json
from functools import wraps


@dataclass
class BenchmarkResult:
    """
    Container for a single benchmark measurement.
    
    Attributes:
        operation: Name of the operation benchmarked
        scheme: HE scheme used (BFV, CKKS, etc.)
        param_set: Parameter configuration name
        data_size: Size of input data
        time_ms: Execution time in milliseconds
        memory_mb: Peak memory usage in MB
        memory_delta_mb: Change in memory usage
        metadata: Additional operation-specific data
        timestamp: When the benchmark was run
    """
    operation: str
    scheme: str
    param_set: str
    data_size: int
    time_ms: float
    memory_mb: float
    memory_delta_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        base = asdict(self)
        # Flatten metadata for CSV
        if self.metadata:
            base['metadata_json'] = json.dumps(self.metadata)
        del base['metadata']
        return base


@dataclass
class AggregatedResults:
    """
    Aggregated statistics from multiple benchmark runs.
    
    Attributes:
        operation: Name of the operation
        num_runs: Number of runs
        mean_time_ms: Mean execution time
        std_time_ms: Standard deviation of time
        min_time_ms: Minimum time
        max_time_ms: Maximum time
        mean_memory_mb: Mean memory usage
        std_memory_mb: Standard deviation of memory
    """
    operation: str
    num_runs: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    mean_memory_mb: float
    std_memory_mb: float


class TimeTracker:
    """High-resolution timing for operations."""
    
    def __init__(self):
        self._start_time = None
        self._end_time = None
    
    def start(self):
        """Start timing."""
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop timing and return elapsed time.
        
        Returns:
            Elapsed time in milliseconds
        """
        self._end_time = time.perf_counter()
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        return (self._end_time - self._start_time) * 1000  # Convert to ms
    
    @contextmanager
    def measure(self):
        """
        Context manager for timing a code block.
        
        Usage:
            with timer.measure() as t:
                # code to time
                pass
            print(f"Time: {t['time_ms']} ms")
        """
        result = {'time_ms': 0}
        self.start()
        try:
            yield result
        finally:
            result['time_ms'] = self.stop()


class MemoryTracker:
    """Memory profiling for operations."""
    
    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._start_memory = None
        self._peak_memory = None
        self._tracemalloc_enabled = False
    
    def start(self):
        """Start memory tracking."""
        # Get baseline memory
        self._start_memory = self._process.memory_info().rss / (1024 * 1024)  # MB
        
        # Start tracemalloc for detailed tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_enabled = True
    
    def stop(self) -> Dict[str, float]:
        """
        Stop memory tracking and return results.
        
        Returns:
            Dictionary with current_mb, peak_mb, delta_mb
        """
        current_memory = self._process.memory_info().rss / (1024 * 1024)
        
        if self._tracemalloc_enabled:
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            tracemalloc.stop()
            self._tracemalloc_enabled = False
        else:
            peak_mb = current_memory
        
        delta = current_memory - self._start_memory if self._start_memory else 0
        
        return {
            'current_mb': current_memory,
            'peak_mb': peak_mb,
            'delta_mb': delta
        }
    
    @contextmanager
    def measure(self):
        """
        Context manager for memory tracking.
        
        Usage:
            with mem_tracker.measure() as m:
                # code to profile
                pass
            print(f"Memory: {m['peak_mb']} MB")
        """
        result = {'current_mb': 0, 'peak_mb': 0, 'delta_mb': 0}
        self.start()
        try:
            yield result
        finally:
            result.update(self.stop())


class BenchmarkSuite:
    """
    Main benchmarking orchestrator.
    
    Manages multiple benchmark runs, collects results, and exports data.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "results/benchmarks"):
        """
        Initialize the benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.time_tracker = TimeTracker()
        self.memory_tracker = MemoryTracker()
    
    @contextmanager
    def measure_operation(
        self,
        operation: str,
        scheme: str,
        param_set: str,
        data_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager to benchmark an operation.
        
        Args:
            operation: Name of the operation
            scheme: HE scheme name
            param_set: Parameter configuration name
            data_size: Size of input data
            metadata: Additional metadata to store
            
        Usage:
            with suite.measure_operation("encrypt", "BFV", "small", 100):
                # code to benchmark
                result = encrypt(data)
        """
        timing = {}
        memory = {}
        
        # Start tracking
        with self.time_tracker.measure() as timing:
            with self.memory_tracker.measure() as memory:
                yield
        
        # Record result
        result = BenchmarkResult(
            operation=operation,
            scheme=scheme,
            param_set=param_set,
            data_size=data_size,
            time_ms=timing['time_ms'],
            memory_mb=memory['peak_mb'],
            memory_delta_mb=memory['delta_mb'],
            metadata=metadata or {}
        )
        self.results.append(result)
    
    def benchmark_function(
        self,
        func: Callable,
        operation: str,
        scheme: str,
        param_set: str,
        data_size: int,
        num_runs: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark a function multiple times.
        
        Args:
            func: Function to benchmark (should take no arguments)
            operation: Operation name
            scheme: Scheme name
            param_set: Parameter set name
            data_size: Data size
            num_runs: Number of times to run
            metadata: Additional metadata
            
        Returns:
            List of benchmark results
        """
        run_results = []
        
        for i in range(num_runs):
            run_metadata = {**(metadata or {}), 'run_number': i + 1}
            
            with self.measure_operation(
                operation, scheme, param_set, data_size, run_metadata
            ):
                func()
            
            run_results.append(self.results[-1])
        
        return run_results
    
    def export_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export all results to a CSV file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the created CSV file
        """
        if not self.results:
            raise ValueError("No results to export")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Get all field names from results
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        return output_path
    
    def export_by_scheme(self) -> Dict[str, Path]:
        """
        Export results separated by scheme.
        
        Returns:
            Dictionary mapping scheme names to CSV file paths
        """
        schemes = set(r.scheme for r in self.results)
        paths = {}
        
        for scheme in schemes:
            scheme_results = [r for r in self.results if r.scheme == scheme]
            scheme_dir = self.output_dir / scheme.lower()
            scheme_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{scheme.lower()}_results_{timestamp}.csv"
            output_path = scheme_dir / filename
            
            fieldnames = list(scheme_results[0].to_dict().keys())
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in scheme_results:
                    writer.writerow(result.to_dict())
            
            paths[scheme] = output_path
        
        return paths
    
    def get_summary_statistics(self, operation: Optional[str] = None) -> List[AggregatedResults]:
        """
        Calculate summary statistics across multiple runs.
        
        Args:
            operation: Filter by specific operation (None for all)
            
        Returns:
            List of aggregated statistics
        """
        import numpy as np
        
        results_to_analyze = self.results
        if operation:
            results_to_analyze = [r for r in self.results if r.operation == operation]
        
        # Group by operation
        operations = {}
        for result in results_to_analyze:
            key = f"{result.scheme}_{result.operation}_{result.param_set}"
            if key not in operations:
                operations[key] = []
            operations[key].append(result)
        
        # Calculate statistics
        summaries = []
        for key, results in operations.items():
            times = [r.time_ms for r in results]
            memories = [r.memory_mb for r in results]
            
            summaries.append(AggregatedResults(
                operation=key,
                num_runs=len(results),
                mean_time_ms=float(np.mean(times)),
                std_time_ms=float(np.std(times)),
                min_time_ms=float(np.min(times)),
                max_time_ms=float(np.max(times)),
                mean_memory_mb=float(np.mean(memories)),
                std_memory_mb=float(np.std(memories))
            ))
        
        return summaries
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()
    
    def __len__(self) -> int:
        """Return number of stored results."""
        return len(self.results)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BenchmarkSuite(results={len(self.results)}, output_dir='{self.output_dir}')"


def benchmark_decorator(
    suite: BenchmarkSuite,
    operation: str,
    scheme: str,
    param_set: str,
    data_size: int
):
    """
    Decorator to automatically benchmark a function.
    
    Args:
        suite: BenchmarkSuite instance
        operation: Operation name
        scheme: Scheme name
        param_set: Parameter set
        data_size: Data size
        
    Usage:
        @benchmark_decorator(suite, "encrypt", "BFV", "small", 100)
        def my_encrypt_function(data):
            return encrypt(data)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with suite.measure_operation(operation, scheme, param_set, data_size):
                return func(*args, **kwargs)
        return wrapper
    return decorator