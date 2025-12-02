"""
Additional utilities for metric collection and analysis.

This module provides specialized metric collectors for:
- Ciphertext properties
- Accuracy tracking (especially for CKKS)
- Noise budget monitoring
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class CiphertextMetrics:
    """
    Metrics specific to ciphertext properties.
    
    Tracks size, noise, and other ciphertext characteristics.
    """
    operation: str
    size_bytes: int
    noise_budget_bits: Optional[float] = None
    scale: Optional[float] = None
    level: Optional[int] = None
    num_slots: Optional[int] = None
    expansion_factor: Optional[float] = None  # ciphertext_size / plaintext_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'size_bytes': self.size_bytes,
            'size_kb': self.size_bytes / 1024,
            'noise_budget_bits': self.noise_budget_bits,
            'scale': self.scale,
            'level': self.level,
            'num_slots': self.num_slots,
            'expansion_factor': self.expansion_factor
        }


@dataclass
class AccuracyMetrics:
    """
    Metrics for tracking numerical accuracy (important for CKKS).
    
    Compares encrypted computation results with plaintext ground truth.
    """
    operation: str
    expected_values: np.ndarray
    actual_values: np.ndarray
    absolute_error: Optional[np.ndarray] = field(init=False)
    relative_error: Optional[np.ndarray] = field(init=False)
    max_absolute_error: Optional[float] = field(init=False)
    mean_absolute_error: Optional[float] = field(init=False)
    max_relative_error: Optional[float] = field(init=False)
    mean_relative_error: Optional[float] = field(init=False)
    rmse: Optional[float] = field(init=False)
    
    def __post_init__(self):
        """Calculate error metrics."""
        self.expected_values = np.array(self.expected_values)
        self.actual_values = np.array(self.actual_values)
        
        # Absolute errors
        self.absolute_error = np.abs(self.expected_values - self.actual_values)
        self.max_absolute_error = float(np.max(self.absolute_error))
        self.mean_absolute_error = float(np.mean(self.absolute_error))
        
        # Relative errors (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.relative_error = np.abs(
                (self.expected_values - self.actual_values) / self.expected_values
            )
            self.relative_error[~np.isfinite(self.relative_error)] = 0
        
        self.max_relative_error = float(np.max(self.relative_error))
        self.mean_relative_error = float(np.mean(self.relative_error))
        
        # RMSE
        self.rmse = float(np.sqrt(np.mean((self.expected_values - self.actual_values) ** 2)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'max_absolute_error': self.max_absolute_error,
            'mean_absolute_error': self.mean_absolute_error,
            'max_relative_error': self.max_relative_error,
            'mean_relative_error': self.mean_relative_error,
            'rmse': self.rmse,
            'num_samples': len(self.expected_values)
        }
    
    def is_acceptable(self, tolerance: float = 1e-3) -> bool:
        """
        Check if errors are within acceptable tolerance.
        
        Args:
            tolerance: Maximum acceptable relative error
            
        Returns:
            True if all errors are within tolerance
        """
        return self.max_relative_error < tolerance


@dataclass
class NoiseBudgetTracker:
    """
    Track noise budget consumption throughout a computation.
    
    Useful for understanding when operations might fail or need bootstrapping.
    """
    scheme: str
    param_set: str
    initial_budget: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_operation(self, operation: str, noise_budget: float, operation_depth: int = 1):
        """
        Record noise budget after an operation.
        
        Args:
            operation: Name of the operation
            noise_budget: Remaining noise budget
            operation_depth: Current circuit depth
        """
        consumption = self.initial_budget - noise_budget if self.history else 0
        if self.history:
            consumption = self.history[-1]['noise_budget'] - noise_budget
        
        self.history.append({
            'operation': operation,
            'noise_budget': noise_budget,
            'consumption': consumption,
            'depth': operation_depth,
            'percentage_remaining': (noise_budget / self.initial_budget) * 100
        })
    
    def get_consumption_summary(self) -> Dict[str, Any]:
        """
        Get summary of noise budget consumption.
        
        Returns:
            Dictionary with consumption statistics
        """
        if not self.history:
            return {}
        
        total_consumption = self.initial_budget - self.history[-1]['noise_budget']
        
        return {
            'scheme': self.scheme,
            'param_set': self.param_set,
            'initial_budget': self.initial_budget,
            'final_budget': self.history[-1]['noise_budget'],
            'total_consumption': total_consumption,
            'percentage_consumed': (total_consumption / self.initial_budget) * 100,
            'num_operations': len(self.history),
            'final_depth': self.history[-1]['depth']
        }
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export history as list of dictionaries."""
        return [{
            'scheme': self.scheme,
            'param_set': self.param_set,
            **entry
        } for entry in self.history]


class OperationProfiler:
    """
    Profile specific operations to understand bottlenecks.
    
    Tracks individual operation types separately.
    """
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
    
    def record(self, operation: str, time_ms: float):
        """
        Record time for an operation.
        
        Args:
            operation: Operation name
            time_ms: Execution time in milliseconds
        """
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(time_ms)
        self.operation_counts[operation] += 1
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all operations.
        
        Returns:
            Dictionary mapping operation names to statistics
        """
        summary = {}
        
        for op, times in self.operation_times.items():
            times_array = np.array(times)
            summary[op] = {
                'count': self.operation_counts[op],
                'total_time_ms': float(np.sum(times_array)),
                'mean_time_ms': float(np.mean(times_array)),
                'std_time_ms': float(np.std(times_array)),
                'min_time_ms': float(np.min(times_array)),
                'max_time_ms': float(np.max(times_array)),
                'median_time_ms': float(np.median(times_array))
            }
        
        return summary
    
    def get_percentage_breakdown(self) -> Dict[str, float]:
        """
        Get percentage of total time spent on each operation.
        
        Returns:
            Dictionary mapping operation names to percentage of total time
        """
        total_time = sum(sum(times) for times in self.operation_times.values())
        
        if total_time == 0:
            return {}
        
        return {
            op: (sum(times) / total_time) * 100
            for op, times in self.operation_times.items()
        }
    
    def clear(self):
        """Clear all recorded data."""
        self.operation_times.clear()
        self.operation_counts.clear()


@dataclass
class ThroughputMetrics:
    """
    Calculate throughput metrics for operations.
    """
    operation: str
    num_operations: int
    total_time_ms: float
    data_size: int
    
    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second."""
        return (self.num_operations / self.total_time_ms) * 1000
    
    @property
    def time_per_operation_ms(self) -> float:
        """Calculate time per operation."""
        return self.total_time_ms / self.num_operations
    
    @property
    def throughput_mb_per_second(self) -> float:
        """Calculate data throughput (if applicable)."""
        data_size_mb = (self.data_size * self.num_operations) / (1024 * 1024)
        time_seconds = self.total_time_ms / 1000
        return data_size_mb / time_seconds if time_seconds > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'num_operations': self.num_operations,
            'total_time_ms': self.total_time_ms,
            'operations_per_second': self.operations_per_second,
            'time_per_operation_ms': self.time_per_operation_ms,
            'throughput_mb_per_second': self.throughput_mb_per_second
        }


class MetricsCollector:
    """
    Central collector for all types of metrics.
    
    Aggregates ciphertext metrics, accuracy metrics, noise budgets, etc.
    """
    
    def __init__(self):
        self.ciphertext_metrics: List[CiphertextMetrics] = []
        self.accuracy_metrics: List[AccuracyMetrics] = []
        self.noise_trackers: List[NoiseBudgetTracker] = []
        self.throughput_metrics: List[ThroughputMetrics] = []
    
    def add_ciphertext_metrics(self, metrics: CiphertextMetrics):
        """Add ciphertext metrics."""
        self.ciphertext_metrics.append(metrics)
    
    def add_accuracy_metrics(self, metrics: AccuracyMetrics):
        """Add accuracy metrics."""
        self.accuracy_metrics.append(metrics)
    
    def add_noise_tracker(self, tracker: NoiseBudgetTracker):
        """Add noise budget tracker."""
        self.noise_trackers.append(tracker)
    
    def add_throughput_metrics(self, metrics: ThroughputMetrics):
        """Add throughput metrics."""
        self.throughput_metrics.append(metrics)
    
    def export_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Export all metrics.
        
        Returns:
            Dictionary with all metric types
        """
        return {
            'ciphertext_metrics': [m.to_dict() for m in self.ciphertext_metrics],
            'accuracy_metrics': [m.to_dict() for m in self.accuracy_metrics],
            'noise_budgets': [t.get_consumption_summary() for t in self.noise_trackers],
            'throughput_metrics': [m.to_dict() for m in self.throughput_metrics]
        }
    
    def clear(self):
        """Clear all collected metrics."""
        self.ciphertext_metrics.clear()
        self.accuracy_metrics.clear()
        self.noise_trackers.clear()
        self.throughput_metrics.clear()