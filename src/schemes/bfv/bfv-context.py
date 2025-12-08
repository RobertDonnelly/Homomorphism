"""
BFV Context and Parameter Management.

This module handles the creation and management of BFV cryptographic contexts,
including parameter selection, validation, and optimization for different use cases.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from Pyfhel import Pyfhel
import numpy as np


@dataclass
class BFVParameterSet:
    """
    Predefined BFV parameter sets for different security/performance tradeoffs.
    
    Attributes:
        name: Identifier for the parameter set
        poly_modulus_degree: Polynomial modulus degree (n)
        plain_modulus: Plaintext modulus (t)
        security_level: Security level in bits (128, 192, 256)
        coeff_modulus_bits: Bit-lengths for coefficient modulus chain
        description: Human-readable description
    """
    name: str
    poly_modulus_degree: int
    plain_modulus: int
    security_level: int
    coeff_modulus_bits: Optional[List[int]] = None
    description: str = ""


# Predefined parameter sets based on common use cases
PARAMETER_SETS = {
    # Small - fast operations, limited depth
    "small": BFVParameterSet(
        name="small",
        poly_modulus_degree=4096,
        plain_modulus=65537,  # 2^16 + 1
        security_level=128,
        coeff_modulus_bits=None,  # Let Pyfhel choose
        description="Fast operations, multiplicative depth ~2-3"
    ),
    
    # Medium - balanced performance and depth
    "medium": BFVParameterSet(
        name="medium",
        poly_modulus_degree=8192,
        plain_modulus=65537,
        security_level=128,
        coeff_modulus_bits=None,
        description="Balanced performance, multiplicative depth ~4-5"
    ),
    
    # Large - high depth computations
    "large": BFVParameterSet(
        name="large",
        poly_modulus_degree=16384,
        plain_modulus=65537,
        security_level=128,
        coeff_modulus_bits=None,
        description="Deep computations, multiplicative depth ~6-8"
    ),
    
    # High security - 192-bit security
    "high_security": BFVParameterSet(
        name="high_security",
        poly_modulus_degree=8192,
        plain_modulus=65537,
        security_level=192,
        coeff_modulus_bits=None,
        description="192-bit security, moderate depth"
    ),
    
    # Very high security - 256-bit security
    "max_security": BFVParameterSet(
        name="max_security",
        poly_modulus_degree=16384,
        plain_modulus=65537,
        security_level=256,
        coeff_modulus_bits=None,
        description="256-bit security, expensive operations"
    ),
    
    # Small plaintext modulus - for binary/small integer operations
    "binary": BFVParameterSet(
        name="binary",
        poly_modulus_degree=4096,
        plain_modulus=2,  # Binary operations
        security_level=128,
        coeff_modulus_bits=None,
        description="Binary operations only"
    ),
    
    # Large plaintext modulus - for large integers
    "large_plaintext": BFVParameterSet(
        name="large_plaintext",
        poly_modulus_degree=8192,
        plain_modulus=40961,  # Larger plaintext space
        security_level=128,
        coeff_modulus_bits=None,
        description="Large plaintext space, moderate depth"
    ),
}


class BFVContext:
    """
    Manages BFV cryptographic context and parameters.
    
    This class handles the initialization and configuration of the Pyfhel
    instance for BFV encryption, including parameter validation and optimization.
    """
    
    def __init__(self, param_set: Optional[BFVParameterSet] = None):
        """
        Initialize BFV context.
        
        Args:
            param_set: Parameter set to use (defaults to "medium")
        """
        self.param_set = param_set or PARAMETER_SETS["medium"]
        self.pyfhel = Pyfhel()
        self._context_created = False
        self._keys_generated = False
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'BFVContext':
        """
        Create context from a predefined parameter set.
        
        Args:
            preset_name: Name of preset ("small", "medium", "large", etc.)
            
        Returns:
            Initialized BFVContext
            
        Raises:
            ValueError: If preset name is invalid
        """
        if preset_name not in PARAMETER_SETS:
            available = ", ".join(PARAMETER_SETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        return cls(PARAMETER_SETS[preset_name])
    
    @classmethod
    def from_custom(
        cls,
        name: str,
        poly_modulus_degree: int,
        plain_modulus: int,
        security_level: int = 128,
        coeff_modulus_bits: Optional[List[int]] = None,
        description: str = "Custom parameters"
    ) -> 'BFVContext':
        """
        Create context with custom parameters.
        
        Args:
            name: Name for this parameter set
            poly_modulus_degree: Polynomial modulus degree (must be power of 2)
            plain_modulus: Plaintext modulus (should be prime)
            security_level: Security level in bits (128, 192, or 256)
            coeff_modulus_bits: Optional coefficient modulus bit-lengths
            description: Description of parameter set
            
        Returns:
            BFVContext with custom parameters
        """
        param_set = BFVParameterSet(
            name=name,
            poly_modulus_degree=poly_modulus_degree,
            plain_modulus=plain_modulus,
            security_level=security_level,
            coeff_modulus_bits=coeff_modulus_bits,
            description=description
        )
        
        return cls(param_set)
    
    def create_context(self) -> None:
        """
        Create the cryptographic context with current parameters.
        
        This initializes the Pyfhel instance with BFV scheme parameters.
        
        Raises:
            RuntimeError: If context creation fails
        """
        try:
            # Build context generation parameters
            context_params = {
                'scheme': 'BFV',
                'n': self.param_set.poly_modulus_degree,
                't': self.param_set.plain_modulus,
                'sec': self.param_set.security_level,
            }
            
            # Add coefficient modulus if specified
            if self.param_set.coeff_modulus_bits:
                context_params['t_bits'] = self.param_set.coeff_modulus_bits
            
            # Generate context
            self.pyfhel.contextGen(**context_params)
            self._context_created = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to create BFV context: {e}")
    
    def generate_keys(self) -> None:
        """
        Generate all necessary keys for BFV operations.
        
        Generates:
        - Public/Secret key pair
        - Relinearization keys (for multiplication)
        - Galois keys (for rotations, optional)
        
        Raises:
            RuntimeError: If context not created or key generation fails
        """
        if not self._context_created:
            raise RuntimeError("Context must be created before generating keys")
        
        try:
            # Generate public and secret keys
            self.pyfhel.keyGen()
            
            # Generate relinearization keys for multiplication
            self.pyfhel.relinKeyGen()
            
            # Generate Galois keys for rotations (optional but useful)
            # This enables SIMD-style operations
            self.pyfhel.rotateKeyGen()
            
            self._keys_generated = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate keys: {e}")
    
    def get_context_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current context.
        
        Returns:
            Dictionary containing context parameters and status
        """
        info = {
            'param_set_name': self.param_set.name,
            'description': self.param_set.description,
            'poly_modulus_degree': self.param_set.poly_modulus_degree,
            'plain_modulus': self.param_set.plain_modulus,
            'security_level': self.param_set.security_level,
            'context_created': self._context_created,
            'keys_generated': self._keys_generated,
        }
        
        if self._context_created:
            # Add runtime context information from Pyfhel
            info.update({
                'slots': self.pyfhel.get_nSlots() if hasattr(self.pyfhel, 'get_nSlots') else None,
            })
        
        return info
    
    def validate_parameters(self) -> Dict[str, Any]:
        """
        Validate that current parameters are secure and practical.
        
        Returns:
            Dictionary with validation results and warnings
        """
        warnings = []
        errors = []
        
        # Check polynomial modulus degree
        n = self.param_set.poly_modulus_degree
        if n < 1024:
            errors.append(f"poly_modulus_degree={n} is too small for security")
        elif n > 32768:
            warnings.append(f"poly_modulus_degree={n} is very large, operations will be slow")
        
        if not (n & (n - 1) == 0):  # Check if power of 2
            errors.append(f"poly_modulus_degree={n} must be a power of 2")
        
        # Check plaintext modulus
        t = self.param_set.plain_modulus
        if not self._is_prime(t):
            warnings.append(f"plain_modulus={t} should be prime for optimal security")
        
        # Check security level
        sec = self.param_set.security_level
        if sec not in [128, 192, 256]:
            errors.append(f"security_level={sec} must be 128, 192, or 256")
        
        # Parameter relationship checks
        if n == 1024 and sec > 128:
            warnings.append(f"n=1024 with sec={sec} may not achieve target security")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """
        Simple primality test for validation.
        
        Args:
            n: Number to test
            
        Returns:
            True if n is likely prime
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Trial division up to sqrt(n)
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True
    
    def estimate_noise_budget(self, num_additions: int, num_multiplications: int) -> Dict[str, Any]:
        """
        Estimate remaining noise budget after operations.
        
        This is a rough estimate based on typical BFV noise growth.
        
        Args:
            num_additions: Number of additions
            num_multiplications: Number of multiplications
            
        Returns:
            Dictionary with noise budget estimates
        """
        # These are rough estimates and vary based on actual parameters
        # Real noise tracking should be done during operations
        
        # Approximate noise growth factors
        # These are simplified; actual growth depends on many factors
        initial_noise = 10  # Fresh ciphertext noise (arbitrary units)
        add_growth = 1.01    # Additive growth factor
        mult_growth = 2.5    # Multiplicative growth factor
        
        noise = initial_noise
        noise *= (add_growth ** num_additions)
        noise *= (mult_growth ** num_multiplications)
        
        # Estimate maximum tolerable noise (depends on q/t ratio)
        # For n=4096, t=65537, typical max noise is around 2^30-40
        q_bits = self._estimate_coeff_modulus_bits()
        t_bits = self.param_set.plain_modulus.bit_length()
        max_noise_bits = q_bits - t_bits - 10  # Safety margin
        
        estimated_noise_bits = np.log2(noise) + 20  # Base offset
        
        return {
            'estimated_noise_bits': estimated_noise_bits,
            'max_noise_bits': max_noise_bits,
            'safe': estimated_noise_bits < max_noise_bits,
            'margin_bits': max_noise_bits - estimated_noise_bits,
            'note': 'This is a rough estimate. Track actual noise during operations.'
        }
    
    def _estimate_coeff_modulus_bits(self) -> int:
        """
        Estimate coefficient modulus size based on parameters.
        
        Returns:
            Estimated coefficient modulus size in bits
        """
        # Approximate bit sizes based on security level and n
        # These are based on HE standard recommendations
        n = self.param_set.poly_modulus_degree
        sec = self.param_set.security_level
        
        # Rough approximations
        if sec == 128:
            if n == 1024:
                return 27
            elif n == 2048:
                return 54
            elif n == 4096:
                return 109
            elif n == 8192:
                return 218
            elif n == 16384:
                return 438
            elif n == 32768:
                return 881
        elif sec == 192:
            if n == 1024:
                return 19
            elif n == 2048:
                return 37
            elif n == 4096:
                return 75
            elif n == 8192:
                return 152
            elif n == 16384:
                return 305
        elif sec == 256:
            if n == 1024:
                return 14
            elif n == 2048:
                return 29
            elif n == 4096:
                return 58
            elif n == 8192:
                return 118
            elif n == 16384:
                return 237
        
        # Default fallback
        return 100
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"BFVContext(param_set='{self.param_set.name}', "
                f"n={self.param_set.poly_modulus_degree}, "
                f"t={self.param_set.plain_modulus}, "
                f"sec={self.param_set.security_level})")


def get_available_parameter_sets() -> List[str]:
    """
    Get list of available predefined parameter sets.
    
    Returns:
        List of parameter set names
    """
    return list(PARAMETER_SETS.keys())


def get_parameter_set_info(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a parameter set.
    
    Args:
        name: Parameter set name
        
    Returns:
        Dictionary with parameter set details
        
    Raises:
        ValueError: If parameter set name is invalid
    """
    if name not in PARAMETER_SETS:
        available = ", ".join(PARAMETER_SETS.keys())
        raise ValueError(f"Unknown parameter set '{name}'. Available: {available}")
    
    param_set = PARAMETER_SETS[name]
    return {
        'name': param_set.name,
        'poly_modulus_degree': param_set.poly_modulus_degree,
        'plain_modulus': param_set.plain_modulus,
        'security_level': param_set.security_level,
        'description': param_set.description
    }
