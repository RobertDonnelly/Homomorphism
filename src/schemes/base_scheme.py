"""
Base abstract class for Homomorphic Encryption schemes.

This module provides the foundational interface that all SHE schemes (BFV, CKKS, etc.)
must implement. It ensures consistency across different schemes and enables
systematic benchmarking and comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class SchemeType(Enum):
    """Enumeration of supported HE schemes."""
    BFV = "BFV"
    CKKS = "CKKS"
    BGV = "BGV"  # Possbibly could add later


@dataclass
class SchemeParameters:
    """
    Container for encryption scheme parameters.
    
    Attributes:
        scheme_type: Type of encryption scheme
        poly_modulus_degree: Polynomial modulus degree (n)
        coeff_modulus: Coefficient modulus bit-lengths
        plain_modulus: Plaintext modulus (for BFV/BGV)
        scale: Scale factor (for CKKS)
        security_level: Security level in bits (128, 192, 256)
        extra_params: Additional scheme-specific parameters
    """
    scheme_type: SchemeType
    poly_modulus_degree: int
    coeff_modulus: Optional[List[int]] = None
    plain_modulus: Optional[int] = None
    scale: Optional[float] = None
    security_level: int = 128
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.poly_modulus_degree not in [1024, 2048, 4096, 8192, 16384, 32768]:
            raise ValueError(f"poly_modulus_degree must be power of 2 between 1024 and 32768")
        
        if self.security_level not in [128, 192, 256]:
            raise ValueError(f"security_level must be 128, 192, or 256")


@dataclass
class EncryptionKeys:
    """
    Container for encryption keys.
    
    Attributes:
        public_key: Public key for encryption
        secret_key: Secret key for decryption
        relin_keys: Relinearization keys for multiplication
        galois_keys: Galois keys for rotations (optional)
    """
    public_key: Any
    secret_key: Any
    relin_keys: Optional[Any] = None
    galois_keys: Optional[Any] = None


@dataclass
class CiphertextMetadata:
    """
    Metadata about a ciphertext for tracking and analysis.
    
    Attributes:
        size_bytes: Size of ciphertext in bytes
        noise_budget: Remaining noise budget (scheme-dependent)
        scale: Current scale (for CKKS)
        level: Current level in modulus chain
        num_slots: Number of slots used
    """
    size_bytes: int
    noise_budget: Optional[float] = None
    scale: Optional[float] = None
    level: Optional[int] = None
    num_slots: Optional[int] = None


class BaseHEScheme(ABC):
    """
    Abstract base class for Homomorphic Encryption schemes.
    
    All HE scheme implementations (BFV, CKKS, etc.) must inherit from this class
    and implement all abstract methods. This ensures a consistent interface for
    benchmarking and comparison.
    """
    
    def __init__(self, params: SchemeParameters):
        """
        Initialize the HE scheme with given parameters.
        
        Args:
            params: Scheme parameters
        """
        self.params = params
        self.context = None
        self.keys: Optional[EncryptionKeys] = None
        self._initialized = False
    
    @abstractmethod
    def setup_context(self) -> None:
        """
        Set up the cryptographic context with the given parameters.
        
        This should initialize all necessary structures for the HE scheme
        including parameter validation and context creation.
        """
        pass
    
    @abstractmethod
    def generate_keys(self) -> EncryptionKeys:
        """
        Generate all necessary keys for the scheme.
        
        Returns:
            EncryptionKeys object containing all generated keys
        """
        pass
    
    @abstractmethod
    def encrypt(self, data: Union[int, float, List[int], List[float], np.ndarray]) -> Any:
        """
        Encrypt plaintext data.
        
        Args:
            data: Data to encrypt (single value or array)
            
        Returns:
            Encrypted ciphertext
        """
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: Any) -> Union[int, float, List[int], List[float], np.ndarray]:
        """
        Decrypt ciphertext.
        
        Args:
            ciphertext: Encrypted data
            
        Returns:
            Decrypted plaintext data
        """
        pass
    
    @abstractmethod
    def add(self, ct1: Any, ct2: Any) -> Any:
        """
        Homomorphic addition of two ciphertexts.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Result ciphertext (ct1 + ct2)
        """
        pass
    
    @abstractmethod
    def add_plain(self, ct: Any, pt: Union[int, float]) -> Any:
        """
        Add a plaintext value to a ciphertext.
        
        Args:
            ct: Ciphertext
            pt: Plaintext value
            
        Returns:
            Result ciphertext (ct + pt)
        """
        pass
    
    @abstractmethod
    def multiply(self, ct1: Any, ct2: Any) -> Any:
        """
        Homomorphic multiplication of two ciphertexts.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Result ciphertext (ct1 * ct2)
        """
        pass
    
    @abstractmethod
    def multiply_plain(self, ct: Any, pt: Union[int, float]) -> Any:
        """
        Multiply a ciphertext by a plaintext value.
        
        Args:
            ct: Ciphertext
            pt: Plaintext scalar
            
        Returns:
            Result ciphertext (ct * pt)
        """
        pass
    
    @abstractmethod
    def negate(self, ct: Any) -> Any:
        """
        Negate a ciphertext.
        
        Args:
            ct: Ciphertext
            
        Returns:
            Result ciphertext (-ct)
        """
        pass
    
    @abstractmethod
    def serialize_ciphertext(self, ct: Any) -> bytes:
        """
        Serialize a ciphertext to bytes for transmission/storage.
        
        Args:
            ct: Ciphertext to serialize
            
        Returns:
            Serialized bytes
        """
        pass
    
    @abstractmethod
    def deserialize_ciphertext(self, data: bytes) -> Any:
        """
        Deserialize bytes back to a ciphertext.
        
        Args:
            data: Serialized ciphertext bytes
            
        Returns:
            Deserialized ciphertext
        """
        pass
    
    @abstractmethod
    def serialize_public_key(self) -> bytes:
        """
        Serialize public key to bytes.
        
        Returns:
            Serialized public key
        """
        pass
    
    @abstractmethod
    def get_ciphertext_metadata(self, ct: Any) -> CiphertextMetadata:
        """
        Get metadata about a ciphertext for analysis.
        
        Args:
            ct: Ciphertext to analyze
            
        Returns:
            Metadata about the ciphertext
        """
        pass
    
    # Optional operations (may not be supported by all schemes)
    
    def rotate(self, ct: Any, steps: int) -> Any:
        """
        Rotate ciphertext slots (if supported).
        
        Args:
            ct: Ciphertext
            steps: Number of positions to rotate
            
        Returns:
            Rotated ciphertext
            
        Raises:
            NotImplementedError: If scheme doesn't support rotation
        """
        raise NotImplementedError(f"Rotation not implemented for {self.params.scheme_type}")
    
    def relinearize(self, ct: Any) -> Any:
        """
        Relinearize a ciphertext after multiplication.
        
        Args:
            ct: Ciphertext to relinearize
            
        Returns:
            Relinearized ciphertext
            
        Raises:
            NotImplementedError: If scheme doesn't support relinearization
        """
        raise NotImplementedError(f"Relinearization not implemented for {self.params.scheme_type}")
    
    def rescale(self, ct: Any) -> Any:
        """
        Rescale a ciphertext (CKKS-specific operation).
        
        Args:
            ct: Ciphertext to rescale
            
        Returns:
            Rescaled ciphertext
            
        Raises:
            NotImplementedError: If scheme doesn't support rescaling
        """
        raise NotImplementedError(f"Rescaling not implemented for {self.params.scheme_type}")
    
    # Utility methods
    
    def get_scheme_info(self) -> Dict[str, Any]:
        """
        Get information about the current scheme configuration.
        
        Returns:
            Dictionary containing scheme information
        """
        return {
            "scheme_type": self.params.scheme_type.value,
            "poly_modulus_degree": self.params.poly_modulus_degree,
            "security_level": self.params.security_level,
            "plain_modulus": self.params.plain_modulus,
            "scale": self.params.scale,
            "initialized": self._initialized,
            "has_keys": self.keys is not None
        }
    
    def __repr__(self) -> str:
        """String representation of the scheme."""
        return f"{self.__class__.__name__}(scheme={self.params.scheme_type.value}, n={self.params.poly_modulus_degree})"