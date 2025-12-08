"""
BFV Encryption and Decryption Operations.

This module handles encryption, decryption, and encoding/decoding of data
for the BFV scheme.
"""

from typing import Union, List, Optional
import numpy as np
from Pyfhel import Pyfhel, PyCtxt


class BFVEncryptor:
    """
    Handles encryption and decryption operations for BFV scheme.
    
    This class provides methods for encrypting plaintext data into ciphertexts
    and decrypting ciphertexts back to plaintext.
    """
    
    def __init__(self, pyfhel: Pyfhel):
        """
        Initialize the encryptor.
        
        Args:
            pyfhel: Configured Pyfhel instance with generated keys
        """
        self.pyfhel = pyfhel
        
        if not hasattr(pyfhel, '_public_key') or pyfhel._public_key is None:
            raise RuntimeError("Pyfhel instance must have keys generated")
    
    def encrypt_int(self, value: int) -> PyCtxt:
        """
        Encrypt a single integer value.
        
        Args:
            value: Integer to encrypt (must be in range [0, t))
            
        Returns:
            Encrypted ciphertext
            
        Raises:
            ValueError: If value is out of valid range
        """
        t = self.pyfhel.t
        
        # Validate input
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"Value must be an integer, got {type(value)}")
        
        # Handle negative numbers using modular arithmetic
        value = int(value)
        if value < 0:
            value = value % t
        
        if value >= t:
            raise ValueError(f"Value {value} exceeds plaintext modulus {t}")
        
        # Encrypt
        return self.pyfhel.encryptInt(value)
    
    def encrypt_array(self, values: Union[List[int], np.ndarray]) -> PyCtxt:
        """
        Encrypt an array of integers using SIMD batching.
        
        BFV supports batching multiple values into a single ciphertext using
        the Chinese Remainder Theorem (CRT). This enables parallel operations.
        
        Args:
            values: Array of integers to encrypt
            
        Returns:
            Ciphertext containing all values in parallel slots
            
        Raises:
            ValueError: If array is too large or values are invalid
        """
        if isinstance(values, list):
            values = np.array(values, dtype=np.int64)
        
        # Get number of available slots
        n_slots = self.pyfhel.get_nSlots()
        
        if len(values) > n_slots:
            raise ValueError(f"Array size {len(values)} exceeds available slots {n_slots}")
        
        # Pad array to fill slots if needed
        if len(values) < n_slots:
            padded = np.zeros(n_slots, dtype=np.int64)
            padded[:len(values)] = values
            values = padded
        
        # Validate all values
        t = self.pyfhel.t
        for i, val in enumerate(values):
            val_mod = int(val) % t if val < 0 else int(val)
            if val_mod >= t:
                raise ValueError(f"Value at index {i} ({val}) exceeds plaintext modulus {t}")
            values[i] = val_mod
        
        # Encrypt array
        return self.pyfhel.encryptArray(values.astype(np.int64))
    
    def encrypt_batch(self, values: Union[List[int], np.ndarray]) -> List[PyCtxt]:
        """
        Encrypt multiple values, automatically batching them efficiently.
        
        This method intelligently groups values into ciphertexts to minimize
        the number of ciphertexts while maximizing SIMD usage.
        
        Args:
            values: List or array of integers to encrypt
            
        Returns:
            List of ciphertexts containing all values
        """
        if isinstance(values, list):
            values = np.array(values, dtype=np.int64)
        
        n_slots = self.pyfhel.get_nSlots()
        ciphertexts = []
        
        # Split values into chunks that fit in slots
        for i in range(0, len(values), n_slots):
            chunk = values[i:i + n_slots]
            ct = self.encrypt_array(chunk)
            ciphertexts.append(ct)
        
        return ciphertexts
    
    def decrypt_int(self, ciphertext: PyCtxt) -> int:
        """
        Decrypt a ciphertext to a single integer.
        
        Args:
            ciphertext: Ciphertext to decrypt
            
        Returns:
            Decrypted integer value
        """
        result = self.pyfhel.decryptInt(ciphertext)
        
        # Handle potential negative values (convert from modular representation)
        t = self.pyfhel.t
        if result > t // 2:
            result = result - t
        
        return int(result)
    
    def decrypt_array(self, ciphertext: PyCtxt) -> np.ndarray:
        """
        Decrypt a batched ciphertext to an array of integers.
        
        Args:
            ciphertext: Batched ciphertext
            
        Returns:
            Array of decrypted integers
        """
        result = self.pyfhel.decryptArray(ciphertext)
        
        # Handle negative values
        t = self.pyfhel.t
        mask = result > t // 2
        result[mask] = result[mask] - t
        
        return result.astype(np.int64)
    
    def decrypt_batch(
        self,
        ciphertexts: List[PyCtxt],
        original_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Decrypt multiple batched ciphertexts.
        
        Args:
            ciphertexts: List of ciphertexts to decrypt
            original_size: Original data size (to remove padding)
            
        Returns:
            Concatenated array of all decrypted values
        """
        results = []
        
        for ct in ciphertexts:
            decrypted = self.decrypt_array(ct)
            results.append(decrypted)
        
        all_results = np.concatenate(results)
        
        # Trim to original size if specified
        if original_size is not None:
            all_results = all_results[:original_size]
        
        return all_results


class BFVEncoder:
    """
    Handles encoding of various data types for BFV encryption.
    
    BFV works with integers modulo t. This class helps encode other data types
    (floats, strings, etc.) into integers for encryption.
    """
    
    def __init__(self, plain_modulus: int):
        """
        Initialize encoder.
        
        Args:
            plain_modulus: Plaintext modulus (t)
        """
        self.t = plain_modulus
    
    def encode_float_to_int(
        self,
        value: float,
        precision: int = 3
    ) -> int:
        """
        Encode a floating-point value as an integer.
        
        This multiplies the float by 10^precision and rounds to nearest integer.
        Note: This is a simple encoding. For better float support, use CKKS.
        
        Args:
            value: Float value to encode
            precision: Number of decimal places to preserve
            
        Returns:
            Encoded integer
            
        Example:
            encode_float_to_int(3.14159, precision=3) -> 3142
        """
        scale = 10 ** precision
        encoded = int(round(value * scale))
        
        # Ensure it fits in plaintext modulus
        encoded = encoded % self.t
        
        return encoded
    
    def decode_int_to_float(
        self,
        value: int,
        precision: int = 3
    ) -> float:
        """
        Decode an integer back to floating-point.
        
        Args:
            value: Encoded integer
            precision: Number of decimal places used in encoding
            
        Returns:
            Decoded float value
        """
        scale = 10 ** precision
        
        # Handle negative values
        if value > self.t // 2:
            value = value - self.t
        
        return float(value) / scale
    
    def encode_float_array(
        self,
        values: Union[List[float], np.ndarray],
        precision: int = 3
    ) -> np.ndarray:
        """
        Encode an array of floats to integers.
        
        Args:
            values: Array of floats
            precision: Decimal precision
            
        Returns:
            Array of encoded integers
        """
        if isinstance(values, list):
            values = np.array(values)
        
        scale = 10 ** precision
        encoded = np.round(values * scale).astype(np.int64)
        encoded = encoded % self.t
        
        return encoded
    
    def decode_float_array(
        self,
        values: Union[List[int], np.ndarray],
        precision: int = 3
    ) -> np.ndarray:
        """
        Decode an array of integers to floats.
        
        Args:
            values: Array of encoded integers
            precision: Decimal precision used in encoding
            
        Returns:
            Array of decoded floats
        """
        if isinstance(values, list):
            values = np.array(values, dtype=np.int64)
        
        # Handle negative values
        mask = values > self.t // 2
        values = values.astype(np.int64)
        values[mask] = values[mask] - self.t
        
        scale = 10 ** precision
        return values.astype(np.float64) / scale
    
    def validate_value(self, value: int) -> bool:
        """
        Check if a value is valid for the current plaintext modulus.
        
        Args:
            value: Integer to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 0 <= value < self.t
    
    def get_valid_range(self) -> tuple:
        """
        Get the valid range of values for this encoder.
        
        Returns:
            Tuple of (min_value, max_value)
        """
        # For signed integers, we use negative wrapping
        return (-self.t // 2, self.t // 2 - 1)


class BFVKeyManager:
    """
    Manages key serialization and storage for client-server applications.
    """
    
    def __init__(self, pyfhel: Pyfhel):
        """
        Initialize key manager.
        
        Args:
            pyfhel: Pyfhel instance with generated keys
        """
        self.pyfhel = pyfhel
    
    def serialize_public_key(self) -> bytes:
        """
        Serialize public key to bytes.
        
        Returns:
            Serialized public key
        """
        return self.pyfhel.to_bytes_public_key()
    
    def serialize_secret_key(self) -> bytes:
        """
        Serialize secret key to bytes.
        
        WARNING: Secret key should never be transmitted!
        This is for local storage only.
        
        Returns:
            Serialized secret key
        """
        return self.pyfhel.to_bytes_secret_key()
    
    def serialize_relin_key(self) -> bytes:
        """
        Serialize relinearization keys.
        
        Returns:
            Serialized relin keys
        """
        return self.pyfhel.to_bytes_relin_key()
    
    def serialize_rotate_key(self) -> bytes:
        """
        Serialize rotation (Galois) keys.
        
        Returns:
            Serialized rotation keys
        """
        return self.pyfhel.to_bytes_rotate_key()
    
    def serialize_context(self) -> bytes:
        """
        Serialize context parameters.
        
        Returns:
            Serialized context
        """
        return self.pyfhel.to_bytes_context()
    
    def load_public_key(self, key_bytes: bytes) -> None:
        """
        Load public key from bytes.
        
        Args:
            key_bytes: Serialized public key
        """
        self.pyfhel.from_bytes_public_key(key_bytes)
    
    def load_secret_key(self, key_bytes: bytes) -> None:
        """
        Load secret key from bytes.
        
        Args:
            key_bytes: Serialized secret key
        """
        self.pyfhel.from_bytes_secret_key(key_bytes)
    
    def load_relin_key(self, key_bytes: bytes) -> None:
        """
        Load relinearization keys from bytes.
        
        Args:
            key_bytes: Serialized relin keys
        """
        self.pyfhel.from_bytes_relin_key(key_bytes)
    
    def load_rotate_key(self, key_bytes: bytes) -> None:
        """
        Load rotation keys from bytes.
        
        Args:
            key_bytes: Serialized rotation keys
        """
        self.pyfhel.from_bytes_rotate_key(key_bytes)
    
    def load_context(self, context_bytes: bytes) -> None:
        """
        Load context from bytes.
        
        Args:
            context_bytes: Serialized context
        """
        self.pyfhel.from_bytes_context(context_bytes)
    
    def save_keys_to_files(self, prefix: str = "bfv_key") -> Dict[str, str]:
        """
        Save all keys to separate files.
        
        Args:
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping key type to filename
        """
        filenames = {}
        
        # Save context
        context_file = f"{prefix}_context.bin"
        self.pyfhel.save_context(context_file)
        filenames['context'] = context_file
        
        # Save public key
        pubkey_file = f"{prefix}_public.bin"
        self.pyfhel.save_public_key(pubkey_file)
        filenames['public_key'] = pubkey_file
        
        # Save secret key
        seckey_file = f"{prefix}_secret.bin"
        self.pyfhel.save_secret_key(seckey_file)
        filenames['secret_key'] = seckey_file
        
        # Save relin keys
        relin_file = f"{prefix}_relin.bin"
        self.pyfhel.save_relin_key(relin_file)
        filenames['relin_key'] = relin_file
        
        # Save rotate keys
        rotate_file = f"{prefix}_rotate.bin"
        self.pyfhel.save_rotate_key(rotate_file)
        filenames['rotate_key'] = rotate_file
        
        return filenames
    
    def load_keys_from_files(self, prefix: str = "bfv_key") -> None:
        """
        Load all keys from files.
        
        Args:
            prefix: Filename prefix used when saving
        """
        # Load context first
        self.pyfhel.load_context(f"{prefix}_context.bin")
        
        # Load keys
        self.pyfhel.load_public_key(f"{prefix}_public.bin")
        self.pyfhel.load_secret_key(f"{prefix}_secret.bin")
        self.pyfhel.load_relin_key(f"{prefix}_relin.bin")
        self.pyfhel.load_rotate_key(f"{prefix}_rotate.bin")
