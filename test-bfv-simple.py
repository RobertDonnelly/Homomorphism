from Pyfhel import Pyfhel
import numpy as np

def test_bfv_basic_safe():
    """Robust BFV test: encode integers into plaintext objects before encryption."""
    
    print("\n=== BFV Safe Test ===")
    
    # Step 1: Initialize Pyfhel
    HE = Pyfhel()
    HE.contextGen(scheme='BFV', n=4096, t=65537, sec=128)
    HE.keyGen()
    HE.relinKeyGen()
    
    # Step 2: Define integers (wrapped as numpy int64 arrays)
    x = np.array([42], dtype=np.int64)
    y = np.array([17], dtype=np.int64)
    
    print(f"Debug: x type = {type(x)}, x dtype = {x.dtype}")
    print(f"Debug: y type = {type(y)}, y dtype = {y.dtype}")
    
    # Step 3: Encode integers into PyPtxt plaintexts
    ptxt_x = HE.encodeInt(x)
    ptxt_y = HE.encodeInt(y)
    
    # Step 4: Encrypt plaintexts
    ct_x = HE.encryptPtxt(ptxt_x)
    ct_y = HE.encryptPtxt(ptxt_y)
    
    print("Encryption successful")
    
    # Step 5: Homomorphic operations
    ct_sum = ct_x + ct_y
    ct_prod = ct_x * ct_y
    HE.relinearize(ct_prod)
    ct_diff = ct_x - ct_y
    ct_scalar = ct_x * 5
    
    # Step 6: Decrypt results
    sum_result = HE.decryptInt(ct_sum)[0]      # decryptInt returns an array
    prod_result = HE.decryptInt(ct_prod)[0]
    diff_result = HE.decryptInt(ct_diff)[0]
    scalar_result = HE.decryptInt(ct_scalar)[0]
    
    # Step 7: Print results
    print(f"42 + 17 = {sum_result}")
    print(f"42 * 17 = {prod_result}")
    print(f"42 - 17 = {diff_result}")
    print(f"42 * 5 = {scalar_result}")
    
    print("✓ All operations completed successfully.")

# Run the test
if __name__ == "__main__":
    test_bfv_basic_safe()
