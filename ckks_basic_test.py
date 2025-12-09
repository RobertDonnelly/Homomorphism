from Pyfhel import Pyfhel
import numpy as np

def simple_ckks_demo():
    # 1. Create Pyfhel instance
    HE = Pyfhel()

    # 2. CKKS context generation
    #   - ckks_scheme = 'CKKS'
    #   - n = poly modulus degree (must be power of 2)
    #   - scale = 2^scale_bits (used internally)
    HE.contextGen(
        scheme='CKKS',
        n=2**14,          # typical values: 2^13, 2^14, 2^15
        scale=2**40,      # commonly 2^30 to 2^50
        qi_sizes=[60, 40, 40, 60]  # modulus chain
    )

    # 3. Generate keys
    HE.keyGen()
    HE.relinKeyGen()

    # 4. Prepare data (floats for CKKS)
    x = np.array([5.5, -2.1, 3.14])
    y = np.array([1.1, 3.2, -0.5])

    # 5. Encode
    ptxt_x = HE.encodeFrac(x)
    ptxt_y = HE.encodeFrac(y)

    # 6. Encrypt
    ctxt_x = HE.encryptPtxt(ptxt_x)
    ctxt_y = HE.encryptPtxt(ptxt_y)

    # 7. Perform operations
    ctxt_sum  = ctxt_x + ctxt_y
    ctxt_mul  = ctxt_x * ctxt_y
    ctxt_scale = ctxt_x * 2.0  # multiply ciphertext by scalar

    # 8. Decrypt & decode
    r_sum  = HE.decryptFrac(ctxt_sum)
    r_mul  = HE.decryptFrac(ctxt_mul)
    r_scale = HE.decryptFrac(ctxt_scale)

    print("\nOriginal x:", x)
    print("Original y:", y)

    print("\nDecrypted Results (CKKS):")
    print("x + y     =", r_sum)
    print("x * y     =", r_mul)
    print("x * 2     =", r_scale)


if __name__ == "__main__":
    simple_ckks_demo()
