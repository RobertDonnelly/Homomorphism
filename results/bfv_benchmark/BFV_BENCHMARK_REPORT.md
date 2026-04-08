# BFV Benchmarking Report

Generated: 2026-04-09 00:06:43

> Integer scale factor: ×1000  (floats multiplied before int64 conversion)

> Encryption/decryption times for sizes > 200 are extrapolated from a timed probe.

> BFV parameters: n=16384, t_bits=17, sec=128

## 1. Encryption Performance

| Data Size | Est. Time (s) | Throughput (values/s) | ms/value |
|----------:|--------------:|----------------------:|---------:|
| 100 | 0.9431 | 106 | 9.431 |
| 500 | 4.6942 | 107 | 9.388 |
| 1,000 | 9.3753 | 107 | 9.375 |
| 5,000 | 46.8082 | 107 | 9.362 |
| 10,000 | 94.9089 | 105 | 9.491 |
| 20,000 | 191.3361 | 105 | 9.567 |

## 2. Decryption Performance

| Data Size | Est. Time (s) | Throughput (values/s) |
|----------:|--------------:|----------------------:|
| 100 | 0.3495 | 286 |
| 500 | 1.7784 | 281 |
| 1,000 | 3.4635 | 289 |
| 5,000 | 17.9240 | 279 |
| 10,000 | 34.7006 | 288 |
| 20,000 | 69.0359 | 290 |

## 3. Aggregation Scalability

| Clients | Agg Time (s) | Per-Client (ms) | Total Samples |
|--------:|-------------:|----------------:|--------------:|
| 2 | 0.0048 | 2.406 | 2,000 |
| 5 | 0.0059 | 1.171 | 5,000 |
| 10 | 0.0095 | 0.953 | 10,000 |
| 20 | 0.0144 | 0.721 | 20,000 |
| 50 | 0.0303 | 0.606 | 50,000 |
| 100 | 0.0540 | 0.540 | 100,000 |
| 200 | 0.1028 | 0.514 | 200,000 |

## 4. Homomorphic Multiplication Performance

> BFV uses exact integer arithmetic. Noise budget (not modulus chain depth) is the limiting factor for chained multiplications.

- **add\_encrypted baseline**: 0.5745 ms
- **Noise budget exhausted at**: not reached within tested range

| Depth | Single Op (ms) | Cumulative (s) | vs add\_encrypted | Correct |
|------:|---------------:|---------------:|------------------:|--------:|
| 1 | 50.4608 | 0.0693 | 87.8x | Yes |
| 2 | 50.0659 | 0.2022 | 87.2x | Yes |
| 3 | 49.9228 | 0.3953 | 86.9x | Yes |
| 4 | 49.8185 | 0.6468 | 86.7x | Yes |
| 5 | 50.0143 | 0.9574 | 87.1x | Yes |

**Average multiply\_encrypted**: 50.0565 ms  
**Average overhead vs add**: 87.1x  

## 5. Communication Overhead

| Data Size | Plaintext (KB) | Ciphertext (KB) | Overhead | # Ciphertexts |
|----------:|---------------:|----------------:|---------:|--------------:|
| 100 | 0.8 | 204835.3 | 262189.1× | 100 |
| 500 | 3.9 | 1024176.3 | 262189.1× | 500 |
| 1,000 | 7.8 | 2048352.5 | 262189.1× | 1,000 |
| 5,000 | 39.1 | 10241762.7 | 262189.1× | 5,000 |
| 10,000 | 78.1 | 20483525.4 | 262189.1× | 10,000 |

## 6. End-to-End Workflow

- **Clients**: 20
- **Samples / client**: 1,000
- **Total time**: 16.952 s
- **Global mean** (decrypted & descaled): 0.000327

### Phase Breakdown

| Phase | Time (s) | Share (%) |
|:------|----------:|----------:|
| crypto_setup | 0.8658 | 5.1 |
| client_data_gen | 0.0003 | 0.0 |
| local_computation | 0.0005 | 0.0 |
| encryption | 0.1912 | 1.1 |
| communication | 15.8794 | 93.7 |
| aggregation | 0.0137 | 0.1 |
