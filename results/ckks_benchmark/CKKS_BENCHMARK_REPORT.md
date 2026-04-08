# CKKS Benchmarking Report

Generated: 2026-04-08 23:01:31

## 1. Encryption Performance

| Data Size | Time (s) | Throughput (values/s) | Chunks |
|----------:|---------:|----------------------:|-------:|
| 100 | 0.0077 | 13,034 | 1 |
| 500 | 0.0075 | 67,063 | 1 |
| 1,000 | 0.0079 | 126,929 | 1 |
| 5,000 | 0.0077 | 646,964 | 1 |
| 10,000 | 0.0150 | 665,575 | 2 |
| 20,000 | 0.0236 | 849,167 | 3 |
| 50,000 | 0.0537 | 931,543 | 7 |
| 100,000 | 0.0967 | 1,034,470 | 13 |

## 2. Decryption Performance

| Data Size | Time (s) | Throughput (values/s) |
|----------:|---------:|----------------------:|
| 100 | 0.0035 | 28,624 |
| 500 | 0.0029 | 169,929 |
| 1,000 | 0.0028 | 357,961 |
| 5,000 | 0.0028 | 1,762,736 |
| 10,000 | 0.0054 | 1,845,563 |
| 20,000 | 0.0082 | 2,426,625 |
| 50,000 | 0.0188 | 2,654,745 |
| 100,000 | 0.0347 | 2,883,174 |

## 3. Aggregation Scalability

| Clients | Agg Time (s) | Per-Client (ms) | Total Samples |
|--------:|-------------:|----------------:|--------------:|
| 2 | 0.0032 | 1.617 | 2,000 |
| 5 | 0.0035 | 0.691 | 5,000 |
| 10 | 0.0043 | 0.435 | 10,000 |
| 20 | 0.0068 | 0.340 | 20,000 |
| 50 | 0.0143 | 0.285 | 50,000 |
| 100 | 0.0251 | 0.251 | 100,000 |
| 200 | 0.0480 | 0.240 | 200,000 |

## 4. Homomorphic Multiplication Performance

- **add\_encrypted baseline**: 0.605 ms
- **Relinearization cost**: 7.7158 ms
- **Depth limit**: not reached within tested range

| Depth | Single Op (ms) | Cumulative (s) | vs add\_encrypted |
|------:|---------------:|---------------:|------------------:|
| 1 | 11.1378 | 0.0573 | 18.4x |
| 2 | 8.9039 | 0.1020 | 14.7x |
| 3 | 5.8390 | 0.1403 | 9.7x |
| 4 | 3.4564 | 0.1754 | 5.7x |

**Average multiply\_encrypted**: 7.3343 ms  
**Average overhead vs add**: 12.1x  

## 5. Communication Overhead

| Data Size | Plaintext (KB) | Ciphertext (KB) | Overhead | # Ciphertexts |
|----------:|---------------:|----------------:|---------:|--------------:|
| 100 | 0.8 | 1024.3 | 1311.1× | 1 |
| 500 | 3.9 | 1024.3 | 262.2× | 1 |
| 1,000 | 7.8 | 1024.3 | 131.1× | 1 |
| 5,000 | 39.1 | 1024.3 | 26.2× | 1 |
| 10,000 | 78.1 | 2048.6 | 26.2× | 2 |
| 50,000 | 390.6 | 7170.2 | 18.4× | 7 |

## 6. End-to-End Workflow

- **Clients**: 20
- **Samples / client**: 1,000
- **Total time**: 6.279 s
- **Global mean** (decrypted): 1.72395
### Phase Breakdown

| Phase | Time (s) | Share (%) |
|:------|----------:|----------:|
| crypto_setup | 0.6906 | 11.0 |
| client_data_gen | 0.0003 | 0.0 |
| local_computation | 0.0001 | 0.0 |
| encryption | 0.1474 | 2.3 |
| communication | 5.4321 | 86.5 |
| aggregation | 0.0072 | 0.1 |
