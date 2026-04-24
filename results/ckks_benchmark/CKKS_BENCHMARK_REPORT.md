# CKKS Benchmarking Report

Generated: 2026-04-24 17:25:09

## 1. Encryption Performance

| Data Size | Time (s) | Throughput (values/s) | Chunks |
|----------:|---------:|----------------------:|-------:|
| 100 | 0.0082 | 12,171 | 1 |
| 500 | 0.0083 | 60,041 | 1 |
| 1,000 | 0.0079 | 126,275 | 1 |
| 5,000 | 0.0079 | 634,469 | 1 |
| 10,000 | 0.0157 | 637,186 | 2 |
| 20,000 | 0.0231 | 864,110 | 3 |
| 50,000 | 0.0516 | 968,208 | 7 |
| 100,000 | 0.0950 | 1,052,677 | 13 |

## 2. Decryption Performance

| Data Size | Time (s) | Throughput (values/s) |
|----------:|---------:|----------------------:|
| 100 | 0.0030 | 33,400 |
| 500 | 0.0030 | 166,019 |
| 1,000 | 0.0027 | 364,724 |
| 5,000 | 0.0027 | 1,825,350 |
| 10,000 | 0.0061 | 1,628,877 |
| 20,000 | 0.0081 | 2,460,690 |
| 50,000 | 0.0189 | 2,646,217 |
| 100,000 | 0.0345 | 2,901,494 |

## 3. Aggregation Scalability

| Clients | Agg Time (s) | Per-Client (ms) | Total Samples |
|--------:|-------------:|----------------:|--------------:|
| 2 | 0.0027 | 1.355 | 2,000 |
| 5 | 0.0032 | 0.636 | 5,000 |
| 10 | 0.0043 | 0.432 | 10,000 |
| 20 | 0.0067 | 0.338 | 20,000 |
| 50 | 0.0136 | 0.271 | 50,000 |
| 100 | 0.0249 | 0.249 | 100,000 |
| 200 | 0.0474 | 0.237 | 200,000 |

## 4. Homomorphic Multiplication Performance

- **add\_encrypted baseline**: 0.4479 ms
- **Relinearization cost**: 6.74 ms
- **Depth limit**: not reached within tested range

| Depth | Single Op (ms) | Cumulative (s) | vs add\_encrypted |
|------:|---------------:|---------------:|------------------:|
| 1 | 10.3688 | 0.0544 | 23.1x |
| 2 | 7.7644 | 0.0968 | 17.3x |
| 3 | 5.5688 | 0.1345 | 12.4x |
| 4 | 3.2825 | 0.1683 | 7.3x |

**Average multiply\_encrypted**: 6.7461 ms  
**Average overhead vs add**: 15.1x  

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
- **Total time**: 6.193 s
- **Global mean** (decrypted): 1.72395
### Phase Breakdown

| Phase | Time (s) | Share (%) |
|:------|----------:|----------:|
| crypto_setup | 0.6712 | 10.8 |
| client_data_gen | 0.0002 | 0.0 |
| local_computation | 0.0001 | 0.0 |
| encryption | 0.1453 | 2.3 |
| communication | 5.3680 | 86.7 |
| aggregation | 0.0077 | 0.1 |
