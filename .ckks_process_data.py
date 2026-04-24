#CKKS Data Processing — Benchmarking Suite
import sys
import pickle
import time
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.schemes.ckks.ckks_crypto import CKKSCrypto

# config
RUN_MULTIPLICATION_BENCHMARK = True

# Output directories
RESULTS_DIR = project_root / 'results' / 'ckks_benchmark'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Data loader
def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/raw/ and return a DataFrame."""
    filepath = project_root / 'data' / 'raw' / filename
    if not filepath.exists():
        print(f"\n!!!  File not found: {filepath}")
        print(f"    Place '{filename}' in data/raw/ and retry.")
        sys.exit(1)
    print(f"...  Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓   {len(df):,} rows × {len(df.columns)} columns  |  "
          f"columns: {list(df.columns)}")
    return df


def extract_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """Validate column exists and return a float64 array with NaNs filled."""
    if column not in df.columns:
        print(f"\n!!!  Column '{column}' not found.")
        print(f"    Available: {list(df.columns)}")
        sys.exit(1)
    return df[column].fillna(0).astype(np.float64).values


# Helper: chunk a 1-D array into SIMD-sized pieces

def _chunk(data: np.ndarray, max_slots: int) -> List[np.ndarray]:
    n = len(data)
    return [data[i * max_slots: min((i + 1) * max_slots, n)]
            for i in range(int(np.ceil(n / max_slots)))]


# Benchmark 1 — Encryption performance
def benchmark_encryption(ckks: CKKSCrypto,
                          data: np.ndarray,
                          data_sizes: List[int]) -> Dict:

    #Returns a dict ready to be serialised to JSON.
    print("\n" + "=" * 70)
    print("BENCHMARK 1: ENCRYPTION PERFORMANCE")
    print("=" * 70)

    max_slots = ckks.n // 2
    results = {
        'data_sizes': data_sizes,
        'encryption_times': [],
        'throughput_values_per_sec': [],
        'time_per_chunk_s': [],
        'num_chunks': [],
        'single_value_time_ms': [],   # None for sizes > 1 000
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = np.resize(data, size).astype(np.float64)

        # ── Single-value baseline (small sizes only) ─────────────────────
        if size <= 1_000:
            probe = sample[:100]
            t0 = time.perf_counter()
            for v in probe:
                _ = ckks.encrypt(float(v))
            sv_ms = (time.perf_counter() - t0) / len(probe) * 1_000
            results['single_value_time_ms'].append(sv_ms)
            print(f"    single-value: {sv_ms:.3f} ms/value")
        else:
            results['single_value_time_ms'].append(None)

        # ── SIMD vector encryption ────────────────────────────────────────
        chunks = _chunk(sample, max_slots)
        t0 = time.perf_counter()
        for ch in chunks:
            _ = ckks.encrypt_vector(ch)
        enc_time = time.perf_counter() - t0

        throughput = size / enc_time if enc_time > 0 else float('inf')
        time_per_chunk = enc_time / len(chunks) if len(chunks) > 0 else 0.0

        results['encryption_times'].append(round(enc_time, 6))
        results['throughput_values_per_sec'].append(round(throughput, 2))
        results['time_per_chunk_s'].append(round(time_per_chunk, 6))
        results['num_chunks'].append(len(chunks))

        print(f"    SIMD enc:  {enc_time:.3f} s total | "
              f"{throughput:,.0f} values/s | "
              f"{len(chunks)} chunk(s)")

    print("\n✓  Encryption benchmark complete.")
    return results


# Benchmark 2 — Decryption performance
def benchmark_decryption(ckks: CKKSCrypto,
                          data: np.ndarray,
                          data_sizes: List[int]) -> Dict:
    #Pre-encrypt chunks then measure decryption throughput for each data size.
    print("\n" + "=" * 70)
    print("BENCHMARK 2: DECRYPTION PERFORMANCE")
    print("=" * 70)

    max_slots = ckks.n // 2
    results = {
        'data_sizes': data_sizes,
        'decryption_times': [],
        'throughput_values_per_sec': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = np.resize(data, size).astype(np.float64)
        chunks = _chunk(sample, max_slots)

        # Pre-encrypt (not timed)
        enc_chunks = [ckks.encrypt_vector(ch) for ch in chunks]

        # ── Timed decryption ──────────────────────────────────────────────
        t0 = time.perf_counter()
        for i, enc in enumerate(enc_chunks):
            ch_size = len(chunks[i])
            _ = ckks.decrypt_vector(enc, ch_size)
        dec_time = time.perf_counter() - t0

        throughput = size / dec_time if dec_time > 0 else float('inf')
        results['decryption_times'].append(round(dec_time, 6))
        results['throughput_values_per_sec'].append(round(throughput, 2))

        print(f"    dec: {dec_time:.3f} s | {throughput:,.0f} values/s")

    print("\n✓  Decryption benchmark complete.")
    return results


# Benchmark 3 — Aggregation scalability
def benchmark_aggregation_scalability(ckks: CKKSCrypto,
                                       data: np.ndarray,
                                       num_clients_list: List[int],
                                       samples_per_client: int = 1_000) -> Dict:
    """
    Measure server-side homomorphic aggregation time as the number of clients
    scales.  Each client contributes a single encrypted scalar sum, matching
    the BFV contribution format so aggregation costs are directly comparable.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: AGGREGATION SCALABILITY")
    print(f"  samples/client = {samples_per_client:,}")
    print("=" * 70)

    results = {
        'num_clients': num_clients_list,
        'samples_per_client': samples_per_client,
        'aggregation_times': [],
        'per_client_overhead_ms': [],
        'total_samples': [],
    }

    for n_clients in num_clients_list:
        print(f"\n  clients = {n_clients}")

        contributions = []
        for _ in range(n_clients):
            client_data = np.resize(data, samples_per_client).astype(np.float64)
            enc_sum = ckks.encrypt(float(np.sum(client_data)))
            contributions.append(enc_sum)

        # ── Timed aggregation ─────────────────────────────────────────────
        t0 = time.perf_counter()
        agg_sum = contributions[0]
        for enc_sum in contributions[1:]:
            agg_sum = ckks.add_encrypted(agg_sum, enc_sum)
        _ = ckks.decrypt(agg_sum)
        agg_time = time.perf_counter() - t0

        per_client_ms = agg_time / n_clients * 1_000
        total_samples = n_clients * samples_per_client

        results['aggregation_times'].append(round(agg_time, 6))
        results['per_client_overhead_ms'].append(round(per_client_ms, 4))
        results['total_samples'].append(total_samples)

        print(f"    agg: {agg_time:.3f} s | "
              f"{per_client_ms:.3f} ms/client | "
              f"{total_samples:,} total samples")

    print("\n✓  Scalability benchmark complete.")
    return results


# Benchmark 4 — Homomorphic multiplication (opt-in)
def benchmark_multiplication(ckks: CKKSCrypto,
                              data: np.ndarray,
                              mul_depths: List[int] = None) -> Dict:
    """
    Benchmark multiply_encrypted performance across increasing multiplication
    depths.

    Measures:
      - Single add_encrypted call as a baseline for comparison
      - Single multiply_encrypted call in isolation
      - Isolated relinearization cost per level
      - Per-operation time at each depth in the chain sweep
      - Cumulative time to reach each depth
      - The depth at which the modulus chain is exhausted (if hit)

    Uses a deeper modulus chain [60,40,40,40,40,60] to allow up to 4 sequential multiplications. 
    """
    if mul_depths is None:
        mul_depths = [1, 2, 3, 4]

    print("\n" + "=" * 70)
    print("BENCHMARK 4: HOMOMORPHIC MULTIPLICATION PERFORMANCE")
    print("  Note: each multiplication consumes one modulus chain level.")
    print(f"  Testing depths: {mul_depths}")
    print("=" * 70)

    # Dedicated CKKS instance with a deeper chain for this benchmark
    qi_sizes = [60, 40, 40, 40, 40, 60]
    ckks_mul = CKKSCrypto(n=2**14, scale=2**40, qi_sizes=qi_sizes)
    ckks_mul.setup()
    max_depth = len(qi_sizes) - 2   # first and last slots are reserved
    print(f"  Modulus chain depth available: {max_depth} levels\n")

    results = {
        'mul_depths_tested': [],
        'single_op_time_ms': [],       # time for one multiply_encrypted (ms)
        'cumulative_time_s': [],        # total elapsed reaching that depth
        'relinearization_time_ms': None,  # isolated relin cost
        'add_baseline_time_ms': None,   # single add_encrypted for comparison
        'depth_limit': None,            # first depth that failed, or None
        'overhead_vs_add': [],          # multiply/add ratio per depth
    }

    # ── Baseline: single add_encrypted ───────────────────────────────────
    enc_a = ckks_mul.encrypt(float(data[0]))
    enc_b = ckks_mul.encrypt(float(data[1]))
    t0 = time.perf_counter()
    _ = ckks_mul.add_encrypted(enc_a, enc_b)
    add_ms = (time.perf_counter() - t0) * 1_000
    results['add_baseline_time_ms'] = round(add_ms, 4)
    print(f"  Baseline — add_encrypted:      {add_ms:.4f} ms")

    # ── Isolated relinearization cost ────────────────────────────────────
    enc_x = ckks_mul.encrypt(float(data[0]))
    enc_y = ckks_mul.encrypt(float(data[1]))
    enc_raw = enc_x * enc_y   # raw multiply, no relin yet
    t0 = time.perf_counter()
    ckks_mul.HE.relinearize(enc_raw)
    relin_ms = (time.perf_counter() - t0) * 1_000
    results['relinearization_time_ms'] = round(relin_ms, 4)
    print(f"  Relinearization cost:          {relin_ms:.4f} ms")

    # ── Single multiply_encrypted in isolation ────────────────────────────
    enc_x = ckks_mul.encrypt(float(data[0]))
    enc_y = ckks_mul.encrypt(float(data[1]))
    t0 = time.perf_counter()
    _ = ckks_mul.multiply_encrypted(enc_x, enc_y)
    single_mul_ms = (time.perf_counter() - t0) * 1_000
    print(f"  Single multiply_encrypted:     {single_mul_ms:.4f} ms")
    print(f"  Overhead vs add:               {single_mul_ms / add_ms:.1f}x\n")

    # ── Depth sweep ───────────────────────────────────────────────────────
    print(f"  {'Depth':<8} {'Single op (ms)':<18} {'Cumulative (s)':<18} {'vs add':<10} Status")
    print(f"  {'-'*8} {'-'*18} {'-'*18} {'-'*10} {'-'*8}")

    cumulative_start = time.perf_counter()
    enc_running = ckks_mul.encrypt(float(data[0]))
    depth_limit = None

    for depth in mul_depths:
        if depth > max_depth:
            print(f"  {depth:<8} {'---':<18} {'---':<18} {'---':<10} SKIPPED (exceeds chain)")
            if depth_limit is None:
                depth_limit = depth
            continue

        # Fresh pair at the correct chain level for an isolated op time
        enc_fa = ckks_mul.encrypt(float(data[0]))
        enc_fb = ckks_mul.encrypt(float(data[1]))
        for _ in range(depth - 1):
            ckks_mul.HE.mod_switch_to_next(enc_fa)
            ckks_mul.HE.mod_switch_to_next(enc_fb)

        try:
            t0 = time.perf_counter()
            _ = ckks_mul.multiply_encrypted(enc_fa, enc_fb)
            op_ms = (time.perf_counter() - t0) * 1_000

            # Advance the running chain ciphertext by one more multiply
            enc_running = ckks_mul.multiply_encrypted(
                enc_running,
                ckks_mul.encrypt(float(data[depth % len(data)]))
            )
            cumulative_s = time.perf_counter() - cumulative_start
            ratio = op_ms / add_ms

            results['mul_depths_tested'].append(depth)
            results['single_op_time_ms'].append(round(op_ms, 4))
            results['cumulative_time_s'].append(round(cumulative_s, 6))
            results['overhead_vs_add'].append(round(ratio, 2))

            print(f"  {depth:<8} {op_ms:<18.4f} {cumulative_s:<18.4f} {ratio:<10.1f} OK")

        except Exception as e:
            print(f"  {depth:<8} {'---':<18} {'---':<18} {'---':<10} FAILED: {e}")
            if depth_limit is None:
                depth_limit = depth
            break

    results['depth_limit'] = depth_limit

    if depth_limit:
        print(f"\n  ⚠  Depth limit hit at depth {depth_limit}.")
    else:
        print(f"\n  ✓  All tested depths completed within chain.")

    if results['single_op_time_ms']:
        avg = np.mean(results['single_op_time_ms'])
        print(f"\n  avg multiply_encrypted: {avg:.4f} ms  |  "
              f"avg overhead vs add: {avg / add_ms:.1f}x")

    print("\n✓  Multiplication benchmark complete.")
    return results


# Benchmark 5 — Communication overhead
def benchmark_communication_overhead(ckks: CKKSCrypto,
                                      data: np.ndarray,
                                      data_sizes: List[int]) -> Dict:
    #Compare plaintext byte size against the serialised ciphertext byte size, yielding an overhead ratio.
    print("\n" + "=" * 70)
    print("BENCHMARK 5: COMMUNICATION OVERHEAD")
    print("=" * 70)

    max_slots = ckks.n // 2
    results = {
        'data_sizes': data_sizes,
        'plaintext_bytes': [],
        'ciphertext_bytes': [],
        'overhead_ratio': [],
        'num_ciphertexts': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = np.resize(data, size).astype(np.float64)
        plaintext_bytes = size * 8   # float64 = 8 bytes

        chunks = _chunk(sample, max_slots)
        enc_first = ckks.encrypt_vector(chunks[0])
        bytes_per_ctxt = len(pickle.dumps(enc_first))
        total_ctxt_bytes = bytes_per_ctxt * len(chunks)
        overhead = total_ctxt_bytes / plaintext_bytes

        results['plaintext_bytes'].append(plaintext_bytes)
        results['ciphertext_bytes'].append(total_ctxt_bytes)
        results['overhead_ratio'].append(round(overhead, 2))
        results['num_ciphertexts'].append(len(chunks))

        print(f"    plaintext: {plaintext_bytes / 1_024:.1f} KB | "
              f"ciphertext: {total_ctxt_bytes / 1_024:.1f} KB | "
              f"overhead: {overhead:.1f}× | "
              f"{len(chunks)} ciphertext(s)")

    print("\n✓  Communication overhead benchmark complete.")
    return results


# Benchmark 6 — End-to-end workflow
def benchmark_end_to_end(ckks: CKKSCrypto,
                          data: np.ndarray,
                          num_clients: int = 5,
                          samples_per_client: int = 1_000) -> Dict:
    # Time each phase of a complete FL round: setup, local computation, encryption, communication (submission), aggregation.
    print("\n" + "=" * 70)
    print("BENCHMARK 6: END-TO-END WORKFLOW")
    print(f"  clients = {num_clients} | samples/client = {samples_per_client:,}")
    print("=" * 70)

    total_start = time.perf_counter()

    # Phase 1 — Crypto context (already set up; measure a fresh setup)
    print("\n  Phase 1: Crypto setup")
    t0 = time.perf_counter()
    _ckks_fresh = CKKSCrypto()
    _ckks_fresh.setup()
    phase_setup = time.perf_counter() - t0
    print(f"    {phase_setup:.3f} s")

    # Phase 2 — Generate client datasets
    print("  Phase 2: Client data generation")
    t0 = time.perf_counter()
    client_datasets = [np.resize(data, samples_per_client).astype(np.float64)
                       for _ in range(num_clients)]
    phase_data_gen = time.perf_counter() - t0
    print(f"    {phase_data_gen:.3f} s")

    # Phase 3 — Local computation
    print("  Phase 3: Local computation")
    t0 = time.perf_counter()
    local_stats = [
        {'sum': float(np.sum(d)), 'count': len(d)}
        for d in client_datasets
    ]
    phase_local = time.perf_counter() - t0
    print(f"    {phase_local:.3f} s")

    # Phase 4 — Client-side encryption
    print("  Phase 4: Encryption")
    t0 = time.perf_counter()
    contributions = [
        {'enc_sum': ckks.encrypt(s['sum']), 'count': s['count']}
        for s in local_stats
    ]
    phase_enc = time.perf_counter() - t0
    print(f"    {phase_enc:.3f} s  ({phase_enc / num_clients * 1_000:.2f} ms/client)")

    # Phase 5 — Simulated submission (serialise / deserialise round-trip)
    print("  Phase 5: Communication (serialisation round-trip)")
    t0 = time.perf_counter()
    for c in contributions:
        _ = pickle.loads(pickle.dumps(c['enc_sum']))
    phase_comm = time.perf_counter() - t0
    print(f"    {phase_comm:.3f} s")

    # Phase 6 — Server aggregation + decryption
    print("  Phase 6: Aggregation + decryption")
    t0 = time.perf_counter()
    agg_sum = contributions[0]['enc_sum']
    total_count = contributions[0]['count']
    for c in contributions[1:]:
        agg_sum = ckks.add_encrypted(agg_sum, c['enc_sum'])
        total_count += c['count']
    global_sum = ckks.decrypt(agg_sum)
    global_mean = global_sum / total_count
    phase_agg = time.perf_counter() - t0
    print(f"    {phase_agg:.3f} s")

    total_time = time.perf_counter() - total_start

    phases = {
        'crypto_setup':        phase_setup,
        'client_data_gen':     phase_data_gen,
        'local_computation':   phase_local,
        'encryption':          phase_enc,
        'communication':       phase_comm,
        'aggregation':         phase_agg,
    }

    print(f"\n  Total: {total_time:.3f} s")
    print("  Breakdown:")
    for name, t in phases.items():
        print(f"    {name:<22}: {t:.3f} s  ({t / total_time * 100:.1f}%)")

    print("\n✓  End-to-end benchmark complete.")
    return {
        'num_clients': num_clients,
        'samples_per_client': samples_per_client,
        'total_samples': num_clients * samples_per_client,
        'phases': {k: round(v, 6) for k, v in phases.items()},
        'total_time': round(total_time, 6),
        'global_mean': round(float(global_mean), 6),
    }


def save_json(data: Dict, filename: str) -> None:
    path = RESULTS_DIR / filename
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f"...  Saved → {path}")


def save_markdown_report(all_results: Dict) -> None:
    path = RESULTS_DIR / 'CKKS_BENCHMARK_REPORT.md'
    lines = [
        "# CKKS Benchmarking Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    # Encryption
    r = all_results.get('encryption', {})
    if r:
        lines += [
            "## 1. Encryption Performance\n",
            "| Data Size | Time (s) | Throughput (values/s) | Chunks |",
            "|----------:|---------:|----------------------:|-------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['encryption_times'][i]:.4f} | "
                f"{r['throughput_values_per_sec'][i]:,.0f} | "
                f"{r['num_chunks'][i]} |"
            )
        lines.append("")

    # Decryption
    r = all_results.get('decryption', {})
    if r:
        lines += [
            "## 2. Decryption Performance\n",
            "| Data Size | Time (s) | Throughput (values/s) |",
            "|----------:|---------:|----------------------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['decryption_times'][i]:.4f} | "
                f"{r['throughput_values_per_sec'][i]:,.0f} |"
            )
        lines.append("")

    # Scalability
    r = all_results.get('scalability', {})
    if r:
        lines += [
            "## 3. Aggregation Scalability\n",
            "| Clients | Agg Time (s) | Per-Client (ms) | Total Samples |",
            "|--------:|-------------:|----------------:|--------------:|",
        ]
        for i, n in enumerate(r['num_clients']):
            lines.append(
                f"| {n} | {r['aggregation_times'][i]:.4f} | "
                f"{r['per_client_overhead_ms'][i]:.3f} | "
                f"{r['total_samples'][i]:,} |"
            )
        lines.append("")

    # Multiplication (opt-in)
    r = all_results.get('multiplication', {})
    if r:
        lines += [
            "## 4. Homomorphic Multiplication Performance\n",
            f"- **add\\_encrypted baseline**: {r['add_baseline_time_ms']} ms",
            f"- **Relinearization cost**: {r['relinearization_time_ms']} ms",
            f"- **Depth limit**: {r['depth_limit'] or 'not reached within tested range'}",
            "",
            "| Depth | Single Op (ms) | Cumulative (s) | vs add\\_encrypted |",
            "|------:|---------------:|---------------:|------------------:|",
        ]
        for i, depth in enumerate(r['mul_depths_tested']):
            lines.append(
                f"| {depth} | {r['single_op_time_ms'][i]:.4f} | "
                f"{r['cumulative_time_s'][i]:.4f} | "
                f"{r['overhead_vs_add'][i]:.1f}x |"
            )
        if r['single_op_time_ms']:
            avg = round(float(np.mean(r['single_op_time_ms'])), 4)
            avg_ratio = round(avg / r['add_baseline_time_ms'], 1)
            lines += [
                "",
                f"**Average multiply\\_encrypted**: {avg} ms  ",
                f"**Average overhead vs add**: {avg_ratio}x  ",
            ]
        lines.append("")

    # Communication
    r = all_results.get('communication', {})
    if r:
        lines += [
            "## 5. Communication Overhead\n",
            "| Data Size | Plaintext (KB) | Ciphertext (KB) | Overhead | # Ciphertexts |",
            "|----------:|---------------:|----------------:|---------:|--------------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['plaintext_bytes'][i]/1_024:.1f} | "
                f"{r['ciphertext_bytes'][i]/1_024:.1f} | "
                f"{r['overhead_ratio'][i]:.1f}× | "
                f"{r['num_ciphertexts'][i]} |"
            )
        lines.append("")

    # End-to-end
    r = all_results.get('end_to_end', {})
    if r:
        lines += [
            "## 6. End-to-End Workflow\n",
            f"- **Clients**: {r['num_clients']}",
            f"- **Samples / client**: {r['samples_per_client']:,}",
            f"- **Total time**: {r['total_time']:.3f} s",
            f"- **Global mean** (decrypted): {r['global_mean']}",
            "### Phase Breakdown\n",
            "| Phase | Time (s) | Share (%) |",
            "|:------|----------:|----------:|",
        ]
        total = r['total_time']
        for phase, t in r['phases'].items():
            lines.append(f"| {phase} | {t:.4f} | {t/total*100:.1f} |")
        lines.append("")

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f"...  Report saved to: {path}")



def main():
    print("=" * 70)
    print("CKKS DATA PROCESSING — BENCHMARKING SUITE")
    print("=" * 70)
    print(f"  Multiplication benchmark: "
          f"{'ENABLED' if RUN_MULTIPLICATION_BENCHMARK else 'DISABLED'} "
          f"(set RUN_MULTIPLICATION_BENCHMARK to change)")

    # ── Configuration ────────────────────────────────────────────────────────
    DATASET_FILENAME   = 'Final_data.csv'
    TARGET_COLUMN      = 'Height (m)'
    ENCRYPTION_SIZES   = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    DECRYPTION_SIZES   = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    COMM_SIZES         = [100, 500, 1_000, 5_000, 10_000, 50_000]
    CLIENT_COUNTS      = [2, 5, 10, 20, 50, 100, 200]
    MUL_DEPTHS         = [1, 2, 3, 4]
    E2E_CLIENTS        = 20
    E2E_SAMPLES        = 1_000
    SAMPLES_PER_CLIENT = 1_000

    # ── Initialise cryptosystem ──────────────────────────────────────────────
    print("\nInitialising CKKS cryptosystem...")
    ckks = CKKSCrypto()
    ckks.setup()
    print(f"✓  SIMD slots: {ckks.n // 2:,}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"\n--- Loading dataset ---")
    df   = load_csv(DATASET_FILENAME)
    data = extract_column(df, TARGET_COLUMN)
    print(f"✓  Using column '{TARGET_COLUMN}': {len(data):,} values")

    all_results: Dict = {}

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results['encryption'] = benchmark_encryption(
        ckks, data, ENCRYPTION_SIZES)

    all_results['decryption'] = benchmark_decryption(
        ckks, data, DECRYPTION_SIZES)

    all_results['scalability'] = benchmark_aggregation_scalability(
        ckks, data, CLIENT_COUNTS, SAMPLES_PER_CLIENT)

    if RUN_MULTIPLICATION_BENCHMARK:
        all_results['multiplication'] = benchmark_multiplication(
            ckks, data, MUL_DEPTHS)

    all_results['communication'] = benchmark_communication_overhead(
        ckks, data, COMM_SIZES)

    all_results['end_to_end'] = benchmark_end_to_end(
        ckks, data, E2E_CLIENTS, E2E_SAMPLES)

    # ── Persist results ──────────────────────────────────────────────────────
    print("\n--- Saving results ---")
    save_json(all_results['encryption'],    'ckks_encryption_benchmark.json')
    save_json(all_results['decryption'],    'ckks_decryption_benchmark.json')
    save_json(all_results['scalability'],   'ckks_scalability_benchmark.json')
    if RUN_MULTIPLICATION_BENCHMARK:
        save_json(all_results['multiplication'], 'ckks_multiplication_benchmark.json')
    save_json(all_results['communication'], 'ckks_communication_benchmark.json')
    save_json(all_results['end_to_end'],    'ckks_end_to_end_benchmark.json')
    save_json(all_results,                  'ckks_all_benchmarks.json')
    save_markdown_report(all_results)

    print("\n" + "=" * 70)
    print("✅  CKKS BENCHMARKING COMPLETE")
    print("=" * 70)
    print(f"\nArtefacts saved in: {RESULTS_DIR.absolute()}/")
    print("  ckks_encryption_benchmark.json")
    print("  ckks_decryption_benchmark.json")
    print("  ckks_scalability_benchmark.json")
    if RUN_MULTIPLICATION_BENCHMARK:
        print("  ckks_multiplication_benchmark.json")
    print("  ckks_communication_benchmark.json")
    print("  ckks_end_to_end_benchmark.json")
    print("  ckks_all_benchmarks.json")
    print("  CKKS_BENCHMARK_REPORT.md")


if __name__ == "__main__":
    main()