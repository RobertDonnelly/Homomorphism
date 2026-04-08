"""
BFV Data Processing — Benchmarking Suite
=========================================
Loads real-world CSV datasets and benchmarks BFV encryption performance:
  - Encryption throughput across data sizes
  - Decryption throughput across data sizes
  - Aggregation scalability across client counts (opt-out)
  - Homomorphic multiplication performance (opt-in)
  - Communication overhead (ciphertext vs plaintext size)
  - End-to-end federated learning round timing
"""

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

from src.schemes.bfv.bfv_crypto import BFVCrypto

# Set to True to run the multiplication benchmark.
RUN_MULTIPLICATION_BENCHMARK = True

# Set to False to skip the aggregation scalability benchmark.
RUN_SCALABILITY_BENCHMARK = True

BFV_N      = 2**14
BFV_T_BITS = 17
BFV_SEC    = 128


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = project_root / 'results' / 'bfv_benchmark'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Integer scaling factor used when converting float data to int64 for BFV
SCALE_FACTOR = 1_000


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/raw/ and return a DataFrame."""
    filepath = project_root / 'data' / 'raw' / filename
    if not filepath.exists():
        print(f"\n!!  File not found: {filepath}")
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
        print(f"\n!!  Column '{column}' not found.")
        print(f"    Available: {list(df.columns)}")
        sys.exit(1)
    return df[column].fillna(0).astype(np.float64).values


def to_int64(data: np.ndarray, scale: int = SCALE_FACTOR) -> np.ndarray:
    """Scale floats and cast to int64 for BFV encryption."""
    return (data * scale).astype(np.int64)


# ---------------------------------------------------------------------------
# Benchmark 1 — Encryption performance
# ---------------------------------------------------------------------------

def benchmark_encryption(bfv: BFVCrypto,
                          data: np.ndarray,
                          data_sizes: List[int]) -> Dict:
    """
    Measure BFV encryption throughput for a range of data sizes drawn from
    *data*.

    BFV encrypts values element-wise (no SIMD batching), so this benchmark
    uses a sample of up to MAX_SAMPLE values per size to keep wall-clock
    time manageable, then extrapolates throughput.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 1: ENCRYPTION PERFORMANCE")
    print("=" * 70)

    MAX_SAMPLE = 200

    results = {
        'data_sizes': data_sizes,
        'encryption_times': [],
        'throughput_values_per_sec': [],
        'timed_sample_size': [],
        'single_value_time_ms': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = to_int64(np.resize(data, size))

        probe_size = min(size, MAX_SAMPLE)
        probe = sample[:probe_size]

        t0 = time.perf_counter()
        for v in probe:
            _ = bfv.encrypt(int(v))
        elapsed_probe = time.perf_counter() - t0

        sv_ms = elapsed_probe / probe_size * 1_000 if elapsed_probe > 0 else 0.0
        enc_time = elapsed_probe / probe_size * size if elapsed_probe > 0 else 0.0
        throughput = size / enc_time if enc_time > 0 else float('inf')

        results['encryption_times'].append(round(enc_time, 6))
        results['throughput_values_per_sec'].append(round(throughput, 2))
        results['timed_sample_size'].append(probe_size)
        results['single_value_time_ms'].append(round(sv_ms, 4))

        print(f"    single-value: {sv_ms:.3f} ms | "
              f"extrapolated total: {enc_time:.3f} s | "
              f"{throughput:,.0f} values/s  "
              f"(timed {probe_size} values)")

    print("\n✓  Encryption benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 2 — Decryption performance
# ---------------------------------------------------------------------------

def benchmark_decryption(bfv: BFVCrypto,
                          data: np.ndarray,
                          data_sizes: List[int]) -> Dict:
    """
    Pre-encrypt a fixed probe set then measure decryption throughput.
    Extrapolates to the full data size in the same way as the encryption
    benchmark.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: DECRYPTION PERFORMANCE")
    print("=" * 70)

    MAX_SAMPLE = 200

    results = {
        'data_sizes': data_sizes,
        'decryption_times': [],
        'throughput_values_per_sec': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = to_int64(np.resize(data, size))

        probe_size = min(size, MAX_SAMPLE)
        probe = sample[:probe_size]

        # Pre-encrypt probe (not timed)
        enc_probe = [bfv.encrypt(int(v)) for v in probe]

        # ── Timed decryption ──────────────────────────────────────────────
        t0 = time.perf_counter()
        for enc in enc_probe:
            _ = bfv.decrypt(enc)
        elapsed_probe = time.perf_counter() - t0

        dec_time = elapsed_probe / probe_size * size if elapsed_probe > 0 else 0.0
        throughput = size / dec_time if dec_time > 0 else float('inf')

        results['decryption_times'].append(round(dec_time, 6))
        results['throughput_values_per_sec'].append(round(throughput, 2))

        print(f"    dec: {dec_time:.3f} s (extrapolated) | "
              f"{throughput:,.0f} values/s")

    print("\n✓  Decryption benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 3 — Aggregation scalability
# ---------------------------------------------------------------------------

def benchmark_aggregation_scalability(bfv: BFVCrypto,
                                       data: np.ndarray,
                                       num_clients_list: List[int],
                                       samples_per_client: int = 1_000) -> Dict:
    """
    Measure server-side homomorphic aggregation time as the number of clients
    scales.  Each client submits an encrypted local sum computed via
    BFV homomorphic addition over their sample.
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

        client_enc_sums = []
        for _ in range(n_clients):
            client_data = to_int64(np.resize(data, samples_per_client))
            enc_vals = bfv.encrypt_vector(client_data)
            enc_sum = bfv.sum_encrypted(enc_vals)
            client_enc_sums.append(enc_sum)

        # ── Timed server aggregation ──────────────────────────────────────
        t0 = time.perf_counter()
        global_enc_sum = bfv.sum_encrypted(client_enc_sums)
        _ = bfv.decrypt(global_enc_sum)
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


# ---------------------------------------------------------------------------
# Benchmark 4 — Homomorphic multiplication (opt-in)
# ---------------------------------------------------------------------------

def benchmark_multiplication(bfv: BFVCrypto,
                              data: np.ndarray,
                              num_multiplications: List[int] = None) -> Dict:
    """
    Benchmark multiply_encrypted performance across increasing multiplication
    counts.

    BFV uses exact integer arithmetic with no modulus chain, but each
    multiplication consumes noise budget. This benchmark measures:
      - Single add_encrypted call as a baseline for comparison
      - Single multiply_encrypted call in isolation
      - Per-operation time at each multiplication count
      - Cumulative time to reach each count
      - Whether decryption remains correct at each depth (noise budget check)
      - The first depth at which decryption produces an incorrect result

    Args:
        bfv: Initialised BFVCrypto instance.
        data: Source data array (used to draw sample scalar values).
        num_multiplications: List of sequential multiplication counts to test.

    Returns a dict ready to be serialised to JSON.
    """
    if num_multiplications is None:
        num_multiplications = [1, 2, 3, 4, 5]

    print("\n" + "=" * 70)
    print("BENCHMARK 4: HOMOMORPHIC MULTIPLICATION PERFORMANCE")
    print("  Note: each multiplication consumes BFV noise budget.")
    print("  Decryption correctness is verified at each depth.")
    print(f"  Testing depths: {num_multiplications}")
    print("=" * 70)

    results = {
        'mul_depths_tested': [],
        'single_op_time_ms': [],
        'cumulative_time_s': [],
        'add_baseline_time_ms': None,
        'overhead_vs_add': [],
        'decryption_correct': [],
        'noise_budget_exhausted_at': None,
    }

    # Use small fixed integers that stay within the plaintext modulus t
    # after repeated multiplication across all tested depths.
    # With t_bits=17 (t=131,072): val_a * val_b^5 = 3 * 4^5 = 3,072 — well within t.
    # Do NOT use scaled real-world data values here — with SCALE_FACTOR=1000
    # height values become ~1500-2000, and 1500 * 2000^1 = 3,000,000 which
    # overflows t=131,072 and causes incorrect results that mimic noise failure.
    val_a = 3
    val_b = 4

    # ── Baseline: single add_encrypted ───────────────────────────────────
    enc_a = bfv.encrypt(val_a)
    enc_b = bfv.encrypt(val_b)
    t0 = time.perf_counter()
    _ = bfv.add_encrypted(enc_a, enc_b)
    add_ms = (time.perf_counter() - t0) * 1_000
    results['add_baseline_time_ms'] = round(add_ms, 4)
    print(f"\n  Baseline — add_encrypted:      {add_ms:.4f} ms")

    # ── Single multiply_encrypted in isolation ────────────────────────────
    enc_x = bfv.encrypt(val_a)
    enc_y = bfv.encrypt(val_b)
    t0 = time.perf_counter()
    _ = bfv.multiply_encrypted(enc_x, enc_y)
    single_mul_ms = (time.perf_counter() - t0) * 1_000
    print(f"  Single multiply_encrypted:     {single_mul_ms:.4f} ms")
    print(f"  Overhead vs add:               {single_mul_ms / add_ms:.1f}x\n")

    # ── Depth sweep ───────────────────────────────────────────────────────
    print(f"  {'Depth':<8} {'Single op (ms)':<18} {'Cumulative (s)':<18} "
          f"{'vs add':<10} {'Correct':<10} Status")
    print(f"  {'-'*8} {'-'*18} {'-'*18} {'-'*10} {'-'*10} {'-'*8}")

    noise_budget_exhausted_at = None
    cumulative_start = time.perf_counter()

    for depth in num_multiplications:
        try:
            # Chain (depth-1) multiplications to reach the right noise level,
            # then time the final one in isolation
            enc_running_a = bfv.encrypt(val_a)
            enc_running_b = bfv.encrypt(val_b)
            for _ in range(depth - 1):
                enc_running_a = bfv.multiply_encrypted(
                    enc_running_a, bfv.encrypt(val_b))

            t0 = time.perf_counter()
            enc_result = bfv.multiply_encrypted(enc_running_a, enc_running_b)
            op_ms = (time.perf_counter() - t0) * 1_000

            cumulative_s = time.perf_counter() - cumulative_start

            # Noise budget / correctness check
            try:
                decrypted = bfv.decrypt(enc_result)
                expected = val_a * (val_b ** depth)
                correct = (decrypted == expected)
            except Exception:
                correct = False

            ratio = op_ms / add_ms

            results['mul_depths_tested'].append(depth)
            results['single_op_time_ms'].append(round(op_ms, 4))
            results['cumulative_time_s'].append(round(cumulative_s, 6))
            results['overhead_vs_add'].append(round(ratio, 2))
            results['decryption_correct'].append(correct)

            status = "OK" if correct else "NOISE EXCEEDED"
            print(f"  {depth:<8} {op_ms:<18.4f} {cumulative_s:<18.4f} "
                  f"{ratio:<10.1f} {'Yes' if correct else 'No':<10} {status}")

            if not correct and noise_budget_exhausted_at is None:
                noise_budget_exhausted_at = depth

        except Exception as e:
            print(f"  {depth:<8} {'---':<18} {'---':<18} "
                  f"{'---':<10} {'---':<10} FAILED: {e}")
            if noise_budget_exhausted_at is None:
                noise_budget_exhausted_at = depth
            break

    results['noise_budget_exhausted_at'] = noise_budget_exhausted_at

    if noise_budget_exhausted_at:
        print(f"\n  ⚠  Noise budget exhausted at depth {noise_budget_exhausted_at}.")
    else:
        print(f"\n  ✓  Decryption correct at all tested depths.")

    if results['single_op_time_ms']:
        avg = np.mean(results['single_op_time_ms'])
        print(f"\n  avg multiply_encrypted: {avg:.4f} ms  |  "
              f"avg overhead vs add: {avg / add_ms:.1f}x")

    print("\n✓  Multiplication benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 5 — Communication overhead
# ---------------------------------------------------------------------------

def benchmark_communication_overhead(bfv: BFVCrypto,
                                      data: np.ndarray,
                                      data_sizes: List[int]) -> Dict:
    """
    Compare plaintext byte size (8 bytes × n float64 values) against the
    serialised ciphertext byte size.  Because BFV encrypts element-wise,
    we time a small probe and extrapolate total ciphertext bytes.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 5: COMMUNICATION OVERHEAD")
    print("=" * 70)

    MAX_SAMPLE = 20

    results = {
        'data_sizes': data_sizes,
        'plaintext_bytes': [],
        'ciphertext_bytes': [],
        'overhead_ratio': [],
        'num_ciphertexts': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = to_int64(np.resize(data, size))
        plaintext_bytes = size * 8   # float64 = 8 bytes

        probe = sample[:MAX_SAMPLE]
        enc_probe = [bfv.encrypt(int(v)) for v in probe]
        bytes_per_ctxt = max(len(pickle.dumps(e)) for e in enc_probe)
        total_ctxt_bytes = bytes_per_ctxt * size
        overhead = total_ctxt_bytes / plaintext_bytes

        results['plaintext_bytes'].append(plaintext_bytes)
        results['ciphertext_bytes'].append(total_ctxt_bytes)
        results['overhead_ratio'].append(round(overhead, 2))
        results['num_ciphertexts'].append(size)

        print(f"    plaintext: {plaintext_bytes / 1_024:.1f} KB | "
              f"ciphertext: {total_ctxt_bytes / 1_024:.1f} KB | "
              f"overhead: {overhead:.1f}× | "
              f"{size:,} ciphertexts")

    print("\n..  Communication overhead benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 6 — End-to-end workflow
# ---------------------------------------------------------------------------

def benchmark_end_to_end(bfv: BFVCrypto,
                          data: np.ndarray,
                          num_clients: int = 5,
                          samples_per_client: int = 1_000) -> Dict:
    """
    Time each phase of a complete BFV FL round.  Client encryption uses
    encrypt_vector (element-wise) to stay consistent with the BFV pipeline.
    The crypto setup phase uses the same parameters as the rest of the run
    so the timing is representative.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 6: END-TO-END WORKFLOW")
    print(f"  clients = {num_clients} | samples/client = {samples_per_client:,}")
    print("=" * 70)

    total_start = time.perf_counter()

    # Phase 1 — Crypto context setup (uses same params as the rest of the run)
    print("\n  Phase 1: Crypto setup")
    t0 = time.perf_counter()
    _bfv_fresh = BFVCrypto(
        n=BFV_N, t_bits=BFV_T_BITS, sec=BFV_SEC)
    _bfv_fresh.setup()
    phase_setup = time.perf_counter() - t0
    print(f"    {phase_setup:.3f} s")

    # Phase 2 — Generate client datasets
    print("  Phase 2: Client data generation")
    t0 = time.perf_counter()
    client_datasets = [
        to_int64(np.resize(data, samples_per_client))
        for _ in range(num_clients)
    ]
    phase_data_gen = time.perf_counter() - t0
    print(f"    {phase_data_gen:.3f} s")

    # Phase 3 — Local computation (integer sum via numpy, pre-encryption)
    print("  Phase 3: Local computation")
    t0 = time.perf_counter()
    local_sums_plain = [int(np.sum(d)) for d in client_datasets]
    phase_local = time.perf_counter() - t0
    print(f"    {phase_local:.3f} s")

    # Phase 4 — Client-side encryption (encrypt each local sum as a scalar)
    print("  Phase 4: Encryption")
    t0 = time.perf_counter()
    enc_sums = [bfv.encrypt(s) for s in local_sums_plain]
    phase_enc = time.perf_counter() - t0
    print(f"    {phase_enc:.3f} s  ({phase_enc / num_clients * 1_000:.2f} ms/client)")

    # Phase 5 — Simulated communication (serialisation round-trip)
    print("  Phase 5: Communication (serialisation round-trip)")
    t0 = time.perf_counter()
    for enc in enc_sums:
        _ = pickle.loads(pickle.dumps(enc))
    phase_comm = time.perf_counter() - t0
    print(f"    {phase_comm:.3f} s")

    # Phase 6 — Server aggregation + decryption
    print("  Phase 6: Aggregation + decryption")
    t0 = time.perf_counter()
    global_enc_sum = bfv.sum_encrypted(enc_sums)
    decrypted_global_sum = bfv.decrypt(global_enc_sum)
    total_count = num_clients * samples_per_client
    global_mean = decrypted_global_sum / (total_count * SCALE_FACTOR)
    phase_agg = time.perf_counter() - t0
    print(f"    {phase_agg:.3f} s")

    total_time = time.perf_counter() - total_start

    phases = {
        'crypto_setup':       phase_setup,
        'client_data_gen':    phase_data_gen,
        'local_computation':  phase_local,
        'encryption':         phase_enc,
        'communication':      phase_comm,
        'aggregation':        phase_agg,
    }

    print(f"\n  Total: {total_time:.3f} s")
    print("  Breakdown:")
    for name, t in phases.items():
        print(f"    {name:<22}: {t:.3f} s  ({t / total_time * 100:.1f}%)")

    print("\n✓  End-to-end benchmark complete.")
    return {
        'num_clients': num_clients,
        'samples_per_client': samples_per_client,
        'total_samples': total_count,
        'phases': {k: round(v, 6) for k, v in phases.items()},
        'total_time': round(total_time, 6),
        'global_mean': round(float(global_mean), 6),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_json(data: Dict, filename: str) -> None:
    path = RESULTS_DIR / filename
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f"...  Saved to: {path}")


def save_markdown_report(all_results: Dict) -> None:
    path = RESULTS_DIR / 'BFV_BENCHMARK_REPORT.md'
    lines = [
        "# BFV Benchmarking Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"> Integer scale factor: ×{SCALE_FACTOR}  "
        "(floats multiplied before int64 conversion)\n",
        "> Encryption/decryption times for sizes > 200 are extrapolated "
        "from a timed probe.\n",
        f"> BFV parameters: n={BFV_N}, t_bits={BFV_T_BITS}, sec={BFV_SEC}\n",
    ]

    # Encryption
    r = all_results.get('encryption', {})
    if r:
        lines += [
            "## 1. Encryption Performance\n",
            "| Data Size | Est. Time (s) | Throughput (values/s) | ms/value |",
            "|----------:|--------------:|----------------------:|---------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['encryption_times'][i]:.4f} | "
                f"{r['throughput_values_per_sec'][i]:,.0f} | "
                f"{r['single_value_time_ms'][i]:.3f} |"
            )
        lines.append("")

    # Decryption
    r = all_results.get('decryption', {})
    if r:
        lines += [
            "## 2. Decryption Performance\n",
            "| Data Size | Est. Time (s) | Throughput (values/s) |",
            "|----------:|--------------:|----------------------:|",
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
            "> BFV uses exact integer arithmetic. Noise budget (not modulus "
            "chain depth) is the limiting factor for chained multiplications.\n",
            f"- **add\\_encrypted baseline**: {r['add_baseline_time_ms']} ms",
            f"- **Noise budget exhausted at**: "
            f"{r['noise_budget_exhausted_at'] or 'not reached within tested range'}",
            "",
            "| Depth | Single Op (ms) | Cumulative (s) | vs add\\_encrypted | Correct |",
            "|------:|---------------:|---------------:|------------------:|--------:|",
        ]
        for i, depth in enumerate(r['mul_depths_tested']):
            correct_str = "Yes" if r['decryption_correct'][i] else "No"
            lines.append(
                f"| {depth} | {r['single_op_time_ms'][i]:.4f} | "
                f"{r['cumulative_time_s'][i]:.4f} | "
                f"{r['overhead_vs_add'][i]:.1f}x | "
                f"{correct_str} |"
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
                f"{r['num_ciphertexts'][i]:,} |"
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
            f"- **Global mean** (decrypted & descaled): {r['global_mean']}\n",
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
    print(f"📄  Report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BFV DATA PROCESSING — BENCHMARKING SUITE")
    print("=" * 70)
    print(f"  Multiplication benchmark: "
          f"{'ENABLED' if RUN_MULTIPLICATION_BENCHMARK else 'DISABLED'} "
          f"(set RUN_MULTIPLICATION_BENCHMARK to change)")
    print(f"  Scalability benchmark: "
          f"{'ENABLED' if RUN_SCALABILITY_BENCHMARK else 'DISABLED'} "
          f"(set RUN_SCALABILITY_BENCHMARK to change)")
    print(f"  BFV parameters: n={BFV_N}, t_bits={BFV_T_BITS}, sec={BFV_SEC}")

    # ── Configuration ────────────────────────────────────────────────────────
    DATASET_FILENAME   = 'Final_data.csv'
    TARGET_COLUMN      = 'Height (m)'
    ENCRYPTION_SIZES   = [100, 500, 1_000, 5_000, 10_000, 20_000]
    DECRYPTION_SIZES   = [100, 500, 1_000, 5_000, 10_000, 20_000]
    COMM_SIZES         = [100, 500, 1_000, 5_000, 10_000]
    CLIENT_COUNTS      = [2, 5, 10, 20, 50, 100, 200]
    MUL_DEPTHS         = [1, 2, 3, 4, 5]
    SAMPLES_PER_CLIENT = 1_000
    E2E_CLIENTS        = 20
    E2E_SAMPLES        = 1_000

    # ── Initialise cryptosystem ──────────────────────────────────────────────
    print("\nInitialising BFV cryptosystem...")
    bfv = BFVCrypto(n=BFV_N, t_bits=BFV_T_BITS, sec=BFV_SEC)
    bfv.setup()
    print("..  BFV ready (element-wise encryption, exact integer arithmetic)")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"\n--- Loading dataset ---")
    df   = load_csv(DATASET_FILENAME)
    data = extract_column(df, TARGET_COLUMN)
    print(f"..  Using column '{TARGET_COLUMN}': {len(data):,} values")

    all_results: Dict = {}

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results['encryption'] = benchmark_encryption(
        bfv, data, ENCRYPTION_SIZES)

    all_results['decryption'] = benchmark_decryption(
        bfv, data, DECRYPTION_SIZES)

    if RUN_SCALABILITY_BENCHMARK:
        all_results['scalability'] = benchmark_aggregation_scalability(
            bfv, data, CLIENT_COUNTS, SAMPLES_PER_CLIENT)

    if RUN_MULTIPLICATION_BENCHMARK:
        all_results['multiplication'] = benchmark_multiplication(
            bfv, data, MUL_DEPTHS)

    all_results['communication'] = benchmark_communication_overhead(
        bfv, data, COMM_SIZES)

    all_results['end_to_end'] = benchmark_end_to_end(
        bfv, data, E2E_CLIENTS, E2E_SAMPLES)

    # ── Persist results ──────────────────────────────────────────────────────
    print("\n--- Saving results ---")
    save_json(all_results['encryption'],    'bfv_encryption_benchmark.json')
    save_json(all_results['decryption'],    'bfv_decryption_benchmark.json')
    if RUN_SCALABILITY_BENCHMARK:
        save_json(all_results['scalability'], 'bfv_scalability_benchmark.json')
    if RUN_MULTIPLICATION_BENCHMARK:
        save_json(all_results['multiplication'], 'bfv_multiplication_benchmark.json')
    save_json(all_results['communication'], 'bfv_communication_benchmark.json')
    save_json(all_results['end_to_end'],    'bfv_end_to_end_benchmark.json')
    save_json(all_results,                  'bfv_all_benchmarks.json')
    save_markdown_report(all_results)

    print("\n" + "=" * 70)
    print("✅  BFV BENCHMARKING COMPLETE")
    print("=" * 70)
    print(f"\nArtefacts saved in: {RESULTS_DIR.absolute()}/")
    print("  bfv_encryption_benchmark.json")
    print("  bfv_decryption_benchmark.json")
    if RUN_SCALABILITY_BENCHMARK:
        print("  bfv_scalability_benchmark.json")
    if RUN_MULTIPLICATION_BENCHMARK:
        print("  bfv_multiplication_benchmark.json")
    print("  bfv_communication_benchmark.json")
    print("  bfv_end_to_end_benchmark.json")
    print("  bfv_all_benchmarks.json")
    print("  BFV_BENCHMARK_REPORT.md")


if __name__ == "__main__":
    main()