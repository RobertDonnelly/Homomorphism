"""
CKKS Data Processing — Benchmarking Suite
==========================================
Loads real-world CSV datasets and benchmarks CKKS encryption performance:
  - Encryption throughput across data sizes
  - Decryption throughput across data sizes
  - Aggregation scalability across client counts
  - Communication overhead (ciphertext vs plaintext size)
  - End-to-end federated learning round timing

Analysis outputs (statistical summaries, polynomial stats, range analysis,
correlation analysis, FL simulation results) have been removed; all outputs
are benchmarking artefacts only.
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

from src.schemes.ckks.ckks_crypto import CKKSCrypto


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = project_root / 'results' / 'ckks_benchmark'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/raw/ and return a DataFrame."""
    filepath = project_root / 'data' / 'raw' / filename
    if not filepath.exists():
        print(f"\n❌  File not found: {filepath}")
        print(f"    Place '{filename}' in data/raw/ and retry.")
        sys.exit(1)
    print(f"📂  Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓   {len(df):,} rows × {len(df.columns)} columns  |  "
          f"columns: {list(df.columns)}")
    return df


def extract_column(df: pd.DataFrame, column: str) -> np.ndarray:
    """Validate column exists and return a float64 array with NaNs filled."""
    if column not in df.columns:
        print(f"\n❌  Column '{column}' not found.")
        print(f"    Available: {list(df.columns)}")
        sys.exit(1)
    return df[column].fillna(0).astype(np.float64).values


# ---------------------------------------------------------------------------
# Helper: chunk a 1-D array into SIMD-sized pieces
# ---------------------------------------------------------------------------

def _chunk(data: np.ndarray, max_slots: int) -> List[np.ndarray]:
    n = len(data)
    return [data[i * max_slots: min((i + 1) * max_slots, n)]
            for i in range(int(np.ceil(n / max_slots)))]


# ---------------------------------------------------------------------------
# Benchmark 1 — Encryption performance
# ---------------------------------------------------------------------------

def benchmark_encryption(ckks: CKKSCrypto,
                          data: np.ndarray,
                          data_sizes: List[int]) -> Dict:
    """
    Measure CKKS encryption throughput for a range of data sizes drawn from
    *data*.  For sizes ≤ 1 000 the single-value baseline is also sampled.

    Returns a dict ready to be serialised to JSON.
    """
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
        # Slice or tile source data to the required size
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


# ---------------------------------------------------------------------------
# Benchmark 2 — Decryption performance
# ---------------------------------------------------------------------------

def benchmark_decryption(ckks: CKKSCrypto,
                          data: np.ndarray,
                          data_sizes: List[int]) -> Dict:
    """
    Pre-encrypt chunks then measure decryption throughput for each data size.
    """
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


# ---------------------------------------------------------------------------
# Benchmark 3 — Aggregation scalability
# ---------------------------------------------------------------------------

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

        # Build encrypted contributions for each client — one scalar sum only
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
        # Single decrypt — mirrors the BFV aggregation path exactly
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


# ---------------------------------------------------------------------------
# Benchmark 4 — Communication overhead
# ---------------------------------------------------------------------------

def benchmark_communication_overhead(ckks: CKKSCrypto,
                                      data: np.ndarray,
                                      data_sizes: List[int]) -> Dict:
    """
    Compare plaintext byte size (8 bytes × n float64 values) against the
    serialised ciphertext byte size, yielding an overhead ratio.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 4: COMMUNICATION OVERHEAD")
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
        # Encrypt first chunk to measure per-ciphertext size
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


# ---------------------------------------------------------------------------
# Benchmark 5 — End-to-end workflow
# ---------------------------------------------------------------------------

def benchmark_end_to_end(ckks: CKKSCrypto,
                          data: np.ndarray,
                          num_clients: int = 5,
                          samples_per_client: int = 1_000) -> Dict:
    """
    Time each phase of a complete FL round: setup, local computation,
    encryption, communication (submission), aggregation.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 5: END-TO-END WORKFLOW")
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

    # Phase 3 — Local computation (sum only — mirrors single-scalar BFV path)
    print("  Phase 3: Local computation")
    t0 = time.perf_counter()
    local_stats = [
        {'sum': float(np.sum(d)), 'count': len(d)}
        for d in client_datasets
    ]
    phase_local = time.perf_counter() - t0
    print(f"    {phase_local:.3f} s")

    # Phase 4 — Client-side encryption (single scalar sum per client)
    print("  Phase 4: Encryption")
    t0 = time.perf_counter()
    contributions = [
        {'enc_sum': ckks.encrypt(s['sum']), 'count': s['count']}
        for s in local_stats
    ]
    phase_enc = time.perf_counter() - t0
    print(f"    {phase_enc:.3f} s  ({phase_enc / num_clients * 1_000:.2f} ms/client)")

    # Phase 5 — Simulated submission (deserialise / reserialise round-trip)
    print("  Phase 5: Communication (serialisation round-trip)")
    t0 = time.perf_counter()
    for c in contributions:
        _ = pickle.loads(pickle.dumps(c['enc_sum']))
    #_ = pickle.loads(pickle.dumps(c['enc_sum_sq']))
    phase_comm = time.perf_counter() - t0
    print(f"    {phase_comm:.3f} s")

    # Phase 6 — Server aggregation + decryption
    print("  Phase 6: Aggregation + decryption")
    t0 = time.perf_counter()
    agg_sum = contributions[0]['enc_sum']
    #agg_sum_sq = contributions[0]['enc_sum_sq']
    total_count = contributions[0]['count']
    for c in contributions[1:]:
        agg_sum = ckks.add_encrypted(agg_sum, c['enc_sum'])
        #agg_sum_sq = ckks.add_encrypted(agg_sum_sq, c['enc_sum_sq'])
        total_count += c['count']
    global_sum = ckks.decrypt(agg_sum)
    #global_sum_sq = ckks.decrypt(agg_sum_sq)
    global_mean = global_sum / total_count
    #global_var = (global_sum_sq / total_count) - global_mean ** 2
    #global_std = float(np.sqrt(abs(global_var)))
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
        #'global_std': round(global_std, 6),
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_json(data: Dict, filename: str) -> None:
    path = RESULTS_DIR / filename
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f"💾  Saved → {path}")


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

    # Communication
    r = all_results.get('communication', {})
    if r:
        lines += [
            "## 4. Communication Overhead\n",
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
            "## 5. End-to-End Workflow\n",
            f"- **Clients**: {r['num_clients']}",
            f"- **Samples / client**: {r['samples_per_client']:,}",
            f"- **Total time**: {r['total_time']:.3f} s",
            f"- **Global mean** (decrypted): {r['global_mean']}",
            #f"- **Global std** (decrypted): {r['global_std']}\n",
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
    print("CKKS DATA PROCESSING — BENCHMARKING SUITE")
    print("=" * 70)

    # ── Configuration ───────────────────────────────────────────────────────
    DATASET_FILENAME   = 'Final_data.csv'
    TARGET_COLUMN      = 'Height (m)'
    ENCRYPTION_SIZES   = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]  # CKKS only
    DECRYPTION_SIZES   = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]  # CKKS only
    COMM_SIZES         = [100, 500, 1_000, 5_000, 10_000, 50_000]                   # CKKS only
    CLIENT_COUNTS      = [2, 5, 10, 20, 50, 100, 200]                               # all schemes
    E2E_CLIENTS        = 20                                                          # all schemes
    E2E_SAMPLES        = 1_000
    SAMPLES_PER_CLIENT = 1_000

    # ── Initialise cryptosystem ─────────────────────────────────────────────
    print("\nInitialising CKKS cryptosystem...")
    ckks = CKKSCrypto()
    ckks.setup()
    print(f"✓  SIMD slots: {ckks.n // 2:,}")

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"\n--- Loading dataset ---")
    df   = load_csv(DATASET_FILENAME)
    data = extract_column(df, TARGET_COLUMN)
    print(f"✓  Using column '{TARGET_COLUMN}': {len(data):,} values")

    all_results: Dict = {}

    # ── Run benchmarks ──────────────────────────────────────────────────────
    all_results['encryption'] = benchmark_encryption(
        ckks, data, ENCRYPTION_SIZES)

    all_results['decryption'] = benchmark_decryption(
        ckks, data, DECRYPTION_SIZES)

    all_results['scalability'] = benchmark_aggregation_scalability(
        ckks, data, CLIENT_COUNTS, SAMPLES_PER_CLIENT)

    all_results['communication'] = benchmark_communication_overhead(
        ckks, data, COMM_SIZES)

    all_results['end_to_end'] = benchmark_end_to_end(
        ckks, data, E2E_CLIENTS, E2E_SAMPLES)

    # ── Persist results ─────────────────────────────────────────────────────
    print("\n--- Saving results ---")
    save_json(all_results['encryption'],    'ckks_encryption_benchmark.json')
    save_json(all_results['decryption'],    'ckks_decryption_benchmark.json')
    save_json(all_results['scalability'],   'ckks_scalability_benchmark.json')
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
    print("  ckks_communication_benchmark.json")
    print("  ckks_end_to_end_benchmark.json")
    print("  ckks_all_benchmarks.json")
    print("  CKKS_BENCHMARK_REPORT.md")


if __name__ == "__main__":
    main()