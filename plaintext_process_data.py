"""
Plaintext Baseline — Benchmarking Suite
========================================
Mirrors the five benchmark dimensions in ckks_process_data.py and
bfv_process_data.py but operates entirely on unencrypted data using
standard numpy operations.  Results serve as the plaintext baseline
against which CKKS and BFV overhead is measured in Chapter 5.
"""

import sys
import copy
import pickle
import time
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent
RESULTS_DIR  = project_root / 'results' / 'plaintext_benchmark'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Number of repeats for very fast operations (keeps timing stable)
REPEATS = 5


# ---------------------------------------------------------------------------
# Data loader  (identical to the encrypted benchmarks)
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
# Helpers
# ---------------------------------------------------------------------------

# CKKS uses n=2**14, so SIMD capacity = n/2 = 8192.  We replicate the same
# chunk size so results are directly comparable.
SIMD_CHUNK_SIZE = 8_192


def _chunk(data: np.ndarray, chunk_size: int = SIMD_CHUNK_SIZE) -> List[np.ndarray]:
    """Split *data* into chunk_size-length slices (last may be shorter)."""
    n = len(data)
    return [data[i * chunk_size: min((i + 1) * chunk_size, n)]
            for i in range(int(np.ceil(n / chunk_size)))]


def _mean_of_repeats(times: List[float]) -> float:
    return round(sum(times) / len(times), 8)


# ---------------------------------------------------------------------------
# Benchmark 1 — "Write" performance  (plaintext analogue of encryption)
# ---------------------------------------------------------------------------

def benchmark_write(data: np.ndarray, data_sizes: List[int]) -> Dict:
    """
    Plaintext analogue of Benchmark 1 (encryption performance).

    Two paths are timed, matching the two CKKS paths:
      • Single-value path: iterate over values, boxing each float into a
        Python float object (mirrors element-wise ciphertext creation).
      • SIMD-chunk path: copy each chunk into a new contiguous float64 array
        (mirrors vectorised SIMD encryption).

    All timings are averaged over REPEATS runs for stability.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 1: WRITE PERFORMANCE  (plaintext analogue of encryption)")
    print("=" * 70)

    results = {
        'data_sizes': data_sizes,
        'write_times_s': [],
        'throughput_values_per_sec': [],
        'time_per_chunk_s': [],
        'num_chunks': [],
        'single_value_time_ms': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = np.resize(data, size).astype(np.float64)

        # ── Single-value path (small sizes only, matches CKKS benchmark) ──
        if size <= 1_000:
            probe = sample[:100]
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                _ = [float(v) for v in probe]
                times.append(time.perf_counter() - t0)
            sv_ms = _mean_of_repeats(times) / len(probe) * 1_000
            results['single_value_time_ms'].append(round(sv_ms, 6))
            print(f"    single-value: {sv_ms:.6f} ms/value")
        else:
            results['single_value_time_ms'].append(None)

        # ── SIMD-chunk path ───────────────────────────────────────────────
        chunks = _chunk(sample)
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for ch in chunks:
                _ = np.array(ch, dtype=np.float64)   # contiguous copy
            times.append(time.perf_counter() - t0)

        write_time = _mean_of_repeats(times)
        throughput  = size / write_time
        per_chunk   = write_time / len(chunks)

        results['write_times_s'].append(round(write_time, 8))
        results['throughput_values_per_sec'].append(round(throughput, 2))
        results['time_per_chunk_s'].append(round(per_chunk, 8))
        results['num_chunks'].append(len(chunks))

        print(f"    chunk write: {write_time:.6f} s | "
              f"{throughput:,.0f} values/s | "
              f"{len(chunks)} chunk(s)")

    print("\n✓  Write benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 2 — "Read" performance  (plaintext analogue of decryption)
# ---------------------------------------------------------------------------

def benchmark_read(data: np.ndarray, data_sizes: List[int]) -> Dict:
    """
    Plaintext analogue of Benchmark 2 (decryption performance).

    Pre-build chunk arrays, then time reading every value back from them —
    mirroring the decryption benchmark's pattern of pre-encrypting then
    timing the decrypt pass.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: READ PERFORMANCE  (plaintext analogue of decryption)")
    print("=" * 70)

    results = {
        'data_sizes': data_sizes,
        'read_times_s': [],
        'throughput_values_per_sec': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = np.resize(data, size).astype(np.float64)
        # Pre-build chunks (not timed — mirrors "pre-encrypt" step)
        stored_chunks = [np.array(ch, dtype=np.float64) for ch in _chunk(sample)]

        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            for ch in stored_chunks:
                _ = ch.tolist()          # materialise every value
            times.append(time.perf_counter() - t0)

        read_time  = _mean_of_repeats(times)
        throughput = size / read_time

        results['read_times_s'].append(round(read_time, 8))
        results['throughput_values_per_sec'].append(round(throughput, 2))

        print(f"    read: {read_time:.6f} s | {throughput:,.0f} values/s")

    print("\n✓  Read benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 3 — Aggregation scalability
# ---------------------------------------------------------------------------

def benchmark_aggregation_scalability(data: np.ndarray,
                                       num_clients_list: List[int],
                                       samples_per_client: int = 1_000) -> Dict:
    """
    Plaintext analogue of Benchmark 3.

    Each client holds a local array.  The server aggregates by summing the
    per-client (sum, sum_of_squares) scalars and computing global mean and
    std — the same statistics the FL server decrypts in the encrypted runs.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: AGGREGATION SCALABILITY")
    print(f"  samples/client = {samples_per_client:,}")
    print("=" * 70)

    results = {
        'num_clients': num_clients_list,
        'samples_per_client': samples_per_client,
        'aggregation_times_s': [],
        'per_client_overhead_ms': [],
        'total_samples': [],
    }

    for n_clients in num_clients_list:
        print(f"\n  clients = {n_clients}")

        # Build client contributions (pre-computed, not timed)
        contributions = []
        for _ in range(n_clients):
            client_data = np.resize(data, samples_per_client).astype(np.float64)
            contributions.append({
                'sum':    float(np.sum(client_data)),
                'sum_sq': float(np.sum(client_data ** 2)),
                'count':  samples_per_client,
            })

        # ── Timed aggregation ─────────────────────────────────────────────
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            total_sum    = sum(c['sum']    for c in contributions)
            #total_sum_sq = sum(c['sum_sq'] for c in contributions)
            total_count  = sum(c['count']  for c in contributions)
            global_mean  = total_sum / total_count
            #global_var   = (total_sum_sq / total_count) - global_mean ** 2
            #_global_std  = float(np.sqrt(abs(global_var)))
            times.append(time.perf_counter() - t0)

        agg_time      = _mean_of_repeats(times)
        per_client_ms = agg_time / n_clients * 1_000
        total_samples = n_clients * samples_per_client

        results['aggregation_times_s'].append(round(agg_time, 8))
        results['per_client_overhead_ms'].append(round(per_client_ms, 6))
        results['total_samples'].append(total_samples)

        print(f"    agg: {agg_time:.6f} s | "
              f"{per_client_ms:.6f} ms/client | "
              f"{total_samples:,} total samples")

    print("\n✓  Scalability benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 4 — Communication overhead
# ---------------------------------------------------------------------------

def benchmark_communication_overhead(data: np.ndarray,
                                      data_sizes: List[int]) -> Dict:
    """
    Plaintext analogue of Benchmark 4.

    Measures the byte cost of transmitting raw float64 data via two methods:
      • raw_bytes  : size × 8  (theoretical minimum — raw float64 binary)
      • pickle_bytes: pickle.dumps of the numpy array (realistic Python IPC)

    The overhead ratio here is pickle_bytes / raw_bytes, illustrating Python
    serialisation overhead without any encryption cost.  These numbers form
    the denominator when computing the CKKS/BFV overhead multiplier.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 4: COMMUNICATION OVERHEAD  (plaintext)")
    print("=" * 70)

    results = {
        'data_sizes': data_sizes,
        'raw_bytes': [],
        'pickle_bytes': [],
        'pickle_overhead_ratio': [],
        'num_chunks': [],
        'serialisation_times_s': [],
    }

    for size in data_sizes:
        print(f"\n  size = {size:,}")
        sample = np.resize(data, size).astype(np.float64)
        chunks = _chunk(sample)

        raw_bytes    = size * 8   # float64 = 8 bytes exactly

        # Measure pickle size and serialisation time
        t0 = time.perf_counter()
        total_pickle_bytes = sum(len(pickle.dumps(ch)) for ch in chunks)
        ser_time = time.perf_counter() - t0

        overhead = total_pickle_bytes / raw_bytes

        results['raw_bytes'].append(raw_bytes)
        results['pickle_bytes'].append(total_pickle_bytes)
        results['pickle_overhead_ratio'].append(round(overhead, 4))
        results['num_chunks'].append(len(chunks))
        results['serialisation_times_s'].append(round(ser_time, 8))

        print(f"    raw:    {raw_bytes / 1_024:.1f} KB | "
              f"pickle: {total_pickle_bytes / 1_024:.1f} KB | "
              f"overhead: {overhead:.3f}× | "
              f"ser: {ser_time*1_000:.3f} ms | "
              f"{len(chunks)} chunk(s)")

    print("\n✓  Communication overhead benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Benchmark 5 — End-to-end workflow
# ---------------------------------------------------------------------------

def benchmark_end_to_end(data: np.ndarray,
                          num_clients: int = 5,
                          samples_per_client: int = 1_000) -> Dict:
    """
    Plaintext analogue of Benchmark 5.

    Mirrors every named phase from the encrypted end-to-end benchmark so
    that per-phase overhead attributable to encryption is directly visible.

      Phase 1 — "setup"         : no cryptographic context to build; timed
                                   as the cost of initialising Python lists
                                   and dicts that would hold client state.
      Phase 2 — data generation : same as encrypted.
      Phase 3 — local computation: numpy sum + sum-of-squares per client.
      Phase 4 — "serialisation" : pickle each client's contribution dict
                                   (replaces encryption).
      Phase 5 — communication   : pickle round-trip (deserialise each dict).
      Phase 6 — aggregation     : plain Python sum + mean + std.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 5: END-TO-END WORKFLOW  (plaintext)")
    print(f"  clients = {num_clients} | samples/client = {samples_per_client:,}")
    print("=" * 70)

    total_start = time.perf_counter()

    # Phase 1 — "Setup" (plaintext: initialise client state containers)
    print("\n  Phase 1: Setup")
    t0 = time.perf_counter()
    client_state = [{} for _ in range(num_clients)]
    phase_setup = time.perf_counter() - t0
    print(f"    {phase_setup:.6f} s")

    # Phase 2 — Client data generation
    print("  Phase 2: Client data generation")
    t0 = time.perf_counter()
    client_datasets = [
        np.resize(data, samples_per_client).astype(np.float64)
        for _ in range(num_clients)
    ]
    phase_data_gen = time.perf_counter() - t0
    print(f"    {phase_data_gen:.6f} s")

    # Phase 3 — Local computation
    print("  Phase 3: Local computation")
    t0 = time.perf_counter()
    local_stats = [
        {
            'sum':    float(np.sum(d)),
            'sum_sq': float(np.sum(d ** 2)),
            'count':  len(d),
        }
        for d in client_datasets
    ]
    phase_local = time.perf_counter() - t0
    print(f"    {phase_local:.6f} s")

    # Phase 4 — Serialisation (plaintext analogue of encryption)
    print("  Phase 4: Serialisation  (plaintext analogue of encryption)")
    t0 = time.perf_counter()
    serialised = [pickle.dumps(s) for s in local_stats]
    phase_ser = time.perf_counter() - t0
    print(f"    {phase_ser:.6f} s  "
          f"({phase_ser / num_clients * 1_000:.4f} ms/client)")

    # Phase 5 — Communication (deserialisation round-trip)
    print("  Phase 5: Communication (deserialisation round-trip)")
    t0 = time.perf_counter()
    received = [pickle.loads(s) for s in serialised]
    phase_comm = time.perf_counter() - t0
    print(f"    {phase_comm:.6f} s")

    # Phase 6 — Aggregation
    print("  Phase 6: Aggregation")
    t0 = time.perf_counter()
    total_sum    = sum(r['sum']    for r in received)
    #total_sum_sq = sum(r['sum_sq'] for r in received)
    total_count  = sum(r['count']  for r in received)
    global_mean  = total_sum / total_count
    #global_var   = (total_sum_sq / total_count) - global_mean ** 2
    #global_std   = float(np.sqrt(abs(global_var)))
    phase_agg = time.perf_counter() - t0
    print(f"    {phase_agg:.6f} s")

    total_time = time.perf_counter() - total_start

    phases = {
        'setup':              phase_setup,
        'client_data_gen':    phase_data_gen,
        'local_computation':  phase_local,
        'serialisation':      phase_ser,
        'communication':      phase_comm,
        'aggregation':        phase_agg,
    }

    print(f"\n  Total: {total_time:.6f} s")
    print("  Breakdown:")
    for name, t in phases.items():
        pct = t / total_time * 100 if total_time > 0 else 0
        print(f"    {name:<22}: {t:.6f} s  ({pct:.1f}%)")

    print("\n✓  End-to-end benchmark complete.")
    return {
        'num_clients':        num_clients,
        'samples_per_client': samples_per_client,
        'total_samples':      num_clients * samples_per_client,
        'phases':             {k: round(v, 8) for k, v in phases.items()},
        'total_time':         round(total_time, 8),
        'global_mean':        round(global_mean, 6),
        #'global_std':         round(global_std, 6),
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
    path = RESULTS_DIR / 'PLAINTEXT_BENCHMARK_REPORT.md'
    lines = [
        "# Plaintext Baseline Benchmark Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "> All timings are averaged over "
        f"{REPEATS} repeat runs where applicable.\n",
        f"> SIMD chunk size: {SIMD_CHUNK_SIZE:,} values "
        "(matches CKKS n/2 = 2^14 / 2).\n",
    ]

    # Write
    r = all_results.get('write', {})
    if r:
        lines += [
            "## 1. Write Performance  (plaintext analogue of encryption)\n",
            "| Data Size | Time (s) | Throughput (values/s) | Chunks |",
            "|----------:|---------:|----------------------:|-------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['write_times_s'][i]:.8f} | "
                f"{r['throughput_values_per_sec'][i]:,.0f} | "
                f"{r['num_chunks'][i]} |"
            )
        lines.append("")

    # Read
    r = all_results.get('read', {})
    if r:
        lines += [
            "## 2. Read Performance  (plaintext analogue of decryption)\n",
            "| Data Size | Time (s) | Throughput (values/s) |",
            "|----------:|---------:|----------------------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['read_times_s'][i]:.8f} | "
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
                f"| {n} | {r['aggregation_times_s'][i]:.8f} | "
                f"{r['per_client_overhead_ms'][i]:.6f} | "
                f"{r['total_samples'][i]:,} |"
            )
        lines.append("")

    # Communication
    r = all_results.get('communication', {})
    if r:
        lines += [
            "## 4. Communication Overhead\n",
            "| Data Size | Raw (KB) | Pickle (KB) | Overhead | Chunks | Ser. Time (ms) |",
            "|----------:|---------:|------------:|---------:|-------:|---------------:|",
        ]
        for i, sz in enumerate(r['data_sizes']):
            lines.append(
                f"| {sz:,} | {r['raw_bytes'][i]/1_024:.1f} | "
                f"{r['pickle_bytes'][i]/1_024:.1f} | "
                f"{r['pickle_overhead_ratio'][i]:.3f}× | "
                f"{r['num_chunks'][i]} | "
                f"{r['serialisation_times_s'][i]*1_000:.3f} |"
            )
        lines.append("")

    # End-to-end
    r = all_results.get('end_to_end', {})
    if r:
        lines += [
            "## 5. End-to-End Workflow\n",
            f"- **Clients**: {r['num_clients']}",
            f"- **Samples / client**: {r['samples_per_client']:,}",
            f"- **Total time**: {r['total_time']:.8f} s",
            f"- **Global mean**: {r['global_mean']}",
            #f"- **Global std**: {r['global_std']}\n",
            "### Phase Breakdown\n",
            "| Phase | Time (s) | Share (%) |",
            "|:------|----------:|----------:|",
        ]
        total = r['total_time']
        for phase, t in r['phases'].items():
            pct = t / total * 100 if total > 0 else 0
            lines.append(f"| {phase} | {t:.8f} | {pct:.1f} |")
        lines.append("")

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f"📄  Report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PLAINTEXT BASELINE — BENCHMARKING SUITE")
    print("=" * 70)
    print(f"SIMD chunk size : {SIMD_CHUNK_SIZE:,}  (= CKKS n/2 = 2^14 / 2)")
    print(f"Repeat runs     : {REPEATS}  (averaged for stable timings)")

    # ── Configuration — keep identical to the encrypted benchmarks ─────────
    DATASET_FILENAME   = 'Final_data.csv'
    TARGET_COLUMN      = 'Height (m)'
    WRITE_SIZES        = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    READ_SIZES         = [100, 500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    COMM_SIZES         = [100, 500, 1_000, 5_000, 10_000, 50_000]
    CLIENT_COUNTS      = [2, 5, 10, 20, 50, 100, 200]
    SAMPLES_PER_CLIENT = 1_000
    E2E_CLIENTS        = 20
    E2E_SAMPLES        = 1_000

    # ── Load dataset ────────────────────────────────────────────────────────
    print(f"\n--- Loading dataset ---")
    df   = load_csv(DATASET_FILENAME)
    data = extract_column(df, TARGET_COLUMN)
    print(f"✓  Using column '{TARGET_COLUMN}': {len(data):,} values")

    all_results: Dict = {}

    # ── Run benchmarks ──────────────────────────────────────────────────────
    all_results['write'] = benchmark_write(data, WRITE_SIZES)

    all_results['read'] = benchmark_read(data, READ_SIZES)

    all_results['scalability'] = benchmark_aggregation_scalability(
        data, CLIENT_COUNTS, SAMPLES_PER_CLIENT)

    all_results['communication'] = benchmark_communication_overhead(
        data, COMM_SIZES)

    all_results['end_to_end'] = benchmark_end_to_end(
        data, E2E_CLIENTS, E2E_SAMPLES)

    # ── Persist results ─────────────────────────────────────────────────────
    print("\n--- Saving results ---")
    save_json(all_results['write'],         'plaintext_write_benchmark.json')
    save_json(all_results['read'],          'plaintext_read_benchmark.json')
    save_json(all_results['scalability'],   'plaintext_scalability_benchmark.json')
    save_json(all_results['communication'], 'plaintext_communication_benchmark.json')
    save_json(all_results['end_to_end'],    'plaintext_end_to_end_benchmark.json')
    save_json(all_results,                  'plaintext_all_benchmarks.json')
    save_markdown_report(all_results)

    print("\n" + "=" * 70)
    print("✅  PLAINTEXT BENCHMARKING COMPLETE")
    print("=" * 70)
    print(f"\nArtefacts saved in: {RESULTS_DIR.absolute()}/")
    print("  plaintext_write_benchmark.json")
    print("  plaintext_read_benchmark.json")
    print("  plaintext_scalability_benchmark.json")
    print("  plaintext_communication_benchmark.json")
    print("  plaintext_end_to_end_benchmark.json")
    print("  plaintext_all_benchmarks.json")
    print("  PLAINTEXT_BENCHMARK_REPORT.md")


if __name__ == "__main__":
    main()
