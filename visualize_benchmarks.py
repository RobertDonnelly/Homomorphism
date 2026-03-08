"""
Benchmark Results Visualizer
==============================
Reads the JSON output from ckks_process_data.py, bfv_process_data.py, or
plaintext_process_data.py and produces a complete set of benchmark plots.

Usage
-----
    # visualise a single scheme
    python visualize_benchmarks.py --scheme ckks
    python visualize_benchmarks.py --scheme bfv
    python visualize_benchmarks.py --scheme plaintext

    # overlay all three schemes on shared axes for direct comparison
    python visualize_benchmarks.py --scheme all

Output
------
    results/<scheme>_benchmark/  (or results/comparison/ for --scheme all)
        benchmark_1_write_read_throughput_<ts>.png
        benchmark_2_aggregation_scalability_<ts>.png
        benchmark_3_communication_overhead_<ts>.png
        benchmark_4_end_to_end_phases_<ts>.png
        benchmark_dashboard_<ts>.png
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Style (mirrors original visualiser) ─────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 100

project_root = Path(__file__).parent

# ── Colour palette — one identity colour per scheme ─────────────────────────
SCHEME_COLOURS = {
    'ckks':      '#3498db',   # blue
    'bfv':       '#e74c3c',   # red
    'plaintext': '#2ecc71',   # green
}
SCHEME_LABELS = {
    'ckks':      'CKKS (Encrypted)',
    'bfv':       'BFV (Encrypted)',
    'plaintext': 'Plaintext (Baseline)',
}

# Header colour used for table cells (matches original #4472C4)
TABLE_HEADER_COLOUR = '#4472C4'
TABLE_ALT_ROW       = '#E7E6E6'


# ============================================================================
# Data loading
# ============================================================================

def _results_dir(scheme: str) -> Path:
    mapping = {
        'ckks':      project_root / 'results' / 'ckks_benchmark',
        'bfv':       project_root / 'results' / 'bfv_benchmark',
        'plaintext': project_root / 'results' / 'plaintext_benchmark',
    }
    return mapping[scheme]


def load_scheme(scheme: str) -> Dict:
    """
    Load the combined *_all_benchmarks.json for *scheme*.
    Falls back to loading individual files if the combined file is absent.
    Returns a dict with keys: write/encryption, read/decryption,
    scalability, communication, end_to_end.
    """
    rdir = _results_dir(scheme)
    combined = rdir / f'{scheme}_all_benchmarks.json'

    if combined.exists():
        with open(combined) as fh:
            data = json.load(fh)
        print(f"  ✓  {scheme:>10}  ← {combined.relative_to(project_root)}")
        return data

    # Fallback: load individual files
    key_map = {
        'ckks': {
            'write':         'ckks_encryption_benchmark.json',
            'read':          'ckks_decryption_benchmark.json',
            'scalability':   'ckks_scalability_benchmark.json',
            'communication': 'ckks_communication_benchmark.json',
            'end_to_end':    'ckks_end_to_end_benchmark.json',
        },
        'bfv': {
            'write':         'bfv_encryption_benchmark.json',
            'read':          'bfv_decryption_benchmark.json',
            'scalability':   'bfv_scalability_benchmark.json',
            'communication': 'bfv_communication_benchmark.json',
            'end_to_end':    'bfv_end_to_end_benchmark.json',
        },
        'plaintext': {
            'write':         'plaintext_write_benchmark.json',
            'read':          'plaintext_read_benchmark.json',
            'scalability':   'plaintext_scalability_benchmark.json',
            'communication': 'plaintext_communication_benchmark.json',
            'end_to_end':    'plaintext_end_to_end_benchmark.json',
        },
    }
    data = {}
    for key, fname in key_map[scheme].items():
        fpath = rdir / fname
        if fpath.exists():
            with open(fpath) as fh:
                data[key] = json.load(fh)
            print(f"  ✓  {scheme:>10} / {key:<14} ← {fpath.name}")
        else:
            print(f"  ⚠️   {scheme:>10} / {key:<14}   not found — skipping")
    return data


def _throughput_key(d: Dict) -> Optional[List[float]]:
    """Return the throughput list regardless of which scheme produced it."""
    for k in ('throughput_values_per_sec', 'throughput'):
        if k in d:
            return d[k]
    return None


def _time_key_write(d: Dict) -> Optional[List[float]]:
    for k in ('write_times_s', 'encryption_times'):
        if k in d:
            return d[k]
    return None


def _time_key_read(d: Dict) -> Optional[List[float]]:
    for k in ('read_times_s', 'decryption_times'):
        if k in d:
            return d[k]
    return None


def _agg_time_key(d: Dict) -> Optional[List[float]]:
    for k in ('aggregation_times_s', 'aggregation_times'):
        if k in d:
            return d[k]
    return None


def _per_client_key(d: Dict) -> Optional[List[float]]:
    for k in ('per_client_overhead_ms',):
        if k in d:
            return d[k]
    return None


# ============================================================================
# Save helper
# ============================================================================

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = out_dir / f'{name}_{ts}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  💾  Saved → {path.relative_to(project_root)}")


# ============================================================================
# Styled table helper  (matches original visualiser style)
# ============================================================================

def _draw_table(ax: plt.Axes, rows: List[List[str]],
                col_widths: List[float] = None) -> None:
    """Draw a styled header+data table on *ax* (axis must be off)."""
    ax.axis('off')
    if col_widths is None:
        col_widths = [1 / len(rows[0])] * len(rows[0])
    tbl = ax.table(cellText=rows, cellLoc='left', loc='center',
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    # Header row
    for j in range(len(rows[0])):
        tbl[(0, j)].set_facecolor(TABLE_HEADER_COLOUR)
        tbl[(0, j)].set_text_props(weight='bold', color='white')
    # Alternating data rows
    for i in range(1, len(rows)):
        for j in range(len(rows[0])):
            if i % 2 == 0:
                tbl[(i, j)].set_facecolor(TABLE_ALT_ROW)


# ============================================================================
# Plot 1 — Write / Read throughput
# ============================================================================

def plot_write_read_throughput(datasets: Dict[str, Dict],
                                out_dir: Path) -> None:
    """
    2 × 2 grid:
      [0,0] Write throughput (log-log)        [0,1] Read throughput (log-log)
      [1,0] Write time (linear)               [1,1] Read time (linear)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Benchmark 1 & 2 — Write / Read Performance\n'
                 '(Encryption & Decryption Throughput)', fontsize=14,
                 fontweight='bold')

    for scheme, data in datasets.items():
        colour = SCHEME_COLOURS[scheme]
        label  = SCHEME_LABELS[scheme]

        # ── Write (encryption analogue) ──────────────────────────────────
        wd = data.get('write') or data.get('encryption', {})
        if wd:
            sizes      = wd.get('data_sizes', [])
            throughput = _throughput_key(wd)
            times      = _time_key_write(wd)

            if throughput:
                axes[0, 0].plot(sizes, throughput, 'o-', color=colour,
                                label=label, linewidth=2, markersize=7)
            if times:
                axes[1, 0].plot(sizes, times, 's--', color=colour,
                                label=label, linewidth=2, markersize=7)

        # ── Read (decryption analogue) ────────────────────────────────────
        rd = data.get('read') or data.get('decryption', {})
        if rd:
            sizes      = rd.get('data_sizes', [])
            throughput = _throughput_key(rd)
            times      = _time_key_read(rd)

            if throughput:
                axes[0, 1].plot(sizes, throughput, 'o-', color=colour,
                                label=label, linewidth=2, markersize=7)
            if times:
                axes[1, 1].plot(sizes, times, 's--', color=colour,
                                label=label, linewidth=2, markersize=7)

    # ── Formatting ────────────────────────────────────────────────────────
    titles    = ['Write Throughput', 'Read Throughput',
                 'Write Time', 'Read Time']
    ylabels   = ['Throughput (values/s)', 'Throughput (values/s)',
                 'Time (s)', 'Time (s)']
    log_axes  = [True, True, False, False]

    for ax, title, ylabel, use_log in zip(axes.flat, titles, ylabels, log_axes):
        ax.set_title(title)
        ax.set_xlabel('Data Size (values)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        if use_log:
            ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    _save(fig, out_dir, 'benchmark_1_write_read_throughput')
    plt.show()


# ============================================================================
# Plot 2 — Aggregation scalability
# ============================================================================

def plot_aggregation_scalability(datasets: Dict[str, Dict],
                                  out_dir: Path) -> None:
    """
    1 × 2:
      [0] Total aggregation time vs. client count
      [1] Per-client overhead (ms) vs. client count
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Benchmark 3 — Aggregation Scalability',
                 fontsize=14, fontweight='bold')

    for scheme, data in datasets.items():
        colour = SCHEME_COLOURS[scheme]
        label  = SCHEME_LABELS[scheme]
        sd     = data.get('scalability', {})
        if not sd:
            continue

        clients    = sd.get('num_clients', [])
        agg_times  = _agg_time_key(sd)
        per_client = _per_client_key(sd)

        if agg_times:
            axes[0].plot(clients, agg_times, '^-', color=colour,
                         label=label, linewidth=2, markersize=8)
        if per_client:
            axes[1].plot(clients, per_client, 'd-', color=colour,
                         label=label, linewidth=2, markersize=8)

    for ax, title, ylabel in zip(
        axes,
        ['Total Aggregation Time vs. Client Count',
         'Per-Client Overhead vs. Client Count'],
        ['Aggregation Time (s)', 'Per-Client Overhead (ms)']
    ):
        ax.set_title(title)
        ax.set_xlabel('Number of Clients')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, 'benchmark_2_aggregation_scalability')
    plt.show()


# ============================================================================
# Plot 3 — Communication overhead
# ============================================================================

def plot_communication_overhead(datasets: Dict[str, Dict],
                                 out_dir: Path) -> None:
    """
    2 × 2:
      [0,0] Overhead ratio vs. data size (bar groups)
      [0,1] Ciphertext / pickle bytes vs. data size (line)
      [1,0] Raw vs serialised bytes (stacked bar)
      [1,1] Summary table
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Benchmark 4 — Communication Overhead',
                 fontsize=14, fontweight='bold')

    scheme_list = list(datasets.keys())
    bar_width   = 0.8 / max(len(scheme_list), 1)

    # Collect sizes from first available dataset
    ref_sizes = []
    for data in datasets.values():
        cd = data.get('communication', {})
        if cd.get('data_sizes'):
            ref_sizes = cd['data_sizes']
            break

    x = np.arange(len(ref_sizes))

    # ── [0,0] Overhead ratio — grouped bars ──────────────────────────────
    for idx, (scheme, data) in enumerate(datasets.items()):
        cd = data.get('communication', {})
        if not cd:
            continue
        ratios = cd.get('overhead_ratio') or cd.get('overhead_ratios', [])
        if ratios:
            offset = (idx - len(scheme_list) / 2 + 0.5) * bar_width
            axes[0, 0].bar(x + offset, ratios, bar_width,
                           label=SCHEME_LABELS[scheme],
                           color=SCHEME_COLOURS[scheme],
                           edgecolor='black', alpha=0.8)

    axes[0, 0].set_title('Overhead Ratio vs. Data Size')
    axes[0, 0].set_xlabel('Data Size (values)')
    axes[0, 0].set_ylabel('Overhead Ratio (×)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'{s:,}' for s in ref_sizes], rotation=30)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, axis='y', alpha=0.3)

    # ── [0,1] Serialised bytes line ───────────────────────────────────────
    for scheme, data in datasets.items():
        cd = data.get('communication', {})
        if not cd:
            continue
        sizes = cd.get('data_sizes', [])
        ctxt  = cd.get('ciphertext_bytes') or cd.get('pickle_bytes', [])
        if sizes and ctxt:
            ctxt_kb = [b / 1_024 for b in ctxt]
            axes[0, 1].plot(sizes, ctxt_kb, 'o-',
                            color=SCHEME_COLOURS[scheme],
                            label=SCHEME_LABELS[scheme],
                            linewidth=2, markersize=7)

    # Add raw plaintext reference line
    if ref_sizes:
        raw_kb = [s * 8 / 1_024 for s in ref_sizes]
        axes[0, 1].plot(ref_sizes, raw_kb, 'k--', linewidth=1.5,
                        label='Raw float64 (8 B/value)', alpha=0.6)

    axes[0, 1].set_title('Serialised Size vs. Data Size')
    axes[0, 1].set_xlabel('Data Size (values)')
    axes[0, 1].set_ylabel('Size (KB)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, which='both')

    # ── [1,0] Plaintext vs serialised at largest size ─────────────────────
    if ref_sizes:
        largest = ref_sizes[-1]
        raw_kb_val = largest * 8 / 1_024
        scheme_labels_bar, ser_kb_vals = [], []

        for scheme, data in datasets.items():
            cd = data.get('communication', {})
            if not cd:
                continue
            sizes = cd.get('data_sizes', [])
            if largest in sizes:
                idx_l = sizes.index(largest)
                ctxt  = cd.get('ciphertext_bytes') or cd.get('pickle_bytes', [])
                if ctxt:
                    scheme_labels_bar.append(SCHEME_LABELS[scheme])
                    ser_kb_vals.append(ctxt[idx_l] / 1_024)

        if scheme_labels_bar:
            # Assign colours by matching scheme order to the labels collected above
            bar_colors = []
            for s in datasets:
                cd_s = datasets[s].get('communication', {})
                sizes_s = cd_s.get('data_sizes', [])
                if largest in sizes_s:
                    ctxt_s = cd_s.get('ciphertext_bytes') or cd_s.get('pickle_bytes', [])
                    if ctxt_s:
                        bar_colors.append(SCHEME_COLOURS[s])
            all_labels = ['Raw\nfloat64'] + scheme_labels_bar
            all_vals   = [raw_kb_val]    + ser_kb_vals
            all_colors = ['#95a5a6']     + bar_colors

            bars = axes[1, 0].bar(all_labels, all_vals,
                                  color=all_colors, edgecolor='black', alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, h,
                                f'{h:,.1f} KB', ha='center', va='bottom',
                                fontsize=9, fontweight='bold')

        axes[1, 0].set_title(f'Transmission Size at n={largest:,}')
        axes[1, 0].set_ylabel('Size (KB)')
        axes[1, 0].grid(True, axis='y', alpha=0.3)

    # ── [1,1] Summary table ───────────────────────────────────────────────
    rows = [['Scheme', 'Max Size', 'Payload (KB)', 'Overhead']]
    for scheme, data in datasets.items():
        cd = data.get('communication', {})
        if not cd:
            continue
        sizes  = cd.get('data_sizes', [])
        ratios = cd.get('overhead_ratio') or cd.get('overhead_ratios', [])
        ctxt   = cd.get('ciphertext_bytes') or cd.get('pickle_bytes', [])
        if sizes and ctxt and ratios:
            rows.append([
                SCHEME_LABELS[scheme],
                f'{sizes[-1]:,}',
                f'{ctxt[-1]/1_024:,.1f}',
                f'{ratios[-1]:.1f}×',
            ])
    _draw_table(axes[1, 1], rows, col_widths=[0.35, 0.20, 0.25, 0.20])
    axes[1, 1].set_title('Overhead Summary (Largest Size)',
                         fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    _save(fig, out_dir, 'benchmark_3_communication_overhead')
    plt.show()


# ============================================================================
# Plot 4 — End-to-end phase breakdown
# ============================================================================

def plot_end_to_end(datasets: Dict[str, Dict], out_dir: Path) -> None:
    """
    One horizontal bar chart per scheme (stacked phases) + a total-time
    comparison bar chart + per-scheme summary tables.
    """
    scheme_list = [s for s in datasets if datasets[s].get('end_to_end')]
    n = len(scheme_list)
    if n == 0:
        print("  ⚠️   No end-to-end data available — skipping.")
        return

    # Figure: (n rows of phase bars) + 1 row for total comparison
    fig = plt.figure(figsize=(16, 5 * (n + 1)))
    gs  = gridspec.GridSpec(n + 1, 2, figure=fig,
                            hspace=0.45, wspace=0.35)
    fig.suptitle('Benchmark 5 — End-to-End Workflow Phase Breakdown',
                 fontsize=14, fontweight='bold', y=1.01)

    phase_colours = ['#3498db', '#2ecc71', '#e74c3c',
                     '#f39c12', '#9b59b6', '#1abc9c']

    for row, scheme in enumerate(scheme_list):
        e2e    = datasets[scheme]['end_to_end']
        phases = e2e.get('phases', {})
        total  = e2e.get('total_time', sum(phases.values()))

        p_names = list(phases.keys())
        p_times = list(phases.values())

        # ── Horizontal stacked bar ────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[row, 0])
        left   = 0.0
        for i, (pname, ptime) in enumerate(zip(p_names, p_times)):
            colour = phase_colours[i % len(phase_colours)]
            bar    = ax_bar.barh(0, ptime, left=left, color=colour,
                                 edgecolor='white', height=0.5,
                                 label=pname.replace('_', ' ').title())
            # Label segment if wide enough
            if ptime / total > 0.04:
                ax_bar.text(left + ptime / 2, 0,
                            f'{ptime*1000:.1f}ms\n({ptime/total*100:.0f}%)',
                            ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')
            left += ptime

        ax_bar.set_title(f'{SCHEME_LABELS[scheme]}  —  total: {total:.4f} s')
        ax_bar.set_xlabel('Time (s)')
        ax_bar.set_yticks([])
        ax_bar.legend(loc='lower right', fontsize=8, ncol=3)
        ax_bar.grid(True, axis='x', alpha=0.3)

        # ── Phase table ───────────────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[row, 1])
        rows_t = [['Phase', 'Time (s)', 'Share']]
        for pname, ptime in phases.items():
            rows_t.append([
                pname.replace('_', ' ').title(),
                f'{ptime:.6f}',
                f'{ptime/total*100:.1f}%',
            ])
        rows_t.append(['TOTAL', f'{total:.6f}', '100%'])
        _draw_table(ax_tbl, rows_t, col_widths=[0.45, 0.30, 0.25])
        ax_tbl.set_title(f'Phase Times — {SCHEME_LABELS[scheme]}',
                         fontsize=10, fontweight='bold', pad=10)

    # ── Total time comparison bar (bottom row, full width) ────────────────
    ax_total = fig.add_subplot(gs[n, :])
    totals   = []
    labels   = []
    colours  = []
    for scheme in scheme_list:
        e2e = datasets[scheme].get('end_to_end', {})
        phases = e2e.get('phases', {})
        total  = e2e.get('total_time', sum(phases.values()))
        totals.append(total)
        labels.append(SCHEME_LABELS[scheme])
        colours.append(SCHEME_COLOURS[scheme])

    bars = ax_total.bar(labels, totals, color=colours,
                        edgecolor='black', alpha=0.85)
    for bar, t in zip(bars, totals):
        h = bar.get_height()
        ax_total.text(bar.get_x() + bar.get_width() / 2, h,
                      f'{t:.4f} s', ha='center', va='bottom',
                      fontsize=10, fontweight='bold')

    ax_total.set_title('Total End-to-End Time Comparison')
    ax_total.set_ylabel('Total Time (s)')
    ax_total.grid(True, axis='y', alpha=0.3)

    # Add speedup annotation relative to slowest
    if len(totals) > 1:
        slowest = max(totals)
        for bar, t, lbl in zip(bars, totals, labels):
            speedup = slowest / t
            if speedup > 1.0:
                ax_total.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() / 2,
                              f'{speedup:.0f}×\nfaster',
                              ha='center', va='center',
                              fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    _save(fig, out_dir, 'benchmark_4_end_to_end_phases')
    plt.show()


# ============================================================================
# Plot 5 — Dashboard  (one-page overview)
# ============================================================================

def plot_dashboard(datasets: Dict[str, Dict], out_dir: Path) -> None:
    """
    Single-page dashboard summarising all five benchmarks at a glance.

    Layout (3 × 3 grid):
      [0,0] Write throughput         [0,1] Aggregation time       [0,2] Summary table
      [1,0] Read throughput          [1,1] Per-client overhead     [1,2] (continues)
      [2,0] Overhead ratio           [2,1] E2E total time          [2,2] (continues)
    """
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle('Benchmark Results Dashboard\n'
                 'CKKS  ·  BFV  ·  Plaintext Baseline',
                 fontsize=16, fontweight='bold')

    # ── [0,0] Write throughput ────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    for scheme, data in datasets.items():
        wd = data.get('write') or data.get('encryption', {})
        if wd:
            sizes = wd.get('data_sizes', [])
            tp    = _throughput_key(wd)
            if tp:
                ax00.plot(sizes, tp, 'o-', color=SCHEME_COLOURS[scheme],
                          label=SCHEME_LABELS[scheme], linewidth=2, markersize=6)
    ax00.set_title('Write Throughput')
    ax00.set_xlabel('Data Size')
    ax00.set_ylabel('Values / s')
    ax00.set_xscale('log')
    ax00.set_yscale('log')
    ax00.legend(fontsize=8)
    ax00.grid(True, alpha=0.3, which='both')

    # ── [1,0] Read throughput ─────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    for scheme, data in datasets.items():
        rd = data.get('read') or data.get('decryption', {})
        if rd:
            sizes = rd.get('data_sizes', [])
            tp    = _throughput_key(rd)
            if tp:
                ax10.plot(sizes, tp, 's-', color=SCHEME_COLOURS[scheme],
                          label=SCHEME_LABELS[scheme], linewidth=2, markersize=6)
    ax10.set_title('Read Throughput')
    ax10.set_xlabel('Data Size')
    ax10.set_ylabel('Values / s')
    ax10.set_xscale('log')
    ax10.set_yscale('log')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3, which='both')

    # ── [0,1] Aggregation time ────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    for scheme, data in datasets.items():
        sd = data.get('scalability', {})
        if sd:
            clients   = sd.get('num_clients', [])
            agg_times = _agg_time_key(sd)
            if agg_times:
                ax01.plot(clients, agg_times, '^-', color=SCHEME_COLOURS[scheme],
                          label=SCHEME_LABELS[scheme], linewidth=2, markersize=6)
    ax01.set_title('Aggregation Time')
    ax01.set_xlabel('Number of Clients')
    ax01.set_ylabel('Time (s)')
    ax01.legend(fontsize=8)
    ax01.grid(True, alpha=0.3)

    # ── [1,1] Per-client overhead ─────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    for scheme, data in datasets.items():
        sd = data.get('scalability', {})
        if sd:
            clients   = sd.get('num_clients', [])
            per_cl    = _per_client_key(sd)
            if per_cl:
                ax11.plot(clients, per_cl, 'd-', color=SCHEME_COLOURS[scheme],
                          label=SCHEME_LABELS[scheme], linewidth=2, markersize=6)
    ax11.set_title('Per-Client Overhead')
    ax11.set_xlabel('Number of Clients')
    ax11.set_ylabel('Overhead (ms)')
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    # ── [2,0] Overhead ratio ──────────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    # Re-use first available sizes list for x-axis
    ref_sizes_dash = []
    for data in datasets.values():
        cd = data.get('communication', {})
        if cd.get('data_sizes'):
            ref_sizes_dash = cd['data_sizes']
            break

    x_dash = np.arange(len(ref_sizes_dash))
    bar_w  = 0.8 / max(len(datasets), 1)

    for idx, (scheme, data) in enumerate(datasets.items()):
        cd = data.get('communication', {})
        if not cd:
            continue
        ratios = cd.get('overhead_ratio') or cd.get('overhead_ratios', [])
        if ratios:
            offset = (idx - len(datasets) / 2 + 0.5) * bar_w
            ax20.bar(x_dash + offset, ratios, bar_w,
                     label=SCHEME_LABELS[scheme],
                     color=SCHEME_COLOURS[scheme],
                     edgecolor='black', alpha=0.8)

    ax20.set_title('Communication Overhead Ratio')
    ax20.set_xlabel('Data Size')
    ax20.set_ylabel('Overhead (×)')
    ax20.set_xticks(x_dash)
    ax20.set_xticklabels([f'{s:,}' for s in ref_sizes_dash], rotation=30,
                         fontsize=8)
    ax20.legend(fontsize=8)
    ax20.grid(True, axis='y', alpha=0.3)

    # ── [2,1] E2E total time comparison ───────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    totals_dash, labels_dash, colours_dash = [], [], []
    for scheme, data in datasets.items():
        e2e    = data.get('end_to_end', {})
        phases = e2e.get('phases', {})
        total  = e2e.get('total_time', sum(phases.values()) if phases else 0)
        if total:
            totals_dash.append(total)
            labels_dash.append(SCHEME_LABELS[scheme])
            colours_dash.append(SCHEME_COLOURS[scheme])

    if totals_dash:
        bars_e2e = ax21.bar(labels_dash, totals_dash, color=colours_dash,
                            edgecolor='black', alpha=0.85)
        for bar, t in zip(bars_e2e, totals_dash):
            ax21.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f'{t:.4f}s', ha='center', va='bottom',
                      fontsize=9, fontweight='bold')
        ax21.set_title('E2E Total Time')
        ax21.set_ylabel('Time (s)')
        ax21.tick_params(axis='x', labelsize=8)
        ax21.grid(True, axis='y', alpha=0.3)

    # ── [0,2] + [1,2] + [2,2]  Summary table (spans full right column) ────
    ax_tbl = fig.add_subplot(gs[:, 2])
    rows_s = [['Benchmark', 'Scheme', 'Key Metric']]

    for scheme, data in datasets.items():
        lbl = SCHEME_LABELS[scheme]
        # Write throughput at largest size
        wd = data.get('write') or data.get('encryption', {})
        if wd:
            tp = _throughput_key(wd)
            if tp:
                rows_s.append(['Write TP', lbl, f'{tp[-1]:,.0f} v/s'])
        # Decryption throughput at largest size
        rd = data.get('read') or data.get('decryption', {})
        if rd:
            tp = _throughput_key(rd)
            if tp:
                rows_s.append(['Read TP', lbl, f'{tp[-1]:,.0f} v/s'])
        # Aggregation time at max clients
        sd = data.get('scalability', {})
        if sd:
            agg = _agg_time_key(sd)
            if agg:
                rows_s.append(['Agg (max)', lbl, f'{agg[-1]:.4f} s'])
        # Overhead at largest size
        cd = data.get('communication', {})
        if cd:
            ratios = cd.get('overhead_ratio') or []
            if ratios:
                rows_s.append(['Overhead', lbl, f'{ratios[-1]:.1f}×'])
        # E2E total
        e2e = data.get('end_to_end', {})
        phases = e2e.get('phases', {})
        total  = e2e.get('total_time', sum(phases.values()) if phases else 0)
        if total:
            rows_s.append(['E2E Total', lbl, f'{total:.4f} s'])

    _draw_table(ax_tbl, rows_s, col_widths=[0.30, 0.40, 0.30])
    ax_tbl.set_title('Key Metrics Summary', fontsize=11,
                     fontweight='bold', pad=12)

    plt.tight_layout()
    _save(fig, out_dir, 'benchmark_dashboard')
    plt.show()


# ============================================================================
# Orchestrator
# ============================================================================

class BenchmarkVisualizer:
    """Load JSON results and produce all benchmark plots."""

    def __init__(self, schemes: List[str]):
        self.schemes  = schemes
        self.datasets: Dict[str, Dict] = {}

        if len(schemes) == 1:
            scheme = schemes[0]
            self.out_dir = _results_dir(scheme)
        else:
            self.out_dir = project_root / 'results' / 'comparison'

    def load(self) -> None:
        print("\n" + "=" * 70)
        print("BENCHMARK VISUALIZER — loading results")
        print("=" * 70)
        for scheme in self.schemes:
            data = load_scheme(scheme)
            if data:
                self.datasets[scheme] = data
            else:
                print(f"  ❌  No data loaded for '{scheme}' — skipping.")

        if not self.datasets:
            print("\n❌  No data loaded. "
                  "Run the benchmark scripts first then retry.")
            sys.exit(1)

        print(f"\n✓  Loaded {len(self.datasets)} scheme(s): "
              f"{', '.join(self.datasets)}")

    def run(self) -> None:
        self.load()

        print(f"\n📊  Generating plots → {self.out_dir.relative_to(project_root)}/\n")

        print("  Plot 1/5 — Write / Read throughput …")
        plot_write_read_throughput(self.datasets, self.out_dir)

        print("  Plot 2/5 — Aggregation scalability …")
        plot_aggregation_scalability(self.datasets, self.out_dir)

        print("  Plot 3/5 — Communication overhead …")
        plot_communication_overhead(self.datasets, self.out_dir)

        print("  Plot 4/5 — End-to-end phase breakdown …")
        plot_end_to_end(self.datasets, self.out_dir)

        print("  Plot 5/5 — Dashboard …")
        plot_dashboard(self.datasets, self.out_dir)

        print("\n" + "=" * 70)
        print("✅  ALL PLOTS COMPLETE")
        print("=" * 70)
        print(f"\nAll figures saved in: {self.out_dir.absolute()}/")


# ============================================================================
# Entry point
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Visualise CKKS / BFV / Plaintext benchmark results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_benchmarks.py --scheme ckks
  python visualize_benchmarks.py --scheme bfv
  python visualize_benchmarks.py --scheme plaintext
  python visualize_benchmarks.py --scheme all
        """
    )
    p.add_argument(
        '--scheme',
        choices=['ckks', 'bfv', 'plaintext', 'all'],
        default='all',
        help="Which benchmark results to plot (default: all)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.scheme == 'all':
        schemes = ['ckks', 'bfv', 'plaintext']
    else:
        schemes = [args.scheme]

    viz = BenchmarkVisualizer(schemes)
    viz.run()


if __name__ == '__main__':
    main()