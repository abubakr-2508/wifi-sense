"""Offline analysis of recorded CSI -- produces the figures and tables for the report.

Runs the same DSP and detection code the live dashboard uses, but over a whole
recording at once, and writes everything to `output/`:

    fig1_dataset_overview.png     amplitude heatmap + frame timing
    fig2_subcarrier_variance.png  which subcarriers carry the signal
    fig3_respiration_extraction.png  raw -> bandpassed -> spectrum, one window
    fig4_respiration_timeline.png breathing rate and motion evidence over time
    fig5_node_agreement.png       node 1 vs node 2 cross-check
    results_summary.csv           dataset and detection statistics
    results_breathing.csv         per-window breathing estimates
    RESULTS.md                    everything above, written out in prose

Usage
-----
    python analyze_csi.py --file ../RuView/data/recordings/pretrain-1775182186.csi.jsonl

Every number in the output is measured from the recording. Nothing is
simulated, and where a quantity cannot be measured the script says so rather
than filling the gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write straight to file
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wifisense.detector import MotionDetector
from wifisense.dsp import (
    RESPIRATORY_BAND_HZ,
    bandpass,
    detrend,
    extract_features,
    hampel,
    respiratory_estimate,
)
from wifisense.sources import decode_iq_hex, decode_iq_hex_phase

# Report-friendly styling: light background, restrained palette, readable at
# the size figures end up in a document.
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "600",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)

C_PRIMARY = "#1f6fb4"
C_ACCENT = "#d1495b"
C_MUTED = "#8d99ae"
C_GREEN = "#2a9d8f"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_recording(path: Path) -> dict:
    """Read a .csi.jsonl capture into per-node amplitude/phase matrices.

    Captures mix frame geometries: this recording contains 64-, 128- and
    192-value payloads (1, 2 and 3 antennas) plus empty frames. Those
    populations are NOT interchangeable -- measured on node 1, subcarrier 56
    has mean 10.04 / SD 4.53 in 64-value frames but mean 12.61 / SD 0.97 in
    128-value ones. Concatenating them injects variance that comes from
    switching between populations rather than from anything in the room, and a
    dispersion-based detector will happily report that artefact as motion.

    So we keep only the modal frame length per node: one geometry, one
    population, no manufactured variance. Discarded counts are returned so the
    report can state exactly how much data was set aside and why.
    """
    by_node: dict[int, dict] = defaultdict(
        lambda: {"t": [], "amp": [], "phase": [], "rssi": []}
    )

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            iq = rec.get("iq_hex", "")
            amps = decode_iq_hex(iq)
            if amps.size == 0:
                continue

            node = int(rec.get("node_id", 0))
            by_node[node]["t"].append(float(rec.get("timestamp", 0.0)))
            by_node[node]["amp"].append(amps)
            by_node[node]["phase"].append(decode_iq_hex_phase(iq))
            by_node[node]["rssi"].append(float(rec.get("rssi", 0)))

    out = {}
    for node, d in by_node.items():
        if len(d["amp"]) < 10:
            continue

        lengths = defaultdict(int)
        for a in d["amp"]:
            lengths[a.size] += 1
        modal = max(lengths, key=lengths.get)
        keep = [i for i, a in enumerate(d["amp"]) if a.size == modal]
        if len(keep) < 10:
            continue

        out[node] = {
            "t": np.asarray([d["t"][i] for i in keep], dtype=float),
            "amp": np.stack([d["amp"][i] for i in keep]),
            "phase": np.stack([d["phase"][i] for i in keep]),
            "rssi": np.asarray([d["rssi"][i] for i in keep], dtype=float),
            "frame_geometry": modal,
            "frames_kept": len(keep),
            "frames_discarded": sum(v for k, v in lengths.items() if k != modal),
            "geometry_counts": dict(lengths),
        }
    return out


def effective_rate(t: np.ndarray) -> float:
    """Median-based sample rate -- robust to gaps in the capture."""
    if t.size < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_dataset_overview(nodes: dict, out: Path) -> dict:
    n = len(nodes)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.1 * n), squeeze=False)
    stats = {}

    for row, (node, d) in enumerate(sorted(nodes.items())):
        amp, t = d["amp"], d["t"]
        fs = effective_rate(t)
        dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
        stats[node] = {"frames": amp.shape[0], "subcarriers": amp.shape[1],
                       "fs": fs, "duration": dur}

        ax = axes[row][0]
        im = ax.imshow(
            amp.T, aspect="auto", origin="lower", cmap="viridis",
            extent=[0, dur, 0, amp.shape[1]],
        )
        ax.set_title(f"Node {node} — CSI amplitude ({amp.shape[0]} frames)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("subcarrier")
        fig.colorbar(im, ax=ax, label="|H|", fraction=0.046, pad=0.03)

        ax = axes[row][1]
        dt = np.diff(t) * 1000.0
        dt = dt[(dt > 0) & (dt < np.percentile(dt, 99.5))]
        ax.hist(dt, bins=50, color=C_PRIMARY, alpha=0.85)
        ax.axvline(1000.0 / fs, color=C_ACCENT, ls="--", lw=1.3,
                   label=f"median {1000.0/fs:.1f} ms ({fs:.1f} Hz)")
        ax.set_title(f"Node {node} — inter-frame interval")
        ax.set_xlabel("Δt (ms)")
        ax.set_ylabel("count")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Dataset overview — real CSI captured from an ESP32-S3 mesh",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig1_dataset_overview.png")
    plt.close(fig)
    return stats


def fig_subcarrier_variance(nodes: dict, out: Path) -> dict:
    """Justify subcarrier selection: variance is far from uniform."""
    fig, axes = plt.subplots(1, len(nodes), figsize=(5.4 * len(nodes), 3.4), squeeze=False)
    selected = {}

    for col, (node, d) in enumerate(sorted(nodes.items())):
        amp = d["amp"]
        var = amp.var(axis=0)
        best = int(np.argmax(var))
        selected[node] = {
            "best": best,
            "best_var": float(var[best]),
            "mean_var": float(var.mean()),
            "ratio": float(var[best] / var.mean()) if var.mean() > 0 else 0.0,
            "dead": int(np.sum(var < var.mean() * 0.01)),
        }

        ax = axes[0][col]
        ax.bar(np.arange(var.size), var, color=C_PRIMARY, alpha=0.8, width=1.0)
        ax.bar([best], [var[best]], color=C_ACCENT, width=1.6,
               label=f"selected #{best} ({var[best]/var.mean():.1f}× mean)")
        ax.axhline(var.mean(), color=C_MUTED, ls=":", lw=1.2, label="mean variance")
        ax.set_title(f"Node {node} — temporal variance per subcarrier")
        ax.set_xlabel("subcarrier index")
        ax.set_ylabel("variance")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Subcarrier selection — respiration concentrates in a few subcarriers, "
        "so averaging all of them dilutes the signal",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out / "fig2_subcarrier_variance.png")
    plt.close(fig)
    return selected


def fig_respiration_extraction(nodes: dict, selected: dict, out: Path, window_s: float):
    """Show the full chain on one window: raw -> conditioned -> spectrum."""
    node = sorted(nodes)[0]
    d = nodes[node]
    amp, t = d["amp"], d["t"]
    fs = effective_rate(t)
    sc = selected[node]["best"]

    n = int(window_s * fs)
    # Pick the window whose selected-subcarrier variance is highest, i.e. the
    # segment where there is actually something to extract.
    best_start, best_var = 0, -1.0
    for start in range(0, max(1, amp.shape[0] - n), max(1, n // 4)):
        seg = amp[start : start + n, sc]
        if seg.size == n and seg.var() > best_var:
            best_var, best_start = seg.var(), start

    raw = amp[best_start : best_start + n, sc].astype(float)
    tt = np.arange(raw.size) / fs
    clean = detrend(hampel(raw))
    band = bandpass(clean, fs, *RESPIRATORY_BAND_HZ)
    est = respiratory_estimate(raw, fs)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2))

    axes[0].plot(tt, raw, color=C_MUTED, lw=0.9, label="raw amplitude")
    axes[0].plot(tt, hampel(raw), color=C_PRIMARY, lw=1.1, label="Hampel-filtered")
    axes[0].set_title(f"Node {node}, subcarrier {sc} — raw signal ({window_s:.0f}s window)")
    axes[0].set_ylabel("|H|")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(tt, band, color=C_GREEN, lw=1.4)
    axes[1].axhline(0, color=C_MUTED, lw=0.7)
    axes[1].set_title(
        f"Band-pass {RESPIRATORY_BAND_HZ[0]}–{RESPIRATORY_BAND_HZ[1]} Hz "
        "(zero-phase Butterworth) — respiratory component"
    )
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("time (s)")

    # Plot the SAME spectrum the estimator searches: a Welch PSD of the
    # BAND-PASSED signal, with the identical segment length dsp._band_power
    # uses. An earlier version drew a Hann-windowed raw FFT of the unfiltered
    # series while annotating it with the peak returned by
    # respiratory_estimate -- two different estimators over two different
    # signals on one axis. On some windows they disagreed badly enough that the
    # marked peak landed in a trough of the drawn curve, which is worse than
    # having no figure at all.
    nper = int(min(band.size, max(8, 2 ** int(math.log2(max(band.size // 2, 8))))))
    freqs, spec = sp_signal.welch(band, fs=fs, nperseg=nper)
    keep = freqs <= 1.2
    axes[2].plot(freqs[keep], spec[keep], color=C_PRIMARY, lw=1.3)
    axes[2].axvspan(*RESPIRATORY_BAND_HZ, color=C_GREEN, alpha=0.13,
                    label="respiratory band 0.1–0.5 Hz")
    if est.supported and est.bpm > 0:
        axes[2].axvline(est.bpm / 60.0, color=C_ACCENT, ls="--", lw=1.4,
                        label=f"peak {est.bpm:.1f} br/min")
    axes[2].set_title("Spectrum — respiratory peak")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("PSD (Welch)")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig3_respiration_extraction.png")
    plt.close(fig)
    return {"node": node, "subcarrier": sc, "window_s": window_s,
            "bpm": est.bpm, "confidence": est.confidence, "supported": est.supported}


def fig_timeline(results: dict, out: Path):
    """Breathing rate and motion evidence over the whole recording."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for node, r in sorted(results.items()):
        if r["bpm_t"]:
            axes[0].plot(r["bpm_t"], r["bpm"], lw=1.2, label=f"node {node}", alpha=0.9)
        if r["z_t"]:
            axes[1].plot(r["z_t"], r["z"], lw=0.9, label=f"node {node}", alpha=0.9)

    axes[0].axhspan(12, 20, color=C_GREEN, alpha=0.12,
                    label="normal adult resting (12–20)")
    axes[0].set_ylabel("breaths / min")
    axes[0].set_title("Estimated respiration rate over the recording")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)

    axes[1].axhline(3.0, color=C_ACCENT, ls="--", lw=1.1, label="enter threshold (3σ)")
    axes[1].axhline(1.5, color=C_GREEN, ls="--", lw=1.1, label="exit threshold (1.5σ)")
    axes[1].set_ylabel("z-score (σ)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Motion evidence — window dispersion vs ambient baseline")
    axes[1].legend(frameon=False, fontsize=8, ncol=4)

    fig.tight_layout()
    fig.savefig(out / "fig4_respiration_timeline.png")
    plt.close(fig)


def fig_node_agreement(results: dict, out: Path) -> dict:
    """Cross-check the two nodes against each other.

    The nodes observe the same person over different propagation paths, so
    agreement between their independent estimates is evidence the estimate
    reflects the subject rather than path-specific artefacts. This is the
    closest thing to validation available without a reference sensor.
    """
    ns = sorted(results)
    if len(ns) < 2:
        return {}

    a, b = results[ns[0]], results[ns[1]]
    if not a["bpm"] or not b["bpm"]:
        return {}

    # Resample both onto a common time grid before comparing.
    t0 = max(min(a["bpm_t"]), min(b["bpm_t"]))
    t1 = min(max(a["bpm_t"]), max(b["bpm_t"]))
    if t1 <= t0:
        return {}

    # Grid density is not cosmetic: it changes the measured correlation. A
    # fixed 240 points spans 12.8 s per sample on a 51-minute capture, which
    # undersamples estimates that update about once a second and depresses the
    # correlation (+0.235 at 240 points, converging to +0.30 for >=1000).
    # Sampling at the estimate update rate lands in the converged regime; the
    # bounds keep short and very long recordings sane.
    n_grid = int(np.clip(round(t1 - t0), 200, 8000))
    grid = np.linspace(t0, t1, n_grid)
    ia = np.interp(grid, a["bpm_t"], a["bpm"])
    ib = np.interp(grid, b["bpm_t"], b["bpm"])

    diff = ia - ib
    corr = float(np.corrcoef(ia, ib)[0, 1]) if ia.std() > 0 and ib.std() > 0 else 0.0
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    loa = 1.96 * float(np.std(diff))

    # Stacked, not side by side. At 11 in wide this had to be scaled to 0.57 to
    # reach the report's 451 pt column, printing its 9 pt labels at 5.2 pt. It
    # carries more than the headline correlation: the scatter shows the
    # estimator's quantisation lattice directly, and the Bland-Altman shows the
    # same thing as diagonal banding, so it needs to be legible.
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.4))

    axes[0].scatter(ia, ib, s=13, color=C_PRIMARY, alpha=0.55, edgecolors="none")
    lim = [min(ia.min(), ib.min()) - 1, max(ia.max(), ib.max()) + 1]
    axes[0].plot(lim, lim, color=C_MUTED, ls="--", lw=1.1, label="perfect agreement")
    axes[0].set_xlabel(f"node {ns[0]} (br/min)")
    axes[0].set_ylabel(f"node {ns[1]} (br/min)")
    axes[0].set_title(f"Node agreement — r = {corr:.3f}, MAE = {mae:.2f} br/min")
    axes[0].legend(frameon=False, fontsize=8)

    # Bland-Altman: the standard way to show agreement between two methods
    # measuring the same quantity.
    mean_ab = (ia + ib) / 2.0
    axes[1].scatter(mean_ab, diff, s=13, color=C_ACCENT, alpha=0.55, edgecolors="none")
    axes[1].axhline(bias, color=C_PRIMARY, lw=1.3, label=f"bias {bias:+.2f}")
    axes[1].axhline(bias + loa, color=C_MUTED, ls="--", lw=1.0,
                    label=f"±1.96 SD ({loa:.2f})")
    axes[1].axhline(bias - loa, color=C_MUTED, ls="--", lw=1.0)
    axes[1].set_xlabel("mean of the two nodes (br/min)")
    axes[1].set_ylabel("difference (br/min)")
    axes[1].set_title("Bland–Altman agreement")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig5_node_agreement.png")
    plt.close(fig)

    controls = agreement_controls(grid, ia, ib, t1 - t0)

    return {"nodes": ns, "corr": corr, "mae": mae, "bias": bias, "loa": loa,
            "points": int(grid.size), "controls": controls}


def _median_note(controls: list[dict]) -> str:
    """State what the median-difference column does or does not demonstrate.

    The shuffle control is the fair comparison for the median: it reorders one
    series without dropping any samples, so both medians are preserved exactly.
    The time-shift controls truncate the arrays, which moves the medians for
    reasons unrelated to signal -- so they cannot be used to make this point.
    """
    aligned = next((c for c in controls if c["pairing"].startswith("aligned")), None)
    shuffled = next((c for c in controls if "shuffled" in c["pairing"]), None)
    if not aligned or not shuffled:
        return ("Median agreement alone cannot separate shared signal from "
                "distributional similarity and is not treated as validation here.")

    if abs(aligned["median_diff"] - shuffled["median_diff"]) < 1e-6:
        return (
            f"**The median difference is unchanged by shuffling "
            f"({aligned['median_diff']:.3f} in both rows).** This is an algebraic identity, "
            "not a measurement: a permutation does not change the multiset of values, so "
            "any statistic computed from the two marginal distributions alone -- median "
            "difference, mean difference, SD ratio -- is invariant under re-pairing by "
            "construction. Median agreement therefore carries no information about temporal "
            "correspondence. The temporal correlation is the quantity that responds to "
            "pairing; the median is not."
        )
    return (
        f"Median difference is {aligned['median_diff']:.3f} aligned versus "
        f"{shuffled['median_diff']:.3f} shuffled. Any statistic of the marginal "
        "distributions alone is invariant under re-pairing, so medians are reported for "
        "completeness only; the temporal correlation is the quantity that responds to "
        "pairing."
    )


def agreement_controls(grid: np.ndarray, ia: np.ndarray, ib: np.ndarray,
                       span: float) -> list[dict]:
    """Decorrelate the two nodes and re-measure agreement.

    Two nodes in the same room see similar interference and similar noise
    statistics, so their *distributions* will resemble each other whether or
    not they are tracking a shared signal. Comparing aligned series alone
    therefore cannot distinguish "both observing the same chest" from "both
    observing similar noise".

    The control breaks the temporal pairing -- by time-shifting one series, and
    by shuffling it outright -- while leaving both distributions untouched. Any
    agreement that survives decorrelation is distributional and carries no
    evidential weight; agreement that *collapses* under the control is
    pairing-dependent, which is necessary for a shared signal but not sufficient
    to establish one.

    Two limitations, recorded here because they bound what this function can
    show. The shifts truncate rather than wrap, so every control is computed on
    fewer samples and over a different span than the aligned pairing; and a
    shift shorter than the decorrelation time of the input series retains shared
    slow structure, which inflates the control. A circular shift with a guard
    band measured from the autocorrelation would fix both, and is named as
    future work rather than pretended away here.

    The median column exists to make one point: it is identical under every
    control, because any statistic of the marginals alone is invariant under
    re-pairing. That is an algebraic fact, not a result.
    """
    out = []

    def measure(x, y, label):
        if x.size < 10 or x.std() == 0 or y.std() == 0:
            return
        out.append({
            "pairing": label,
            "corr": float(np.corrcoef(x, y)[0, 1]),
            "mae": float(np.mean(np.abs(x - y))),
            "median_diff": float(abs(np.median(x) - np.median(y))),
        })

    measure(ia, ib, "aligned (true pairing)")
    # Shift offsets scale with the recording, so short captures still get real
    # controls instead of silently falling back to the shuffle alone.
    for frac in (0.08, 0.17, 0.33, 0.5):
        shift = span * frac
        k = int(grid.size * frac)
        if 0 < k < grid.size - 10:
            measure(ia[:-k], ib[k:], f"shifted +{shift:.0f}s")

    rng = np.random.default_rng(0)  # fixed seed: the control must be reproducible
    shuffled = ib.copy()
    rng.shuffle(shuffled)
    measure(ia, shuffled, "fully shuffled")

    return out


# ---------------------------------------------------------------------------
# Detection pass
# ---------------------------------------------------------------------------


def run_detection(nodes: dict, window_s: float, calib_s: float,
                  resp_window_s: float, enter_z: float, exit_z: float) -> dict:
    """Run the live detector over each node's stream and collect the outputs."""
    results = {}

    for node, d in sorted(nodes.items()):
        amp, t = d["amp"], d["t"]
        fs = effective_rate(t)
        det = MotionDetector(
            fs=fs, window_s=window_s, calibration_s=calib_s,
            enter_z=enter_z, exit_z=exit_z, source_kind="csi",
            respiration_window_s=resp_window_s,
        )

        t0 = t[0]
        bpm, bpm_t, conf, zs, z_t, states = [], [], [], [], [], defaultdict(int)

        for i in range(amp.shape[0]):
            vec = amp[i]
            r = det.update(float(vec.mean()), d["rssi"][i], vec)
            states[r.state.value] += 1
            if r.calibration_progress >= 1.0:
                zs.append(r.z_score)
                z_t.append(float(t[i] - t0))
                if r.respiration.supported and r.respiration.bpm > 0:
                    bpm.append(r.respiration.bpm)
                    bpm_t.append(float(t[i] - t0))
                    conf.append(r.respiration.confidence)

        results[node] = {
            "fs": fs, "frames": amp.shape[0], "subcarriers": amp.shape[1],
            "duration": float(t[-1] - t[0]),
            "bpm": bpm, "bpm_t": bpm_t, "conf": conf,
            "z": zs, "z_t": z_t, "states": dict(states),
            "baseline": det.baseline.as_dict(),
            "selected": det.selected_subcarrier,
        }
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_outputs(path: Path, out: Path, ds: dict, sel: dict,
                  results: dict, agreement: dict, demo: dict) -> None:
    # --- summary CSV ---
    with (out / "results_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "node", "value", "unit"])
        for node, r in sorted(results.items()):
            arr = np.asarray(r["bpm"]) if r["bpm"] else np.zeros(0)
            rows = [
                ("frames", r["frames"], "count"),
                ("subcarriers", r["subcarriers"], "count"),
                ("duration", round(r["duration"], 2), "s"),
                ("sample_rate", round(r["fs"], 3), "Hz"),
                ("selected_subcarrier", r["selected"], "index"),
                ("baseline_mean_std", r["baseline"]["mean_std"], "-"),
                ("baseline_sigma", r["baseline"]["sigma_std"], "-"),
                ("z_max", round(float(np.max(r["z"])), 3) if r["z"] else "", "sigma"),
                ("z_mean", round(float(np.mean(r["z"])), 3) if r["z"] else "", "sigma"),
                ("breathing_n", arr.size, "count"),
                ("breathing_mean", round(float(arr.mean()), 2) if arr.size else "", "br/min"),
                ("breathing_median", round(float(np.median(arr)), 2) if arr.size else "", "br/min"),
                ("breathing_std", round(float(arr.std()), 2) if arr.size else "", "br/min"),
                ("breathing_min", round(float(arr.min()), 2) if arr.size else "", "br/min"),
                ("breathing_max", round(float(arr.max()), 2) if arr.size else "", "br/min"),
                ("in_normal_range_pct",
                 round(float(np.mean((arr >= 12) & (arr <= 20)) * 100), 1) if arr.size else "", "%"),
            ]
            for k, v, u in rows:
                w.writerow([k, node, v, u])

        for k, v in (agreement or {}).items():
            if k != "nodes":
                w.writerow([f"agreement_{k}", "-", round(v, 4) if isinstance(v, float) else v, "-"])

    # --- per-window breathing CSV ---
    with (out / "results_breathing.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["node", "time_s", "breaths_per_min", "band_power_ratio"])
        for node, r in sorted(results.items()):
            for tt, bb, cc in zip(r["bpm_t"], r["bpm"], r["conf"]):
                w.writerow([node, round(tt, 3), round(bb, 3), round(cc, 5)])

    # --- prose summary ---
    lines = [
        "# Results — WiFi CSI human sensing",
        "",
        f"Source recording: `{path.name}`",
        "",
        "All values below are measured from real Channel State Information captured",
        "by an ESP32-S3 mesh. Nothing here is simulated. Where a quantity could not",
        "be measured, that is stated rather than estimated.",
        "",
        "## Dataset",
        "",
        "| Node | Frames | Subcarriers | Duration (s) | Sample rate (Hz) |",
        "|------|--------|-------------|--------------|------------------|",
    ]
    for node, r in sorted(results.items()):
        lines.append(
            f"| {node} | {r['frames']} | {r['subcarriers']} | "
            f"{r['duration']:.1f} | {r['fs']:.2f} |"
        )

    lines += ["", "## Subcarrier selection", "",
              "| Node | Selected | Variance vs mean | Near-zero subcarriers |",
              "|------|----------|------------------|----------------------|"]
    for node, s in sorted(sel.items()):
        lines.append(f"| {node} | {s['best']} | {s['ratio']:.1f}× | {s['dead']} |")
    lines += ["",
              "Variance is strongly non-uniform across subcarriers, which is why the",
              "pipeline selects the highest-variance subcarrier rather than averaging.",
              "Averaging dilutes the respiratory component into subcarriers that carry none.",
              ""]

    lines += ["## Respiration", "",
              "| Node | Estimates | Mean | Median | SD | Range | Within 12–20 br/min |",
              "|------|-----------|------|--------|----|-------|---------------------|"]
    for node, r in sorted(results.items()):
        a = np.asarray(r["bpm"]) if r["bpm"] else np.zeros(0)
        if a.size:
            pct = float(np.mean((a >= 12) & (a <= 20)) * 100)
            lines.append(
                f"| {node} | {a.size} | {a.mean():.2f} | {np.median(a):.2f} | "
                f"{a.std():.2f} | {a.min():.1f}–{a.max():.1f} | {pct:.1f}% |"
            )
        else:
            lines.append(f"| {node} | 0 | — | — | — | — | — |")

    if agreement:
        lines += ["", "## Node agreement", "",
                  f"Nodes {agreement['nodes'][0]} and {agreement['nodes'][1]} observe the same",
                  "subject over independent propagation paths, so agreement between their",
                  "estimates is evidence the measurement reflects the subject rather than a",
                  "path-specific artefact.",
                  "",
                  f"- Pearson correlation: **{agreement['corr']:.3f}**",
                  f"- Mean absolute difference: **{agreement['mae']:.2f} br/min**",
                  f"- Bias: {agreement['bias']:+.2f} br/min",
                  f"- 95% limits of agreement: ±{agreement['loa']:.2f} br/min",
                  f"- Compared over {agreement['points']} resampled points",
                  ""]

        controls = agreement.get("controls") or []
        if controls:
            lines += [
                "### Control: is the agreement real, or distributional?",
                "",
                "Two nodes in the same room see similar noise, so similar *distributions*",
                "are expected whether or not they track a shared signal. The control breaks",
                "the temporal pairing (time-shift, shuffle) while leaving both distributions",
                "unchanged. Agreement that survives decorrelation is distributional and",
                "carries no evidential weight. Agreement that collapses is pairing-dependent,",
                "which is necessary for a shared signal but not sufficient to establish one.",
                "",
                "The shifts below truncate rather than wrap, so each control is computed on",
                "fewer samples and over a different span than the aligned pairing; and a shift",
                "shorter than the decorrelation time of a respiration-*rate* series retains",
                "shared slow structure, which inflates the control rather than the aligned",
                "value. This table is therefore descriptive, not a significance test.",
                "",
                "| Pairing | Correlation | MAE (br/min) | Median difference |",
                "|---------|-------------|--------------|-------------------|",
            ]
            for c in controls:
                lines.append(
                    f"| {c['pairing']} | {c['corr']:+.3f} | {c['mae']:.2f} | {c['median_diff']:.3f} |"
                )

            aligned = controls[0]["corr"]
            decorr = [c["corr"] for c in controls[1:]]
            worst = max(abs(v) for v in decorr) if decorr else 0.0

            # The verdict follows the measurement. A shared signal requires the
            # aligned pairing to be POSITIVE and clearly above every
            # decorrelated control; anything else is stated as unsupported.
            # Even when it clears that bar the statement stays descriptive: with
            # a handful of control rows the table cannot reach significance.
            if aligned > 0 and aligned > max(worst * 2.0, 0.1):
                n_ctrl = len(decorr)
                verdict = [
                    f"Aligned correlation is **{aligned:+.3f}**; the strongest decorrelated",
                    f"control reaches **{worst:.3f}**. The correlation is pairing-dependent --",
                    "it is not reproduced once the temporal correspondence between the two",
                    "nodes is broken.",
                    "",
                    f"This margin is descriptive, not inferential. With {n_ctrl} decorrelated",
                    f"rows the finest attainable p-value is 1/{n_ctrl + 1} = "
                    f"{1.0 / (n_ctrl + 1):.2f}, so no conventional significance threshold is",
                    "reachable from this table however large the margin looks.",
                    "",
                    "Pairing-dependence over the full recording does **not** establish",
                    "per-window respiration tracking. See the segment-stability test in",
                    "`ABLATION.md`, which re-measures this on disjoint segments and does not",
                    "support that interpretation.",
                ]
            elif aligned <= 0:
                verdict = [
                    f"Aligned correlation is **{aligned:+.3f}** -- negative. Two nodes observing",
                    "the same subject would correlate positively, so this recording provides",
                    "**no evidence of a shared respiratory signal**. The per-window estimates are",
                    "consistent with band-limited noise rather than respiration.",
                ]
            else:
                verdict = [
                    f"Aligned correlation is **{aligned:+.3f}**, against a strongest decorrelated",
                    f"control of **{worst:.3f}**. The margin is too small to distinguish shared",
                    "signal from distributional similarity, so agreement here is **not** evidence",
                    "of a real respiratory measurement.",
                ]
            lines += [""] + verdict + ["",
                _median_note(controls),
                "",
                (
                    "This does not establish accuracy. No reference respiration sensor was "
                    "recorded, so nothing here can support the estimate as a correct one -- "
                    "only as a consistency check between two observers."
                    if aligned > 0 and aligned > max(worst * 2.0, 0.1)
                    else
                    "Accuracy is not assessable either way: no reference respiration sensor "
                    "was recorded. The control establishes only that these estimates are not "
                    "supported as a shared measurement."
                ),
                "",
            ]

    lines += ["## Limitations", "",
              "- **No ground truth.** No reference respiration sensor was recorded alongside",
              "  the CSI, so accuracy cannot be stated. Node agreement is a consistency check,",
              "  not a validation against truth.",
              "- **Presence detection is a threshold, not a classifier.** It compares window",
              "  dispersion against a learned ambient baseline. It will respond to any",
              "  channel disturbance, including non-human sources such as fans.",
              "- **Pose estimation is not implemented.** No trained keypoint weights are",
              "  loaded, and the system draws no skeleton rather than drawing a fabricated one.",
              "- **Cardiac estimation is not attempted** at this sample rate and source.",
              "- **Commodity RSSI was evaluated and rejected.** On a MediaTek MT7922 adapter,",
              "  RSSI was static across 99 samples over 30 s via netsh and 236 samples over",
              "  12 s via direct WLAN API queries — driver-level smoothing removes the",
              "  fluctuation sensing depends on. This motivates dedicated CSI hardware.",
              "",
              "## Figures", "",
              "1. `fig1_dataset_overview.png` — amplitude heatmap and frame timing",
              "2. `fig2_subcarrier_variance.png` — variance per subcarrier, selection justified",
              "3. `fig3_respiration_extraction.png` — raw → band-passed → spectrum",
              "4. `fig4_respiration_timeline.png` — breathing rate and motion evidence over time",
              "5. `fig5_node_agreement.png` — scatter and Bland–Altman agreement",
              ""]

    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Offline CSI analysis -> report figures and tables")
    p.add_argument("--file", required=True, help="path to a .csi.jsonl recording")
    p.add_argument("--out", default="output", help="output directory")
    p.add_argument("--window", type=float, default=4.0, help="motion window, s")
    p.add_argument("--calibration", type=float, default=15.0, help="ambient calibration, s")
    p.add_argument("--resp-window", type=float, default=30.0, help="respiration window, s")
    p.add_argument("--enter-z", type=float, default=3.0)
    p.add_argument("--exit-z", type=float, default=1.5)
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"recording not found: {path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {path.name} ...")
    nodes = load_recording(path)
    if not nodes:
        raise SystemExit("no usable frames found")
    for node, d in sorted(nodes.items()):
        print(f"  node {node}: {d['amp'].shape[0]} frames x {d['amp'].shape[1]} subcarriers "
              f"@ {effective_rate(d['t']):.2f} Hz")

    print("fig1 dataset overview ...")
    ds = fig_dataset_overview(nodes, out)

    print("fig2 subcarrier variance ...")
    sel = fig_subcarrier_variance(nodes, out)
    for node, s in sorted(sel.items()):
        print(f"  node {node}: subcarrier {s['best']} carries {s['ratio']:.1f}x mean variance")

    print("fig3 respiration extraction ...")
    demo = fig_respiration_extraction(nodes, sel, out, args.resp_window)
    print(f"  node {demo['node']} sc {demo['subcarrier']}: {demo['bpm']:.1f} br/min")

    print("running detector over all nodes ...")
    results = run_detection(nodes, args.window, args.calibration,
                            args.resp_window, args.enter_z, args.exit_z)
    for node, r in sorted(results.items()):
        a = np.asarray(r["bpm"]) if r["bpm"] else np.zeros(0)
        if a.size:
            print(f"  node {node}: {a.size} estimates, median {np.median(a):.2f} br/min")
        else:
            print(f"  node {node}: no respiration estimates")

    print("fig4 timeline ...")
    fig_timeline(results, out)

    print("fig5 node agreement ...")
    agreement = fig_node_agreement(results, out)
    if agreement:
        print(f"  r = {agreement['corr']:.3f}, MAE = {agreement['mae']:.2f} br/min")
    else:
        print("  skipped (needs two nodes with estimates)")

    print("writing tables and RESULTS.md ...")
    write_outputs(path, out, ds, sel, results, agreement, demo)

    print(f"\ndone -> {out.resolve()}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
