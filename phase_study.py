"""Phase-domain investigation: can CSI phase improve respiration on this data?

In the CSI sensing literature respiration is usually recovered from *phase*,
not amplitude -- millimetric chest displacement shifts subcarrier phase while
barely moving magnitude. This study asks whether that holds on the ESP32
recordings used here, and answers honestly.

Raw ESP32 phase is not directly usable: it carries a carrier-frequency offset
(CFO, a drift over time) and a sampling-time offset (STO, a linear ramp across
subcarriers), both hardware artefacts far larger than any respiratory signal.
Two standard remedies exist:

  1. Single-antenna linear sanitization -- remove the linear-across-subcarrier
     term per frame. Feasible on every frame.
  2. Multi-antenna conjugate multiplication -- multiply one antenna's CSI by the
     conjugate of another, cancelling the common CFO/STO. Needs contiguous
     multi-antenna frames.

The script evaluates (1) against amplitude and tests whether (2) is even
possible on this capture, then writes figures and a report section.

Usage
-----
    python phase_study.py --file ../RuView/data/recordings/overnight-1775217646.csi.jsonl

Every result is measured. Where phase does not help, the study says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)

C_AMP = "#1f6fb4"
C_PHA = "#d1495b"
C_MUT = "#8d99ae"
C_GRN = "#2a9d8f"

RESP_BAND = (0.1, 0.5)  # Hz -> 6-30 breaths/min


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _complex_from_hex(iq_hex: str) -> np.ndarray:
    """Decode an ADR-018 I/Q hex payload into a complex CSI vector."""
    b = bytes.fromhex(iq_hex)
    if len(b) < 2:
        return np.zeros(0, dtype=complex)
    iq = np.frombuffer(b[: (len(b) // 2) * 2], dtype=np.int8).astype(float).reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


def load_single_antenna(path: Path, node: int, limit: int = 4000):
    """Load frames of the modal (single-antenna, 64-value) geometry for a node.

    Returns complex CSI, timestamps, and the detected geometry. Restricting to
    one geometry keeps the amplitude-vs-phase comparison on identical frames.
    """
    frames, ts = [], []
    counts: dict[int, int] = defaultdict(int)

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("node_id") != node:
                continue
            c = _complex_from_hex(rec.get("iq_hex", ""))
            if c.size:
                counts[c.size] += 1

    modal = max(counts, key=counts.get) if counts else 64

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("node_id") != node:
                continue
            c = _complex_from_hex(rec.get("iq_hex", ""))
            if c.size != modal:
                continue
            frames.append(c)
            ts.append(float(rec.get("timestamp", 0.0)))
            if len(frames) >= limit:
                break

    return np.array(frames), np.array(ts), modal


def multi_antenna_runs(path: Path, node: int) -> np.ndarray:
    """Return the run lengths of consecutive multi-antenna (>=128-value) frames.

    Conjugate multiplication needs a contiguous stretch of multi-antenna frames
    long enough for a respiratory window; this reports whether such stretches
    exist.
    """
    runs, run = [], 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("node_id") != node:
                continue
            size = len(rec.get("iq_hex", "")) // 4  # values per frame
            if size >= 128:
                run += 1
            elif run:
                runs.append(run)
                run = 0
    if run:
        runs.append(run)
    return np.array(runs) if runs else np.zeros(0, dtype=int)


def effective_rate(ts: np.ndarray) -> float:
    if ts.size < 2:
        return 20.0
    dt = np.diff(ts)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else 20.0


# ---------------------------------------------------------------------------
# Phase sanitization
# ---------------------------------------------------------------------------


def sanitize_phase(csi: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Single-antenna linear phase sanitization.

    For each frame, unwrap the phase across the active subcarriers and subtract
    a least-squares linear fit. This removes the sampling-time offset (the slope
    across subcarriers) and a constant offset -- the standard single-device
    remedy from the CSI literature. What remains is the phase variation that is
    not explained by those linear hardware terms.
    """
    idx = np.arange(active.size)
    out = np.zeros((csi.shape[0], active.size))
    for t in range(csi.shape[0]):
        ph = np.unwrap(np.angle(csi[t, active]))
        coef = np.polyfit(idx, ph, 1)
        out[t] = ph - np.polyval(coef, idx)
    return out


def band_concentration(x: np.ndarray, fs: float, band=RESP_BAND):
    """Return (peak frequency in br/min, in-band power fraction).

    The in-band fraction is the share of total spectral power inside the
    respiratory band -- a scale-free measure of how concentrated the signal is
    where breathing would be, comparable between amplitude and phase.
    """
    x = sp.detrend(np.asarray(x, dtype=float))
    if x.std() == 0 or x.size < 16:
        return 0.0, 0.0
    nperseg = int(min(x.size, 512))
    f, pxx = sp.welch(x, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return 0.0, 0.0
    peak = f[m][np.argmax(pxx[m])]
    frac = float(np.trapezoid(pxx[m], f[m]) / max(np.trapezoid(pxx, f), 1e-12))
    return peak * 60.0, frac


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_sanitization(csi, ts, active, out: Path):
    """Show the CFO/STO corruption and what sanitization removes."""
    # Stacked rather than side by side. At 13 in wide the figure had to be
    # scaled to 0.49 to reach the report's 451 pt column, which put its 9 pt
    # labels on the page at 4.4 pt -- well under the reference report's own
    # ~6-7 pt. At 6.2 in it needs no scaling at all.
    fig, ax = plt.subplots(3, 1, figsize=(6.2, 8.1))
    idx = np.arange(active.size)

    # The nulls are excluded from `active`, so the series jumps from raw 26 to
    # raw 38 partway along. Against an ACTIVE index that gap plots as a step,
    # and the step is large enough to read as the dominant feature. Marking it
    # is the difference between a reader seeing an artefact and seeing a fault.
    gaps = np.where(np.diff(active) > 1)[0]

    # Panel 1: raw phase across subcarriers at a few instants -> STO ramp
    for t in (10, 200, 500):
        if t < csi.shape[0]:
            ph = np.unwrap(np.angle(csi[t, active]))
            ax[0].plot(idx, ph, lw=1.1, alpha=0.85, label=f"frame {t}")
    # Line only, no in-plot label: a rotated caption here runs straight through
    # the three curves and the panel title already names what the step is.
    for g in gaps:
        ax[0].axvline(g + 0.5, color=C_MUT, lw=0.8, ls=":")
    ax[0].set_title("Raw phase across subcarriers\n"
                    "(STO ramp; the step is the excised guard band)")
    ax[0].set_xlabel("active subcarrier index")
    ax[0].set_ylabel("unwrapped phase (rad)")
    ax[0].legend(frameon=False, fontsize=8)

    # Panel 2: after sanitization -> ramp removed
    san = sanitize_phase(csi[: min(600, csi.shape[0])], active)
    for t in (10, 200, 500):
        if t < san.shape[0]:
            ax[1].plot(idx, san[t], lw=1.1, alpha=0.85, label=f"frame {t}")
    ax[1].axhline(0, color=C_MUT, lw=0.7)
    for g in gaps:
        ax[1].axvline(g + 0.5, color=C_MUT, lw=0.8, ls=":")
    # "noise" would be the wrong word for what this panel shows: the three
    # frames lie almost on top of one another, so what survives the linear fit
    # is a repeatable frequency-dependent error, not a random one.
    ax[1].set_title("After linear sanitization\n"
                    "(residual is structured, and repeats across frames)")
    ax[1].set_xlabel("active subcarrier index")
    ax[1].set_ylabel("residual phase (rad)")
    ax[1].legend(frameon=False, fontsize=8)

    # Panel 3: raw phase over time on one subcarrier -> CFO drift
    # `sc` is a RAW index while the panels above use an ACTIVE one; naming both
    # stops a reader looking for active index 38, which is a different carrier.
    mid = active.size // 2
    sc = active[mid]
    raw_t = np.angle(csi[:, sc])
    ax[2].plot(np.arange(raw_t.size), raw_t, lw=0.6, color=C_PHA)
    ax[2].set_title(f"Raw phase over time, raw subcarrier {sc} (active index {mid})\n"
                    "(CFO drift, wraps to noise)")
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("phase (rad)")

    fig.suptitle("Raw ESP32 phase is dominated by hardware offsets (CFO / STO)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "fig_phase1_sanitization.png")
    plt.close(fig)


def fig_amp_vs_phase(csi, ts, active, out: Path) -> dict:
    """Compare respiratory-band concentration: amplitude vs sanitized phase."""
    fs = effective_rate(ts)
    amp = np.abs(csi[:, active])
    san = sanitize_phase(csi, active)

    amp_frac, pha_frac = [], []
    amp_bpm, pha_bpm = [], []
    for j in range(active.size):
        ab, af = band_concentration(amp[:, j], fs)
        pb, pf = band_concentration(san[:, j], fs)
        amp_frac.append(af)
        pha_frac.append(pf)
        amp_bpm.append(ab)
        pha_bpm.append(pb)

    amp_frac = np.array(amp_frac)
    pha_frac = np.array(pha_frac)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    x = np.arange(active.size)
    ax[0].bar(x - 0.2, amp_frac, width=0.4, color=C_AMP, label="amplitude")
    ax[0].bar(x + 0.2, pha_frac, width=0.4, color=C_PHA, label="sanitized phase")
    ax[0].set_title("Respiratory-band power fraction per subcarrier")
    ax[0].set_xlabel("active subcarrier (index)")
    ax[0].set_ylabel("in-band power / total")
    ax[0].legend(frameon=False, fontsize=8)

    # Distribution summary
    ax[1].boxplot([amp_frac, pha_frac], tick_labels=["amplitude", "sanitized\nphase"],
                  patch_artist=True,
                  boxprops=dict(facecolor="#e8eef5"),
                  medianprops=dict(color=C_PHA, linewidth=1.6))
    ax[1].set_title("Distribution across subcarriers")
    ax[1].set_ylabel("in-band power fraction")
    ax[1].text(0.5, 0.94,
               f"amp median {np.median(amp_frac):.3f}   "
               f"phase median {np.median(pha_frac):.3f}",
               transform=ax[1].transAxes, ha="center", fontsize=8, color=C_MUT)

    fig.suptitle("Amplitude vs single-antenna sanitized phase — no phase advantage on this data",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_phase2_amp_vs_phase.png")
    plt.close(fig)

    return {
        "fs": fs,
        "amp_frac_mean": float(amp_frac.mean()),
        "amp_frac_max": float(amp_frac.max()),
        "pha_frac_mean": float(pha_frac.mean()),
        "pha_frac_max": float(pha_frac.max()),
        "amp_above_010": int((amp_frac > 0.10).sum()),
        "pha_above_010": int((pha_frac > 0.10).sum()),
        "n_active": int(active.size),
    }


def fig_contiguity(runs: np.ndarray, fs: float, resp_window_s: float, out: Path) -> dict:
    """Show that multi-antenna frames are too fragmented for conjugate mult."""
    need = int(resp_window_s * fs)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    if runs.size:
        maxr = int(runs.max())
        bins = np.arange(1, maxr + 2) - 0.5
        ax.hist(runs, bins=bins, color=C_AMP, alpha=0.85)
        ax.set_xlim(0.5, max(maxr + 0.5, 8.5))
    ax.axvline(need, color=C_PHA, ls="--", lw=1.6,
               label=f"needed for {resp_window_s:.0f}s window: {need} frames")
    ax.set_title("Contiguous multi-antenna runs — far short of a respiratory window")
    ax.set_xlabel("consecutive multi-antenna frames per run")
    ax.set_ylabel("number of runs")
    ax.legend(frameon=False, fontsize=9)

    # If the requirement is off the right edge, annotate it explicitly.
    if runs.size and need > runs.max() * 1.5:
        ax.text(0.5, 0.6,
                f"longest run = {int(runs.max())} frames ({runs.max()/fs:.2f}s)\n"
                f"requirement = {need} frames ({resp_window_s:.0f}s)\n"
                "conjugate multiplication is infeasible on this capture",
                transform=ax.transAxes, ha="center", fontsize=9, color=C_PHA,
                bbox=dict(boxstyle="round", fc="#fdecea", ec=C_PHA, alpha=0.9))

    fig.tight_layout()
    fig.savefig(out / "fig_phase3_contiguity.png")
    plt.close(fig)

    return {
        "runs": int(runs.size),
        "max_run": int(runs.max()) if runs.size else 0,
        "median_run": float(np.median(runs)) if runs.size else 0.0,
        "need_frames": need,
        "max_run_seconds": float(runs.max() / fs) if runs.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(path: Path, out: Path, cmp: dict, cont: dict) -> None:
    phase_helps = cmp["pha_frac_max"] > cmp["amp_frac_max"] * 1.2

    lines = [
        "# Phase-domain investigation",
        "",
        f"Source recording: `{path.name}`",
        "",
        "## Motivation",
        "",
        "In the CSI sensing literature, respiration is usually recovered from",
        "subcarrier **phase** rather than amplitude: chest displacement is",
        "millimetric and rotates the phase of the reflected path, while barely",
        "changing received power. This section tests whether phase offers that",
        "advantage on the ESP32 recordings used here.",
        "",
        "## The problem with raw ESP32 phase",
        "",
        "Raw phase is not directly usable. It carries two hardware artefacts far",
        "larger than any respiratory signal:",
        "",
        "- **Carrier-frequency offset (CFO)** — a drift over time that, once the",
        "  phase wraps, makes a single subcarrier's phase series look random.",
        "- **Sampling-time offset (STO)** — a linear ramp across subcarriers.",
        "",
        "`fig_phase1_sanitization.png` shows both: the raw phase across",
        "subcarriers is a sloped line (STO), and over time a single subcarrier",
        "wraps repeatedly (CFO).",
        "",
        "## Method 1 — single-antenna linear sanitization",
        "",
        "The standard single-device remedy removes, per frame, a linear fit of",
        "phase against subcarrier index. This cancels the STO slope and a",
        "constant offset. It is implemented in `sanitize_phase()` and applied on",
        "every frame.",
        "",
        "Comparing the respiratory-band power fraction of amplitude against",
        f"sanitized phase across {cmp['n_active']} active subcarriers "
        f"(sample rate {cmp['fs']:.1f} Hz):",
        "",
        "| Signal | Mean in-band fraction | Max | Subcarriers > 0.10 |",
        "|--------|-----------------------|-----|--------------------|",
        f"| Amplitude | {cmp['amp_frac_mean']:.3f} | {cmp['amp_frac_max']:.3f} | {cmp['amp_above_010']} |",
        f"| Sanitized phase | {cmp['pha_frac_mean']:.3f} | {cmp['pha_frac_max']:.3f} | {cmp['pha_above_010']} |",
        "",
    ]

    if phase_helps:
        lines += [
            "On this data sanitized phase concentrates respiratory-band power more",
            "than amplitude, consistent with the literature.",
        ]
    else:
        lines += [
            "**Sanitized phase does not improve on amplitude here** — the two are",
            "comparable and both weak. Single-antenna linear sanitization removes",
            "the linear hardware terms but leaves residual phase noise that is not",
            "smaller than the amplitude fluctuation, so it confers no advantage for",
            "respiration on this single-antenna capture. See",
            "`fig_phase2_amp_vs_phase.png`.",
        ]

    lines += [
        "",
        "## Method 2 — multi-antenna conjugate multiplication",
        "",
        "The technique that reliably recovers phase in the literature multiplies",
        "one antenna's CSI by the complex conjugate of another. The two antennas",
        "share the same CFO and STO, so the product cancels them and leaves a",
        "clean relative phase. This requires a contiguous run of multi-antenna",
        "frames spanning a respiratory window.",
        "",
        "This capture does not provide that. Multi-antenna frames occur, but as",
        "isolated frames scattered among single-antenna frames:",
        "",
        f"- Multi-antenna runs found: **{cont['runs']}**",
        f"- Median run length: **{cont['median_run']:.0f} frame(s)**",
        f"- Longest run: **{cont['max_run']} frames** "
        f"(~{cont['max_run_seconds']:.2f} s)",
        f"- Frames needed for one respiratory window: **{cont['need_frames']}**",
        "",
        "The longest contiguous multi-antenna stretch is therefore roughly two",
        f"orders of magnitude short of the {cont['need_frames']}-frame window a",
        "respiratory estimate needs. **Conjugate multiplication is infeasible on",
        "this capture.** See `fig_phase3_contiguity.png`.",
        "",
        "(An earlier attempt that pooled the scattered multi-antenna frames as if",
        "they were a continuous series produced a spuriously strong respiratory",
        "peak; treating irregularly-sampled frames as uniform is what created it.",
        "This is recorded here because it is the exact trap the contiguity check",
        "exists to catch.)",
        "",
        "## Conclusion",
        "",
        "- Raw ESP32 phase is corrupted by CFO and STO and must be sanitized.",
        "- Single-antenna linear sanitization is correct but yields **no",
        "  respiratory advantage over amplitude** on this data.",
        "- Multi-antenna conjugate multiplication — the method that works in the",
        "  literature — is **infeasible here** because the capture interleaves",
        "  single- and multi-antenna frames rather than providing contiguous",
        "  multi-antenna streams.",
        "",
        "Amplitude therefore remains the practical channel for this hardware and",
        "dataset, which is why the main pipeline uses it. Robust phase-based",
        "sensing needs **uniform multi-antenna capture** — a concrete hardware",
        "requirement for future work: an ESP32 configured to stream a fixed",
        "multi-antenna geometry every frame, or a Raspberry Pi with Nexmon CSI,",
        "which exposes wider bandwidth and consistent multi-antenna CSI.",
        "",
        "## Figures",
        "",
        "1. `fig_phase1_sanitization.png` — CFO/STO corruption and its removal",
        "2. `fig_phase2_amp_vs_phase.png` — amplitude vs sanitized phase",
        "3. `fig_phase3_contiguity.png` — multi-antenna frames too fragmented",
        "",
    ]

    (out / "PHASE_STUDY.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Phase-domain investigation for CSI respiration")
    p.add_argument("--file", required=True, help="path to a .csi.jsonl recording")
    p.add_argument("--node", type=int, default=1, help="node id to analyse")
    p.add_argument("--out", default="output_phase", help="output directory")
    p.add_argument("--resp-window", type=float, default=30.0, help="respiratory window, s")
    p.add_argument("--limit", type=int, default=4000, help="max frames to load")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"recording not found: {path}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {path.name}, node {args.node} ...")
    csi, ts, geo = load_single_antenna(path, args.node, args.limit)
    if csi.shape[0] < 200:
        raise SystemExit("not enough single-antenna frames")
    fs = effective_rate(ts)
    print(f"  {csi.shape[0]} frames x {geo} values, {fs:.2f} Hz")

    amp = np.abs(csi)
    mean_amp = amp.mean(axis=0)
    active = np.where(mean_amp > mean_amp.max() * 0.15)[0]
    print(f"  {active.size} active subcarriers")

    print("fig1 sanitization ...")
    fig_sanitization(csi, ts, active, out)

    print("fig2 amplitude vs phase ...")
    cmp = fig_amp_vs_phase(csi, ts, active, out)
    print(f"  amplitude in-band max {cmp['amp_frac_max']:.3f}, "
          f"phase max {cmp['pha_frac_max']:.3f}")

    print("fig3 multi-antenna contiguity ...")
    runs = multi_antenna_runs(path, args.node)
    cont = fig_contiguity(runs, fs, args.resp_window, out)
    print(f"  multi-antenna runs: {cont['runs']}, longest {cont['max_run']} frames, "
          f"need {cont['need_frames']}")

    print("writing PHASE_STUDY.md ...")
    write_report(path, out, cmp, cont)

    print(f"\ndone -> {out.resolve()}")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
