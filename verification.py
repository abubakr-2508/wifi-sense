"""Verification study: check what the pipeline assumes, using its own outputs.

Three questions the main studies take for granted, answered here so that a reader
can reproduce the answers rather than take them on trust:

  1. Is the dataset genuine measured CSI, or synthesised? The project that
     published these recordings has been publicly accused of shipping fabricated
     data, so this cannot be assumed either way.
  2. Does variance-based subcarrier selection concentrate on deep fades? A known
     objection to the top-K rule, which the ablation never tested because it
     swept how MANY subcarriers to take, not which.
  3. How many distinct values can the respiration estimator actually return, and
     how much of the cross-node agreement survives correction for that?

Writes output_verification/VERIFICATION.md plus three figures.

    ../.venv/Scripts/python.exe verification.py \
        --dir ../RuView/data/recordings --out output_verification
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_MAIN, C_ACC, C_MUT = "#1f77b4", "#d62728", "#888888"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_node(path: Path, node: int):
    """Amplitudes and timestamps for one node, modal frame geometry only.

    Same rule the analysis pipeline applies: captures interleave 1-, 2- and
    3-antenna frames whose amplitude statistics differ, so mixing them injects
    variance from population switching rather than from the channel.
    """
    rows, meta = [], {"empty": 0, "total": 0, "rssi": [], "payloads": Counter()}
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("node_id") != node:
            continue
        meta["total"] += 1
        h = rec.get("iq_hex", "")
        if not h:
            meta["empty"] += 1
            continue
        meta["payloads"][h] += 1
        if isinstance(rec.get("rssi"), (int, float)):
            meta["rssi"].append(rec["rssi"])
        rows.append((float(rec.get("timestamp", 0.0)), h))

    geo = Counter(len(h) // 4 for _, h in rows)
    modal = max(geo, key=geo.get) if geo else 0
    t, amps, iq = [], [], []
    for ts, h in rows:
        if len(h) // 4 != modal:
            continue
        v = np.frombuffer(bytes.fromhex(h), dtype=np.int8).reshape(-1, 2).astype(float)
        iq.append(v)
        amps.append(np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2))
        t.append(ts)
    meta["geometry"] = dict(sorted(geo.items()))
    meta["modal"] = modal
    return np.array(t), np.array(amps), np.concatenate(iq) if iq else np.zeros((0, 2)), meta


# ---------------------------------------------------------------------------
# 1. Is the data real?
# ---------------------------------------------------------------------------

def authenticity(t, amp, iq, meta):
    d = np.diff(np.sort(t))
    d = d[(d > 0) & (d < 5)]
    mean_amp = amp.mean(axis=0)
    nulls = np.where(mean_amp == 0)[0]
    dup = sum(c for c in meta["payloads"].values() if c > 1)
    rails = int(np.count_nonzero((iq >= 127) | (iq <= -128)))
    return {
        "frames": int(amp.shape[0]),
        "geometry": meta["geometry"],
        "dt_cv": float(d.std() / d.mean()) if d.size else 0.0,
        "dt_median_ms": float(np.median(d) * 1000) if d.size else 0.0,
        "span_min": float((t.max() - t.min()) / 60) if t.size else 0.0,
        "empty_frames": meta["empty"],
        "repeated_payloads": int(dup - meta["empty"]) if dup >= meta["empty"] else int(dup),
        "null_subcarriers": nulls.tolist(),
        "active_subcarriers": int(mean_amp.size - nulls.size),
        "iq_at_rails_pct": 100.0 * rails / max(iq.size, 1),
        "rssi_distinct": len(set(meta["rssi"])),
        "mean_amp": mean_amp,
        "var": amp.var(axis=0),
    }


# ---------------------------------------------------------------------------
# 2. Does top-K by variance land in the fades?
# ---------------------------------------------------------------------------

def fade_relation(mean_amp, var, k=8):
    active = np.where(mean_amp > 0)[0]
    m, v = mean_amp[active], var[active]
    top = active[np.argsort(v)[::-1][:k]]
    # Pearson without scipy: the pipeline already depends on numpy only here.
    r = float(np.corrcoef(m, v)[0, 1])
    below = int(sum(1 for sc in top if mean_amp[sc] < np.median(m)))
    return {"r": r, "top_k": sorted(top.tolist()), "below_median": below, "k": k,
            "active": active, "m": m, "v": v}


# ---------------------------------------------------------------------------
# 3. Estimator resolution and chance-corrected agreement
# ---------------------------------------------------------------------------

def quantisation(csv_path: Path):
    """Read the pipeline's own breathing output and measure what it can express."""
    if not csv_path.exists():
        return None
    by = defaultdict(dict)
    for row in csv.DictReader(csv_path.open()):
        by[row["node"]][round(float(row["time_s"]), 1)] = float(row["breaths_per_min"])
    if len(by) < 2:
        return None

    stats = {}
    for n, series in sorted(by.items()):
        c = Counter(round(x, 3) for x in series.values())
        vals = sorted(c)
        stats[n] = {
            "n": len(series),
            "distinct": len(c),
            "values": vals,
            "spacing": float(np.median(np.diff(vals))) if len(vals) > 1 else 0.0,
            "top3_share": 100.0 * sum(sorted(c.values(), reverse=True)[:3]) / len(series),
        }

    # Which of the observed values are real Welch bins, and which are artefacts?
    # Respiration is the median of K = 8 per-subcarrier estimates. Each estimate
    # is quantised to the Welch grid, but a median over an EVEN count averages
    # the two central values, so two adjacent bins produce a point exactly
    # halfway between them. The output grid is therefore the true grid plus its
    # midpoints, and only the even-indexed values are frequencies the estimator
    # can actually resolve.
    for n, s in stats.items():
        idx = [int(round(v / s["spacing"])) for v in s["values"]]
        s["indices"] = idx
        s["real_bins"] = [v for v, i in zip(s["values"], idx) if i % 2 == 0]
        s["midpoints"] = [v for v, i in zip(s["values"], idx) if i % 2 == 1]
        s["true_spacing"] = s["spacing"] * 2

    a, b = [by[k] for k in sorted(by)][:2]
    common = sorted(set(a) & set(b))
    A = np.array([a[k] for k in common])
    B = np.array([b[k] for k in common])
    # Compare bin INDEX, so the two nodes' slightly different grids align.
    ia = np.rint(A / stats[sorted(by)[0]]["spacing"]).astype(int)
    ib = np.rint(B / stats[sorted(by)[1]]["spacing"]).astype(int)
    n = ia.size

    labels = sorted(set(ia.tolist()) | set(ib.tolist()))
    pos = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    O = np.zeros((K, K))
    for x, y in zip(ia, ib):
        O[pos[x], pos[y]] += 1
    O /= O.sum()
    row, col = O.sum(axis=1), O.sum(axis=0)
    E = np.outer(row, col)

    I, J = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")

    def weighted(w):
        """Cohen's weighted kappa for a disagreement-weight matrix."""
        den = float((w * E).sum())
        return 1.0 - float((w * O).sum()) / den if den > 0 else float("nan")

    kappa = weighted((I != J).astype(float))          # unweighted
    k_lin = weighted(np.abs(I - J) / max(K - 1, 1))    # linear
    k_quad = weighted(((I - J) / max(K - 1, 1)) ** 2)  # quadratic == ICC

    # Diagnostics that separate the competing explanations for the bimodality.
    near = float(O[np.abs(I - J) <= 1].sum())
    ratio2 = 0.0
    for x in range(K):
        for y in range(K):
            if x == y:
                continue
            lo, hi = sorted((labels[x], labels[y]))
            if lo > 0 and hi == 2 * lo:
                ratio2 += O[x, y]
    off = float(O.sum() - np.trace(O))

    # Do the nodes change bin at the same moments? Synchronised changes point to
    # a property of the signal; independent ones to noise-driven peak selection.
    ca_, cb_ = np.diff(ia) != 0, np.diff(ib) != 0
    both = float((ca_ & cb_).mean())
    indep = float(ca_.mean() * cb_.mean())
    phi = float(np.corrcoef(ca_.astype(float), cb_.astype(float))[0, 1])

    return {"per_node": stats, "windows": n,
            "po": float((ia == ib).mean()), "pe": float(E.trace()),
            "kappa": kappa, "kappa_linear": k_lin, "kappa_quadratic": k_quad,
            "pearson": float(np.corrcoef(A, B)[0, 1]),
            "labels": labels, "table": O,
            "near_diagonal": near, "ratio2_of_off": ratio2 / off if off > 0 else 0.0,
            "max_row": float(row.max()), "max_col": float(col.max()),
            "flip_both": both, "flip_independent": indep, "flip_phi": phi}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_profile(res, out: Path):
    fig, axes = plt.subplots(1, len(res), figsize=(5.6 * len(res), 3.4), squeeze=False)
    for i, (label, a) in enumerate(res.items()):
        ax = axes[0][i]
        ax.bar(np.arange(a["mean_amp"].size), a["mean_amp"], color=C_MAIN, width=1.0)
        for sc in a["null_subcarriers"]:
            ax.axvspan(sc - 0.5, sc + 0.5, color=C_ACC, alpha=0.18, lw=0)
        ax.set_title(f"{label} — mean amplitude per subcarrier")
        ax.set_xlabel("subcarrier index")
        ax.set_ylabel("mean amplitude")
        ax.text(0.02, 0.94, f"{a['active_subcarriers']} active, "
                            f"{len(a['null_subcarriers'])} null (shaded)",
                transform=ax.transAxes, fontsize=8, va="top")
    fig.suptitle("Null pattern matches 802.11n HT20: DC null plus guard band, 52 active",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_v1_null_pattern.png", dpi=140)
    plt.close(fig)


def fig_fade(res, out: Path):
    fig, axes = plt.subplots(1, len(res), figsize=(4.6 * len(res), 3.6), squeeze=False)
    for i, (label, a) in enumerate(res.items()):
        f = a["fade"]
        ax = axes[0][i]
        ax.scatter(f["m"], f["v"], s=14, color=C_MUT, label="subcarrier")
        sel = [list(f["active"]).index(s) for s in f["top_k"]]
        ax.scatter(f["m"][sel], f["v"][sel], s=42, color=C_ACC, zorder=3,
                   label=f"selected top-{f['k']}")
        ax.axvline(np.median(f["m"]), color=C_MUT, ls=":", lw=1)
        ax.set_title(f"{label}   r = {f['r']:+.3f}", fontsize=9)
        ax.set_xlabel("mean amplitude")
        ax.set_ylabel("temporal variance")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Does variance-based selection land in the fades?", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_v2_variance_vs_amplitude.png", dpi=140)
    plt.close(fig)


def fig_bins(q, out: Path):
    if not q:
        return
    # The SPLIT is the finding, so the figure has to draw it. A median over an
    # even K averages the two central values, so every second observed value
    # falls between two Welch bins, at a frequency the transform never offered.
    # Plotting all nine as identical dots showed a reader nothing at all --
    # found by looking at the printed figure beside the text it illustrates.
    # 6.2 in wide rather than 7.4 so the report's 451 pt column needs no
    # scaling and the 9 pt labels stay 9 pt on paper.
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for i, (n, s) in enumerate(sorted(q["per_node"].items())):
        col = C_MAIN if i == 0 else C_ACC
        ax.plot(s["real_bins"], np.full(len(s["real_bins"]), i), "o", ms=9,
                color=col, zorder=3,
                label=f"node {n}: {len(s['real_bins'])} resolvable")
        ax.plot(s["midpoints"], np.full(len(s["midpoints"]), i), "o", ms=9,
                mfc="none", mec=col, mew=1.5, zorder=3,
                label=f"node {n}: {len(s['midpoints'])} median artefact")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["node 1", "node 2"])
    ax.set_ylim(-0.7, 1.9)
    ax.set_xlabel("breaths per minute")
    ax.set_title("Only half of what the estimator returns\n"
                 "is a frequency it can resolve", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(out / "fig_v3_estimator_bins.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------

def fig_confusion(q, out: Path):
    if not q:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    lab = q["labels"]
    im = ax.imshow(q["table"], cmap="viridis", origin="lower")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=7)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("node 2 bin index"); ax.set_ylabel("node 1 bin index")
    ax.set_title("Where the two nodes land, window by window", fontsize=10)
    # Mark the cells an octave error would populate.
    for x, lx in enumerate(lab):
        for y, ly in enumerate(lab):
            if lx > 0 and ly == 2 * lx:
                ax.add_patch(plt.Rectangle((y - .5, x - .5), 1, 1, fill=False,
                                           ec=C_ACC, lw=1.4))
            if ly > 0 and lx == 2 * ly:
                ax.add_patch(plt.Rectangle((y - .5, x - .5), 1, 1, fill=False,
                                           ec=C_ACC, lw=1.4))
    fig.colorbar(im, ax=ax, label="fraction of windows")
    fig.tight_layout()
    fig.savefig(out / "fig_v4_confusion.png", dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Verification study")
    p.add_argument("--dir", required=True, help="directory of .csi.jsonl recordings")
    p.add_argument("--breathing", default="output_overnight/results_breathing.csv")
    p.add_argument("--out", default="output_verification")
    p.add_argument("--file", default="overnight-1775217646.csi.jsonl")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = Path(args.dir) / args.file

    res = {}
    for node in (1, 2):
        print(f"  loading node {node} ...")
        t, amp, iq, meta = load_node(src, node)
        a = authenticity(t, amp, iq, meta)
        a["fade"] = fade_relation(a["mean_amp"], a["var"])
        res[f"node {node}"] = a
        print(f"    {a['frames']} frames, {a['active_subcarriers']} active subcarriers, "
              f"dt CV {a['dt_cv']:.2f}, fade r {a['fade']['r']:+.3f}")

    print("  reading the pipeline's breathing output ...")
    q = quantisation(Path(args.breathing))
    if q:
        print(f"    kappa = {q['kappa']:+.3f}  (po {q['po']:.3f}, chance floor {q['pe']:.3f})")

    print("  figures ...")
    fig_profile(res, out)
    fig_fade(res, out)
    fig_bins(q, out)
    fig_confusion(q, out)

    write_report(out / "VERIFICATION.md", src, res, q)
    print(f"\ndone -> {out.resolve()}")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


def write_report(path: Path, src: Path, res: dict, q):
    L = ["# Verification", "",
         f"Source recording: `{src.name}`", "",
         "Three things the other studies assume. Checked here so a reader can",
         "repeat the checks rather than take them on trust.", "",
         "## 1. Is the data genuine measured CSI?", ""]

    for label, a in res.items():
        L += [f"**{label}** — {a['frames']} frames over {a['span_min']:.1f} minutes.", "",
              f"| Test | Result | Reading |",
              f"|---|---|---|",
              f"| Always-zero subcarriers | {a['null_subcarriers']} | "
              f"DC null plus guard band |",
              f"| Active subcarriers | **{a['active_subcarriers']}** | "
              f"802.11n HT20 carries exactly 52 |",
              f"| Inter-frame timing CV | **{a['dt_cv']:.2f}** | "
              f"bursty and contended; a generator is regular |",
              f"| Repeated payloads | {a['repeated_payloads']} | mock data cycles |",
              f"| I/Q samples at the int8 rails | **{a['iq_at_rails_pct']:.3f}%** | "
              f"random fill would saturate |",
              f"| Distinct RSSI values | {a['rssi_distinct']} | a real link drifts |",
              f"| Frame geometries | {a['geometry']} | occasional multi-antenna frames |",
              ""]

    L += ["The null pattern is the decisive test. Twelve subcarriers are zero in every",
          "frame and they sit at index 0 and indices 27-37 — the DC null and the guard",
          "band of an 802.11n 20 MHz channel — leaving 52 active, which is exactly the",
          "48 data plus 4 pilot subcarriers the standard defines. Nothing synthetic",
          "reproduces that pattern and a physically plausible fading profile by accident.",
          "",
          "This matters because the project that published these recordings has been",
          "publicly criticised for shipping fabricated output. That criticism is",
          "well-founded for its *derived* frames, which are not used here: across the",
          "whole recording the `vitals` packets report a constant occupancy of 4 people",
          "and a `presence_score` identical to `motion_energy` in every single frame.",
          "This project reads only the `raw_csi` payloads and computes everything else",
          "itself.", "",
          "See `fig_v1_null_pattern.png`.", "",
          "## 2. Does variance-based subcarrier selection land in deep fades?", ""]

    L += ["A known objection to top-K-by-variance is that the highest-variance",
          "subcarriers tend to sit in deep fades, where they are over-sensitive. The",
          "ablation swept how *many* subcarriers to take but never which rule picks",
          "them, so the objection is tested here directly.", "",
          "| Node | corr(mean amplitude, variance) | selected top-8 | below median amplitude |",
          "|---|---|---|---|"]
    for label, a in res.items():
        f = a["fade"]
        L.append(f"| {label} | {f['r']:+.3f} | {f['top_k']} | {f['below_median']} of {f['k']} |")
    L += ["", "See `fig_v2_variance_vs_amplitude.png`.", ""]

    if q:
        n1 = q["per_node"][sorted(q["per_node"])[0]]
        L += ["## 3. What can the respiration estimator actually express?", "",
              "Respiration is the median of K = 8 per-subcarrier estimates, and each of",
              "those is the strongest Welch bin inside 0.1-0.5 Hz. The output is therefore",
              "discrete. It is also finer-grained than it looks, for a reason worth",
              "stating: a median over an EVEN number of values averages the two central",
              "ones, so whenever those fall on adjacent bins the result lands exactly",
              "halfway between them. The observed grid is the true Welch grid plus its",
              "midpoints. Only the even-indexed values are frequencies the estimator can",
              "resolve; the rest are averaging artefacts. An odd K would remove them.", ""]
        for n, s in sorted(q["per_node"].items()):
            L += [f"- **Node {n}**: {s['n']} windows taking **{s['distinct']} distinct "
                  f"values**, apparently spaced {s['spacing']:.3f} bpm apart, but resting "
                  f"on only **{len(s['real_bins'])} real bins** at {s['true_spacing']:.3f} "
                  f"bpm. Three values hold {s['top3_share']:.1f}% of the output.",
                  f"  Resolvable: {', '.join(f'{v:.2f}' for v in s['real_bins'])}",
                  f"  Median artefacts: {', '.join(f'{v:.2f}' for v in s['midpoints'])}"]
        L += ["",
              "Two of these values stand in an exact 2:1 ratio. That is **not** evidence",
              "of an octave ambiguity, and an earlier draft of this file wrongly said it",
              "was: a uniform grid contains 2:1 index pairs by construction, so the",
              "exactness of the ratio carries no information about the signal. What does",
              "need explaining is the *bimodality* -- two non-adjacent values holding",
              "roughly half the mass -- and the contingency table below is what",
              "distinguishes the candidate explanations.", "",
              "Because the output is discrete, agreement between the nodes carries a floor",
              "that owes nothing to physiology: two independent estimators restricted to",
              "the same handful of bins coincide some of the time by construction. Kappa",
              "corrects for that.", "",
              "| Quantity | Value |", "|---|---|",
              f"| Windows where both nodes reported | {q['windows']} |",
              f"| Exact agreement | {100*q['po']:.1f}% |",
              f"| Chance floor from the bin structure | **{100*q['pe']:.1f}%** |",
              f"| Cohen's kappa, unweighted | {q['kappa']:+.3f} |",
              f"| Cohen's kappa, linear weights | {q['kappa_linear']:+.3f} |",
              f"| **Cohen's kappa, quadratic weights** | **{q['kappa_quadratic']:+.3f}** |",
              f"| Pearson r on the same series | {q['pearson']:+.3f} |", "",
              "The Pearson value reproduces the correlation in `RESULTS.md`, so this is",
              "the same comparison seen two ways.", "",
              "The bins are ORDERED, so unweighted kappa is the wrong variant on its own --",
              "it treats a one-bin miss and a four-bin miss as identical failures. The",
              "quadratic-weighted figure is the appropriate headline; note that it is",
              "equivalent to the intraclass correlation. All three are given because the",
              "gap between them measures how much of the disagreement is large-magnitude",
              "rather than adjacent-bin.", "",
              "**Two caveats, both of which apply here.** The conventional interpretive",
              "bands for kappa are convention rather than derived result -- their authors",
              "offered no evidence for them -- so no verbal label is attached to these",
              "numbers. And kappa is known to misbehave when the marginal distributions",
              "are unbalanced, which is exactly this case: three of the nine values hold",
              "three quarters of the mass. The raw agreement and the chance floor are",
              "given above so the coefficient can be checked against its inputs.", "",
              "### What the contingency table says", "",
              f"- Mass on the diagonal or its immediate neighbours: **{100*q['near_diagonal']:.1f}%**",
              f"- Share of the OFF-diagonal mass in cells where the bin indices stand in a",
              f"  2:1 ratio: **{100*q['ratio2_of_off']:.1f}%**",
              f"- Largest single row / column share: {100*q['max_row']:.1f}% / {100*q['max_col']:.1f}%",
              f"- Windows where both nodes changed bin at once: {100*q['flip_both']:.1f}%,",
              f"  against {100*q['flip_independent']:.1f}% if the two changed independently",
              f"  (phi = {q['flip_phi']:+.3f})", "",
              "Mass concentrated near the diagonal indicates coarse quantisation of a",
              "single underlying distribution; mass in the 2:1 cells would indicate octave",
              "confusion; mass concentrated in one row or column would indicate one node",
              "locking onto something non-respiratory. Synchronised changes would suggest",
              "a property of the signal, independent ones noise-driven peak selection.",
              "See `fig_v4_confusion.png`, where the 2:1 cells are outlined.", "",
              "**A limitation not addressed here.** Parabolic interpolation of the",
              "periodogram peak would give sub-bin resolution and remove the need for a",
              "categorical statistic altogether. It is not implemented: doing so would",
              "change every respiration number in this report, and the pipeline is frozen.",
              "It is the first thing to try if this work is continued.", "",
              "None of this replaces the segment-stability test in `ABLATION.md`, which is",
              "direct empirical evidence and stands on its own. This section describes a",
              "mechanism by which the apparent agreement was possible.", "",
              "See `fig_v3_estimator_bins.png`.", ""]

    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
