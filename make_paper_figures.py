"""Figures drawn for the journal paper rather than for the thesis.

    ../.venv/Scripts/python.exe make_paper_figures.py

The paper reuses most of its figures from the analysis studies. This file holds
the ones that have no analysis behind them because they describe the method
rather than a measurement.

Same conventions as `make_design_figures.py`, whose drawing layer is imported
rather than copied so that a change to the visual language reaches both files:
one data unit is one printed point, so `fontsize=9` really is 9 pt on paper,
and the document inserts the image at its native size with no scaling.

The one measured difference is the column. The thesis text column is 451 pt;
the journal template is A4 with 20 mm side margins, giving **481.9 pt**,
measured from the template's own sectPr rather than assumed. Authoring at
460 pt leaves the figure comfortably inside it and still needs no scaling, so
the 9 pt floor survives to the printed page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from make_design_figures import (
    INK,
    LW,
    WHITE,
    arrow,
    canvas,
    note,
    poly_arrow,
    save,
)

# The loader is imported rather than rewritten. Chapter 7 of the thesis records
# that the I/Q decode already exists three times over and names consolidating it
# as outstanding work; adding a fourth copy here would make that worse.
from verification import C_ACC, C_MAIN, authenticity, load_node

# FONT, and the reasoning is worth recording because the obvious choice is the
# wrong one. `make_design_figures` sets Arial; the five figures this paper
# reuses from the analysis studies set no family and so render in matplotlib's
# default, which is wider. Forcing the diagram into that default for the sake of
# a uniform look was tried and measured: four of its boxes overflowed at once
# and one label ran off the canvas, and the only way to fit them was to chop
# every label into a stack of two-word lines. A block diagram and a data plot
# carrying different sans faces is an ordinary distinction; an unreadable block
# diagram is not. So the diagram keeps Arial and the plot keeps the default,
# scoped per figure rather than set globally.
DATA_FONT = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}

# The journal column, measured from Example_paper_template-IJWMT.docx.
COL_PT = 481.9
W = 460.0
H = 300.0
M = 8.0             # side margin inside the canvas
FS = 9.0            # the floor, as in the design figures


def _row(n, width_total, gap):
    """Centres for n equal boxes spanning the usable width."""
    w = (width_total - (n - 1) * gap) / n
    return w, [M + w / 2 + i * (w + gap) for i in range(n)]


def flow_box(ax, x, y, w, h, text, sink):
    """A labelled box that remembers its label so the fit can be checked.

    Written because switching this file's font broke four boxes at once and
    nothing complained: the text simply ran through the outline and, in one
    case, off the canvas. A diagram whose labels are laid out by hand needs a
    machine to confirm they fit, in the same way every other number in this
    project is measured rather than eyeballed.
    """
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=WHITE,
                           edgecolor=INK, linewidth=LW, zorder=2))
    t = ax.text(x, y, text, ha="center", va="center", fontsize=FS, color=INK,
                linespacing=1.30, zorder=5)
    sink.append((t, w, h))
    return t


def check_fit(fig, sink, pad=8.0):
    """Report any label wider or taller than the box drawn around it."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, w, h in sink:
        bb = t.get_window_extent(r)
        tw = bb.width / fig.dpi * 72.0
        th = bb.height / fig.dpi * 72.0
        if tw > w - pad or th > h - pad:
            bad += 1
            print(f"    !! overflow: {t.get_text()!r:<44} "
                  f"needs {tw:.0f}x{th:.0f} pt in a {w:.0f}x{h:.0f} pt box")
    return bad


def fig_method():
    """Fig. 1 -- the estimator, and where its output set comes from.

    The diagram has to carry one thing the prose cannot show compactly: each of
    the eight per-subcarrier estimates is confined to the Welch grid, and the
    median of an EVEN number of them can land between two grid points. That is
    the mechanism Section 5 measures, so it is drawn rather than described.
    """
    fig, ax = canvas(W, H)
    usable = W - 2 * M
    fitted = []

    # --- lane label -------------------------------------------------------
    ax.text(M, H - 9, "For each receiver, independently",
            ha="left", va="center", fontsize=FS, color=INK, style="italic")

    # --- row A: conditioning ---------------------------------------------
    ya, ha_ = 244.0, 50.0
    wa, xa = _row(5, usable, 17.0)
    labels_a = [
        "CSI frames\none receiver",
        "modal geometry\nfilter",
        "amplitude\nper subcarrier",
        "outlier removal,\nzero-phase\n0.1–0.5 Hz",
        "rank by\nvariance,\nkeep K = 8",
    ]
    for x, t in zip(xa, labels_a):
        flow_box(ax, x, ya, wa, ha_, t, fitted)
    for i in range(4):
        arrow(ax, (xa[i] + wa / 2, ya), (xa[i + 1] - wa / 2, ya))

    # --- row A -> row B ---------------------------------------------------
    yb, hb = 158.0, 50.0
    wb, xb = _row(4, usable, 20.0)
    poly_arrow(ax, [(xa[4], ya - ha_ / 2), (xa[4], 208.0),
                    (xb[0], 208.0), (xb[0], yb + hb / 2)])

    # --- row B: estimation ------------------------------------------------
    labels_b = [
        "Welch periodogram\n30 s window\n(one per subcarrier)",
        "largest in-band\nordinate\n→ breaths/min",
        "median of\nthe eight",
        "rate series\nfor this receiver",
    ]
    for x, t in zip(xb, labels_b):
        flow_box(ax, x, yb, wb, hb, t, fitted)
    for i in range(3):
        arrow(ax, (xb[i] + wb / 2, yb), (xb[i + 1] - wb / 2, yb))

    # --- the output-set schematic ----------------------------------------
    # Nine positions: five on the transform's own grid, four sitting between
    # adjacent pairs. Drawn to scale with the measured result rather than
    # decoratively -- filled is resolvable, hollow is a median artefact.
    gx0, gx1, gy = xb[1] - wb / 2, xb[2] + wb / 2, 104.0
    ax.add_line(Line2D([gx0, gx1], [gy, gy], color=INK, linewidth=0.7,
                       zorder=2))
    step = (gx1 - gx0) / 8.0
    for i in range(9):
        x = gx0 + i * step
        if i % 2 == 0:
            ax.plot([x], [gy], marker="o", ms=5.0, color=INK, zorder=3)
        else:
            ax.plot([x], [gy], marker="o", ms=5.0, markerfacecolor=WHITE,
                    markeredgecolor=INK, markeredgewidth=0.9, zorder=3)

    # Name the first of each kind with a short leader, so the alternation is
    # labelled where it happens rather than in a floating legend.
    ax.add_line(Line2D([gx0, gx0], [gy + 4, gy + 10], color=INK,
                       linewidth=0.6, zorder=2))
    ax.text(gx0, gy + 17, "resolvable", ha="center", va="center",
            fontsize=FS, color=INK)
    ax.add_line(Line2D([gx0 + step, gx0 + step], [gy - 4, gy - 10], color=INK,
                       linewidth=0.6, zorder=2))
    ax.text(gx0 + step, gy - 17, "interpolated", ha="center", va="center",
            fontsize=FS, color=INK)

    poly_arrow(ax, [(xb[2], yb - hb / 2), (xb[2], gy + 12)], dashed=True)

    note(ax, (gx0 + gx1) / 2, 46.0, 250.0, 44.0,
         "K = 8 is even, so the median of two\n"
         "neighbouring bins falls between them:\n"
         "only alternate outputs are resolvable")

    bad = check_fit(fig, fitted)
    print(f"    label fit: {len(fitted) - bad} of {len(fitted)} inside their "
          f"boxes" + ("" if not bad else f"  <-- {bad} OVERFLOWING"))
    return save(fig, "paper_fig1_method")


def fig_null_pattern(src: Path):
    """Fig. 2 -- the subcarrier null pattern, stacked for a single column.

    The study's own `fig_v1_null_pattern` puts the two receivers SIDE BY SIDE at
    806 pt. Any column has to shrink that by 40%, which drags its 8 pt labels
    down to 4.8 pt -- below the ~6-7 pt the reference report's own figures
    print at. Stacking the same two panels makes the figure 403 pt wide, so it
    needs no scaling at all and the labels survive at full size.

    Type size depends on WIDTH alone, because the scale factor is
    column / native_width. Trading width for height is therefore free here, and
    that is the whole reason this variant exists.

    The thesis keeps the wide version deliberately: there Table 4.2 states the
    null indices and the active count in text beside it, so the figure
    corroborates an argument the table already carries. This paper has no such
    table and calls the pattern decisive, so it needs the legible one.
    """
    w_pt, h_pt = 403.0, 330.0
    got = [(node, authenticity(*load_node(src, node))) for node in (1, 2)]

    with plt.rc_context(DATA_FONT):
        return _draw_null_pattern(got, w_pt, h_pt)


def _draw_null_pattern(got, w_pt, h_pt):
    fig, axes = plt.subplots(2, 1, figsize=(w_pt / 72.0, h_pt / 72.0),
                             sharex=True)

    # One y-scale across both panels, so a reader comparing the two receivers is
    # comparing amplitudes rather than two different axes. The headroom keeps
    # the annotation off the bars -- black on dark blue is legible only just,
    # and fig_ablation1 needed the same fix for the same reason.
    top = max(a["mean_amp"].max() for _, a in got) * 1.18

    for ax, (node, a) in zip(axes, got):
        ax.bar(np.arange(a["mean_amp"].size), a["mean_amp"], color=C_MAIN,
               width=1.0)
        for sc in a["null_subcarriers"]:
            ax.axvspan(sc - 0.5, sc + 0.5, color=C_ACC, alpha=0.18, lw=0)
        ax.set_ylim(0, top)
        ax.set_ylabel("mean amplitude", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.text(0.02, 0.95,
                f"receiver {node} — {a['active_subcarriers']} active, "
                f"{len(a['null_subcarriers'])} null (shaded)",
                transform=ax.transAxes, fontsize=8, va="top")
    axes[1].set_xlabel("subcarrier index", fontsize=8)
    # No in-figure title: the journal prints an 8 pt caption below every figure,
    # and repeating the claim inside the artwork duplicates it.
    # pad stays at matplotlib's own default: a tighter value clipped the y-axis
    # label off the canvas once the shared scale widened the tick labels.
    # bbox_inches="tight" would fix it too, but it also changes the saved size,
    # and the whole point of this file is that the authored size IS the printed
    # size.
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "paper_fig2_null_pattern.png"
    fig.savefig(path, dpi=300, facecolor=WHITE)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Figures drawn for the paper")
    p.add_argument("--dir", default="../RuView/data/recordings")
    p.add_argument("--file", default="overnight-1775217646.csi.jsonl")
    p.add_argument("--only", choices=("method", "null"), default=None)
    args = p.parse_args()

    if args.only in (None, "method"):
        path = fig_method()
        print(f"  paper_fig1_method       -> {path}")
        print(f"    authored {W:.0f} x {H:.0f} pt; column {COL_PT:.1f} pt "
              f"-> scale {min(1.0, COL_PT / W):.3f}, floor {FS:.0f} pt prints "
              f"{FS * min(1.0, COL_PT / W):.1f} pt")

    if args.only in (None, "null"):
        print("  decoding both receivers ...")
        path = fig_null_pattern(Path(args.dir) / args.file)
        print(f"  paper_fig2_null_pattern -> {path}")
        print(f"    authored 403 x 330 pt; column {COL_PT:.1f} pt "
              f"-> scale {min(1.0, COL_PT / 403.0):.3f}, floor 8 pt prints "
              f"{8 * min(1.0, COL_PT / 403.0):.1f} pt "
              f"(the wide original prints 4.8 pt)")


if __name__ == "__main__":
    main()
