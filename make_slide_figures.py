"""Figures the PRESENTATION needs that no existing figure supplies.

    ../.venv/Scripts/python.exe make_slide_figures.py [--only architecture]

WHY THIS FILE EXISTS. `ch3_architecture.png` is 1879 x 2041 px -- essentially
square, because it was authored at final printed size for a PORTRAIT A4 page,
four stages stacked vertically. On a 13.33 x 7.5 in slide that either shrinks to
nothing or squeezes the bullets off the slide. The fix is a landscape redraw,
not a rescale: scaling a figure to fit shrinks its labels with it, which is the
whole reason make_design_figures.py authors at final size in the first place.

⭐ The THESIS FIGURE IS NOT TOUCHED. This writes to docs/figures/slides/, the
same precedent make_paper_figures.py set when the paper needed a different
version of a figure the thesis prints. Changing ch3_architecture.png would
repaginate a finished, Turnitin-checked 116-page document.

⚠️ TWO THINGS ABOUT REUSING make_design_figures's drawing layer:
  - `INK` is read inside the function bodies, so reassigning the module
    attribute DOES retint every stroke and label. That is how the diagram picks
    up the deck's navy.
  - `fs=FS_LABEL` is a DEFAULT ARGUMENT, bound when the module was imported.
    Reassigning mdf.FS_LABEL afterwards changes nothing. Every call below
    therefore passes `fs=` explicitly. Getting this wrong fails silently -- the
    figure just comes out at 9 pt.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

import make_design_figures as mdf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "docs" / "figures" / "slides"

# The deck's navy, sampled from the institute mark (61.5% of its opaque pixels).
NAVY = "#002A50"
PANEL = "#EEF2F7"      # stage fill, a navy-tinted light neutral
BAND = "#D6E0EA"       # the interface bar

# Slide type floor. The thesis figures use 9 pt because they are read on paper
# at arm's length; a projected slide is read across a room, so nothing here goes
# below 10 and the stage titles are larger.
FS = 10.0
FS_TITLE = 12.0


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=mdf.DPI, facecolor=mdf.WHITE)
    mdf.plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fit checking
# ---------------------------------------------------------------------------

_BOXES = []


def tracked_box(ax, cx, cy, w, h, text, fill, fs):
    """mdf.box, but remembering the box so check_fit can measure the label."""
    mdf.box(ax, cx, cy, w, h, text, fill=fill, fs=fs)
    _BOXES.append((cx, cy, w, h, text))


def check_fit(fig, ax, pad=3.0):
    """Report any label wider or taller than the box drawn around it.

    ⭐ Written after LOOKING at the first render, which is the only thing that
    catches this class of defect: the figure had correct dimensions, correct
    dpi and correct content, and "motion, presence: validated" ran past its box
    AND past its container, off the edge of the figure. Nothing that measures
    the PNG would have noticed. make_paper_figures.py carries the same pass for
    the same reason.

    Data coordinates are printed points here (see mdf.canvas), so the numbers
    below are directly comparable to the box sizes.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for t in ax.texts:
        tx, ty = t.get_position()
        for cx, cy, w, h, text in _BOXES:
            if abs(tx - cx) < 0.01 and abs(ty - cy) < 0.01:
                bb = t.get_window_extent(renderer=renderer).transformed(inv)
                over_w = bb.width - (w - 2 * pad)
                over_h = bb.height - (h - 2 * pad)
                if over_w > 0 or over_h > 0:
                    bad.append((text.split("\n")[0], bb.width, w,
                                bb.height, h, over_w, over_h))
                break
    print(f"  fit check: {len(_BOXES)} boxes measured")
    if not bad:
        print("             every label fits its box")
    for first, bw_, w, bh_, h, ow, oh in bad:
        why = []
        if ow > 0:
            why.append(f"{ow:.1f} pt too WIDE ({bw_:.1f} in {w:.1f})")
        if oh > 0:
            why.append(f"{oh:.1f} pt too TALL ({bh_:.1f} in {h:.1f})")
        print(f"    !! {first[:34]:<34} " + "; ".join(why))
    return not bad


def fig_architecture_wide():
    """The same four stages as Figure 3.5, laid out left to right.

    Portrait stacks the stages and reads top-down; landscape puts them side by
    side and reads left-to-right, which is also the direction the data actually
    flows. The two explanatory notes from the thesis version are dropped: on the
    slide the bullets carry that text, and repeating it inside the figure would
    be the same words twice on one screen.
    """
    W, H = 504.0, 285.0                       # 7.00 x 3.96 in at the slide
    fig, ax = mdf.canvas(W, H)

    mdf.INK = NAVY                            # read at call time -- retints all

    GAP = 18.0
    X0, X1 = 6.0, W - 6.0
    cw = (X1 - X0 - 3 * GAP) / 4.0            # 109.5 pt per stage
    bw = cw - 12.0                            # 97.5 pt per box

    TOP, CH = 266.0, 240.0                    # container top and height
    BOX_TOP, BOX_BOT = 246.0, 38.0            # vertical span the boxes may use

    def lay_out(n, gap=8.0):
        """Centres and a common height for n boxes filling BOX_TOP..BOX_BOT.

        Computed rather than tabulated so a stage with three boxes fills the
        column instead of leaving the fourth slot empty. Output has three; the
        rest have four. ⚠️ For n = 4 this yields 46 pt, NOT 38: three lines of
        10 pt at 1.30 leading measure 36.6, so a 38 pt box left under a point
        of clearance and read as cramped. The fit check reported it and the box
        grew, rather than the check being relaxed to silence the complaint.
        """
        h = (BOX_TOP - BOX_BOT - (n - 1) * gap) / n
        return [BOX_TOP - h / 2 - i * (h + gap) for i in range(n)], h

    # ⚠️ LINE LENGTH IS THE CONSTRAINT, not the number of lines. A 97.5 pt box
    # holds roughly 16 characters at 10 pt Arial. The first draft used lines of
    # 19-27 ("motion, presence: validated") and three boxes overflowed, one of
    # them clean off the figure. Prefer another line over a longer one.
    # ⚠️ The BAND item keeps its own short height: it is a joining bar, not a
    # stage box, and stretching it to a full slot would read as a fifth source.
    stages = [
        ("1 · Sources", [
            ("Windows RSSI\nnetsh · 4 Hz\nstatic, rejected", mdf.WHITE, None),
            ("Recorded CSI\n.csi.jsonl · 20 Hz\nused throughout", mdf.WHITE, None),
            ("ESP32-S3 UDP\nport 5005\nnot yet verified", mdf.WHITE, None),
            ("one Sample interface", BAND, 22.0),
        ]),
        ("2 · Conditioning", [
            ("Hampel filter\noutliers → median", PANEL, None),
            ("Detrend\nremoves AP drift", PANEL, None),
            ("Butterworth\n0.1–0.5 Hz\nzero-phase", PANEL, None),
            ("Welch PSD\nband power", PANEL, None),
        ]),
        ("3 · Detection", [
            ("Ambient baseline\nmean ± σ\nroom empty", PANEL, None),
            ("z-score\nper window", PANEL, None),
            ("Hysteresis\nenter 3σ\nexit 1.5σ", PANEL, None),
            ("Debounce\nN agreeing\nwindows", PANEL, None),
        ]),
        ("4 · Output", [
            ("Dashboard\nJSON over\nloopback :8000", mdf.WHITE, None),
            ("Figures, CSVs\nand RESULTS.md", mdf.WHITE, None),
            # ⚠️ No tick or cross glyphs anywhere: U+2713 is absent from Arial
            # and prints as a tofu box. Words instead, as in the thesis figure.
            # The three capability states are NOT spelled out here -- slide 14
            # shows the panel itself, and repeating them would be the same
            # content twice in one talk.
            ("Capability gate\nstates what it\ncannot measure", mdf.WHITE, None),
        ]),
    ]

    centres = []
    for i, (title, items) in enumerate(stages):
        x0 = X0 + i * (cw + GAP)
        cx = x0 + cw / 2.0
        centres.append((x0, x0 + cw, cx))
        mdf.container(ax, x0, TOP, x0 + cw, CH, title)

        ys, slot_h = lay_out(len(items))
        for y, (text, fill, h_override) in zip(ys, items):
            # The interface bar spans the container rather than the box column:
            # "one Sample interface" is 20 characters and will not fit 97.5 pt.
            w = cw - 8.0 if fill is BAND else bw
            tracked_box(ax, cx, y, w, h_override or slot_h, text, fill, FS)

    # Stage-to-stage flow, in the 18 pt channels between the containers.
    for (_, right, _), (left, _, _) in zip(centres, centres[1:]):
        mdf.arrow(ax, (right + 2, 146.0), (left - 2, 146.0))

    if not check_fit(fig, ax):
        print("  !! fix the overflows above before using this figure")
    return save(fig, "slide_architecture")


def fig_title_band():
    """A full-bleed strip of the real measured channel, for the title slide.

    ⭐ The title slide was the only slide in the deck with no figure, and it is
    the one people look at longest. This is not decoration: it is the same
    quantity slide 3 shows full size -- subcarrier x time x amplitude -- and the
    dark horizontal stripe across it is the 802.11n guard band, the very thing
    the dataset slide later uses to prove the recordings are genuine.

    ⚠️ GENERATED, NOT CROPPED. The dashboard screenshot is 1337 px wide; run
    across a 13.33 in slide that is 100 dpi and visibly soft. Authored here at
    the final printed size and 300 dpi instead, which is the same rule the
    thesis figures follow.

    The 2-minute capture is used rather than the 51-minute one: it loads in a
    second and a title band shows texture, not a result.
    """
    from verification import load_node                      # loader, not a copy

    src = (Path(__file__).resolve().parent / ".." / "RuView" / "data" /
           "recordings" / "pretrain-1775182186.csi.jsonl").resolve()
    _, amp, _, meta = load_node(src, 1)
    if amp.size == 0:
        raise SystemExit(f"no frames read from {src}")

    W_IN, H_IN = 13.333, 0.85
    fig = mdf.plt.figure(figsize=(W_IN, H_IN), dpi=mdf.DPI)
    ax = fig.add_axes([0, 0, 1, 1])                          # full bleed
    ax.set_axis_off()

    # Clip to percentiles so one hot frame cannot flatten the whole band, which
    # is what the dashboard's own colour scaling does.
    lo, hi = np.percentile(amp[amp > 0], [3, 97])
    ax.imshow(amp.T, aspect="auto", origin="lower", cmap="viridis",
              vmin=lo, vmax=hi, interpolation="nearest")

    path = save(fig, "title_band")
    print(f"  frames {amp.shape[0]}  subcarriers {amp.shape[1]}  "
          f"modal geometry {meta['modal']}")
    return path


FIGURES = {"architecture": fig_architecture_wide,
           "titleband": fig_title_band}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FIGURES))
    args = ap.parse_args()
    names = [args.only] if args.only else sorted(FIGURES)
    for n in names:
        p = FIGURES[n]()
        from PIL import Image
        w, h = Image.open(p).size
        print(f"{p.name:<28} {w} x {h} px   "
              f"authored {w / (mdf.DPI / 72):.0f} x {h / (mdf.DPI / 72):.0f} pt   "
              f"sharp to {w / 150:.2f} in")


if __name__ == "__main__":
    main()
