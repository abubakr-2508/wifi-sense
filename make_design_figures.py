"""Design diagrams for the system-design chapter of the report.

    ../.venv/Scripts/python.exe make_design_figures.py            # all of them
    ../.venv/Scripts/python.exe make_design_figures.py --only activity

Every figure is authored at its FINAL PRINTED SIZE, in points, and rasterised
at 300 dpi. That is the whole reason this file exists rather than a drawing
tool: a diagram drawn at an arbitrary size and scaled to fit the text column
shrinks its labels with it, which is how a 9 pt label becomes an unreadable
6 pt one. Here one data unit is one printed point, `fontsize=9` really is 9 pt
on paper, and the document inserts the image at its native size.

Two conventions, both measured rather than chosen:

  * The text column of the report is 451 pt wide, so that is the ceiling on
    figure width.
  * The reference report's own diagrams are monochrome -- six of its seven
    measure 0.0% non-grey pixels. These match that: black line work over white
    and two greys, so nothing depends on being printed in colour.

Output goes to docs/figures/. The diagrams live in this repository rather than
beside the report because they document this code, `docs/architecture.svg`
already sets that precedent, and a single copy cannot drift out of date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent / "docs" / "figures"

# One place for the visual language, so changing the look of all seven diagrams
# is an edit here rather than seven edits scattered through the drawing code.
INK = "#000000"
WHITE = "#FFFFFF"
GREY_SOFT = "#F2F2F2"
GREY_MID = "#D9D9D9"
LW = 0.9            # box and node outlines, points
LW_EDGE = 0.8       # control-flow edges
# 9 pt is the legibility floor and NOTHING is drawn below it -- edge labels and
# stereotypes included. A smaller secondary size is the usual convention, but a
# single floor is easier to hold across seven figures and easier to check.
FS_LABEL = 9.0      # text inside a node
FS_EDGE = 9.0       # edge labels, decision text, stereotypes, notes
DPI = 300

plt.rcParams.update({
    "font.family": "sans-serif",
    # Arial first: sans-serif inside a figure with serif body text is the
    # standard pairing, and Times at 9 pt inside a box goes muddy.
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "axes.linewidth": LW,
})


# ---------------------------------------------------------------------------
# Drawing layer
# ---------------------------------------------------------------------------


def canvas(w_pt: float, h_pt: float):
    """A figure whose data coordinates ARE printed points.

    figsize is given in inches and matplotlib measures font size in points, so
    fixing the axes to (0, w_pt) x (0, h_pt) with no margins makes one data
    unit equal one point on paper. Every coordinate below is therefore a real
    printed position and can be checked against the page geometry directly.
    """
    fig = plt.figure(figsize=(w_pt / 72.0, h_pt / 72.0), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_pt)
    ax.set_ylim(0, h_pt)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _label(ax, x, y, text, fs=FS_LABEL, weight="normal", ha="center",
           va="center", rot=0):
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=INK,
                   weight=weight, linespacing=1.30, zorder=5, rotation=rot)


def box(ax, x, y, w, h, text="", fill=WHITE, fs=FS_LABEL, weight="normal",
        dashed=False, stereotype=None):
    """Plain rectangle, (x, y) at its CENTRE. Used for classes and components."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fill,
                           edgecolor=INK, linewidth=LW, zorder=2,
                           linestyle=(0, (3, 2)) if dashed else "solid"))
    if stereotype:
        _label(ax, x, y + h / 2 - 9, stereotype, fs=FS_EDGE)
        if text:
            _label(ax, x, y - 4, text, fs=fs, weight=weight)
    elif text:
        _label(ax, x, y, text, fs=fs, weight=weight)


def action(ax, x, y, w, h, text, fill=WHITE, fs=FS_LABEL):
    """Rounded rectangle -- a UML activity action node."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 + 6, y - h / 2 + 6), w - 12, h - 12,
        boxstyle="round,pad=6,rounding_size=7",
        facecolor=fill, edgecolor=INK, linewidth=LW, zorder=2))
    _label(ax, x, y, text, fs=fs)


def diamond(ax, x, y, w, h, text, fill=WHITE, fs=FS_EDGE):
    """Decision node. Text sits inside, so keep it to two short lines."""
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y),
                          (x, y - h / 2), (x - w / 2, y)],
                         closed=True, facecolor=fill, edgecolor=INK,
                         linewidth=LW, zorder=2))
    if text:
        _label(ax, x, y, text, fs=fs)


def merge(ax, x, y, s=12):
    """Small unlabelled diamond: several incoming edges, one outgoing."""
    diamond(ax, x, y, s, s, "")


def start_node(ax, x, y, r=6):
    ax.add_patch(Circle((x, y), r, facecolor=INK, edgecolor=INK, zorder=3))


def end_node(ax, x, y, r=7):
    ax.add_patch(Circle((x, y), r, facecolor=WHITE, edgecolor=INK,
                        linewidth=LW, zorder=3))
    ax.add_patch(Circle((x, y), r - 3, facecolor=INK, edgecolor=INK, zorder=4))


def ellipse(ax, x, y, w, h, text, fill=WHITE, fs=FS_LABEL):
    """Use case."""
    ax.add_patch(Ellipse((x, y), w, h, facecolor=fill, edgecolor=INK,
                         linewidth=LW, zorder=2))
    _label(ax, x, y, text, fs=fs)


def arrow(ax, p0, p1, label=None, dashed=False, open_head=False, lw=LW_EDGE,
          label_dx=0.0, label_dy=5.0, label_ha="center"):
    """Straight control-flow or message edge with a filled arrowhead."""
    ax.annotate("", xy=p1, xytext=p0, zorder=3, annotation_clip=False,
                arrowprops=dict(
                    arrowstyle="-|>" if not open_head else "->",
                    color=INK, linewidth=lw, shrinkA=0, shrinkB=0,
                    mutation_scale=9,
                    linestyle=(0, (4, 2.5)) if dashed else "solid"))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        _label(ax, mx + label_dx, my + label_dy, label, fs=FS_EDGE, ha=label_ha)


def poly_arrow(ax, pts, label=None, label_at=0, dashed=False, lw=LW_EDGE,
               label_dx=0.0, label_dy=5.0, label_ha="center"):
    """Orthogonal multi-segment edge: all but the last segment as plain lines,
    the last one carrying the arrowhead. Used for branch and loop-back edges,
    which have to route around the spine of the diagram."""
    style = (0, (4, 2.5)) if dashed else "solid"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=INK,
                           linewidth=lw, linestyle=style, zorder=3,
                           solid_capstyle="butt"))
    arrow(ax, pts[-2], pts[-1], dashed=dashed, lw=lw)
    if label:
        a, b = pts[label_at], pts[label_at + 1]
        _label(ax, (a[0] + b[0]) / 2 + label_dx, (a[1] + b[1]) / 2 + label_dy,
               label, fs=FS_EDGE, ha=label_ha)


def note(ax, x, y, w, h, text, anchor=None):
    """UML note: a rectangle with the top-right corner folded, joined to what
    it annotates by a dashed line."""
    f = 8.0
    ax.add_patch(Polygon(
        [(x - w / 2, y - h / 2), (x - w / 2, y + h / 2),
         (x + w / 2 - f, y + h / 2), (x + w / 2, y + h / 2 - f),
         (x + w / 2, y - h / 2)],
        closed=True, facecolor=GREY_SOFT, edgecolor=INK, linewidth=0.7, zorder=2))
    ax.add_patch(Polygon([(x + w / 2 - f, y + h / 2), (x + w / 2 - f, y + h / 2 - f),
                          (x + w / 2, y + h / 2 - f)],
                         closed=True, facecolor=GREY_MID, edgecolor=INK,
                         linewidth=0.7, zorder=3))
    _label(ax, x, y, text, fs=FS_EDGE)
    if anchor:
        ax.add_line(Line2D([x - w / 2, anchor[0]], [y, anchor[1]], color=INK,
                           linewidth=0.6, linestyle=(0, (2, 2)), zorder=1))


ROW = 11.5      # one member line; 11.5 pt of leading on 9 pt text
PAD = 5.0


def class_box(ax, x, top, w, name, attrs=(), ops=(), stereotype=None,
              fill=WHITE):
    """UML class box, sized to its content and stacked downward.

    The caller gives the TOP edge and gets the bottom back, so a column is
    built by feeding one box's bottom into the next box's top. Hand-computing
    heights from member counts is how a box ends up overflowing its own text.

    A box with no operations gets two compartments rather than three, which is
    the right rendering for the six dataclasses and the enumeration.
    """
    head = 18.0 + (11.0 if stereotype else 0.0)
    # An empty compartment is legal UML but prints as a stray hairline strip,
    # so a class with no attributes simply does not get one.
    h = head
    if attrs:
        h += PAD + ROW * len(attrs) + PAD
    if ops:
        h += PAD + ROW * len(ops) + PAD
    bottom = top - h

    ax.add_patch(Rectangle((x - w / 2, bottom), w, h, facecolor=fill,
                           edgecolor=INK, linewidth=LW, zorder=2))
    y = top - 12
    if stereotype:
        _label(ax, x, y, stereotype, fs=FS_EDGE)
        y -= 11
    _label(ax, x, y, name, fs=FS_LABEL, weight="bold")

    def rule(yy):
        ax.add_line(Line2D([x - w / 2, x + w / 2], [yy, yy], color=INK,
                           linewidth=LW, zorder=3))

    y = top - head
    rule(y)
    if attrs:
        y -= PAD
        for a in attrs:
            _label(ax, x - w / 2 + 6, y - ROW / 2, a, fs=FS_EDGE, ha="left")
            y -= ROW
        y -= PAD
    if ops:
        rule(y)
        y -= PAD
        for o in ops:
            _label(ax, x - w / 2 + 6, y - ROW / 2, o, fs=FS_EDGE, ha="left")
            y -= ROW
    return bottom


def generalisation(ax, p0, p1, size=9.0):
    """Hollow triangle at p1 -- the UML "is a" arrowhead."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    base = (p1[0] - ux * size, p1[1] - uy * size)
    ax.add_line(Line2D([p0[0], base[0]], [p0[1], base[1]], color=INK,
                       linewidth=LW_EDGE, zorder=3))
    ax.add_patch(Polygon([p1, (base[0] + px * size / 2, base[1] + py * size / 2),
                          (base[0] - px * size / 2, base[1] - py * size / 2)],
                         closed=True, facecolor=WHITE, edgecolor=INK,
                         linewidth=LW_EDGE, zorder=4))


def composition(ax, p0, p1, size=9.0):
    """Filled diamond at p0, the OWNING end, then a plain line to p1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    tip = (p0[0] + ux * size * 2, p0[1] + uy * size * 2)
    ax.add_patch(Polygon([p0, (p0[0] + ux * size + px * size / 2.2,
                               p0[1] + uy * size + py * size / 2.2), tip,
                          (p0[0] + ux * size - px * size / 2.2,
                           p0[1] + uy * size - py * size / 2.2)],
                         closed=True, facecolor=INK, edgecolor=INK,
                         linewidth=LW_EDGE, zorder=4))
    ax.add_line(Line2D([tip[0], p1[0]], [tip[1], p1[1]], color=INK,
                       linewidth=LW_EDGE, zorder=3))


def actor(ax, x, y, label):
    """Stick figure for a human actor, drawn about a 34 pt tall body."""
    seg = dict(color=INK, linewidth=LW, zorder=3)
    ax.add_patch(Circle((x, y + 14), 5.5, facecolor=WHITE, edgecolor=INK,
                        linewidth=LW, zorder=3))
    ax.add_line(Line2D([x, x], [y + 8.5, y - 6], **seg))
    ax.add_line(Line2D([x - 9, x + 9], [y + 3, y + 3], **seg))
    ax.add_line(Line2D([x, x - 8], [y - 6, y - 18], **seg))
    ax.add_line(Line2D([x, x + 8], [y - 6, y - 18], **seg))
    _label(ax, x, y - 28, label, fs=FS_EDGE)


def lifeline(ax, x, top, bottom, label, w=62, h=30, foot=False):
    """Sequence-diagram head box with its dashed lifeline hanging below.

    `foot` repeats the box at the bottom. Worth it on a tall diagram: by the
    last few messages a reader has otherwise lost track of which dashed line
    belongs to which participant.
    """
    box(ax, x, top - h / 2, w, h, label, fs=FS_EDGE)
    ax.add_line(Line2D([x, x], [top - h, bottom], color=INK, linewidth=0.7,
                       linestyle=(0, (3, 3)), zorder=1))
    if foot:
        box(ax, x, bottom - h / 2, w, h, label, fs=FS_EDGE)


def activation(ax, x, top, bottom, w=8):
    ax.add_patch(Rectangle((x - w / 2, bottom), w, top - bottom, facecolor=WHITE,
                           edgecolor=INK, linewidth=0.7, zorder=2))


def fragment(ax, x0, y0, x1, y1, kind, guard="", tab=54):
    """Combined fragment: a box with the operator in a pentagon tab."""
    ax.add_patch(Rectangle((x0, y1), x1 - x0, y0 - y1, facecolor="none",
                           edgecolor=INK, linewidth=0.7, zorder=2))
    ax.add_patch(Polygon([(x0, y0), (x0 + tab, y0), (x0 + tab, y0 - 9),
                          (x0 + tab - 7, y0 - 15), (x0, y0 - 15)],
                         closed=True, facecolor=WHITE, edgecolor=INK,
                         linewidth=0.7, zorder=3))
    _label(ax, x0 + tab / 2 - 3, y0 - 7.5, kind, fs=FS_EDGE, weight="bold")
    if guard:
        _label(ax, x0 + tab + 6, y0 - 7.5, guard, fs=FS_EDGE, ha="left")


def self_message(ax, x, y, text, drop=13, out=22):
    """The little rectangular hook a lifeline draws when it calls itself."""
    seg = dict(color=INK, linewidth=LW_EDGE, zorder=3)
    ax.add_line(Line2D([x, x + out], [y, y], **seg))
    ax.add_line(Line2D([x + out, x + out], [y, y - drop], **seg))
    arrow(ax, (x + out, y - drop), (x + 4, y - drop))
    _label(ax, x + out + 6, y - drop / 2, text, fs=FS_EDGE, ha="left")


def node3d(ax, x, y, w, h, title, stereotype="«device»", d=9.0, dashed=False,
           fill=WHITE):
    """UML deployment node: a box with a depth face on the top and right."""
    # zorder 1 throughout: a node is a CONTAINER, so anything placed inside it
    # afterwards has to sit on top. Drawing the front face above its own
    # contents hid every artifact in the first render.
    ls = (0, (3, 2)) if dashed else "solid"
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d),
                          (x + w, y + h)], closed=True, facecolor=GREY_SOFT,
                         edgecolor=INK, linewidth=LW, linestyle=ls, zorder=1))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d),
                          (x + w + d, y + h + d), (x + w, y + h)], closed=True,
                         facecolor=GREY_MID, edgecolor=INK, linewidth=LW,
                         linestyle=ls, zorder=1))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK,
                           linewidth=LW, linestyle=ls, zorder=1))
    _label(ax, x + w / 2, y + h - 11, stereotype, fs=FS_EDGE)
    _label(ax, x + w / 2, y + h - 22, title, fs=FS_LABEL, weight="bold")


def container(ax, x0, top, x1, h, title):
    """A bordered group with its name inside the top edge.

    Taken from the reference report, which draws its architecture stages this
    way. It is strictly better than a label floating in the left margin: the
    grouping is drawn rather than implied by proximity. Returns the bottom.
    """
    ax.add_patch(Rectangle((x0, top - h), x1 - x0, h, facecolor="none",
                           edgecolor=INK, linewidth=LW, zorder=1))
    _label(ax, (x0 + x1) / 2, top - 9, title, fs=FS_LABEL, weight="bold")
    return top - h


def component(ax, x, y, w, h, name, sub="", fill=WHITE, dashed=False):
    """«component» box with the two tabs on its left edge."""
    box(ax, x, y, w, h, "", fill=fill, dashed=dashed)
    _label(ax, x, y + (6 if sub else 0), name, fs=FS_LABEL, weight="bold")
    if sub:
        _label(ax, x, y - 8, sub, fs=FS_EDGE)
    bx = x - w / 2 - 4
    for dy in (4, -8):
        ax.add_patch(Rectangle((bx, y + dy), 9, 6, facecolor=WHITE,
                               edgecolor=INK, linewidth=0.7, zorder=4))


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor=WHITE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3.4 -- activity diagram of the detection decision path
# ---------------------------------------------------------------------------


def fig_activity():
    """The detector's control flow, drawn from wifisense/detector.py:331-447.

    Three things this shows that a description does not: the analysis window
    must fill before anything is reported, calibration consumes a further fixed
    number of windows, and the threshold test is asymmetric -- which is the
    hysteresis, and the reason the state does not chatter at the boundary.
    """
    W, H = 451.0, 620.0
    fig, ax = canvas(W, H)

    SP = 138.0       # main spine
    BR = 330.0       # right-hand branch column
    RAIL_L = 12.0    # loop-back rail, clear of the widest spine box (30..246)
    RET = 430.0      # single return rail, clear of the widest branch box (415)

    # -- spine -------------------------------------------------------------
    start_node(ax, SP, 606)
    action(ax, SP, 582, 168, 28, "Read sample from source")
    action(ax, SP, 540, 200, 34, "Append to motion, respiration\nand subcarrier buffers")
    diamond(ax, SP, 494, 112, 40, "window\nfull?")
    action(ax, SP, 444, 216, 36,
           "Extract features: Hampel → detrend\n→ σ, MAD → Welch band power")
    diamond(ax, SP, 396, 118, 40, "baseline\ncomplete?")
    action(ax, SP, 346, 216, 34,
           "Compute z-score of window dispersion\nagainst the ambient baseline")
    diamond(ax, SP, 298, 118, 40, "currently in\nMOTION?")

    # -- the asymmetric threshold test, which IS the hysteresis -------------
    diamond(ax, 72, 240, 96, 36, "z > exit_z ?")
    diamond(ax, 222, 240, 96, 36, "z > enter_z ?")
    merge(ax, SP, 192)

    diamond(ax, SP, 150, 132, 40, "target =\ncurrent state?")
    action(ax, SP, 42, 214, 32, "Publish Reading and snapshot")
    # The loop does terminate -- SensingService._run() runs `while not
    # self._stop.is_set()` -- so the diagram gets a proper activity final node
    # rather than looping forever with no exit.
    end_node(ax, SP, 12)

    # -- right-hand column. Every box here is 170 pt wide or less so that the
    #    return rail at x = 430 passes clear of all of them.
    action(ax, BR, 494, 170, 34, "Report CALIBRATING\n(filling analysis window)", GREY_SOFT)
    action(ax, BR, 396, 170, 26, "Accumulate window dispersion", GREY_SOFT)
    diamond(ax, BR, 348, 132, 40, "calibration\nsamples reached?", GREY_SOFT)
    action(ax, BR, 296, 170, 34, "Compute ambient μ and σ;\nstate := IDLE", GREY_SOFT)
    action(ax, BR, 150, 170, 26, "increment debounce counter", GREY_SOFT)
    diamond(ax, BR, 104, 122, 38, "counter ≥\ndebounce?", GREY_SOFT)
    action(ax, BR, 58, 160, 26, "commit state change", GREY_SOFT)

    # -- spine edges ---------------------------------------------------------
    arrow(ax, (SP, 600), (SP, 596))
    arrow(ax, (SP, 568), (SP, 557))
    arrow(ax, (SP, 523), (SP, 514))
    arrow(ax, (SP, 474), (SP, 462), "yes", label_dx=13, label_dy=-4)
    arrow(ax, (SP, 426), (SP, 416))
    arrow(ax, (SP, 376), (SP, 363), "yes", label_dx=13, label_dy=-2)
    arrow(ax, (SP, 329), (SP, 318))

    poly_arrow(ax, [(SP - 59, 298), (72, 298), (72, 258)], "yes",
               label_dx=-14, label_dy=6)
    poly_arrow(ax, [(SP + 59, 298), (222, 298), (222, 258)], "no",
               label_dx=14, label_dy=6)
    poly_arrow(ax, [(72, 222), (72, 192), (SP - 7, 192)],
               "target := MOTION / IDLE", label_at=1, label_dy=10, label_dx=4)
    poly_arrow(ax, [(222, 222), (222, 192), (SP + 7, 192)])

    arrow(ax, (SP, 186), (SP, 170))
    arrow(ax, (SP, 130), (SP, 58), "yes", label_dx=12, label_dy=26)
    arrow(ax, (SP, 26), (SP, 20), "[stop requested]", label_dx=56, label_dy=-2)

    # -- branch edges --------------------------------------------------------
    poly_arrow(ax, [(SP + 56, 494), (BR - 85, 494)], "no", label_dy=6)
    poly_arrow(ax, [(SP + 59, 396), (BR - 85, 396)], "no", label_dy=6)
    poly_arrow(ax, [(SP + 66, 150), (BR - 85, 150)], "no", label_dy=6)
    arrow(ax, (BR, 383), (BR, 368))
    arrow(ax, (BR, 328), (BR, 313), "yes", label_dx=14, label_dy=0)
    arrow(ax, (BR, 137), (BR, 123))
    arrow(ax, (BR, 85), (BR, 71), "yes", label_dx=14, label_dy=0)

    # Every branch that cannot decide yet feeds ONE return rail rather than
    # three parallel ones -- three rails collided with the branch boxes and
    # made the top of the diagram unreadable.
    feed = dict(color=INK, linewidth=LW_EDGE, zorder=3)
    ax.add_line(Line2D([BR + 85, RET], [494, 494], **feed))
    ax.add_line(Line2D([BR + 66, RET], [348, 348], **feed))
    _label(ax, BR + 76, 354, "no", fs=FS_EDGE)
    ax.add_line(Line2D([BR, BR], [279, 266], **feed))
    ax.add_line(Line2D([BR, RET], [266, 266], **feed))
    poly_arrow(ax, [(RET, 266), (RET, 582), (SP + 84, 582)])

    # The two debounce outcomes rejoin the spine at Publish. The bypass runs
    # BELOW "commit state change" (which spans y 45..71) rather than across it.
    poly_arrow(ax, [(BR, 45), (BR, 42), (SP + 107, 42)])
    poly_arrow(ax, [(BR + 61, 104), (RET, 104), (RET, 30), (SP + 107, 30)],
               "no", label_at=0, label_dy=6)

    # main loop back to the top
    poly_arrow(ax, [(SP - 107, 42), (RAIL_L, 42), (RAIL_L, 582), (SP - 84, 582)])

    # -- note ----------------------------------------------------------------
    note(ax, 354, 228, 164, 44,
         "asymmetric by design — the gap\n"
         "between enter_z and exit_z is the\n"
         "hysteresis, and stops chattering",
         anchor=(270, 240))

    return save(fig, "ch3_activity")


# ---------------------------------------------------------------------------
# Figure 3.2 -- class diagram of the wifisense package
# ---------------------------------------------------------------------------


def fig_class():
    """The fourteen classes of the running system, counted from the source:
    four in sources.py, two in dsp.py, four in detector.py, three in server.py,
    plus the Handler defined inside make_handler().

    Members that do not bear on the design are elided -- the point of the
    figure is the shape of the package, not an API listing. Three columns:
    the source hierarchy, the behavioural classes, and the value types.
    """
    W, H = 451.0, 620.0
    fig, ax = canvas(W, H)

    # Column widths are set by the longest member string at 9 pt, then the
    # right-hand gap is made as wide as what is left over: two rails 6 pt apart
    # read as a single line, which is how the first draft looked.
    CL, WL = 76.0, 128.0     # sources          spans  12..140
    CM, WM = 222.0, 140.0    # behaviour        spans 152..292
    CR, WR = 380.0, 126.0    # value types      spans 317..443
    GEN = 146.0              # generalisation rail, in the left gap
    R_STATE, R_STATS = 300.0, 310.0        # two rails, 10 pt apart

    # Tops are kept alongside the returned bottoms so every connector attaches
    # to a real box edge. Deriving edge positions from hand-added offsets is
    # what put the first draft's «creates» label on top of another class.
    def stack(cx, w, top, *boxes):
        out = []
        for i, kw in enumerate(boxes):
            b = class_box(ax, cx, top, w, **kw)
            out.append((top, b))
            top = b - (16 if i < len(boxes) - 1 else 0)
        return out

    # -- column 1: the source hierarchy --------------------------------------
    (src, net, rec, esp) = stack(
        CL, WL, 614,
        dict(name="Source", stereotype="«abstract»",
             attrs=["kind: str", "nominal_rate_hz: float", "verified: bool"],
             ops=["read(): Sample", "describe(): dict"]),
        dict(name="NetshRSSISource",
             attrs=['kind = "rssi"', "nominal_rate_hz = 4.0", "verified = True"]),
        dict(name="RecordedCSISource",
             attrs=['kind = "csi"', "nominal_rate_hz = 20.0", "verified = True"],
             ops=["_detect_geometry()"]),
        dict(name="Esp32UdpSource",
             attrs=['kind = "csi"', "nominal_rate_hz = 20.0", "verified = False"]))

    # -- column 2: behaviour --------------------------------------------------
    (det, svc, hnd) = stack(
        CM, WM, 596,
        dict(name="MotionDetector",
             attrs=["fs: float", "window_n: int", "respiration_n: int",
                    "enter_z = 3.0", "exit_z = 1.5"],
             ops=["update(): Reading", "reset_calibration()",
                  "set_thresholds()", "capabilities(): dict", "snapshot(): dict"]),
        dict(name="SensingService",
             attrs=["source: Source", "detector: MotionDetector",
                    "poll_interval: float"],
             ops=["start()", "stop()", "snapshot(): dict",
                  "waterfall_since(): dict"]),
        dict(name="Handler", ops=["do_GET()", "do_POST()"]))

    # -- column 3: value types, two compartments each -------------------------
    (smp, bas, fea, rdg, rsp, sta, sts) = stack(
        CR, WR, 614,
        dict(name="Sample",
             attrs=["timestamp: float", "value: float", "rssi_dbm: float",
                    "vector: ndarray | None"]),
        dict(name="Baseline",
             attrs=["mean_std: float", "sigma_std: float", "samples: int",
                    "complete: bool"]),
        dict(name="Features",
             attrs=["std: float", "mad: float", "motion_power: float",
                    "dominant_hz: float"]),
        dict(name="Reading",
             attrs=["timestamp: float", "state: State", "z_score: float",
                    "confidence: float"]),
        dict(name="RespiratoryEstimate",
             attrs=["bpm: float", "confidence: float", "supported: bool",
                    "reason: str"]),
        dict(name="State", stereotype="«enumeration»",
             attrs=["CALIBRATING", "IDLE", "MOTION", "NO_SIGNAL"]),
        dict(name="ServiceStats",
             attrs=["samples: int", "dropped: int", "actual_rate_hz: float"]))

    mid = lambda box: (box[0] + box[1]) / 2

    # -- generalisation: three concrete sources share one rail ----------------
    rail = dict(color=INK, linewidth=LW_EDGE, zorder=3)
    for b in (net, rec, esp):
        ax.add_line(Line2D([CL + WL / 2, GEN], [mid(b), mid(b)], **rail))
    ax.add_line(Line2D([GEN, GEN], [mid(esp), src[1] - 9], **rail))
    ax.add_line(Line2D([GEN, CL], [src[1] - 9, src[1] - 9], **rail))
    generalisation(ax, (CL, src[1] - 9), (CL, src[1]))

    # -- compositions and associations ----------------------------------------
    # MotionDetector owns its Baseline. It also creates every Reading, but that
    # is already stated by `update(): Reading` in its own signature, so no
    # dependency arrow is drawn -- the one in the first draft had nowhere to put
    # its label except on top of the Baseline class.
    composition(ax, (CM + WM / 2, mid(bas)), (CR - WR / 2, mid(bas)))

    # A Reading carries its Features and its RespiratoryEstimate, and names a
    # State. Features sits directly above it and the estimate directly below,
    # so only the enumeration has to route around.
    composition(ax, (CR, rdg[0]), (CR, fea[1]))
    composition(ax, (CR, rdg[1]), (CR, rsp[0]))
    y0, y1 = rdg[1] + 14, mid(sta)
    ax.add_line(Line2D([CR - WR / 2, R_STATE], [y0, y0], **rail))
    ax.add_line(Line2D([R_STATE, R_STATE], [y1, y0], **rail))
    ax.add_line(Line2D([R_STATE, CR - WR / 2], [y1, y1], **rail))

    # SensingService owns its ServiceStats and drives the detector; Handler
    # depends on the service it serves.
    y0, y1 = svc[1] + 12, mid(sts)
    composition(ax, (CM + WM / 2, y0), (R_STATS, y0))
    ax.add_line(Line2D([R_STATS, R_STATS], [y1, y0], **rail))
    ax.add_line(Line2D([R_STATS, CR - WR / 2], [y1, y1], **rail))
    composition(ax, (CM, svc[0]), (CM, det[1]))
    arrow(ax, (CM, hnd[0]), (CM, svc[1]), dashed=True)

    # Source declares read(): Sample. Drawn because Sample would otherwise sit
    # in the diagram unconnected to anything.
    poly_arrow(ax, [(CL + WL / 2, 605), (CR - WR / 2, 605)], dashed=True)
    _label(ax, 227, 611, "«creates»", fs=FS_EDGE)

    return save(fig, "ch3_class")


# ---------------------------------------------------------------------------
# Figure 3.1 -- use case diagram
# ---------------------------------------------------------------------------


def fig_usecase():
    """One human actor and one supporting actor.

    What is absent matters as much as what is present: no actor authenticates,
    because there is no login, no registration and no user account anywhere in
    the system. Section 3.2.1 says so in the prose rather than the figure.
    """
    # Two columns rather than one. A single stack of eight ellipses ran to
    # 410 pt of mostly empty boundary; the reference report's use case diagram
    # is 87 pt tall because it lays its cases out sideways. Eight cases and two
    # actors will not fit one row, but the operator's six in a column with the
    # two included cases beside them costs 310 pt instead of 410.
    W, H = 451.0, 310.0
    fig, ax = canvas(W, H)
    X0, X1, Y0, Y1 = 92.0, 376.0, 12.0, 296.0

    ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, facecolor="none",
                           edgecolor=INK, linewidth=LW, zorder=1))
    _label(ax, (X0 + X1) / 2, Y1 - 12, "WiFi Sensing System",
           fs=FS_LABEL, weight="bold")

    CA, CB, EW, EH = 156.0, 312.0, 112.0, 34.0
    main = [(250, "Select source and\nstart pipeline"),
            (206, "Monitor detection\nstate"),
            (162, "Adjust detection\nthresholds"),
            (118, "View channel\nwaterfall"),
            (74, "Review findings\nfigures"),
            (30, "Acquire and\ncondition samples")]
    incl = [(250, "Learn ambient\nbaseline"),
            (206, "Report measurable\nquantities")]
    for y, name in main:
        ellipse(ax, CA, y, EW, EH, name)
    for y, name in incl:
        ellipse(ax, CB, y, EW, EH, name)

    actor(ax, 50, 168, "Operator")
    box(ax, 412, 30, 68, 38, "CSI Source", stereotype="«actor»")

    seg = dict(color=INK, linewidth=LW_EDGE, zorder=1)
    for y, _ in main[:5]:                       # the operator's five
        ax.add_line(Line2D([62, CA - EW / 2], [164, y], **seg))
    # The source participates only in acquisition; the line runs below both
    # included cases, so nothing crosses.
    ax.add_line(Line2D([378, CA + EW / 2], [30, 30], **seg))

    # «include»: starting always calibrates, and every state read carries the
    # statement of what the system can and cannot measure.
    for y in (250, 206):
        arrow(ax, (CA + EW / 2, y), (CB - EW / 2, y), dashed=True)
        _label(ax, (CA + CB) / 2, y + 10, "«include»", fs=FS_EDGE)

    return save(fig, "ch3_use_case")


# ---------------------------------------------------------------------------
# Figure 3.3 -- sequence diagram
# ---------------------------------------------------------------------------


def fig_sequence():
    """One pass of the background sensing loop, and one client poll.

    The two are drawn as separate interactions on purpose: the browser polls a
    published snapshot and never blocks the sensing thread, which is the whole
    reason the server can be stdlib-only.
    """
    # 500 rather than 460: repeating the participant boxes at the foot costs
    # ~40 pt, and without it the client exchange had to be squeezed up into the
    # loop fragment, where the note collided with _publish.
    W, H = 451.0, 500.0
    fig, ax = canvas(W, H)

    # Lifelines are pulled left of centre so the two widest message labels --
    # extract_features(...) and respiratory_estimate(...) -- still land inside
    # the page. At the first spacing the :dsp head box touched the right edge.
    BR, HD, SV, SR, MD, DS = 34.0, 106.0, 178.0, 250.0, 322.0, 400.0
    for x, name in ((BR, ":Browser"), (HD, ":Handler"), (SV, ":Sensing\nService"),
                    (SR, ":Source"), (MD, ":Motion\nDetector"), (DS, ":dsp")):
        lifeline(ax, x, 492, 55, name, w=60, foot=True)

    fragment(ax, 146, 442, 442, 170, "loop", "[every 1/fs s — background thread]")
    fragment(ax, 154, 384, 434, 182, "alt", "[sample is None]")
    fragment(ax, 296, 272, 428, 218, "opt", "[csi source, ~1 Hz]", tab=44)

    activation(ax, SV, 412, 183)
    activation(ax, SR, 412, 396)
    activation(ax, MD, 318, 208)
    activation(ax, DS, 298, 282)
    activation(ax, DS, 242, 226)

    arrow(ax, (SV, 412), (SR, 412), "read()")
    arrow(ax, (SR, 396), (SV, 396), "Sample | None", dashed=True)
    self_message(ax, SV, 356, "stats.dropped += 1")

    ax.add_line(Line2D([154, 434], [340, 340], color=INK, linewidth=0.7,
                       linestyle=(0, (4, 3)), zorder=2))
    _label(ax, 162, 332, "[else]", fs=FS_EDGE, ha="left")

    arrow(ax, (SV, 318), (MD, 318), "update(value, rssi_dbm, vector)")
    arrow(ax, (MD, 298), (DS, 298), "extract_features(window, fs)")
    arrow(ax, (DS, 282), (MD, 282), "Features", dashed=True)
    arrow(ax, (MD, 242), (DS, 242), "respiratory_estimate(top-K)")
    arrow(ax, (DS, 226), (MD, 226), "RespiratoryEstimate", dashed=True)
    arrow(ax, (MD, 208), (SV, 208), "Reading", dashed=True)
    # _publish belongs to the else branch, so its hook has to finish above the
    # alt fragment's lower edge at y=182.
    self_message(ax, SV, 196, "_publish(snapshot)")

    # Below the loop fragment (ends 170) and above the foot boxes (top 55).
    note(ax, 226, 150, 310, 22,
         "the browser polls independently of the loop above")
    activation(ax, BR, 118, 64)
    activation(ax, HD, 118, 64)
    arrow(ax, (BR, 118), (HD, 118), "GET /api/state")
    arrow(ax, (HD, 100), (SV, 100), "snapshot()")
    arrow(ax, (SV, 82), (HD, 82), "dict", dashed=True)
    arrow(ax, (HD, 64), (BR, 64), "200 application/json")

    return save(fig, "ch3_sequence")


# ---------------------------------------------------------------------------
# Figure 3.5 -- architecture of the sensing pipeline
# ---------------------------------------------------------------------------


def fig_architecture():
    """Four stages, redrawn for print from docs/architecture.svg.

    ⚠️ The validation panel of that SVG is deliberately NOT carried over. It
    still reads "margin ~2x, suggestive", which predates the segment-stability
    test and contradicts what detector.py now reports. Validation belongs to
    Chapter 6, not to a design figure.
    """
    W, H = 451.0, 490.0
    fig, ax = canvas(W, H)
    GX0, GX1, MID = 6.0, 445.0, 225.5

    # Each stage is a named container rather than a label in the left margin,
    # so the grouping is drawn instead of implied. That also removes the need
    # for the elbow connectors the first draft used: a plain vertical arrow
    # between two labelled boxes cannot be misread as skipping the chain.
    container(ax, GX0, 482, GX1, 100, "1 · Sources")
    for x, t in ((78, "Windows RSSI\nnetsh · WLAN API · 4 Hz\nstatic on MT7922 — rejected"),
                 (226, "Recorded CSI\n.csi.jsonl · 20 Hz · 52 sc\nused throughout"),
                 (374, "ESP32-S3 UDP\nADR-018 · port 5005\nwritten, unverified")):
        box(ax, x, 439, 140, 46, t)
        arrow(ax, (x, 416), (x, 408))
    box(ax, 226, 397, 300, 18, "one Sample interface", fill=GREY_MID)
    arrow(ax, (MID, 382), (MID, 362))

    container(ax, GX0, 360, GX1, 64, "2 · Conditioning")
    cond = [(62, "Hampel filter\noutliers → median"), (172, "Detrend\nremoves AP drift"),
            (282, "Butterworth\nzero-phase"), (392, "Welch PSD\nband power")]
    for i, (x, t) in enumerate(cond):
        box(ax, x, 321, 104, 38, t, fill=GREY_SOFT)
        if i:
            arrow(ax, (cond[i - 1][0] + 52, 321), (x - 52, 321))
    arrow(ax, (MID, 296), (MID, 276))

    container(ax, GX0, 274, GX1, 64, "3 · Detection")
    det = [(62, "Ambient baseline\nmean ± σ, area empty"), (172, "z-score"),
           (282, "Hysteresis\nenter 3σ · exit 1.5σ"), (392, "Debounce\nN agreeing windows")]
    for i, (x, t) in enumerate(det):
        box(ax, x, 235, 104, 38, t, fill=GREY_SOFT)
        if i:
            arrow(ax, (det[i - 1][0] + 52, 235), (x - 52, 235))
    arrow(ax, (MID, 210), (MID, 190))

    container(ax, GX0, 188, GX1, 74, "4 · Output")
    # ⚠️ No tick or cross glyphs: U+2713 is absent from Arial and prints as a
    # tofu box. The three-state wording is clearer in a report anyway.
    box(ax, 78, 144, 140, 48, "Dashboard\nJSON over loopback\n:8000")
    box(ax, 226, 144, 140, 48, "Figures, CSVs\nand RESULTS.md")
    box(ax, 374, 144, 140, 48,
        "Capability gate\nmotion, presence: validated\nrespiration: unvalidated\n"
        "pose: unavailable")

    note(ax, 118, 60, 212, 72,
         "Separate windows — motion 4 s (80\n"
         "samples at 20 Hz) to stay responsive;\n"
         "respiration 30 s, because resolution is\n"
         "fs/N and 0.1 Hz needs one full cycle")
    note(ax, 338, 60, 212, 72,
         "Top-K median — respiration from the\n"
         "median of the 8 highest-variance\n"
         "subcarriers; one subcarrier alone gave\n"
         "18.7 against 9.3 br/min between nodes")

    return save(fig, "ch3_architecture")


# ---------------------------------------------------------------------------
# Figure 3.6 -- component diagram
# ---------------------------------------------------------------------------


def fig_component():
    """Modules and what each requires of the others.

    The dependency edges come from an exhaustive scan of every import
    statement in the five analysis scripts, which is why two of them are drawn
    with no connector at all: phase_study.py and verification.py share no code
    with the pipeline and decode the capture themselves.

    selftest.py sits outside the offline-studies group deliberately. The five
    scripts inside it measure the data; the self-test drives the modules and
    checks the requirements, which is a different job.
    """
    W, H = 451.0, 390.0
    fig, ax = canvas(W, H)

    component(ax, 226, 356, 150, 34, "web", "index.html · app.js · style.css")
    # h=46 rather than 40: component() puts a sub at y-8, so the lower line of a
    # two-line sub sits ~19 pt below centre and a shorter box leaves it touching
    # the bottom rule.
    component(ax, 380, 352, 128, 46, "selftest.py",
              "server, detector,\nsources, dsp")
    component(ax, 226, 292, 196, 48, "wifisense.server",
              "/api/state · /api/waterfall\n/api/calibrate · /api/thresholds")
    component(ax, 128, 224, 138, 32, "wifisense.detector")
    component(ax, 330, 224, 138, 32, "wifisense.sources")
    component(ax, 226, 166, 138, 32, "wifisense.dsp")
    component(ax, 62, 292, 98, 42, "run_live.py",
              "assembles source,\ndetector, server")

    dep = dict(dashed=True)
    arrow(ax, (226, 339), (226, 316), "«use»", label_dx=24, **dep)
    arrow(ax, (196, 268), (146, 240), **dep)
    arrow(ax, (258, 268), (312, 240), **dep)
    arrow(ax, (152, 208), (206, 182), **dep)
    arrow(ax, (306, 208), (250, 182), **dep)
    arrow(ax, (111, 292), (128, 292), **dep)
    arrow(ax, (356, 329), (314, 317), **dep)

    ax.add_patch(Rectangle((8, 14), 435, 120, facecolor="none", edgecolor=INK,
                           linewidth=0.7, linestyle=(0, (4, 3)), zorder=1))
    _label(ax, 16, 124, "offline studies", fs=FS_EDGE, ha="left")

    scripts = [(50, "analyze_csi.py", "detector, dsp,\nsources"),
               (138, "ablation.py", "dsp"),
               (226, "motion_eval.py", "dsp"),
               (314, "phase_study.py", "— none —"),
               (402, "verification.py", "— none —")]
    for x, name, req in scripts:
        box(ax, x, 96, 84, 44, "", fill=GREY_SOFT)
        _label(ax, x, 108, name, fs=FS_EDGE, weight="bold")
        _label(ax, x, 90, req, fs=FS_EDGE)
    # All three land on the dsp box's bottom edge (x 157..295); the first draft
    # aimed analyze_csi.py at x=110, which is outside the box entirely.
    for x, tx in ((50, 175), (138, 205), (226, 235)):
        arrow(ax, (x, 118), (tx, 150), **dep)

    note(ax, 340, 44, 190, 34,
         "these two decode the capture\nthemselves — see Section 3.2.6")

    return save(fig, "ch3_component")


# ---------------------------------------------------------------------------
# Figure 3.7 -- deployment diagram
# ---------------------------------------------------------------------------


def fig_deployment():
    """One machine, loopback only.

    No identifying detail: this file is committed to a public repository, so
    the workstation is described by architecture rather than by name.
    """
    # The node is 290 pt wide because "«execution environment»" is 101 pt at
    # 9 pt and TWO of them have to sit side by side inside it; at 250 the
    # keyword overflowed the browser box on both edges.
    W, H = 430.0, 300.0
    fig, ax = canvas(W, H)

    node3d(ax, 14, 54, 290, 232, "Workstation (Windows 11, x86-64)")

    box(ax, 88, 182, 120, 140, "", fill=WHITE)
    _label(ax, 88, 241, "«execution environment»", fs=FS_EDGE)
    _label(ax, 88, 230, "Python 3.11 (venv)", fs=FS_LABEL, weight="bold")
    box(ax, 88, 208, 106, 18, "run_live.py", fill=GREY_SOFT, fs=FS_EDGE)
    box(ax, 88, 186, 106, 18, "wifisense", fill=GREY_SOFT, fs=FS_EDGE)
    box(ax, 88, 156, 106, 32, "ThreadingHTTPServer\nbound 127.0.0.1:8000",
        fill=GREY_SOFT, fs=FS_EDGE)

    box(ax, 230, 207, 120, 90, "", fill=WHITE)
    _label(ax, 230, 241, "«execution environment»", fs=FS_EDGE)
    _label(ax, 230, 230, "Web browser", fs=FS_LABEL, weight="bold")
    box(ax, 230, 200, 106, 32, "index.html · app.js\nstyle.css", fill=GREY_SOFT,
        fs=FS_EDGE)

    box(ax, 88, 85, 120, 34, "CSI recordings (.csi.jsonl)\non local disk",
        fill=GREY_SOFT, fs=FS_EDGE)
    ax.add_line(Line2D([88, 88], [102, 112], color=INK, linewidth=LW_EDGE,
                       zorder=3))

    # The communication path runs through a 22 pt gap, so its label is set
    # vertically rather than straddling both boxes. The address it binds to is
    # already on the server artifact, so the path carries only the protocol.
    ax.add_line(Line2D([148, 170], [190, 190], color=INK, linewidth=LW_EDGE,
                       zorder=3))
    _label(ax, 159, 208, "HTTP", fs=FS_EDGE, rot=90)

    node3d(ax, 340, 168, 66, 62, "ESP32-S3 ×2", dashed=True, fill=GREY_SOFT)
    _label(ax, 373, 152, "UDP :5005 (ADR-018)", fs=FS_EDGE)
    _label(ax, 373, 141, "not deployed", fs=FS_EDGE)
    arrow(ax, (313, 199), (340, 199), dashed=True)

    note(ax, 215, 26, 300, 30,
         "no network access is required — the system\nruns with WiFi disabled")

    return save(fig, "ch3_deployment")


FIGURES = {"activity": fig_activity, "class": fig_class,
           "use_case": fig_usecase, "sequence": fig_sequence,
           "architecture": fig_architecture, "component": fig_component,
           "deployment": fig_deployment}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=sorted(FIGURES), help="draw one figure")
    args = p.parse_args()

    for name in ([args.only] if args.only else sorted(FIGURES)):
        path = FIGURES[name]()
        print(f"  {name:<14} -> {path}")


if __name__ == "__main__":
    main()
