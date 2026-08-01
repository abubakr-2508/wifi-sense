# Ablation study

Source recording: `overnight-1775217646.csi.jsonl`

## Objective metric

No ground-truth respiration sensor was recorded, so accuracy cannot be
the objective. Instead each configuration is scored by **cross-node
agreement**: the two nodes observe the same subject over independent
paths, so a configuration that recovers the shared signal makes their
respiration estimates correlate more. Every configuration is also scored
against a decorrelation control (seeded shuffle + time-shifts). The
ranking metric is **excess correlation = aligned - control**: a
configuration is credited only for the agreement that exceeds what
decorrelation already produces. This matters -- raw aligned correlation
can be inflated by distributional similarity (averaging many subcarriers
smooths both nodes' series and makes them look alike regardless of
signal), and the control captures exactly that. Ranking by raw aligned
correlation would, for example, wrongly prefer no preprocessing.

The 51-minute recording is used because it is the case with a recoverable
shared signal. The 2-minute mixed-activity recording is the negative case
(see `output/RESULTS.md`) and offers nothing to optimise.

Two spans appear in this report and they measure different things. The
ablation reads the first 40,000 single-antenna frames per node, which
reaches 38.1 minutes into the 51.3-minute recording; frames of other
antenna geometries are excluded because their amplitude statistics differ
substantially, and mixing them injects variance from population switching
rather than from the channel. Where this report says *n-minute series* it
means the analysed span, not the length of the capture.

## 1. Subcarrier aggregation (headline)

This is the project's key design choice: respiration is estimated from
the median of the top-K highest-variance subcarriers, not from a single
subcarrier and not from the mean of all. The sweep sets K from 1 to 32
and includes the mean-of-all baseline.

| K | Aligned r | Control | Excess | Spread (br/min) |
|---|-----------|---------|--------|-----------------|
| 1 | +0.233 | 0.137 | +0.096 | 0.00 |
| 2 | +0.285 | 0.132 | +0.153 | 1.67 |
| 4 | +0.278 | 0.099 | +0.179 | 2.45 |
| 8 | +0.374 | 0.111 | +0.263 | 3.45 |
| 16 | +0.357 | 0.100 | +0.257 | 4.15 |
| 32 | +0.377 | 0.154 | +0.224 | 4.68 |
| mean | +0.128 | 0.133 | -0.005 | 0.00 |

**Best: K = 8** (excess +0.263: aligned +0.374 over control 0.111).

Both extremes are rejected, but only one of them fails the control test,
and the distinction matters.

**Mean-of-all fails it.** Its control (0.133) is at least
as large as its aligned correlation (+0.128), leaving an
excess of -0.005: the apparent agreement is distributional,
not shared signal. Averaging every subcarrier dilutes the respiratory
component into the majority that carry none.

**K = 1 does not fail that way.** Its excess is +0.096 -- positive,
and clear of its control (0.137 against an aligned
+0.233). It is rejected on two independent grounds instead:
it returns the smallest excess of any K in the sweep, and a single
subcarrier is hostage to whichever spectral peak it happens to land on,
which is what produced the factor-of-two disagreement between the two
nodes reported earlier. A median over several subcarriers outvotes that
failure; one subcarrier cannot.

Excess peaks in the intermediate range.

The spread column is reported for completeness but does **not**
independently identify the best K: it rises monotonically with K, which is
expected because widening the selection admits more disagreeing
subcarriers, and it is zero at K = 1 only trivially (a single estimate has
no spread). Spread is therefore a description of the selection, not a
second line of evidence for it. See `fig_ablation1_topk.png`.

## 2. Respiration window length

| Window (s) | Aligned r | Control | Excess |
|-----------|-----------|---------|--------|
| 15 | +0.259 | 0.078 | +0.181 |
| 20 | +0.322 | 0.118 | +0.204 |
| 30 | +0.374 | 0.111 | +0.263 |
| 45 | +0.391 | 0.204 | +0.188 |
| 60 | +0.310 | 0.248 | +0.063 |

**Best: 30 s** (excess +0.263). Frequency
resolution is fs/N, so a longer window resolves the respiratory band more
finely and raw agreement tends to rise; but past a point the subject no
longer holds still across the window, the control climbs, and the excess
falls. The mid-range windows give the best excess. See
`fig_ablation2_window.png`.

## 3. Preprocessing stages

| Stages | Aligned r | Control | Excess |
|--------|-----------|---------|--------|
| none | +0.401 | 0.072 | +0.329 |
| Hampel | +0.375 | 0.112 | +0.263 |
| detrend | +0.401 | 0.072 | +0.329 |
| both | +0.374 | 0.111 | +0.263 |

**Best by this metric: detrend** (excess +0.329).

Two findings here, both negative for the preprocessing stages, and both
worth stating plainly rather than dressing up:

**Linear detrend is redundant.** Its rows are indistinguishable from the
corresponding no-detrend rows. The reason is structural: the pipeline
band-passes to 0.1-0.5 Hz and then takes the argmax of the spectrum, and
the band-pass already rejects DC and slow trend. Over 200 synthetic
trials with injected linear trends, adding an explicit detrend changed the
extracted peak in 1 trial out of 200 (99.5% identical, mean difference
0.011 br/min). Detrend would matter for a waveform-based estimator such
as zero-crossing counting; for peak extraction it does not.

**Hampel filtering does not improve this metric.** It lowers excess
correlation from +0.329 to +0.263 over this recording. Outlier removal is defensible on
signal-quality grounds -- an impulsive spike genuinely is not channel
information -- but the ablation provides no evidence that it improves
cross-node agreement, and the honest conclusion is that its benefit is
not demonstrated here. See `fig_ablation3_preprocessing.png`.

## 4. Filter order (robustness check)

| Butterworth order | Aligned r | Control |
|-------------------|-----------|---------|
| 2 | +0.359 | 0.109 |
| 3 | +0.374 | 0.111 |
| 5 | +0.346 | 0.110 |

Aligned correlation varies by only 0.028 across orders 2-5, so
the band-pass order is not a material factor for this metric; order 3 is
used as a standard default. (Zero-phase `filtfilt` is retained throughout
for timing fidelity, independent of order.)

## 5. Segment stability — the most important caveat

A correlation measured over a whole recording can arise from slow
structure common to both nodes rather than from breath-by-breath
tracking. Re-measuring the chosen configuration on disjoint segments
distinguishes the two: a genuine respiratory signal should be detectable
*within* a segment, whereas a long-timescale artefact appears only over
the full series.

| Segment | Aligned r | Control | Excess |
|---------|-----------|---------|--------|
| 0 | +0.260 | 0.311 | -0.050 |
| 1 | +0.269 | 0.339 | -0.070 |
| 2 | +0.194 | 0.153 | +0.042 |
| 3 | +0.214 | 0.323 | -0.109 |
| 4 | +0.150 | 0.234 | -0.084 |

Per-segment excess: mean **-0.054**, SD **0.052**, across 5 segments of 6.9 minutes each.

Against the full-recording excess of **+0.263**, 4 of 5 segments have a **negative** excess -- meaning the
decorrelated control correlates *more strongly* than the true pairing.
The per-segment mean is -0.054, and mean ± 1 SD does not reach zero.

**The agreement is therefore scale-dependent: within a single segment
there is no detectable shared respiratory signal at all, and the
positive excess appears only over the full 38-minute series.**

This is a decisive qualification, and it revises the earlier reading of
the node-agreement result. A breath-by-breath measurement should
survive segmentation; one that does not is more consistent with
slowly-varying structure shared by both nodes -- a common
environmental drift, or both estimators settling over time -- than with
respiration. Note also that the segment controls are themselves large
(up to 0.34), which is the signature of exactly such slow common
structure: a time-shift inside a short segment still overlaps the trend
it is meant to break.

That inflation matters for how this result should be read, and it cuts
in our own favour. A control that overlaps the trend it is meant to
break is too high; a control that is too high makes the excess too low;
and an excess that is too low biases this test *towards* the negative
reading drawn from it. The honest statement is therefore that per-window
respiration tracking is **unsupported here**, not that a shared signal
has been shown to be absent. A circular shift with a guard band measured
from the autocorrelation would remove the bias and settle which it is.

The shuffle control destroys all temporal ordering and therefore cannot
separate these two cases. The segment test can, and it does not support
the respiratory interpretation.

**Honest position: full-recording cross-node agreement is measurable and
survives decorrelation, but it does not establish per-window
respiration tracking, and the segment test argues against that
interpretation.** Settling it requires a ground-truth respiration
reference, which none of these recordings provides. This is recorded
here rather than omitted because it is the strongest counter-evidence
the study produced against its own most attractive result.

## Summary of chosen configuration

- Subcarrier aggregation: **top-8** median
- Respiration window: **30 s**
- Preprocessing: Hampel retained on signal-quality grounds; explicit
  detrend retained but shown to be redundant before the band-pass
- Filter: Butterworth order 3, zero-phase

The subcarrier-K and window choices are the ones that maximised validated
cross-node agreement, and both coincide with the values the pipeline
already used -- so the defaults are empirically justified rather than
assumed. The preprocessing stages are **not** vindicated by this metric,
and the segment-stability result above limits how strongly any of it can
be claimed.

## Figures

1. `fig_ablation1_topk.png` — subcarrier aggregation sweep
2. `fig_ablation2_window.png` — respiration window length
3. `fig_ablation3_preprocessing.png` — preprocessing stages
4. `fig_ablation4_stability.png` — per-segment vs full-recording agreement
