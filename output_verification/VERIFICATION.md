# Verification

Source recording: `overnight-1775217646.csi.jsonl`

Three things the other studies assume. Checked here so a reader can
repeat the checks rather than take them on trust.

## 1. Is the data genuine measured CSI?

**node 1** — 52574 frames over 51.3 minutes.

| Test | Result | Reading |
|---|---|---|
| Always-zero subcarriers | [0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] | DC null plus guard band |
| Active subcarriers | **52** | 802.11n HT20 carries exactly 52 |
| Inter-frame timing CV | **0.48** | bursty and contended; a generator is regular |
| Repeated payloads | 0 | mock data cycles |
| I/Q samples at the int8 rails | **0.000%** | random fill would saturate |
| Distinct RSSI values | 0 | a real link drifts |
| Frame geometries | {64: 52574, 128: 13049, 192: 535} | occasional multi-antenna frames |

**node 2** — 54911 frames over 51.3 minutes.

| Test | Result | Reading |
|---|---|---|
| Always-zero subcarriers | [0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37] | DC null plus guard band |
| Active subcarriers | **52** | 802.11n HT20 carries exactly 52 |
| Inter-frame timing CV | **0.45** | bursty and contended; a generator is regular |
| Repeated payloads | 0 | mock data cycles |
| I/Q samples at the int8 rails | **0.000%** | random fill would saturate |
| Distinct RSSI values | 0 | a real link drifts |
| Frame geometries | {64: 54911, 128: 11239, 192: 507} | occasional multi-antenna frames |

The null pattern is the decisive test. Twelve subcarriers are zero in every
frame and they sit at index 0 and indices 27-37 — the DC null and the guard
band of an 802.11n 20 MHz channel — leaving 52 active, which is exactly the
48 data plus 4 pilot subcarriers the standard defines. Nothing synthetic
reproduces that pattern and a physically plausible fading profile by accident.

This matters because the project that published these recordings has been
publicly criticised for shipping fabricated output. That criticism is
well-founded for its *derived* frames, which are not used here: across the
whole recording the `vitals` packets report a constant occupancy of 4 people
and a `presence_score` identical to `motion_energy` in every single frame.
This project reads only the `raw_csi` payloads and computes everything else
itself.

See `fig_v1_null_pattern.png`.

## 2. Does variance-based subcarrier selection land in deep fades?

A known objection to top-K-by-variance is that the highest-variance
subcarriers tend to sit in deep fades, where they are over-sensitive. The
ablation swept how *many* subcarriers to take but never which rule picks
them, so the objection is tested here directly.

| Node | corr(mean amplitude, variance) | selected top-8 | below median amplitude |
|---|---|---|---|
| node 1 | -0.157 | [12, 13, 14, 15, 16, 61, 62, 63] | 4 of 8 |
| node 2 | -0.123 | [3, 12, 13, 14, 19, 60, 61, 62] | 5 of 8 |

See `fig_v2_variance_vs_amplitude.png`.

## 3. What can the respiration estimator actually express?

Respiration is the median of K = 8 per-subcarrier estimates, and each of
those is the strongest Welch bin inside 0.1-0.5 Hz. The output is therefore
discrete. It is also finer-grained than it looks, for a reason worth
stating: a median over an EVEN number of values averages the two central
ones, so whenever those fall on adjacent bins the result lands exactly
halfway between them. The observed grid is the true Welch grid plus its
midpoints. Only the even-indexed values are frequencies the estimator can
resolve; the rest are averaging artefacts. An odd K would remove them.

- **Node 1**: 29585 windows taking **9 distinct values**, apparently spaced 2.271 bpm apart, but resting on only **5 real bins** at 4.543 bpm. Three values hold 79.0% of the output.
  Resolvable: 9.09, 13.63, 18.17, 22.71, 27.26
  Median artefacts: 11.36, 15.90, 20.44, 24.99
- **Node 2**: 29787 windows taking **9 distinct values**, apparently spaced 2.283 bpm apart, but resting on only **5 real bins** at 4.566 bpm. Three values hold 75.3% of the output.
  Resolvable: 9.13, 13.70, 18.26, 22.83, 27.39
  Median artefacts: 11.41, 15.98, 20.54, 25.11

Two of these values stand in an exact 2:1 ratio. That is **not** evidence
of an octave ambiguity, and an earlier draft of this file wrongly said it
was: a uniform grid contains 2:1 index pairs by construction, so the
exactness of the ratio carries no information about the signal. What does
need explaining is the *bimodality* -- two non-adjacent values holding
roughly half the mass -- and the contingency table below is what
distinguishes the candidate explanations.

Because the output is discrete, agreement between the nodes carries a floor
that owes nothing to physiology: two independent estimators restricted to
the same handful of bins coincide some of the time by construction. Kappa
corrects for that.

| Quantity | Value |
|---|---|
| Windows where both nodes reported | 28904 |
| Exact agreement | 33.4% |
| Chance floor from the bin structure | **21.6%** |
| Cohen's kappa, unweighted | +0.151 |
| Cohen's kappa, linear weights | +0.248 |
| **Cohen's kappa, quadratic weights** | **+0.302** |
| Pearson r on the same series | +0.303 |

The Pearson value reproduces the correlation in `RESULTS.md`, so this is
the same comparison seen two ways.

The bins are ORDERED, so unweighted kappa is the wrong variant on its own --
it treats a one-bin miss and a four-bin miss as identical failures. The
quadratic-weighted figure is the appropriate headline; note that it is
equivalent to the intraclass correlation. All three are given because the
gap between them measures how much of the disagreement is large-magnitude
rather than adjacent-bin.

**Two caveats, both of which apply here.** The conventional interpretive
bands for kappa are convention rather than derived result -- their authors
offered no evidence for them -- so no verbal label is attached to these
numbers. And kappa is known to misbehave when the marginal distributions
are unbalanced, which is exactly this case: three of the nine values hold
three quarters of the mass. The raw agreement and the chance floor are
given above so the coefficient can be checked against its inputs.

### What the contingency table says

- Mass on the diagonal or its immediate neighbours: **50.8%**
- Share of the OFF-diagonal mass in cells where the bin indices stand in a
  2:1 ratio: **9.7%**
- Largest single row / column share: 29.9% / 31.0%
- Windows where both nodes changed bin at once: 0.2%,
  against 0.1% if the two changed independently
  (phi = +0.020)

Mass concentrated near the diagonal indicates coarse quantisation of a
single underlying distribution; mass in the 2:1 cells would indicate octave
confusion; mass concentrated in one row or column would indicate one node
locking onto something non-respiratory. Synchronised changes would suggest
a property of the signal, independent ones noise-driven peak selection.
See `fig_v4_confusion.png`, where the 2:1 cells are outlined.

**A limitation not addressed here.** Parabolic interpolation of the
periodogram peak would give sub-bin resolution and remove the need for a
categorical statistic altogether. It is not implemented: doing so would
change every respiration number in this report, and the pipeline is frozen.
It is the first thing to try if this work is continued.

None of this replaces the segment-stability test in `ABLATION.md`, which is
direct empirical evidence and stands on its own. This section describes a
mechanism by which the apparent agreement was possible.

See `fig_v3_estimator_bins.png`.
