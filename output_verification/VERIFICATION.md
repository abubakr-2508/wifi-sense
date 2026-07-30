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

The estimator picks the strongest Welch bin inside 0.1-0.5 Hz. That makes
its output discrete, and the grid is coarse relative to the range it has
to cover.

- **Node 1**: 29585 windows taking **9 distinct values**, spaced 2.271 bpm apart. Three values account for 79.0% of the output.
  Values: 9.09, 11.36, 13.63, 15.90, 18.17, 20.44, 22.71, 24.99, 27.26
- **Node 2**: 29787 windows taking **9 distinct values**, spaced 2.283 bpm apart. Three values account for 75.3% of the output.
  Values: 9.13, 11.41, 13.70, 15.98, 18.26, 20.54, 22.83, 25.11, 27.39

Two of those values stand in an exact 2:1 ratio, which is the signature of
an octave ambiguity rather than of two independent measurements disagreeing.

Because the output is discrete, agreement between the nodes has a floor
that owes nothing to physiology: two independent estimators restricted to
the same handful of bins will coincide some of the time by construction.
Cohen's kappa corrects for exactly that.

| Quantity | Value |
|---|---|
| Windows where both nodes reported | 28904 |
| Exact agreement | 33.4% |
| Chance floor from the bin structure | **21.6%** |
| **Cohen's kappa** | **+0.151** |
| Pearson r on the same series | +0.303 |

The Pearson value reproduces the correlation reported in `RESULTS.md`, so
this is the same comparison seen two ways. On the conventional
interpretation a kappa of this size is *slight* agreement. The correlation
and the chance-corrected statistic disagree about how much the cross-node
result is worth, and the chance-corrected one is the appropriate measure
for a discrete output.

This does not replace the segment-stability test in `ABLATION.md`, which is
direct empirical evidence and stands on its own. It explains a mechanism by
which the apparent agreement was possible.

See `fig_v3_estimator_bins.png`.
