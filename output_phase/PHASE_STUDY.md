# Phase-domain investigation

Source recording: `overnight-1775217646.csi.jsonl`

## Motivation

In the CSI sensing literature, respiration is usually recovered from
subcarrier **phase** rather than amplitude: chest displacement is
millimetric and rotates the phase of the reflected path, while barely
changing received power. This section tests whether phase offers that
advantage on the ESP32 recordings used here.

## The problem with raw ESP32 phase

Raw phase is not directly usable. It carries two hardware artefacts far
larger than any respiratory signal:

- **Carrier-frequency offset (CFO)** — a drift over time that, once the
  phase wraps, makes a single subcarrier's phase series look random.
- **Sampling-time offset (STO)** — a linear ramp across subcarriers.

`fig_phase1_sanitization.png` shows both: the raw phase across
subcarriers is a sloped line (STO), and over time a single subcarrier
wraps repeatedly (CFO).

## Method 1 — single-antenna linear sanitization

The standard single-device remedy removes, per frame, a linear fit of
phase against subcarrier index. This cancels the STO slope and a
constant offset. It is implemented in `sanitize_phase()` and applied on
every frame.

Comparing the respiratory-band power fraction of amplitude against
sanitized phase across 52 active subcarriers (sample rate 19.6 Hz):

| Signal | Mean in-band fraction | Max | Subcarriers > 0.10 |
|--------|-----------------------|-----|--------------------|
| Amplitude | 0.044 | 0.072 | 0 |
| Sanitized phase | 0.046 | 0.069 | 0 |

**Sanitized phase does not improve on amplitude here** — the two are
comparable and both weak. Single-antenna linear sanitization removes
the linear hardware terms but leaves residual phase noise that is not
smaller than the amplitude fluctuation, so it confers no advantage for
respiration on this single-antenna capture. See
`fig_phase2_amp_vs_phase.png`.

## Method 2 — multi-antenna conjugate multiplication

The technique that reliably recovers phase in the literature multiplies
one antenna's CSI by the complex conjugate of another. The two antennas
share the same CFO and STO, so the product cancels them and leaves a
clean relative phase. This requires a contiguous run of multi-antenna
frames spanning a respiratory window.

This capture does not provide that. Multi-antenna frames occur, but as
isolated frames scattered among single-antenna frames:

- Multi-antenna runs found: **10710**
- Median run length: **1 frame(s)**
- Longest run: **9 frames** (~0.46 s)
- Frames needed for one respiratory window: **586**

The longest contiguous multi-antenna stretch is therefore roughly two
orders of magnitude short of the 586-frame window a
respiratory estimate needs. **Conjugate multiplication is infeasible on
this capture.** See `fig_phase3_contiguity.png`.

(An earlier attempt that pooled the scattered multi-antenna frames as if
they were a continuous series produced a spuriously strong respiratory
peak; treating irregularly-sampled frames as uniform is what created it.
This is recorded here because it is the exact trap the contiguity check
exists to catch.)

## Conclusion

- Raw ESP32 phase is corrupted by CFO and STO and must be sanitized.
- Single-antenna linear sanitization is correct but yields **no
  respiratory advantage over amplitude** on this data.
- Multi-antenna conjugate multiplication — the method that works in the
  literature — is **infeasible here** because the capture interleaves
  single- and multi-antenna frames rather than providing contiguous
  multi-antenna streams.

Amplitude therefore remains the practical channel for this hardware and
dataset, which is why the main pipeline uses it. Robust phase-based
sensing needs **uniform multi-antenna capture** — a concrete hardware
requirement for future work: an ESP32 configured to stream a fixed
multi-antenna geometry every frame, or a Raspberry Pi with Nexmon CSI,
which exposes wider bandwidth and consistent multi-antenna CSI.

## Figures

1. `fig_phase1_sanitization.png` — CFO/STO corruption and its removal
2. `fig_phase2_amp_vs_phase.png` — amplitude vs sanitized phase
3. `fig_phase3_contiguity.png` — multi-antenna frames too fragmented
