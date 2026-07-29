# Results — WiFi CSI human sensing

Source recording: `pretrain-1775182186.csi.jsonl`

All values below are measured from real Channel State Information captured
by an ESP32-S3 mesh. Nothing here is simulated. Where a quantity could not
be measured, that is stated rather than estimated.

## Dataset

| Node | Frames | Subcarriers | Duration (s) | Sample rate (Hz) |
|------|--------|-------------|--------------|------------------|
| 1 | 2133 | 64 | 119.9 | 19.49 |
| 2 | 1994 | 64 | 120.0 | 18.94 |

## Subcarrier selection

| Node | Selected | Variance vs mean | Near-zero subcarriers |
|------|----------|------------------|----------------------|
| 1 | 56 | 3.3× | 12 |
| 2 | 1 | 4.0× | 12 |

Variance is strongly non-uniform across subcarriers, which is why the
pipeline selects the highest-variance subcarrier rather than averaging.
Averaging dilutes the respiratory component into subcarriers that carry none.

## Respiration

| Node | Estimates | Mean | Median | SD | Range | Within 12–20 br/min |
|------|-----------|------|--------|----|-------|---------------------|
| 1 | 1765 | 15.52 | 15.98 | 3.44 | 9.1–22.8 | 77.4% |
| 2 | 1636 | 11.15 | 11.10 | 2.27 | 8.9–17.8 | 43.2% |

## Node agreement

Nodes 1 and 2 observe the same
subject over independent propagation paths, so agreement between their
estimates is evidence the measurement reflects the subject rather than a
path-specific artefact.

- Pearson correlation: **-0.157**
- Mean absolute difference: **5.00 br/min**
- Bias: +4.36 br/min
- 95% limits of agreement: ±8.72 br/min
- Compared over 240 resampled points

### Control: is the agreement real, or distributional?

Two nodes in the same room see similar noise, so similar *distributions*
are expected whether or not they track a shared signal. The control breaks
the temporal pairing (time-shift, shuffle) while leaving both distributions
unchanged. Agreement that survives decorrelation is distributional and
carries no evidential weight; agreement that collapses is real.

| Pairing | Correlation | MAE (br/min) | Median difference |
|---------|-------------|--------------|-------------------|
| aligned (true pairing) | -0.157 | 5.00 | 4.888 |
| shifted +8s | -0.134 | 4.93 | 2.604 |
| shifted +17s | +0.185 | 4.65 | 4.329 |
| shifted +33s | -0.072 | 5.27 | 4.952 |
| shifted +50s | -0.193 | 5.03 | 2.668 |
| fully shuffled | -0.079 | 5.03 | 4.888 |

Aligned correlation is **-0.157** -- negative. Two nodes observing
the same subject would correlate positively, so this recording provides
**no evidence of a shared respiratory signal**. The per-window estimates are
consistent with band-limited noise rather than respiration.

**The median difference is unchanged by shuffling (4.888 in both rows).** Median agreement is therefore a property of the two distributions and demonstrates nothing about a shared signal -- it must not be reported as validation. Temporal correlation against these controls is the evidence; the median is not.

Accuracy is not assessable either way: no reference respiration sensor was recorded. The control establishes only that these estimates are not supported as a shared measurement.

## Limitations

- **No ground truth.** No reference respiration sensor was recorded alongside
  the CSI, so accuracy cannot be stated. Node agreement is a consistency check,
  not a validation against truth.
- **Presence detection is a threshold, not a classifier.** It compares window
  dispersion against a learned ambient baseline. It will respond to any
  channel disturbance, including non-human sources such as fans.
- **Pose estimation is not implemented.** No trained keypoint weights are
  loaded, and the system draws no skeleton rather than drawing a fabricated one.
- **Cardiac estimation is not attempted** at this sample rate and source.
- **Commodity RSSI was evaluated and rejected.** On a MediaTek MT7922 adapter,
  RSSI was static across 99 samples over 30 s via netsh and 236 samples over
  12 s via direct WLAN API queries — driver-level smoothing removes the
  fluctuation sensing depends on. This motivates dedicated CSI hardware.

## Figures

1. `fig1_dataset_overview.png` — amplitude heatmap and frame timing
2. `fig2_subcarrier_variance.png` — variance per subcarrier, selection justified
3. `fig3_respiration_extraction.png` — raw → band-passed → spectrum
4. `fig4_respiration_timeline.png` — breathing rate and motion evidence over time
5. `fig5_node_agreement.png` — scatter and Bland–Altman agreement
