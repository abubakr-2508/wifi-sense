# Results — WiFi CSI human sensing

Source recording: `overnight-1775217646.csi.jsonl`

All values below are measured from real Channel State Information captured
by an ESP32-S3 mesh. Nothing here is simulated. Where a quantity could not
be measured, that is stated rather than estimated.

## Dataset

| Node | Frames | Subcarriers | Duration (s) | Sample rate (Hz) |
|------|--------|-------------|--------------|------------------|
| 1 | 52574 | 64 | 3079.1 | 19.38 |
| 2 | 54911 | 64 | 3079.0 | 19.48 |

## Subcarrier selection

| Node | Selected | Variance vs mean | Near-zero subcarriers |
|------|----------|------------------|----------------------|
| 1 | 14 | 3.7× | 12 |
| 2 | 13 | 3.0× | 12 |

Variance is strongly non-uniform across subcarriers, which is why the
pipeline selects the highest-variance subcarrier rather than averaging.
Averaging dilutes the respiratory component into subcarriers that carry none.

## Respiration

| Node | Estimates | Mean | Median | SD | Range | Within 12–20 br/min |
|------|-----------|------|--------|----|-------|---------------------|
| 1 | 52207 | 14.33 | 13.63 | 4.35 | 9.1–27.3 | 56.3% |
| 2 | 54543 | 14.51 | 13.70 | 4.24 | 9.1–27.4 | 58.1% |

## Node agreement

Nodes 1 and 2 observe the same
subject over independent propagation paths, so agreement between their
estimates is evidence the measurement reflects the subject rather than a
path-specific artefact.

- Pearson correlation: **0.302**
- Mean absolute difference: **3.60 br/min**
- Bias: -0.19 br/min
- 95% limits of agreement: ±9.96 br/min
- Compared over 3059 resampled points

### Control: is the agreement real, or distributional?

Two nodes in the same room see similar noise, so similar *distributions*
are expected whether or not they track a shared signal. The control breaks
the temporal pairing (time-shift, shuffle) while leaving both distributions
unchanged. Agreement that survives decorrelation is distributional and
carries no evidential weight; agreement that collapses is real.

| Pairing | Correlation | MAE (br/min) | Median difference |
|---------|-------------|--------------|-------------------|
| aligned (true pairing) | +0.302 | 3.60 | 0.066 |
| shifted +245s | -0.004 | 4.77 | 0.066 |
| shifted +520s | +0.030 | 4.52 | 0.066 |
| shifted +1010s | +0.014 | 4.56 | 0.066 |
| shifted +1530s | +0.144 | 4.19 | 0.066 |
| fully shuffled | -0.013 | 4.83 | 0.066 |

Aligned correlation is **+0.302**; the strongest decorrelated
control reaches only **0.144**. The correlation depends on the true
temporal pairing, which indicates a **shared time-varying signal** between
the two nodes -- consistent with both observing the same subject over
independent propagation paths.

**The median difference is unchanged by shuffling (0.066 in both rows).** Median agreement is therefore a property of the two distributions and demonstrates nothing about a shared signal -- it must not be reported as validation. Temporal correlation against these controls is the evidence; the median is not.

This does not establish accuracy. No reference respiration sensor was recorded, so a positive control supports the estimate as a shared measurement, never as a correct one.

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
