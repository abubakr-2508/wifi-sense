# Motion-detector evaluation

Source recording: `overnight-1775217646.csi.jsonl` — real quiet CSI used as the baseline.

## Why this is a characterisation, not a field-accuracy figure

No ground-truth motion reference was recorded, so accuracy against a real
moving person cannot be stated. The detector is instead characterised
against a controlled contrast on a single recording, which removes the
gain, frame-geometry and sample-rate confounds a cross-recording
comparison would carry:

- **Negatives** — real quiet windows from this recording.
- **Positives** — the same windows with a motion-like perturbation added,
  band-limited to the 0.3–2 Hz motion band and scaled to a target SNR
  against each window's own quiet dispersion.

This measures two honest quantities: the perturbation strength required
for reliable detection, and the false-positive rate on genuine quiet data.

## ROC (n = 195 windows per class)

| Injected SNR | AUC |
|--------------|-----|
| 0.25 | 0.568 |
| 0.5 | 0.740 |
| 1 | 0.941 |
| 2 | 1.000 |
| 4 | 1.000 |

AUC rises with SNR to 1.000 at the strongest perturbation, and
approaches chance (0.5) as the perturbation falls below the quiet noise
floor — exactly the expected behaviour of a dispersion detector. See
`fig_eval1_roc.png`.

## Sensitivity and operating point

- False-positive rate on genuine quiet windows at the z ≥ 3 operating threshold: **0.005**
- Detection reaches 90% at an injected SNR of **≈ 1.83** (perturbation RMS ≈ 1.83× the quiet dispersion).

Detection rate by SNR at the operating threshold:

| SNR | Detection rate |
|-----|----------------|
| 0.25 | 0.015 |
| 0.5 | 0.031 |
| 1 | 0.446 |
| 2 | 0.995 |
| 4 | 1.000 |

See `fig_eval2_sensitivity.png`. The detector is insensitive to motion
well below the quiet noise floor (as it must be to keep false positives
low) and reliable once the perturbation approaches the quiet dispersion.

## Confusion matrix (operating threshold z ≥ 3, SNR 1)

| | Predicted motion | Predicted quiet |
|---|---|---|
| **Actual motion** | 87 (TP) | 108 (FN) |
| **Actual quiet**  | 1 (FP) | 194 (TN) |

- Precision **0.989**, recall **0.446**, F1 **0.615** at this operating point. See `fig_eval3_confusion.png`.

## Real-data cross-check

As an independent check on real data, the detector's firing rate — the
fraction of windows it flags as motion at z ≥ 3, self-calibrated on each
recording's own quietest windows — is compared between the `pretrain`
recording (labelled "mixed-activity", a real moving person) and the quiet
recording:

- Moving-person recording: fires on **36.4%** of windows (n=55)
- Quiet recording: fires on **8.6%** of windows (n=209)

The detector fires substantially more often on the labelled moving-person recording, an independent real-data confirmation that its statistic tracks genuine activity.

## Honest scope

This evaluation establishes the detector's sensitivity and false-positive
behaviour under a controlled, physically-motivated stimulus, and shows on
real data that its statistic tracks genuine activity. It does **not**
establish field accuracy against real human motion, which would require a
synchronised ground-truth sensor. That is the natural next step with the
ESP32 hardware.

## Figures

1. `fig_eval1_roc.png` — ROC by injected SNR
2. `fig_eval2_sensitivity.png` — detection rate vs SNR, false-positive floor
3. `fig_eval3_confusion.png` — confusion matrix at the operating point
