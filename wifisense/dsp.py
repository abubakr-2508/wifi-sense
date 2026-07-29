"""Signal-processing primitives for WiFi-based human sensing.

The functions here are deliberately source-agnostic: they operate on a 1-D
sequence of samples regardless of whether those samples came from coarse RSSI
(Windows `netsh`, ~4 Hz) or from fine-grained CSI subcarrier amplitudes
(ESP32-S3, ~20 Hz). Sample rate is always passed in explicitly so the same
code path serves both.

Terminology used throughout:
    signal quality  Windows reports link quality as an integer percentage.
    RSSI            Received Signal Strength Indicator, in dBm.
    motion band     The frequency range where whole-body movement shows up.
    respiratory band  0.1-0.5 Hz (6-30 breaths/min). Only reachable with CSI
                    phase data -- see `respiratory_estimate` for why RSSI
                    cannot deliver this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

# Windows exposes link quality as 0-100%, not dBm. The mapping used by the
# Windows WLAN API (and documented for WLAN_SIGNAL_QUALITY) is linear between
# -100 dBm (0%) and -50 dBm (100%), i.e. dBm = quality/2 - 100.
QUALITY_MIN_DBM = -100.0
QUALITY_MAX_DBM = -50.0


def quality_to_dbm(quality_pct: float) -> float:
    """Convert a Windows link-quality percentage to approximate RSSI in dBm.

    This is a documented linear mapping, not a measurement. The quantisation is
    coarse: one percentage point equals 0.5 dBm, so RSSI derived this way has
    ~0.5 dBm resolution at best. That coarseness is the fundamental reason
    RSSI supports motion detection but not vital signs.
    """
    q = float(np.clip(quality_pct, 0.0, 100.0))
    return QUALITY_MIN_DBM + (q / 100.0) * (QUALITY_MAX_DBM - QUALITY_MIN_DBM)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove slow drift so that only fast fluctuations remain.

    Slow drift comes from things that are not people: the AP adjusting transmit
    power, thermal drift in the radio, the laptop being carried to another
    room. Subtracting a linear fit keeps the detector from reading drift as
    motion.
    """
    if x.size < 3:
        return x - x.mean() if x.size else x
    return sp_signal.detrend(x, type="linear")


def hampel(x: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace impulsive outliers with the local median (Hampel filter).

    CSI and RSSI streams both contain isolated spikes caused by packet
    corruption or a missed beacon rather than by anything physical. A single
    spike inflates the rolling variance and fires a false detection, so it is
    replaced by the local median before any statistics are computed.

    `n_sigmas` is expressed in robust sigmas: the median absolute deviation is
    scaled by 1.4826 so it estimates the standard deviation of a Gaussian.
    """
    if x.size < window or window < 3:
        return x.copy()

    out = x.copy()
    half = window // 2
    # 1.4826 makes the MAD a consistent estimator of sigma for Gaussian data.
    scale = 1.4826

    for i in range(half, x.size - half):
        chunk = x[i - half : i + half + 1]
        med = np.median(chunk)
        mad = np.median(np.abs(chunk - med))
        sigma = scale * mad
        if sigma > 0 and abs(x[i] - med) > n_sigmas * sigma:
            out[i] = med
    return out


def bandpass(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3
) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    `filtfilt` runs the filter forwards and backwards, which cancels the phase
    delay. That matters here because a delayed detection is a wrong detection:
    we want the reported motion instant to line up with the real one.

    Returns the input unchanged when the requested band is not representable at
    this sample rate (see `nyquist_ok`), rather than silently producing noise.
    """
    nyq = fs / 2.0
    if not nyq > 0:
        return x.copy()

    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high or x.size < 4 * (order + 1):
        return x.copy()

    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, x)


def nyquist_ok(fs: float, high_hz: float) -> bool:
    """True when `high_hz` is actually resolvable at sample rate `fs`.

    Sampling theory: a band whose upper edge is at or above fs/2 cannot be
    recovered. This guard is what stops the pipeline reporting a heart rate
    from a 4 Hz RSSI stream -- the 0.8-2.0 Hz cardiac band needs fs > 4 Hz, and
    claiming otherwise would be fabrication.
    """
    return fs / 2.0 > high_hz


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass
class Features:
    """Per-window summary of the signal, used by the detector."""

    std: float = 0.0             # dispersion; primary motion evidence
    mad: float = 0.0             # robust dispersion, resists residual spikes
    range_db: float = 0.0        # peak-to-peak within the window
    motion_power: float = 0.0    # spectral power inside the motion band
    dominant_hz: float = 0.0     # frequency carrying the most power
    mean_level: float = 0.0      # window mean (dBm) -- context, not evidence
    n: int = 0                   # samples contributing

    def as_dict(self) -> dict:
        return {
            "std": round(self.std, 4),
            "mad": round(self.mad, 4),
            "range_db": round(self.range_db, 3),
            "motion_power": round(self.motion_power, 5),
            "dominant_hz": round(self.dominant_hz, 4),
            "mean_level": round(self.mean_level, 2),
            "n": self.n,
        }


# Whole-body movement (walking, arm gestures, standing up) modulates the
# channel in roughly this range. The lower edge sits above the drift the
# detrender removes; the upper edge is limited by the sample rate in practice.
MOTION_BAND_HZ = (0.3, 2.0)

# Respiration. Reachable from CSI phase, not from RSSI magnitude.
RESPIRATORY_BAND_HZ = (0.1, 0.5)

# Cardiac. Listed for completeness; needs fs > 4 Hz and CSI phase.
CARDIAC_BAND_HZ = (0.8, 2.0)


def extract_features(x: np.ndarray, fs: float) -> Features:
    """Summarise one analysis window.

    The dispersion measures (`std`, `mad`) are what actually drive detection:
    a still room produces a tight distribution, a moving body widens it. The
    spectral terms are reported for the dashboard and for the report's figures,
    and give a second opinion when dispersion alone is ambiguous.
    """
    if x.size == 0:
        return Features()

    x = np.asarray(x, dtype=float)
    clean = hampel(x)
    fluct = detrend(clean)

    std = float(np.std(fluct))
    med = float(np.median(fluct))
    mad = float(np.median(np.abs(fluct - med)))
    rng = float(np.max(clean) - np.min(clean))

    motion_power, dominant = _band_power(fluct, fs, *MOTION_BAND_HZ)

    return Features(
        std=std,
        mad=mad,
        range_db=rng,
        motion_power=motion_power,
        dominant_hz=dominant,
        mean_level=float(np.mean(clean)),
        n=int(x.size),
    )


def _band_power(x: np.ndarray, fs: float, low_hz: float, high_hz: float):
    """Return (power inside the band, frequency of the strongest bin).

    Uses Welch's method, which averages periodograms over overlapping segments.
    Averaging trades frequency resolution for variance reduction -- the right
    trade here, because we care whether energy is present in a band, not about
    resolving two nearby tones.
    """
    if x.size < 8 or fs <= 0:
        return 0.0, 0.0

    nperseg = int(min(x.size, max(8, 2 ** int(math.log2(max(x.size // 2, 8))))))
    try:
        freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    except ValueError:
        return 0.0, 0.0

    band = (freqs >= low_hz) & (freqs <= min(high_hz, fs / 2.0))
    if not np.any(band):
        return 0.0, 0.0

    power = float(np.trapezoid(psd[band], freqs[band]))
    peak_idx = int(np.argmax(psd[band]))
    dominant = float(freqs[band][peak_idx])
    return power, dominant


# ---------------------------------------------------------------------------
# Respiration (CSI only)
# ---------------------------------------------------------------------------


@dataclass
class RespiratoryEstimate:
    """Outcome of a breathing-rate estimate, including refusal cases."""

    bpm: float = 0.0
    confidence: float = 0.0
    supported: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "bpm": round(self.bpm, 2),
            "confidence": round(self.confidence, 3),
            "supported": self.supported,
            "reason": self.reason,
        }


def respiratory_estimate(x: np.ndarray, fs: float) -> RespiratoryEstimate:
    """Estimate breathing rate, or explain why it cannot be estimated.

    This function refuses rather than guesses. Two conditions must hold:

    1. The respiratory band must be resolvable at this sample rate. At the
       ~4 Hz an RSSI poll achieves the band is nominally in range, but see (2).
    2. There must be enough samples to resolve a 0.1 Hz feature. Frequency
       resolution is fs/N, so distinguishing breathing rates needs a window of
       at least ~30 s.

    Even when both hold, an estimate from RSSI *magnitude* is not comparable to
    one from CSI *phase*: chest displacement is millimetric and moves the phase
    of individual subcarriers, while barely touching aggregate signal strength.
    Callers working from RSSI should treat any returned value as indicative
    only -- which is why `supported` is set from the caller's declared source.
    """
    if x.size == 0:
        return RespiratoryEstimate(reason="no samples")

    low, high = RESPIRATORY_BAND_HZ
    if not nyquist_ok(fs, high):
        return RespiratoryEstimate(
            reason=f"sample rate {fs:.1f} Hz cannot resolve {high} Hz (Nyquist)"
        )

    duration = x.size / fs
    min_duration = 1.0 / low  # one full cycle at the slowest rate of interest
    if duration < min_duration:
        return RespiratoryEstimate(
            reason=f"window {duration:.1f}s shorter than one cycle ({min_duration:.1f}s)"
        )

    fluct = detrend(hampel(np.asarray(x, dtype=float)))
    band = bandpass(fluct, fs, low, high)
    power, dominant = _band_power(band, fs, low, high)

    if dominant <= 0:
        return RespiratoryEstimate(reason="no spectral peak in respiratory band")

    total_power, _ = _band_power(fluct, fs, 0.0, fs / 2.0)
    confidence = float(power / total_power) if total_power > 0 else 0.0

    return RespiratoryEstimate(
        bpm=dominant * 60.0,
        confidence=min(confidence, 1.0),
        supported=True,
        reason="ok",
    )
