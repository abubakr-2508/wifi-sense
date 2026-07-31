"""Presence and motion detection from a conditioned signal stream.

The detector is deliberately simple and fully explainable -- there is no
trained model here, and it does not pretend to be one. It works by comparing
the current window's dispersion against a baseline learned from an empty room:

    1. CALIBRATING  Collect ambient samples with nobody moving. Record the
                    mean and standard deviation of the window dispersion.
    2. Running      For each new window compute a z-score of dispersion
                    against that baseline.
    3. Hysteresis   Enter MOTION above a high threshold, leave it only below a
                    lower one. Two thresholds instead of one is what stops the
                    state chattering when the signal sits near the boundary.
    4. Debounce     Require several consecutive agreeing windows before
                    changing state, which suppresses single-window spikes.

Calibration quality is the whole ballgame: a baseline captured while someone
was moving raises the bar so far that real motion never crosses it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

import numpy as np

from .dsp import (
    CARDIAC_BAND_HZ,
    Features,
    RespiratoryEstimate,
    extract_features,
    nyquist_ok,
    respiratory_estimate,
)


class State(str, Enum):
    CALIBRATING = "calibrating"
    IDLE = "idle"          # room reads as still
    MOTION = "motion"      # movement detected
    NO_SIGNAL = "no_signal"


@dataclass
class Baseline:
    """Ambient statistics learned during calibration."""

    mean_std: float = 0.0
    sigma_std: float = 0.0
    samples: int = 0
    captured_at: float = 0.0
    complete: bool = False

    def as_dict(self) -> dict:
        return {
            "mean_std": round(self.mean_std, 5),
            "sigma_std": round(self.sigma_std, 5),
            "samples": self.samples,
            "complete": self.complete,
        }


@dataclass
class Reading:
    """One detector output, ready to serialise to the dashboard."""

    timestamp: float
    state: State
    z_score: float
    confidence: float
    features: Features
    respiration: RespiratoryEstimate
    rssi_dbm: float
    calibration_progress: float = 1.0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "z_score": round(self.z_score, 3),
            "confidence": round(self.confidence, 3),
            "features": self.features.as_dict(),
            "respiration": self.respiration.as_dict(),
            "rssi_dbm": round(self.rssi_dbm, 2),
            "calibration_progress": round(self.calibration_progress, 3),
            "note": self.note,
        }


class MotionDetector:
    """Windowed dispersion detector with learned ambient baseline.

    Parameters
    ----------
    fs
        Sample rate of the incoming stream, Hz. Drives window sizing and
        governs which frequency bands are physically resolvable.
    window_s
        Analysis window length in seconds. Longer windows are steadier but
        slower to react; ~4 s is a reasonable compromise for walking-speed
        motion.
    calibration_s
        Ambient capture duration. Must cover an empty, still room.
    enter_z, exit_z
        Hysteresis thresholds in units of baseline sigma. `enter_z > exit_z`.
    debounce
        Consecutive agreeing windows required before the state flips.
    source_kind
        "rssi" or "csi". Gates whether vital-sign output is even attempted --
        an RSSI stream never reports breathing, by construction rather than by
        threshold.
    """

    def __init__(
        self,
        fs: float,
        window_s: float = 4.0,
        calibration_s: float = 15.0,
        enter_z: float = 3.0,
        exit_z: float = 1.5,
        debounce: int = 2,
        source_kind: str = "rssi",
        respiration_window_s: float = 30.0,
        respiration_top_k: int = 8,
    ):
        self.fs = float(fs)
        self.window_n = max(8, int(round(window_s * fs)))
        self.calibration_n = max(self.window_n * 2, int(round(calibration_s * fs)))
        self.enter_z = enter_z
        self.exit_z = exit_z
        self.debounce = max(1, debounce)
        self.source_kind = source_kind

        # Motion and respiration need different window lengths, so they get
        # separate buffers. Motion wants a short window to stay responsive;
        # respiration wants a long one because frequency resolution is fs/N and
        # separating breathing rates around 0.1-0.5 Hz needs tens of seconds.
        # Using one window for both would force a bad compromise on each.
        self.respiration_n = max(self.window_n, int(round(respiration_window_s * fs)))

        self.buffer: Deque[float] = deque(maxlen=self.window_n)
        self.resp_buffer: Deque[float] = deque(maxlen=self.respiration_n)
        self.history: Deque[float] = deque(maxlen=600)   # z-score trail for the UI
        self.level_history: Deque[float] = deque(maxlen=600)

        # Rolling per-subcarrier amplitudes, kept only for CSI sources. Used to
        # pick the subcarriers that actually carry the respiration signal.
        self.vector_buffer: Deque[np.ndarray] = deque(maxlen=self.respiration_n)
        self.respiration_top_k = max(1, respiration_top_k)
        self.selected_subcarrier: Optional[int] = None
        self.candidate_subcarriers: list[int] = []
        self.respiration_spread = 0.0      # SD across per-subcarrier estimates
        self.respiration_agreement = 0.0   # fraction within 15% of the median

        # Estimating respiration costs K spectral transforms, and breathing
        # does not change measurably between consecutive samples at 20 Hz.
        # Recompute about once a second and serve the cached value in between:
        # same output, a fraction of the work, and the live loop keeps up.
        self._resp_interval = max(1, int(round(self.fs)))
        self._resp_countdown = 0
        self._resp_cached: Optional[RespiratoryEstimate] = None

        self._calib_disp: list[float] = []
        self._calib_count = 0
        self.baseline = Baseline()

        self.state = State.CALIBRATING
        self._pending = 0
        self._pending_state: Optional[State] = None
        self._last_reading: Optional[Reading] = None

    # -- calibration ------------------------------------------------------

    def reset_calibration(self) -> None:
        """Discard the baseline and re-learn it. Room must be empty and still."""
        self._calib_disp.clear()
        self._calib_count = 0
        self.baseline = Baseline()
        self.state = State.CALIBRATING
        self._pending = 0
        self._pending_state = None

    def set_thresholds(self, enter_z=None, exit_z=None, debounce=None) -> dict:
        """Live-adjust the detection thresholds. Safe to call mid-stream.

        Only the decision thresholds are adjustable at runtime, deliberately.
        They are read fresh on every window (see `_target_state`), so changing
        them takes effect immediately without touching the buffers or the
        learned baseline. Window length and top-K are NOT adjustable live -- they
        resize buffers and would discard the calibration -- so they remain
        launch-time flags, and the ablation study covers their effect instead.

        Values are validated and the hysteresis invariant (enter > exit) is
        enforced, so no combination the UI can send leaves the detector in an
        inconsistent state.
        """
        if enter_z is not None:
            self.enter_z = float(np.clip(enter_z, 0.5, 8.0))
        if exit_z is not None:
            self.exit_z = float(np.clip(exit_z, 0.2, 7.5))
        if debounce is not None:
            self.debounce = int(np.clip(debounce, 1, 10))
        # Enforce enter > exit; if the user crosses them, nudge exit below enter.
        if self.exit_z >= self.enter_z:
            self.exit_z = max(0.2, self.enter_z - 0.5)
        return {"enter_z": self.enter_z, "exit_z": self.exit_z, "debounce": self.debounce}

    @property
    def calibration_progress(self) -> float:
        if self.baseline.complete:
            return 1.0
        return min(1.0, self._calib_count / max(1, self.calibration_n))

    def _finish_calibration(self) -> None:
        disp = np.asarray(self._calib_disp, dtype=float)
        if disp.size < 2:
            return

        mean = float(np.mean(disp))
        sigma = float(np.std(disp))
        # A perfectly flat ambient period yields sigma 0, which would make every
        # z-score infinite. Floor it at a small fraction of the mean so the
        # detector stays numerically sane in a very quiet environment.
        if sigma <= 1e-9:
            sigma = max(mean * 0.05, 1e-6)

        self.baseline = Baseline(
            mean_std=mean,
            sigma_std=sigma,
            samples=int(disp.size),
            captured_at=time.time(),
            complete=True,
        )
        self.state = State.IDLE

    # -- main entry point -------------------------------------------------

    def _respiration_matrix(self) -> Optional[np.ndarray]:
        """(time x subcarrier) matrix over the respiration window, or None."""
        if len(self.vector_buffer) < 8:
            return None
        width = min(v.size for v in self.vector_buffer)
        if width == 0:
            return None
        return np.stack([v[:width] for v in self.vector_buffer])

    def _estimate_respiration_cached(self) -> RespiratoryEstimate:
        """Throttled wrapper: recompute roughly once a second, else reuse."""
        self._resp_countdown -= 1
        if self._resp_cached is not None and self._resp_countdown > 0:
            return self._resp_cached
        self._resp_countdown = self._resp_interval
        self._resp_cached = self._estimate_respiration()
        return self._resp_cached

    def _estimate_respiration(self) -> RespiratoryEstimate:
        """Estimate breathing rate by consensus across the top-K subcarriers.

        Averaging all subcarriers dilutes the respiratory component, because
        most of them carry none. But relying on the single highest-variance
        subcarrier is fragile in the opposite way: one subcarrier sitting near
        a band edge, or one whose spectral peak lands on a subharmonic, drags
        the whole estimate with it. Measured on a two-node capture, single-
        subcarrier selection put node 1 at 18.7 br/min and node 2 at 9.3 --
        almost exactly a factor of two, the signature of a subharmonic pick.

        So: estimate independently from each of the K highest-variance
        subcarriers and take the median. A single bad subcarrier is then
        outvoted rather than decisive.

        The spread across those independent estimates is itself the honest
        confidence measure -- when subcarriers observing the same chest through
        different multipath agree, the estimate means something; when they
        scatter, it does not.
        """
        mat = self._respiration_matrix()
        if mat is None:
            return respiratory_estimate(
                np.asarray(self.resp_buffer, dtype=float), self.fs
            )

        variances = mat.var(axis=0)
        # Null/guard subcarriers carry no channel estimate at all; excluding
        # them keeps the ranking from being polluted by structural zeros.
        live = variances > (variances.max() * 1e-3)
        if not np.any(live):
            return RespiratoryEstimate(reason="no live subcarriers")

        idx = np.argsort(np.where(live, variances, -np.inf))[::-1]
        k = int(min(self.respiration_top_k, np.count_nonzero(live)))
        chosen = [int(i) for i in idx[:k]]
        self.selected_subcarrier = chosen[0] if chosen else None
        self.candidate_subcarriers = chosen

        estimates, confidences = [], []
        for sc in chosen:
            est = respiratory_estimate(mat[:, sc].astype(float), self.fs)
            if est.supported and est.bpm > 0:
                estimates.append(est.bpm)
                confidences.append(est.confidence)

        if not estimates:
            return RespiratoryEstimate(reason="no subcarrier yielded a peak")

        arr = np.asarray(estimates, dtype=float)
        median = float(np.median(arr))

        # Agreement: fraction of the chosen subcarriers landing within 15% of
        # the median. This is the number that should be believed or doubted,
        # not the band-power ratio alone.
        if median > 0:
            agree = float(np.mean(np.abs(arr - median) / median <= 0.15))
        else:
            agree = 0.0

        self.respiration_spread = float(arr.std())
        self.respiration_agreement = agree

        return RespiratoryEstimate(
            bpm=median,
            confidence=agree,
            supported=True,
            reason=f"median of {arr.size} subcarriers, {agree*100:.0f}% agree",
        )

    def update(
        self, value: float, rssi_dbm: float, vector: Optional[np.ndarray] = None
    ) -> Reading:
        """Feed one sample and get the current reading back.

        `vector` carries per-subcarrier amplitudes when the source provides
        them. Its presence is what unlocks respiration estimation.
        """
        self.buffer.append(float(value))
        self.resp_buffer.append(float(value))
        # The dashboard labels this trace "mean subcarrier amplitude" for a CSI
        # source and "dBm" for an RSSI one, so it has to be given the quantity
        # it names. Appending rssi_dbm unconditionally made the CSI chart plot
        # the recording's RSSI field instead -- which in these captures is a
        # placeholder constant, so the trace was a flat line under a label
        # promising something else. `value` already IS the mean subcarrier
        # amplitude for a CSI source, and equals rssi_dbm for an RSSI one.
        self.level_history.append(
            float(value) if self.source_kind == "csi" else float(rssi_dbm)
        )
        if vector is not None and vector.size:
            self.vector_buffer.append(np.asarray(vector, dtype=float))

        window = np.asarray(self.buffer, dtype=float)
        feats = extract_features(window, self.fs)

        # Respiration is attempted only for CSI sources, and only over the long
        # buffer. For RSSI we return an explicit refusal rather than a number
        # nobody could defend.
        if self.source_kind == "csi":
            resp = self._estimate_respiration_cached()
        else:
            resp = RespiratoryEstimate(
                supported=False,
                reason="RSSI source: chest displacement perturbs subcarrier phase, "
                "which link-quality percentage does not expose",
            )

        # Still filling the window -- nothing meaningful to report yet.
        if len(self.buffer) < self.window_n:
            reading = Reading(
                timestamp=time.time(),
                state=State.CALIBRATING,
                z_score=0.0,
                confidence=0.0,
                features=feats,
                respiration=resp,
                rssi_dbm=rssi_dbm,
                calibration_progress=self.calibration_progress,
                note="filling analysis window",
            )
            self._last_reading = reading
            return reading

        dispersion = feats.std

        if not self.baseline.complete:
            self._calib_disp.append(dispersion)
            self._calib_count += 1
            if self._calib_count >= self.calibration_n:
                self._finish_calibration()

            reading = Reading(
                timestamp=time.time(),
                state=self.state,
                z_score=0.0,
                confidence=0.0,
                features=feats,
                respiration=resp,
                rssi_dbm=rssi_dbm,
                calibration_progress=self.calibration_progress,
                note="learning ambient baseline - keep the area still",
            )
            self._last_reading = reading
            return reading

        z = (dispersion - self.baseline.mean_std) / self.baseline.sigma_std
        self.history.append(z)

        target = self._target_state(z)
        self._apply_debounce(target)

        # Confidence saturates at the enter threshold: once well past it there
        # is no more information to convey, so it is capped rather than growing
        # without bound.
        confidence = float(np.clip(z / self.enter_z, 0.0, 1.0)) if z > 0 else 0.0

        reading = Reading(
            timestamp=time.time(),
            state=self.state,
            z_score=float(z),
            confidence=confidence,
            features=feats,
            respiration=resp,
            rssi_dbm=rssi_dbm,
            calibration_progress=1.0,
            note="",
        )
        self._last_reading = reading
        return reading

    def _target_state(self, z: float) -> State:
        """Where the z-score says we should be, before debouncing."""
        if self.state == State.MOTION:
            # Already in motion: stay until the signal falls below the LOW
            # threshold. This gap is the hysteresis.
            return State.MOTION if z > self.exit_z else State.IDLE
        return State.MOTION if z > self.enter_z else State.IDLE

    def _apply_debounce(self, target: State) -> None:
        """Only change state after `debounce` consecutive agreeing windows."""
        if target == self.state:
            self._pending = 0
            self._pending_state = None
            return

        if target == self._pending_state:
            self._pending += 1
        else:
            self._pending_state = target
            self._pending = 1

        if self._pending >= self.debounce:
            self.state = target
            self._pending = 0
            self._pending_state = None

    # -- reporting --------------------------------------------------------

    def capabilities(self) -> dict:
        """What this configuration can honestly measure.

        Three states, not two, because "the physics permits it" and "I have
        evidence it works" are different claims and conflating them is how a
        display ends up overstating its own results:

          "validated"   -- resolvable at this sample rate AND supported by
                           evidence in this project.
          "unvalidated" -- physically resolvable, but the evidence does not
                           support it. A number can be computed; it should not
                           be presented as a measurement.
          "unavailable" -- not resolvable, or no model exists to produce it.

        Respiration is deliberately `unvalidated` rather than available. The
        respiratory band is resolvable at 20 Hz and the pipeline does produce a
        rate, but the ablation study's segment test found cross-node agreement
        negative in 4 of 5 disjoint segments -- a breath-by-breath measurement
        would survive segmentation and this does not. Reporting it as a working
        readout would contradict the project's own findings, so the UI labels it
        as computed-but-unvalidated and the reason travels with it.
        """
        csi = self.source_kind == "csi"
        resp_resolvable = csi and nyquist_ok(self.fs, 0.5)
        card_resolvable = csi and nyquist_ok(self.fs, CARDIAC_BAND_HZ[1])

        return {
            # Booleans retained: they answer "can a value be produced at all",
            # which is what the estimator gating uses.
            "motion": True,
            "presence": True,
            "respiration": resp_resolvable,
            "cardiac": card_resolvable,
            "pose": False,  # needs trained keypoint weights; none are loaded
            "sample_rate_hz": self.fs,
            "source_kind": self.source_kind,
            # Evidence state, for display.
            "status": {
                "motion": "validated",
                "presence": "validated",
                "respiration": "unvalidated" if resp_resolvable else "unavailable",
                "cardiac": "unvalidated" if card_resolvable else "unavailable",
                "pose": "unavailable",
            },
            "why": {
                "motion": "dispersion vs learned ambient baseline",
                "presence": "derived from motion evidence",
                "respiration": (
                    "band is resolvable, but cross-node agreement was negative in "
                    "4 of 5 segments — not demonstrated"
                    if resp_resolvable
                    else "needs CSI phase and fs > 1 Hz"
                ),
                "cardiac": (
                    "band is resolvable, but never evaluated against any reference"
                    if card_resolvable
                    else "needs CSI phase and fs > 4 Hz"
                ),
                "pose": "no trained keypoint weights loaded",
            },
        }

    def snapshot(self) -> dict:
        r = self._last_reading
        return {
            "reading": r.as_dict() if r else None,
            "baseline": self.baseline.as_dict(),
            "capabilities": self.capabilities(),
            "z_history": [round(v, 3) for v in self.history],
            "level_history": [round(v, 2) for v in self.level_history],
            "thresholds": {"enter_z": self.enter_z, "exit_z": self.exit_z,
                           "debounce": self.debounce},
            "selected_subcarrier": self.selected_subcarrier,
            "candidate_subcarriers": self.candidate_subcarriers,
            "respiration_spread": round(self.respiration_spread, 3),
            "respiration_agreement": round(self.respiration_agreement, 3),
            "windows": {
                "motion_n": self.window_n,
                "motion_s": round(self.window_n / self.fs, 1),
                "respiration_n": self.respiration_n,
                "respiration_s": round(self.respiration_n / self.fs, 1),
                "respiration_filled": len(self.resp_buffer),
            },
        }
