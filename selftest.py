"""Self-test for the wifisense pipeline -- exercises the requirements directly.

Every other program in this repository measures the *data*. This one measures
the *system*: it drives each requirement the design chapter states, records what
actually came back, and decides pass or fail from that value rather than from an
assertion that hides it.

That distinction is the reason this is a plain script and not a test framework.
A framework reports "ok"; the report needs "longest recursion run 12 against a
limit of 1000". The measured value IS the result, and pass or fail is a
judgement made about it afterwards, so every check returns a number or a string
that can be read on its own.

Three outcomes, not two:

    PASS      the check ran and the measured value met the stated expectation
    FAIL      the check ran and it did not
    NOT RUN   the check could not be run here, with the reason attached

The third exists for the same reason the capability gate has three states: the
ESP32 wire format can be exercised over loopback, but "verified against real
silicon" cannot be tested without silicon, and recording that as a pass would be
the one kind of dishonesty this project has spent five studies avoiding.

Two rules govern what is written below. Nothing here re-implements the pipeline
-- every check imports the module it tests, because a check that decodes a frame
for itself is testing a copy rather than the thing. And every synthetic signal is
seeded, because these numbers are printed in a bound report and must not drift
between one run and the next.

    ../.venv/Scripts/python.exe selftest.py
    ../.venv/Scripts/python.exe selftest.py --slow      # adds the regression check

Writes output_selftest/SELFTEST.md and exits non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import http.client
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wifisense.detector import MotionDetector, State
from wifisense.dsp import nyquist_ok, respiratory_estimate
from wifisense.server import SensingService, serve
from wifisense.sources import (
    ADR018_HEADER,
    ADR018_MAGIC,
    Esp32UdpSource,
    NetshRSSISource,
    RecordedCSISource,
    Source,
)

RECORDINGS = ROOT.parent / "RuView" / "data" / "recordings"
PRETRAIN = "pretrain-1775182186.csi.jsonl"
OVERNIGHT = "overnight-1775217646.csi.jsonl"

SEED = 20260731  # fixed so the printed numbers are stable across runs


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One test case, in the shape the report's test-case tables need."""

    tc: str
    title: str
    requirement: str        # the Section 3.3 requirement, by name
    objective: str
    description: str
    inputs: str
    expected: str
    measured: list = field(default_factory=list)   # (label, value) pairs
    status: str = "PASS"
    note: str = ""

    def record(self, label: str, value) -> None:
        self.measured.append((label, value))

    def require(self, condition: bool, why: str) -> None:
        """Fail the check, keeping every measurement taken so far."""
        if not condition and self.status == "PASS":
            self.status = "FAIL"
            self.note = why

    def skip(self, why: str) -> None:
        self.status = "NOT RUN"
        self.note = why

    @property
    def actual(self) -> str:
        body = "; ".join(f"{k} {v}" for k, v in self.measured)
        if self.note:
            body = f"{body} ({self.note})" if body else self.note
        return body


CHECKS: list = []


def check(tc, title, requirement, objective, description, inputs, expected):
    """Register a test function and give it a pre-filled Check to populate."""

    def wrap(fn):
        def run():
            c = Check(tc, title, requirement, objective, description, inputs, expected)
            try:
                fn(c)
            except Exception as exc:                       # a crash is a failure
                c.status = "FAIL"
                c.note = f"{type(exc).__name__}: {exc}"
            return c

        CHECKS.append(run)
        return fn

    return wrap


def _recording(name: str):
    p = RECORDINGS / name
    return p if p.is_file() else None


def _node_payload_sizes(path: Path, node: int) -> list:
    """Payload size in subcarriers for every record of one node, in file order.

    Mirrors RecordedCSISource._frames(), which filters on node_id only -- the
    derived `vitals` and `feature` records reach the reader too and decode to a
    size of zero, which is why they show up in the discard count.
    """
    sizes = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("node_id") != node:
                continue
            sizes.append(len(rec.get("iq_hex", "")) // 4)
    return sizes


# ---------------------------------------------------------------------------
# TC-01  Source abstraction
# ---------------------------------------------------------------------------


@check(
    "TC-01",
    "Source abstraction",
    "Source abstraction",
    "Verify that all three sources present one interface and declare whether "
    "they have been verified against hardware",
    "Each source is instantiated and its declared interface inspected. The UDP "
    "source is then driven over loopback with a hand-built ADR-018 frame, and "
    "with a frame carrying the wrong magic number.",
    "Three source classes; one well-formed 20-byte ADR-018 header with a "
    "64-subcarrier payload; one frame with magic 0xC5110002",
    "All three expose read/open/close/describe and a kind of rssi or csi; the "
    "ESP32 source reports verified = False; a well-formed frame decodes to 64 "
    "subcarriers; a wrong-magic frame is rejected",
)
def tc01(c: Check) -> None:
    sources = [
        NetshRSSISource(),
        RecordedCSISource(RECORDINGS / PRETRAIN),
        Esp32UdpSource(port=0, bind="127.0.0.1", timeout=2.0),
    ]

    for s in sources:
        for method in ("read", "open", "close", "describe"):
            c.require(callable(getattr(s, method, None)),
                      f"{s.name} is missing {method}()")
        c.require(isinstance(s, Source), f"{s.name} does not implement Source")
        c.require(s.kind in ("rssi", "csi"), f"{s.name} declares kind {s.kind!r}")

    c.record("interface", f"{len(sources)}/3 conform")
    c.record("declared", ", ".join(
        f"{s.name}={s.kind}@{s.nominal_rate_hz:g}Hz verified={s.verified}"
        for s in sources))

    # verified=False is the whole point of the flag: it is what the dashboard
    # surfaces so nothing claims hardware validation it does not have.
    esp = sources[2]
    c.require(esp.verified is False, "the ESP32 source claims to be verified")
    c.require(sources[0].verified and sources[1].verified,
              "a source exercised against real data reports unverified")

    # Exercise the wire format over loopback. This tests the parser, not the
    # hardware -- see the NOT RUN line recorded at the end.
    esp.open()
    try:
        port = esp._sock.getsockname()[1]
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            rng = np.random.default_rng(SEED)
            payload = rng.integers(-100, 100, size=64 * 2, dtype=np.int8).tobytes()
            good = ADR018_HEADER.pack(
                ADR018_MAGIC, 1, 1, 64, 2437, 7, -55, -92, 0) + payload

            tx.sendto(good, ("127.0.0.1", port))
            sample = esp.read()
            c.require(sample is not None, "a well-formed ADR-018 frame was not read")
            if sample is not None:
                c.record("decoded subcarriers", int(sample.vector.size))
                c.record("node id", sample.meta.get("node_id"))
                c.require(sample.vector.size == 64,
                          f"decoded {sample.vector.size} subcarriers, expected 64")
                c.require(sample.meta.get("sequence") == 7, "sequence number lost")

            bad = ADR018_HEADER.pack(
                0xC5110002, 1, 1, 64, 2437, 8, -55, -92, 0) + payload
            tx.sendto(bad, ("127.0.0.1", port))
            c.require(esp.read() is None, "a wrong-magic frame was accepted")
            c.record("wrong magic", "rejected")
        finally:
            tx.close()
    finally:
        esp.close()

    c.record("hardware verification", "NOT RUN - no ESP32-S3 board connected")


# ---------------------------------------------------------------------------
# TC-02  Frame-geometry consistency: the pre-scan earns its keep
# ---------------------------------------------------------------------------


@check(
    "TC-02",
    "Modal geometry selection",
    "Frame-geometry consistency",
    "Verify that the loader selects the most common frame geometry rather than "
    "the first one it encounters",
    "The first decodable payload size on node 1 of the short recording is "
    "compared against the geometry the 400-frame pre-scan selects.",
    "pretrain recording, node 1",
    "The pre-scan selects 64 values, and does so in spite of the first frame "
    "carrying a different geometry",
)
def tc02(c: Check) -> None:
    path = _recording(PRETRAIN)
    if path is None:
        c.skip(f"recording not found at {RECORDINGS / PRETRAIN}")
        return

    src = RecordedCSISource(path, node_id=1)
    modal = src._detect_geometry()

    sizes = _node_payload_sizes(path, 1)
    first = next((s for s in sizes if s), None)
    counts = Counter(s for s in sizes if s)

    c.record("first frame", f"{first} values")
    c.record("pre-scan selects", f"{modal} values")
    c.record("population", f"{counts.get(modal, 0)} of {sum(counts.values())} frames")

    c.require(modal == 64, f"pre-scan selected {modal}, expected 64")
    c.require(
        first != modal,
        "first-seen and modal geometry agree here, so this check no longer "
        "demonstrates anything -- re-examine it",
    )


# ---------------------------------------------------------------------------
# TC-03  Frame-geometry consistency: live and offline select the same population
# ---------------------------------------------------------------------------


@check(
    "TC-03",
    "Pre-scan representativeness",
    "Frame-geometry consistency",
    "Verify that the geometry chosen from a 400-frame pre-scan is the same one "
    "a pass over the whole recording would choose",
    "For every node of both recordings, the pre-scan result is compared against "
    "the mode taken over all decodable frames of that node.",
    "Both recordings, nodes 1 and 2",
    "The pre-scan and the whole-file mode agree in all four cases",
)
def tc03(c: Check) -> None:
    agree, total = 0, 0
    for name in (PRETRAIN, OVERNIGHT):
        path = _recording(name)
        if path is None:
            c.skip(f"recording not found at {RECORDINGS / name}")
            return
        for node in (1, 2):
            prescan = RecordedCSISource(path, node_id=node)._detect_geometry()
            sizes = _node_payload_sizes(path, node)
            full = Counter(s for s in sizes if s).most_common(1)[0][0]
            total += 1
            agree += int(prescan == full)
            c.record(f"{name.split('-')[0]} node {node}",
                     f"pre-scan {prescan} / whole-file {full}")
    c.record("agreement", f"{agree}/{total}")
    c.require(agree == total, "the pre-scan is not representative of the file")


# ---------------------------------------------------------------------------
# TC-04  Frame-geometry consistency: recursion margin on discarded frames
# ---------------------------------------------------------------------------


@check(
    "TC-04",
    "Discarded-frame recursion margin",
    "Frame-geometry consistency",
    "Establish how close the recorded source comes to exhausting the Python "
    "call stack, given that it recurses once per discarded frame",
    "RecordedCSISource.read() calls itself for every frame it discards "
    "(sources.py:323 and :333). The longest consecutive run of discardable "
    "frames is counted for each node of each recording and compared against the "
    "interpreter's recursion limit.",
    "Both recordings, nodes 1 and 2",
    "The longest run stays well below sys.getrecursionlimit()",
)
def tc04(c: Check) -> None:
    limit = sys.getrecursionlimit()
    worst, worst_where = 0, ""
    for name in (PRETRAIN, OVERNIGHT):
        path = _recording(name)
        if path is None:
            c.skip(f"recording not found at {RECORDINGS / name}")
            return
        for node in (1, 2):
            sizes = _node_payload_sizes(path, node)
            counts, seen = {}, 0
            for s in sizes:                      # mirror the 400-frame pre-scan
                if s:
                    counts[s] = counts.get(s, 0) + 1
                    seen += 1
                if seen >= 400:
                    break
            modal = max(counts, key=counts.get) if counts else None

            run = best = 0
            discarded = 0
            for s in sizes:
                if s == 0 or s != modal:
                    run += 1
                    discarded += 1
                    best = max(best, run)
                else:
                    run = 0
            if best > worst:
                worst, worst_where = best, f"{name.split('-')[0]} node {node}"
            c.record(f"{name.split('-')[0]} node {node}",
                     f"{len(sizes) - discarded} kept / {discarded} discarded, "
                     f"longest run {best}")

    c.record("worst case", f"{worst} ({worst_where})")
    c.record("recursion limit", limit)
    c.record("margin", f"{limit / worst:.0f}x" if worst else "no recursion")
    c.require(worst < limit // 2,
              f"longest run {worst} is within half the recursion limit {limit}")


# ---------------------------------------------------------------------------
# TC-05  Ambient calibration
# ---------------------------------------------------------------------------


@check(
    "TC-05",
    "Ambient calibration",
    "Ambient calibration",
    "Verify that the detector learns a baseline over the configured interval "
    "and reports its progress while doing so",
    "A detector is driven with seeded Gaussian samples. The state, the reported "
    "progress and the number of samples consumed before the baseline completes "
    "are recorded.",
    "20 Hz, 4 s window, 15 s calibration, seeded normal(0, 1)",
    "Progress rises monotonically to 1.0, the baseline completes after the "
    "window has filled plus the calibration interval, and no decision is "
    "reported before it does",
)
def tc05(c: Check) -> None:
    fs = 20.0
    det = MotionDetector(fs=fs, window_s=4.0, calibration_s=15.0, source_kind="rssi")
    rng = np.random.default_rng(SEED)

    progress, completed_at, states = [], None, []
    for i in range(1, 601):
        r = det.update(float(rng.normal(0.0, 1.0)), -60.0)
        progress.append(r.calibration_progress)
        states.append(r.state)
        if completed_at is None and det.baseline.complete:
            completed_at = i

    expected = det.window_n - 1 + det.calibration_n
    c.record("window / calibration samples", f"{det.window_n} / {det.calibration_n}")
    c.record("baseline complete after", f"{completed_at} samples "
                                        f"({completed_at / fs:.1f} s)")
    c.record("expected", f"{expected} samples")
    c.record("progress monotonic", all(b >= a for a, b in zip(progress, progress[1:])))
    c.record("final state", det.state.value)
    c.record("baseline", f"mean {det.baseline.mean_std:.4f}, "
                         f"sigma {det.baseline.sigma_std:.4f}")

    # The window that completes the baseline calls _finish_calibration() before
    # it builds its Reading, so it already reports idle while still carrying the
    # "learning ambient baseline" note. One window out of 379, and the state is
    # the field the interface acts on, so it is recorded rather than corrected.
    c.record("windows reporting calibrating", states.count(State.CALIBRATING))
    c.record("completing window reports", states[expected - 1].value)

    c.require(completed_at == expected,
              f"completed after {completed_at} samples, expected {expected}")
    c.require(all(b >= a for a, b in zip(progress, progress[1:])),
              "calibration progress went backwards")
    c.require(det.state is State.IDLE, f"state is {det.state}, expected idle")
    c.require(states.count(State.CALIBRATING) == expected - 1,
              "the detector left the calibrating state at the wrong window")
    c.require(State.MOTION not in states[:expected],
              "the detector reported motion before calibration finished")


# ---------------------------------------------------------------------------
# TC-06  Ambient calibration: the sigma floor
# ---------------------------------------------------------------------------


@check(
    "TC-06",
    "Calibration on a perfectly still input",
    "Ambient calibration",
    "Verify that a zero-variance ambient period cannot produce an infinite "
    "z-score",
    "A constant signal is fed through calibration, which drives the standard "
    "deviation of window dispersion to zero. The floor at detector.py:229 "
    "should keep every subsequent z-score finite.",
    "20 Hz, constant 1.0 for 600 samples",
    "sigma_std is greater than zero and the resulting z-scores are finite",
)
def tc06(c: Check) -> None:
    det = MotionDetector(fs=20.0, window_s=4.0, calibration_s=15.0, source_kind="rssi")
    for _ in range(500):
        det.update(1.0, -60.0)

    zs = [det.update(1.0, -60.0).z_score for _ in range(20)]
    c.record("sigma_std", f"{det.baseline.sigma_std:.3e}")
    c.record("z over 20 windows", f"min {min(zs):.3f}, max {max(zs):.3f}")
    c.record("all finite", bool(np.all(np.isfinite(zs))))

    c.require(det.baseline.complete, "calibration did not complete")
    c.require(det.baseline.sigma_std > 0.0, "sigma floor did not engage")
    c.require(bool(np.all(np.isfinite(zs))), "a z-score was not finite")


# ---------------------------------------------------------------------------
# TC-07  Motion decision: hysteresis
# ---------------------------------------------------------------------------


@check(
    "TC-07",
    "Hysteresis",
    "Motion decision",
    "Verify that the state entered above the upper threshold is left only below "
    "the lower one",
    "The decision rule is driven with supplied z-scores from each state, and "
    "then the whole path is exercised end to end by calibrating on quiet "
    "samples, feeding agitated ones and returning to quiet.",
    "enter 3.0, exit 1.5; z of 2.0 and 3.5 from idle and from motion; then "
    "seeded normal(0, 1) and normal(0, 4)",
    "z = 2.0 does not enter the motion state but does not leave it either; the "
    "end-to-end z rises above the enter threshold under agitation and falls "
    "back below it",
)
def tc07(c: Check) -> None:
    det = MotionDetector(fs=20.0, source_kind="rssi", enter_z=3.0, exit_z=1.5)

    rows = []
    for state, z in ((State.IDLE, 2.0), (State.IDLE, 3.5),
                     (State.MOTION, 2.0), (State.MOTION, 1.0)):
        det.state = state
        rows.append((state.value, z, det._target_state(z).value))
    c.record("decision rule", "; ".join(f"{s}@z={z} -> {t}" for s, z, t in rows))

    c.require(rows[0][2] == "idle", "z=2.0 entered the motion state")
    c.require(rows[1][2] == "motion", "z=3.5 did not enter the motion state")
    c.require(rows[2][2] == "motion", "z=2.0 left the motion state -- no hysteresis")
    c.require(rows[3][2] == "idle", "z=1.0 did not leave the motion state")

    # End to end, so the z-score itself is exercised and not only the rule.
    det = MotionDetector(fs=20.0, window_s=4.0, calibration_s=15.0,
                         source_kind="rssi", enter_z=3.0, exit_z=1.5)
    rng = np.random.default_rng(SEED)
    for _ in range(400):
        det.update(float(rng.normal(0.0, 1.0)), -60.0)
    z_quiet = det.update(float(rng.normal(0.0, 1.0)), -60.0).z_score

    for _ in range(200):
        r = det.update(float(rng.normal(0.0, 4.0)), -60.0)
    z_active, state_active = r.z_score, det.state

    for _ in range(200):
        r = det.update(float(rng.normal(0.0, 1.0)), -60.0)
    z_settled, state_settled = r.z_score, det.state

    c.record("z quiet", f"{z_quiet:.2f}")
    c.record("z agitated", f"{z_active:.2f} -> {state_active.value}")
    c.record("z settled", f"{z_settled:.2f} -> {state_settled.value}")

    c.require(z_quiet < 3.0, f"a quiet window scored {z_quiet:.2f}")
    c.require(z_active > 3.0, f"an agitated window scored only {z_active:.2f}")
    c.require(state_active is State.MOTION, "agitation did not reach the motion state")
    c.require(state_settled is State.IDLE, "the detector did not settle back to idle")


# ---------------------------------------------------------------------------
# TC-08  Motion decision: debounce
# ---------------------------------------------------------------------------


@check(
    "TC-08",
    "Debounce",
    "Motion decision",
    "Verify that a single unusual window cannot change the reported state",
    "With the baseline already learned, a single motion-target window is "
    "applied and the state inspected, then two consecutive ones.",
    "debounce = 2, starting from the idle state",
    "One window leaves the state unchanged; two consecutive windows change it",
)
def tc08(c: Check) -> None:
    det = MotionDetector(fs=20.0, source_kind="rssi", debounce=2)
    det.state = State.IDLE

    det._apply_debounce(State.MOTION)
    after_one = det.state
    det._apply_debounce(State.MOTION)
    after_two = det.state

    c.record("after 1 window", after_one.value)
    c.record("after 2 windows", after_two.value)

    # A spike that is not sustained must also reset the counter rather than
    # accumulating towards a flip across unrelated windows.
    det2 = MotionDetector(fs=20.0, source_kind="rssi", debounce=2)
    det2.state = State.IDLE
    det2._apply_debounce(State.MOTION)
    det2._apply_debounce(State.IDLE)
    det2._apply_debounce(State.MOTION)
    c.record("spike, quiet, spike", det2.state.value)

    c.require(after_one is State.IDLE, "a single window flipped the state")
    c.require(after_two is State.MOTION, "two agreeing windows did not flip the state")
    c.require(det2.state is State.IDLE, "non-consecutive spikes accumulated")


# ---------------------------------------------------------------------------
# TC-09  Runtime adjustment
# ---------------------------------------------------------------------------


@check(
    "TC-09",
    "Threshold adjustment invariant",
    "Runtime adjustment",
    "Verify that no combination the interface can send leaves the detector with "
    "an exit threshold at or above its enter threshold, and that adjusting "
    "thresholds does not discard the learned baseline",
    "Every combination of a grid of enter and exit values, including negative, "
    "crossed and far out-of-range inputs, is applied through set_thresholds and "
    "the invariant re-checked after each.",
    "enter in {-5, 0.5, 2, 3, 8, 100}; exit in {-5, 0.2, 1.5, 5, 100}; "
    "debounce in {0, 2, 50}",
    "enter > exit holds after every combination, values are clamped to their "
    "documented ranges, and the baseline is unchanged",
)
def tc09(c: Check) -> None:
    det = MotionDetector(fs=20.0, source_kind="rssi")
    rng = np.random.default_rng(SEED)
    for _ in range(400):                      # learn a baseline to protect
        det.update(1.0 + float(rng.normal()), -60.0)
    before = (det.baseline.mean_std, det.baseline.sigma_std, det.baseline.complete)

    enters = [-5.0, 0.5, 2.0, 3.0, 8.0, 100.0]
    exits = [-5.0, 0.2, 1.5, 5.0, 100.0]
    debounces = [0, 2, 50]

    held, total, violations = 0, 0, []
    for e in enters:
        for x in exits:
            applied = det.set_thresholds(enter_z=e, exit_z=x,
                                         debounce=debounces[total % len(debounces)])
            total += 1
            if applied["enter_z"] > applied["exit_z"]:
                held += 1
            else:
                violations.append((e, x, applied))
            c.require(0.5 <= applied["enter_z"] <= 8.0,
                      f"enter_z {applied['enter_z']} outside its documented range")
            c.require(1 <= applied["debounce"] <= 10,
                      f"debounce {applied['debounce']} outside its documented range")

    after = (det.baseline.mean_std, det.baseline.sigma_std, det.baseline.complete)
    c.record("combinations", total)
    c.record("invariant held", f"{held}/{total}")
    c.record("clamping", f"enter_z(100) -> {det.set_thresholds(enter_z=100)['enter_z']}, "
                         f"debounce(50) -> {det.set_thresholds(debounce=50)['debounce']}")
    c.record("baseline preserved", before == after)

    c.require(held == total, f"invariant violated in {violations[:2]}")
    c.require(before == after, "adjusting thresholds disturbed the baseline")


# ---------------------------------------------------------------------------
# TC-10  Refusal with a reason
# ---------------------------------------------------------------------------


@check(
    "TC-10",
    "Refusal with a reason",
    "Refusal with a reason",
    "Verify that every quantity the system cannot support is refused with a "
    "stated reason rather than returned as a number",
    "The respiration estimator is called in each of its three refusal "
    "conditions; an RSSI detector is asked for respiration; and the capability "
    "gate is inspected for a CSI source at 20 Hz and an RSSI source at 4 Hz.",
    "Empty array; fs = 0.8 Hz; a 5 s window at 20 Hz; source_kind rssi; both "
    "capability configurations",
    "Each refusal carries supported = False and a distinct non-empty reason; "
    "pose is unavailable in every configuration; every capability has a reason",
)
def tc10(c: Check) -> None:
    rng = np.random.default_rng(SEED)

    refusals = {
        "empty buffer": respiratory_estimate(np.zeros(0), 20.0),
        "sub-Nyquist": respiratory_estimate(rng.normal(size=400), 0.8),
        "window under one cycle": respiratory_estimate(rng.normal(size=100), 20.0),
    }
    for label, est in refusals.items():
        c.require(est.supported is False, f"{label} returned a supported estimate")
        c.require(bool(est.reason.strip()), f"{label} refused without a reason")
        c.record(label, f'"{est.reason}"')

    c.require(len({e.reason for e in refusals.values()}) == 3,
              "two refusal conditions give the same reason")
    c.require(not nyquist_ok(0.8, 0.5), "the Nyquist guard accepted 0.8 Hz")

    det = MotionDetector(fs=4.0, source_kind="rssi")
    resp = det.update(-60.0, -60.0).respiration
    c.record("rssi source", f'supported={resp.supported}, "{resp.reason[:58]}..."')
    c.require(resp.supported is False, "an RSSI source returned a respiration rate")

    csi = MotionDetector(fs=20.0, source_kind="csi").capabilities()
    rssi = MotionDetector(fs=4.0, source_kind="rssi").capabilities()
    c.record("csi @20Hz", json.dumps(csi["status"], separators=(",", ":")))
    c.record("rssi @4Hz", json.dumps(rssi["status"], separators=(",", ":")))

    for name, caps in (("csi", csi), ("rssi", rssi)):
        c.require(caps["status"]["pose"] == "unavailable",
                  f"pose is not unavailable for a {name} source")
        c.require(all(str(v).strip() for v in caps["why"].values()),
                  f"a {name} capability carries no reason")
    c.require(csi["status"]["respiration"] == "unvalidated",
              "respiration is not reported as unvalidated on a CSI source")
    c.require(len(set(csi["status"].values())) >= 3,
              "the capability gate is not using all three states")


# ---------------------------------------------------------------------------
# TC-11  Respiration estimate
# ---------------------------------------------------------------------------


@check(
    "TC-11",
    "Respiration estimate",
    "Respiration estimate",
    "Verify that the estimate is the median across the highest-variance "
    "subcarriers, that structural zeros are excluded from the ranking, and that "
    "the agreement across those subcarriers is reported",
    "A synthetic 64-subcarrier stream is built with the real null pattern held "
    "at zero, six subcarriers carrying a tone placed exactly on a spectral bin, "
    "and the rest seeded noise. The stream is fed through the detector.",
    "20 Hz, 620 samples, tone at 0.3125 Hz (18.75 br/min), nulls at index 0 and "
    "27-37, noise sigma 0.2",
    "Eight subcarriers are chosen, none of them a null, and the returned rate "
    "is within half a spectral bin of 18.75 br/min",
)
def tc11(c: Check) -> None:
    fs, n = 20.0, 620
    nulls = {0, *range(27, 38)}
    carriers = [10, 11, 12, 13, 14, 15]
    tone_hz = 4 * fs / 256          # lands exactly on a Welch bin at nperseg=256

    rng = np.random.default_rng(SEED)
    t = np.arange(n) / fs
    mat = rng.normal(0.0, 0.2, size=(n, 64)) + 20.0
    for sc in carriers:
        mat[:, sc] += 3.0 * np.sin(2 * np.pi * tone_hz * t)
    for sc in nulls:
        mat[:, sc] = 0.0

    det = MotionDetector(fs=fs, source_kind="csi", respiration_window_s=30.0)
    reading = None
    for i in range(n):
        reading = det.update(float(mat[i].mean()), 0.0, mat[i])

    chosen = det.candidate_subcarriers
    est = reading.respiration

    c.record("subcarriers chosen", f"{len(chosen)} -> {sorted(chosen)}")
    c.record("nulls among them", len(set(chosen) & nulls))
    c.record("estimate", f"{est.bpm:.2f} br/min (expected {tone_hz * 60:.2f})")
    c.record("agreement", f"{est.confidence:.2f}")
    c.record("bin width", f"{fs / 256 * 60:.2f} br/min")

    c.require(len(chosen) == 8, f"{len(chosen)} subcarriers chosen, expected 8")
    c.require(not (set(chosen) & nulls), "a structural zero entered the ranking")
    c.require(set(carriers) <= set(chosen), "a tone-carrying subcarrier was not chosen")
    c.require(est.supported, f"the estimator refused: {est.reason}")
    c.require(abs(est.bpm - tone_hz * 60) <= fs / 256 * 60 / 2 + 0.01,
              f"estimate {est.bpm:.2f} is over half a bin from {tone_hz * 60:.2f}")


# ---------------------------------------------------------------------------
# TC-12  Local interface
# ---------------------------------------------------------------------------


def _serve_on_ephemeral_port():
    """Start the real service and server on a port the OS chooses.

    Port 0 rather than 8000 so a demonstration already running on the default
    port cannot collide with the test, and vice versa.
    """
    path = _recording(PRETRAIN)
    if path is None:
        return None, None, None
    src = RecordedCSISource(path, node_id=1, realtime=False, loop=True)
    det = MotionDetector(fs=20.0, source_kind="csi")
    svc = SensingService(src, det, poll_interval=0.001)
    svc.start()
    httpd = serve(svc, host="127.0.0.1", port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return svc, httpd, httpd.server_address[1]


def _get(port, route):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post(port, route, payload=None):
    """POST, sending a body only where the endpoint documents one.

    /api/calibrate takes no body and does not read one, and a body left unread
    in the receive buffer makes the close send an RST rather than a FIN -- which
    surfaces on Windows as an aborted connection. The dashboard sends no body
    there either, so this mirrors the real client rather than working around it.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=data, headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


@check(
    "TC-12",
    "Local interface",
    "Local interface",
    "Verify that the four endpoints the browser calls respond correctly, and "
    "that the waterfall returns only the columns a client has not yet seen",
    "The real service is started on an ephemeral loopback port against the "
    "short recording. Each endpoint is called over HTTP, and the waterfall is "
    "polled twice to confirm the sequence number advances without resending.",
    "GET /api/state, GET /api/waterfall, POST /api/calibrate, "
    "POST /api/thresholds, GET /api/nonexistent",
    "All four return 200 with the documented keys, an unknown route returns "
    "404, and a second waterfall poll returns only newer columns",
)
def tc12(c: Check) -> None:
    svc, httpd, port = _serve_on_ephemeral_port()
    if svc is None:
        c.skip(f"recording not found at {RECORDINGS / PRETRAIN}")
        return
    try:
        time.sleep(1.5)                       # let the loop accumulate samples

        status, state = _get(port, "/api/state")
        c.record("GET /api/state", status)
        for key in ("reading", "baseline", "capabilities", "thresholds", "windows"):
            c.require(key in state, f"/api/state is missing {key!r}")

        status, wf1 = _get(port, "/api/waterfall?since=0")
        first = len(wf1["columns"])
        status2, wf2 = _get(port, f"/api/waterfall?since={wf1['seq']}")
        second = len(wf2["columns"])
        c.record("GET /api/waterfall", f"{status}, {first} columns from seq 0")
        c.record("second poll", f"{second} columns after seq {wf1['seq']}")
        c.require(first > 0, "the waterfall returned nothing on a fresh poll")
        c.require(second < first, "the waterfall resent columns the client held")

        status, cal = _post(port, "/api/calibrate")
        c.record("POST /api/calibrate", f"{status}, state={cal.get('state')}")
        c.require(cal.get("state") == "calibrating", "calibrate did not reset the state")

        status, th = _post(port, "/api/thresholds", {"enter_z": 4.0, "exit_z": 2.0})
        c.record("POST /api/thresholds", f"{status}, {th.get('thresholds')}")
        c.require(th["thresholds"]["enter_z"] == 4.0, "thresholds were not applied")

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/nonexistent")
        r = conn.getresponse()
        r.read()
        c.record("GET /api/nonexistent", r.status)
        c.require(r.status == 404, f"an unknown route returned {r.status}")
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        svc.stop()


# ---------------------------------------------------------------------------
# TC-13  Network exposure
# ---------------------------------------------------------------------------


@check(
    "TC-13",
    "Network exposure",
    "Network exposure",
    "Verify that the server binds to the loopback address by default and "
    "refuses any request for a file outside the web directory",
    "The documented defaults of serve() and of the launcher are inspected, and "
    "path-traversal requests are sent as raw request lines through http.client "
    "so that no client-side normalisation can hide the attempt.",
    "GET /static/style.css, GET /static/../wifisense/detector.py, "
    "GET /static/%2e%2e/wifisense/detector.py",
    "A legitimate static file is served; both traversal attempts are refused "
    "with 403 or 404 and no file content is returned",
)
def tc13(c: Check) -> None:
    import inspect

    default_host = inspect.signature(serve).parameters["host"].default
    c.record("serve() default host", default_host)
    c.require(default_host == "127.0.0.1", f"serve() defaults to {default_host}")

    launcher = (ROOT / "run_live.py").read_text(encoding="utf-8")
    c.record("launcher default", "--host 127.0.0.1"
             if '"--host", default="127.0.0.1"' in launcher else "NOT loopback")
    c.require('"--host", default="127.0.0.1"' in launcher,
              "the launcher does not default to loopback")

    svc, httpd, port = _serve_on_ephemeral_port()
    if svc is None:
        c.skip(f"recording not found at {RECORDINGS / PRETRAIN}")
        return
    try:
        for route, label in (
            ("/static/style.css", "legitimate file"),
            ("/static/../wifisense/detector.py", "traversal"),
            ("/static/%2e%2e/wifisense/detector.py", "encoded traversal"),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", route)          # sent verbatim, not normalised
            r = conn.getresponse()
            body = r.read()
            conn.close()
            c.record(label, f"{r.status} ({len(body)} bytes)")
            if label == "legitimate file":
                c.require(r.status == 200, f"a legitimate static file returned {r.status}")
            else:
                c.require(r.status in (403, 404), f"{label} returned {r.status}")
                c.require(b"MotionDetector" not in body,
                          f"{label} returned source code")
    finally:
        httpd.shutdown()
        httpd.server_close()
        svc.stop()


# ---------------------------------------------------------------------------
# TC-14  Offline operation, dependencies, throughput
# ---------------------------------------------------------------------------

# An XML namespace such as xmlns='http://www.w3.org/2000/svg' is an identifier,
# not a location -- nothing dereferences it. Excluding it explicitly is more
# honest than a scan that quietly reports zero because it never looked.
_FETCHABLE = re.compile(
    r"""(?:src|href)\s*=\s*['"]\s*(https?:|//)|@import\s+(?:url\()?['"]?\s*(https?:|//)"""
    r"""|url\(\s*['"]?\s*(https?:|//)""",
    re.IGNORECASE,
)


@check(
    "TC-14",
    "Offline operation, dependencies and throughput",
    "Offline operation · Dependencies · Throughput",
    "Verify that the interface fetches nothing from the network, that the "
    "server module imports only the standard library, and that the pipeline "
    "keeps pace with a 20 Hz stream on one core",
    "The three files the browser loads are scanned for references that would "
    "cause a fetch. Every module is parsed with ast and its imports classified "
    "against sys.stdlib_module_names. The detector is then driven with 2,000 "
    "synthetic CSI samples and the achieved rate measured.",
    "web/index.html, web/style.css, web/app.js; the five package modules; "
    "2,000 samples of 64 subcarriers at 20 Hz",
    "No fetchable external reference; server.py imports no third-party module; "
    "the achieved rate exceeds 20 Hz",
)
def tc14(c: Check) -> None:
    external = []
    for name in ("index.html", "style.css", "app.js"):
        text = (ROOT / "web" / name).read_text(encoding="utf-8")
        external += [(name, m.group(0)[:40]) for m in _FETCHABLE.finditer(text)]
    ns = sum((ROOT / "web" / n).read_text(encoding="utf-8").count("xmlns=")
             for n in ("index.html", "style.css", "app.js"))
    c.record("fetchable external references", len(external))
    c.record("xml namespaces excluded", ns)
    c.require(not external, f"external assets referenced: {external[:2]}")

    third_party = {}
    for mod in sorted((ROOT / "wifisense").glob("*.py")):
        tree = ast.parse(mod.read_text(encoding="utf-8"))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        third_party[mod.name] = sorted(found - sys.stdlib_module_names - {"__future__"})
    c.record("third-party imports", "; ".join(
        f"{k}: {v or 'none'}" for k, v in third_party.items()))
    c.require(not third_party.get("server.py"),
              f"server.py imports {third_party.get('server.py')}")

    # Throughput is wall-clock and therefore the one measurement here that moves
    # between runs. Two things make the reported figure defensible: the buffers
    # are filled first so every timed sample runs at steady state rather than
    # against a half-empty respiration window, and the SLOWEST of three
    # repetitions is reported, because a claim to keep pace with a stream is
    # only as good as its worst observation.
    fs, block_n = 20.0, 500
    rng = np.random.default_rng(SEED)
    det = MotionDetector(fs=fs, source_kind="csi", respiration_window_s=30.0)
    warm = rng.normal(20.0, 1.0, size=(det.respiration_n, 64))
    for i in range(warm.shape[0]):                       # untimed
        det.update(float(warm[i].mean()), 0.0, warm[i])

    rates = []
    for _ in range(3):
        block = rng.normal(20.0, 1.0, size=(block_n, 64))   # built before timing
        start = time.perf_counter()
        for i in range(block_n):
            det.update(float(block[i].mean()), 0.0, block[i])
        rates.append(block_n / (time.perf_counter() - start))

    achieved = min(rates)
    c.record("throughput", f"{achieved:,.0f} samples/s, slowest of 3 x {block_n} "
                           f"at steady state (range {min(rates):,.0f}-{max(rates):,.0f})")
    c.record("headroom vs 20 Hz", f"{achieved / fs:.1f}x")
    c.require(achieved > fs, f"achieved only {achieved:.1f} samples/s against {fs} Hz")


# ---------------------------------------------------------------------------
# TC-15  Regression (slow -- opt in)
# ---------------------------------------------------------------------------


@check(
    "TC-15",
    "Analysis regression",
    "Reproducibility",
    "Verify that regenerating the offline analysis reproduces the committed "
    "report and tables byte for byte",
    "analyze_csi.py is re-run over the short recording into a temporary "
    "directory and the SHA-256 of RESULTS.md and both CSV tables compared "
    "against the committed copies in output/.",
    "pretrain recording, default parameters",
    "All three files hash identically to the committed versions",
)
def tc15(c: Check) -> None:
    if "--slow" not in sys.argv:
        c.skip("not run by default; pass --slow (takes about a minute)")
        return
    path = _recording(PRETRAIN)
    if path is None:
        c.skip(f"recording not found at {RECORDINGS / PRETRAIN}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "analyze_csi.py"),
             "--file", str(path), "--out", tmp],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        c.require(proc.returncode == 0,
                  f"analyze_csi.py exited {proc.returncode}: {proc.stderr[-200:]}")
        if proc.returncode != 0:
            return

        identical = 0
        for name in ("RESULTS.md", "results_summary.csv", "results_breathing.csv"):
            committed = (ROOT / "output" / name).read_bytes()
            regenerated = (Path(tmp) / name).read_bytes()
            same = hashlib.sha256(committed).hexdigest() == \
                hashlib.sha256(regenerated).hexdigest()
            identical += int(same)
            c.record(name, "identical" if same else
                     f"DIFFERS ({len(committed)} vs {len(regenerated)} bytes)")
        c.record("reproduced", f"{identical}/3")
        c.require(identical == 3, "regeneration did not reproduce the committed output")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slow", action="store_true",
                    help="include the regression check (re-runs the analysis)")
    ap.add_argument("--out", default="output_selftest", help="output directory")
    args = ap.parse_args()

    print("=" * 78)
    print("  wifisense self-test")
    print("=" * 78)

    results = []
    for run in CHECKS:
        c = run()
        results.append(c)
        mark = {"PASS": "PASS", "FAIL": "FAIL", "NOT RUN": "----"}[c.status]
        print(f"  [{mark}] {c.tc}  {c.title}")
        for label, value in c.measured:
            print(f"           {label}: {value}")
        if c.note:
            print(f"           note: {c.note}")

    passed = sum(r.status == "PASS" for r in results)
    failed = sum(r.status == "FAIL" for r in results)
    skipped = sum(r.status == "NOT RUN" for r in results)

    print("=" * 78)
    print(f"  {passed} passed, {failed} failed, {skipped} not run"
          f"   ({len(results)} checks)")
    print("=" * 78)

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    lines = [
        "# Self-test results",
        "",
        "Generated by `selftest.py`. Every value below was measured by running "
        "the check named beside it; nothing here is asserted from inspection.",
        "",
        f"**{passed} passed, {failed} failed, {skipped} not run.**",
        "",
        "| ID | Test case | Requirement | Status |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r.tc} | {r.title} | {r.requirement} | {r.status} |" for r in results]
    lines.append("")
    for r in results:
        lines += [
            f"## {r.tc}: {r.title}",
            "",
            f"| Field | Description |",
            f"|---|---|",
            f"| Test ID | {r.tc} |",
            f"| Requirement | {r.requirement} |",
            f"| Test Objective | {r.objective} |",
            f"| Test Description | {r.description} |",
            f"| Input | {r.inputs} |",
            f"| Expected Result | {r.expected} |",
            f"| Actual Result | {r.actual} |",
            f"| Test Status | {r.status} |",
            "",
        ]
    (out / "SELFTEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out / 'SELFTEST.md'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
