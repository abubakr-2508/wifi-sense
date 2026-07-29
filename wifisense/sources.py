"""Data sources for WiFi sensing.

Every source yields the same `Sample` type, so the detector and dashboard never
learn where the data came from. Three are implemented:

    NetshRSSISource   Live, this laptop, no extra hardware. Coarse RSSI only.
    RecordedCSISource Real CSI captured from a 2-node ESP32 mesh, replayed.
    Esp32UdpSource    Live CSI from ESP32-S3 nodes over UDP (ADR-018 format).

The third is written and ready but untested against real silicon -- there is no
board on this desk yet. It is included so that adding hardware is a change of
one line in `run_live.py`, not a rewrite. Its `verified` flag is False and the
dashboard surfaces that, so nothing claims hardware validation it does not have.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .dsp import quality_to_dbm

# ---------------------------------------------------------------------------
# Common sample type
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One observation from any source.

    `value` is what the detector consumes. For RSSI it is the dBm level; for
    CSI it is the mean subcarrier amplitude, which plays the same role of "how
    strong is the channel right now".

    `vector` carries the full per-subcarrier amplitudes when the source has
    them. RSSI sources leave it None, and that absence is exactly what limits
    RSSI to motion detection.
    """

    timestamp: float
    value: float
    rssi_dbm: float
    vector: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


class Source:
    """Interface every source implements."""

    name = "base"
    kind = "unknown"          # "rssi" or "csi" -- drives capability gating
    nominal_rate_hz = 1.0
    verified = False          # True only for sources exercised on real data

    def open(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def read(self) -> Optional[Sample]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "rate_hz": self.nominal_rate_hz,
            "verified": self.verified,
        }


# ---------------------------------------------------------------------------
# Live RSSI from Windows
# ---------------------------------------------------------------------------

_NETSH = r"C:\Windows\System32\netsh.exe"

# `netsh wlan show interfaces` prints "Key : Value" pairs, indented. Matching
# on the key name rather than line position keeps this robust against the
# extra fields different driver versions emit.
_FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()._-]*?)\s*:\s*(.+?)\s*$")


class NetshRSSISource(Source):
    """Poll the Windows WLAN stack for link quality.

    Measured cost of one `netsh wlan show interfaces` call on this machine is
    ~120 ms, so sustained polling above ~6 Hz is not achievable. We default to
    4 Hz, which leaves headroom and still resolves the motion band.

    What this source can and cannot support:
      * Motion / presence -- yes. A body moving through the path changes
        multipath, and that shows up as increased variance in link quality.
      * Breathing or heart rate -- no. Chest displacement is millimetric and
        perturbs subcarrier *phase*; aggregate signal strength barely moves,
        and the 0.5%-quantised quality value would not capture it if it did.
    """

    name = "netsh-rssi"
    kind = "rssi"
    nominal_rate_hz = 4.0
    verified = True  # exercised against this machine's adapter

    def __init__(self, netsh_path: str = _NETSH):
        self.netsh_path = netsh_path
        self._last_meta: dict = {}

    def _run(self) -> str:
        try:
            proc = subprocess.run(
                [self.netsh_path, "wlan", "show", "interfaces"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return proc.stdout or ""
        except (subprocess.SubprocessError, OSError):
            return ""

    @staticmethod
    def parse(output: str) -> Optional[dict]:
        """Pull the fields we care about out of netsh's key/value block."""
        fields: dict = {}
        for line in output.splitlines():
            m = _FIELD_RE.match(line)
            if m:
                fields[m.group(1).strip().lower()] = m.group(2).strip()

        if fields.get("state", "").lower() != "connected":
            return None

        signal = fields.get("signal", "")
        m = re.match(r"(\d+)\s*%", signal)
        if not m:
            return None

        quality = float(m.group(1))
        return {
            "quality_pct": quality,
            "rssi_dbm": quality_to_dbm(quality),
            "ssid": fields.get("ssid", "?"),
            "bssid": fields.get("ap bssid", fields.get("bssid", "?")),
            "band": fields.get("band", "?"),
            "channel": fields.get("channel", "?"),
            "radio": fields.get("radio type", "?"),
            "rx_mbps": fields.get("receive rate (mbps)", "?"),
        }

    def read(self) -> Optional[Sample]:
        parsed = self.parse(self._run())
        if parsed is None:
            return None
        self._last_meta = parsed
        # RSSI is both the raw level and the detector's input signal.
        return Sample(
            timestamp=time.time(),
            value=parsed["rssi_dbm"],
            rssi_dbm=parsed["rssi_dbm"],
            vector=None,
            meta=parsed,
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            {
                "ssid": self._last_meta.get("ssid", "-"),
                "band": self._last_meta.get("band", "-"),
                "channel": self._last_meta.get("channel", "-"),
                "radio": self._last_meta.get("radio", "-"),
            }
        )
        return d


# ---------------------------------------------------------------------------
# Recorded CSI replay
# ---------------------------------------------------------------------------


def decode_iq_hex(iq_hex: str) -> np.ndarray:
    """Decode an ADR-018 I/Q hex payload into per-subcarrier amplitudes.

    The payload is a flat run of signed 8-bit pairs, one (I, Q) per subcarrier.
    Amplitude is sqrt(I^2 + Q^2). Phase (atan2) is available from the same
    pairs, which is precisely the quantity RSSI cannot provide.
    """
    raw = bytes.fromhex(iq_hex)
    if len(raw) < 2:
        return np.zeros(0, dtype=float)

    pairs = np.frombuffer(raw[: (len(raw) // 2) * 2], dtype=np.int8).reshape(-1, 2)
    i = pairs[:, 0].astype(float)
    q = pairs[:, 1].astype(float)
    return np.sqrt(i * i + q * q)


def decode_iq_hex_phase(iq_hex: str) -> np.ndarray:
    """Decode the same payload into per-subcarrier phase, in radians."""
    raw = bytes.fromhex(iq_hex)
    if len(raw) < 2:
        return np.zeros(0, dtype=float)

    pairs = np.frombuffer(raw[: (len(raw) // 2) * 2], dtype=np.int8).reshape(-1, 2)
    return np.arctan2(pairs[:, 1].astype(float), pairs[:, 0].astype(float))


class RecordedCSISource(Source):
    """Replay a `.csi.jsonl` capture as if it were arriving live.

    This is real measured data -- captured from a 2-node ESP32-S3 mesh -- not
    synthetic. Replaying it exercises exactly the same detector path that live
    hardware would drive, which is what makes the offline results meaningful.
    """

    name = "recorded-csi"
    kind = "csi"
    verified = True

    def __init__(
        self,
        path: Path,
        node_id: Optional[int] = None,
        realtime: bool = True,
        loop: bool = False,
    ):
        self.path = Path(path)
        self.node_id = node_id
        self.realtime = realtime
        self.loop = loop
        self._iter: Optional[Iterator[dict]] = None
        self._t0_wall = 0.0
        self._t0_rec = 0.0
        self._count = 0
        self._laps = 0
        self._frame_len: Optional[int] = None
        self._skipped_geometry = 0
        self.nominal_rate_hz = 20.0

    def _frames(self) -> Iterator[dict]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.node_id is not None and rec.get("node_id") != self.node_id:
                    continue
                yield rec

    def open(self) -> None:
        self._iter = self._frames()
        self._t0_wall = time.time()
        self._t0_rec = 0.0
        self._count = 0

    def read(self) -> Optional[Sample]:
        if self._iter is None:
            self.open()
        assert self._iter is not None

        rec = next(self._iter, None)
        if rec is None:
            if not self.loop:
                return None
            # Restart the file and rebase the clock so playback continues
            # seamlessly -- a demo needs to keep running for as long as it is
            # being watched.
            self._laps += 1
            self._iter = self._frames()
            self._t0_wall = time.time()
            self._t0_rec = 0.0
            rec = next(self._iter, None)
            if rec is None:
                return None

        amps = decode_iq_hex(rec.get("iq_hex", ""))
        if amps.size == 0:
            return self.read()

        # Captures mix frame geometries (1/2/3 antennas). Those populations
        # have different amplitude statistics, so interleaving them injects
        # variance that a dispersion detector misreads as motion. Lock onto the
        # first geometry seen and skip the rest.
        if self._frame_len is None:
            self._frame_len = amps.size
        if amps.size != self._frame_len:
            self._skipped_geometry += 1
            return self.read()

        ts = float(rec.get("timestamp", time.time()))
        if self._t0_rec == 0.0:
            self._t0_rec = ts

        # Pace playback to the capture's own timeline so the dashboard shows
        # the signal evolving at its true rate.
        if self.realtime:
            target = self._t0_wall + (ts - self._t0_rec)
            delay = target - time.time()
            if 0 < delay < 1.0:
                time.sleep(delay)

        self._count += 1
        return Sample(
            timestamp=ts,
            value=float(np.mean(amps)),
            rssi_dbm=float(rec.get("rssi", 0)),
            vector=amps,
            meta={
                "node_id": rec.get("node_id"),
                "subcarriers": int(rec.get("subcarriers", amps.size)),
                "frame": self._count,
                "file": self.path.name,
            },
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "file": self.path.name,
            "frames_read": self._count,
            "loops": self._laps,
            "frame_geometry": self._frame_len,
            "skipped_other_geometry": self._skipped_geometry,
        })
        return d


# ---------------------------------------------------------------------------
# Live ESP32 CSI over UDP
# ---------------------------------------------------------------------------

ADR018_MAGIC = 0xC5110001
ADR018_HEADER = struct.Struct("<IBBHIIbbH")  # 20 bytes, little-endian


class Esp32UdpSource(Source):
    """Receive live CSI frames from ESP32-S3 nodes (ADR-018 wire format).

    Header layout, 20 bytes then the I/Q payload:

        0   u32  magic = 0xC5110001
        4   u8   node id
        5   u8   antenna count
        6   u16  subcarrier count
        8   u32  frequency, MHz
        12  u32  sequence number
        16  i8   RSSI
        17  i8   noise floor
        18  u16  reserved
        20  ..   I/Q pairs, one signed byte each

    NOT YET VERIFIED against real hardware -- no board has been connected. The
    parser follows the documented format and the same decode path the recorded
    captures exercise, but until frames from real silicon arrive this stays
    `verified = False` and the dashboard labels it accordingly.

    Windows note: Docker Desktop collapses multi-source UDP to a single source
    IP at the WSL boundary, silently dropping all but one node. Running this
    natively (as here) avoids that entirely.
    """

    name = "esp32-udp"
    kind = "csi"
    nominal_rate_hz = 20.0
    verified = False

    def __init__(self, port: int = 5005, bind: str = "0.0.0.0", timeout: float = 1.0):
        self.port = port
        self.bind = bind
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._count = 0
        self._nodes: set = set()

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(self.timeout)
        self._sock.bind((self.bind, self.port))

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read(self) -> Optional[Sample]:
        if self._sock is None:
            self.open()
        assert self._sock is not None

        try:
            data, _addr = self._sock.recvfrom(4096)
        except socket.timeout:
            return None

        if len(data) < ADR018_HEADER.size:
            return None

        magic, node, n_ant, n_sub, freq, seq, rssi, noise, _rsv = ADR018_HEADER.unpack(
            data[: ADR018_HEADER.size]
        )
        if magic != ADR018_MAGIC:
            return None  # not a CSI frame (could be the 0xC5110002 vitals packet)

        payload = data[ADR018_HEADER.size :]
        expected = n_ant * n_sub * 2
        if expected and len(payload) >= expected:
            payload = payload[:expected]

        pairs = np.frombuffer(
            payload[: (len(payload) // 2) * 2], dtype=np.int8
        ).reshape(-1, 2)
        if pairs.size == 0:
            return None

        amps = np.sqrt(
            pairs[:, 0].astype(float) ** 2 + pairs[:, 1].astype(float) ** 2
        )

        self._count += 1
        self._nodes.add(node)
        return Sample(
            timestamp=time.time(),
            value=float(np.mean(amps)),
            rssi_dbm=float(rssi),
            vector=amps,
            meta={
                "node_id": node,
                "antennas": n_ant,
                "subcarriers": n_sub,
                "freq_mhz": freq,
                "sequence": seq,
                "noise_floor": noise,
                "frame": self._count,
            },
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            {
                "port": self.port,
                "frames_read": self._count,
                "nodes_seen": sorted(self._nodes),
            }
        )
        return d
