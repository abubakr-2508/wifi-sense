"""WiFi-based human sensing -- presence and motion from commodity hardware.

A compact, source-agnostic sensing pipeline:

    sources  ->  dsp  ->  detector  ->  server/dashboard

Live RSSI from this laptop needs no extra hardware; recorded CSI replays real
captures from an ESP32-S3 mesh; the ESP32 UDP source is wired and waiting for a
board. Capability gating is enforced in the detector, so the system reports
only what its current source and sample rate can physically support.
"""

__version__ = "0.1.0"

from .detector import MotionDetector, Reading, State
from .sources import (
    Esp32UdpSource,
    NetshRSSISource,
    RecordedCSISource,
    Sample,
    Source,
)

__all__ = [
    "MotionDetector",
    "Reading",
    "State",
    "Sample",
    "Source",
    "NetshRSSISource",
    "RecordedCSISource",
    "Esp32UdpSource",
]
