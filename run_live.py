"""Launch the live sensing dashboard.

Examples
--------
Live RSSI from this laptop's WiFi adapter (no extra hardware):

    python run_live.py

Replay real CSI captured from a 2-node ESP32-S3 mesh:

    python run_live.py --source recorded \
        --file ../RuView/data/recordings/pretrain-1775182186.csi.jsonl

Live CSI from ESP32-S3 nodes once hardware is connected:

    python run_live.py --source esp32 --port-udp 5005

Calibration matters more than any parameter here: the first `--calibration`
seconds must be captured with the area empty and still, because everything
afterwards is measured against that baseline.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wifisense.detector import MotionDetector
from wifisense.server import SensingService, serve
from wifisense.sources import Esp32UdpSource, NetshRSSISource, RecordedCSISource


def build_source(args):
    if args.source == "netsh":
        src = NetshRSSISource()
        rate = args.rate or src.nominal_rate_hz
        return src, rate

    if args.source == "recorded":
        if not args.file:
            raise SystemExit("--file is required for --source recorded")
        path = Path(args.file)
        if not path.is_file():
            raise SystemExit(f"recording not found: {path}")
        src = RecordedCSISource(
            path, node_id=args.node, realtime=not args.fast, loop=args.loop
        )
        # The capture metadata reports ~20 Hz per node; allow an override for
        # files recorded at a different rate.
        rate = args.rate or 20.0
        src.nominal_rate_hz = rate
        return src, rate

    if args.source == "esp32":
        src = Esp32UdpSource(port=args.port_udp)
        rate = args.rate or src.nominal_rate_hz
        return src, rate

    raise SystemExit(f"unknown source: {args.source}")


def main():
    p = argparse.ArgumentParser(
        description="WiFi sensing dashboard — presence and motion from commodity WiFi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=["netsh", "recorded", "esp32"],
        default="netsh",
        help="data source (default: netsh — this laptop's WiFi, no hardware needed)",
    )
    p.add_argument("--file", help="path to a .csi.jsonl recording (for --source recorded)")
    p.add_argument("--node", type=int, help="filter a recording to one node id")
    p.add_argument("--fast", action="store_true", help="replay a recording as fast as possible")
    p.add_argument("--loop", action="store_true",
                   help="restart the recording when it ends (keeps a demo running)")
    p.add_argument("--resp-window", type=float, default=30.0,
                   help="respiration analysis window, seconds (needs >= 10 s)")
    p.add_argument("--port-udp", type=int, default=5005, help="UDP port for ESP32 CSI")
    p.add_argument("--rate", type=float, help="override sample rate in Hz")

    p.add_argument("--window", type=float, default=4.0, help="analysis window, seconds")
    p.add_argument("--calibration", type=float, default=15.0, help="ambient calibration, seconds")
    p.add_argument("--enter-z", type=float, default=3.0, help="motion enter threshold, sigmas")
    p.add_argument("--exit-z", type=float, default=1.5, help="motion exit threshold, sigmas")
    p.add_argument("--debounce", type=int, default=2, help="consecutive windows before state flips")

    p.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    p.add_argument("--port", type=int, default=8000, help="dashboard port")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = p.parse_args()

    source, rate = build_source(args)

    detector = MotionDetector(
        fs=rate,
        window_s=args.window,
        calibration_s=args.calibration,
        enter_z=args.enter_z,
        exit_z=args.exit_z,
        debounce=args.debounce,
        source_kind=source.kind,
        respiration_window_s=args.resp_window,
    )

    service = SensingService(source, detector, poll_interval=1.0 / rate)

    # The Windows console defaults to cp1252, which cannot encode Greek letters
    # or em dashes. Console output is kept ASCII-only; the dashboard renders the
    # proper symbols where UTF-8 is guaranteed.
    print("=" * 66)
    print("  WiFi Sensing - live dashboard")
    print("=" * 66)
    print(f"  source        : {source.name} ({source.kind})")
    if not source.verified:
        print("                  ** not yet verified against real hardware **")
    print(f"  sample rate   : {rate:.1f} Hz")
    print(f"  window        : {args.window:.1f} s  ({detector.window_n} samples)")
    print(f"  calibration   : {args.calibration:.1f} s  ({detector.calibration_n} samples)")
    print(f"  thresholds    : enter {args.enter_z} / exit {args.exit_z} sigma, "
          f"debounce {args.debounce}")
    print(f"  dashboard     : http://{args.host}:{args.port}")
    print("=" * 66)
    print("  Keep the area EMPTY AND STILL during calibration.")
    print("  Ctrl+C to stop.")
    print()

    service.start()
    httpd = serve(service, host=args.host, port=args.port)

    if not args.no_browser:
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")
        ).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        service.stop()
        httpd.shutdown()
        print(f"samples processed: {service.stats.samples}  dropped: {service.stats.dropped}")


if __name__ == "__main__":
    main()
