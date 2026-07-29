# WiFi Sensing — device-free presence and motion detection

A signal-processing pipeline that detects human presence and motion from WiFi
Channel State Information (CSI), with a live dashboard and an offline analysis
tool that produces publication-ready figures.

The design principle throughout is that **the system reports only what its
current hardware and sample rate can physically support**. Where a quantity
cannot be measured, it says so instead of producing a number.

---

## What this is

WiFi radios estimate the wireless channel on every packet they receive. That
estimate — Channel State Information — is per-subcarrier amplitude and phase,
and it changes when a human body moves through the signal path. This project
turns that side effect into a sensor: no cameras, no wearables, no line of
sight required.

```
source  ──►  DSP  ──►  detector  ──►  dashboard / report
  │           │           │
  │           │           └─ ambient calibration, z-score, hysteresis, debounce
  │           └─ Hampel filter, detrend, Butterworth band-pass, Welch PSD
  └─ live RSSI  ·  recorded CSI  ·  ESP32-S3 over UDP
```

Sources are interchangeable behind one interface, so moving from replayed
capture to live hardware is a command-line flag rather than a rewrite.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib
```

**Live dashboard on recorded CSI** (no hardware required):

```bash
python run_live.py --source recorded --file path/to/capture.csi.jsonl --node 1 --loop
```

Then open <http://127.0.0.1:8000>. The server binds to loopback only.

**Live from this machine's WiFi adapter** (see the RSSI finding below before
relying on this):

```bash
python run_live.py --source netsh
```

**Live from ESP32-S3 nodes over UDP:**

```bash
python run_live.py --source esp32 --port-udp 5005
```

**Offline analysis → figures and tables:**

```bash
python analyze_csi.py --file path/to/capture.csi.jsonl --out output
```

---

## How detection works

There is no trained model here, and the project does not pretend otherwise.
Detection is a explainable statistical test:

1. **Calibrate.** With the area empty and still, record the mean and standard
   deviation of windowed signal dispersion. This is the ambient baseline.
2. **Score.** For each new window, compute a z-score of dispersion against that
   baseline.
3. **Hysteresis.** Enter MOTION above 3σ, leave it only below 1.5σ. Two
   thresholds rather than one stops the state chattering at the boundary.
4. **Debounce.** Require consecutive agreeing windows before the state flips,
   which suppresses single-window spikes.

Calibration quality dominates everything: a baseline captured while someone was
moving raises the bar so far that real motion never crosses it.

### Window sizing

Motion and respiration use **separate buffers**, because they need different
window lengths and one window would compromise both:

| Phenomenon | Window | Why |
|---|---|---|
| Motion | 4 s | Short, to stay responsive |
| Respiration | 30 s | Frequency resolution is `fs/N`; 0.1 Hz needs one full cycle minimum |

### Subcarrier selection

Respiration modulates a handful of subcarriers strongly and leaves the rest
flat, so averaging all 64 dilutes the signal into noise. But relying on the
single highest-variance subcarrier is fragile in the opposite direction — one
subcarrier landing on a spectral subharmonic drags the whole estimate with it.

Measured on a two-node capture, single-subcarrier selection put node 1 at
**18.7 br/min** and node 2 at **9.3** — almost exactly a factor of two, the
signature of a subharmonic pick.

The pipeline therefore estimates independently from each of the K
highest-variance subcarriers and takes the **median**, so one bad subcarrier is
outvoted rather than decisive. The **spread across those estimates is the
confidence measure**: subcarriers observing the same chest through different
multipath either agree or they don't.

---

## Findings

### 1. Commodity RSSI is unusable for sensing on this adapter

Live sensing was attempted using the host WiFi adapter (MediaTek MT7922). RSSI
proved completely static:

| Method | Samples | Duration | Distinct values |
|---|---|---|---|
| `netsh wlan show interfaces` | 99 | 30 s | **1** |
| WLAN API `wlan_intf_opcode_rssi` via ctypes | 236 | 12 s | **1** |

The second measurement queries the driver directly, bypassing `netsh`, and
still returns a frozen value — so this is **driver-level smoothing**, not a
tooling artefact. There is no software workaround; the fluctuation sensing
depends on never leaves the chip.

This motivates dedicated CSI-capable hardware (ESP32-S3), and the `netsh`
source is retained only to document the negative result.

### 2. Plausible vital signs can be extracted from pure noise

The respiratory band 0.1–0.5 Hz maps exactly onto 6–30 breaths/min. **Any**
spectral peak in that band therefore yields a physiologically plausible
number — so obtaining "14 br/min" demonstrates nothing on its own. Noise
produces that too.

During development, a frame-handling bug produced apparent agreement of
**0.04 br/min** between two independent nodes. The cause was concatenating CSI
frames of different geometries (64/128/192 values = 1/2/3 antennas), whose
amplitude statistics differ substantially — on node 1, subcarrier 56 has
SD 4.53 in 64-value frames but SD 0.97 in 128-value frames. Mixing them
injected variance from population switching rather than from the room, and both
nodes were reading the same artefact.

With frame geometries separated, the nodes **anti-correlate** (r = −0.157).

**Conclusion: plausibility is not validation. Agreement between independent
observers is.** The pipeline reports respiration only where cross-node
agreement supports it, and reports the agreement fraction alongside every
estimate.

---

## Capability gating

The dashboard shows what is measurable at the current source and sample rate,
and refuses the rest:

| Capability | RSSI @ 4 Hz | CSI @ 20 Hz |
|---|---|---|
| Motion detection | ✓ | ✓ |
| Presence (coarse) | ✓ | ✓ |
| Respiration | ✕ — magnitude does not expose chest phase | ✓ subject to agreement |
| Heart rate | ✕ — needs CSI phase and fs > 4 Hz | band resolvable; not validated |
| Pose / skeleton | ✕ | ✕ — no trained keypoint weights |

Pose is never drawn. Rendering a skeleton without trained weights would be
showing the viewer a fabrication.

---

## Limitations

- **No ground truth.** No reference respiration sensor was recorded alongside
  the CSI, so accuracy cannot be stated. Cross-node agreement is a consistency
  check, not validation against truth.
- **Presence is a threshold, not a classifier.** It responds to any channel
  disturbance, including non-human sources such as fans or microwaves.
- **Pose estimation is not implemented.**
- **The ESP32 UDP source is written but unverified** against real silicon. It
  parses the documented ADR-018 frame format and is marked `verified = False`
  until frames from real hardware confirm it.

---

## Data

The CSI recordings used for development and evaluation were **not captured by
this project**. They come from the [RuView](https://github.com/ruvnet/RuView)
open-source project (MIT licence), which published captures from an ESP32-S3
mesh. This project implements its own independent processing pipeline and uses
those recordings as an evaluation dataset, in the same way any public benchmark
dataset would be used.

Recordings are excluded from version control by `.gitignore` due to size; see
the RuView repository under `data/recordings/`.

The ADR-018 wire format implemented in `wifisense/sources.py` is likewise
documented by that project.

---

## Layout

```
wifisense/
  dsp.py         Hampel filter, detrend, Butterworth band-pass, Welch PSD, Nyquist guards
  sources.py     NetshRSSISource · RecordedCSISource · Esp32UdpSource
  detector.py    calibration, z-score, hysteresis, debounce, capability gating
  server.py      stdlib-only HTTP + JSON API
web/             dashboard (no external libraries — nothing to load, nothing to break offline)
run_live.py      live dashboard entry point
analyze_csi.py   offline analysis → figures, CSVs, RESULTS.md
```

No runtime dependencies beyond NumPy, SciPy and Matplotlib. The web server is
standard library only, and the dashboard loads no external assets, so it runs
with no network connection.
