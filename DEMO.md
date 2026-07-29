# Demo runbook

Everything needed to run this in front of an evaluator, including what to do
when something does not start. **Nothing here requires an internet connection**
— rehearse it with WiFi switched off at least once.

---

## Pre-demo checklist

Do this the night before, not on the day.

- [ ] Reboot the laptop, then run through this whole page from a cold start
- [ ] Confirm the Docker image is cached locally (`docker images`) — pulling it
      on the day needs internet you may not have
- [ ] Confirm the recordings exist at `../RuView/data/recordings/`
- [ ] Switch WiFi **off** and confirm both demos still start
- [ ] Charge the laptop; the CSI replay pins one core and drains battery
- [ ] Close Zoom/Teams — they grab the webcam and can fight for ports

---

## Part 1 — your system (lead with this)

This is your own code, running on real captured CSI.

```bash
cd "C:/Users/HP OMEN/OneDrive/Desktop/mtech-mini/wifi-sense"
```

```bash
../.venv/Scripts/python.exe run_live.py --source recorded --file ../RuView/data/recordings/pretrain-1775182186.csi.jsonl --node 1 --loop
```

A browser opens at <http://127.0.0.1:8000>. `--loop` restarts the recording when
it ends, so it runs for as long as you need.

**Wait for calibration to finish** (~15 s) before talking to the screen. The
progress bar reads "learning ambient baseline".

### What to point at, in order

1. **Detection state** — the ring and CLEAR/MOTION, with the z-score beneath it.
   Say: *"This is dispersion in the current window measured against a baseline
   learned from an empty room. Three sigma to enter, one-point-five to leave —
   the gap is hysteresis, so the state doesn't chatter at the boundary."*

2. **Measurable at this configuration** — the honesty panel. This is the part
   that distinguishes the project. Say: *"The system reports only what the
   source and sample rate physically support. Pose shows a cross because no
   trained keypoint weights are loaded — it draws nothing rather than drawing a
   fabrication."*

3. **Channel waterfall** — your strongest visual. Subcarrier against time, colour
   is amplitude, scrolling live. This is the standard display in CSI
   literature, and it is the raw measured channel with nothing added.

   Point at the **dark horizontal band** across the middle and say: *"Those are
   the guard subcarriers — 802.11 transmits nothing there, so they carry no
   channel estimate. Same for the single dark line at subcarrier zero, the DC
   null. My detector excludes those before selecting subcarriers, because a
   structural zero is not a quiet channel."*

   Then point at the vertical texture: *"That variation is the channel changing
   over time. That is the sensing signal — everything else in the pipeline is
   extracting structure from it."*

   This is the moment to pre-empt the skeleton question: *"A reference
   implementation would draw a skeleton here. Mine shows the measurement
   instead, because I have no trained pose model and drawing one would mean
   showing you something that isn't real."*

4. **Subcarrier amplitudes** — 64 live bars, the current frame as a cross-section
   of the waterfall's rightmost column.

5. **Stream panel** — measured rate against nominal. Say: *"Measured throughput,
   not assumed. The frequency analysis depends on the real sample rate, so it's
   reported rather than trusted."*

---

## Part 2 — the results

Open `output/RESULTS.md` and `output_overnight/RESULTS.md`, or show the figures
directly. The two datasets tell opposite stories, and that pairing is the point.

| | 2-min `pretrain` | 51-min `overnight` |
|---|---|---|
| Aligned correlation | **−0.157** | **≈ +0.30** |
| Verdict | no shared signal | ~2× controls — suggestive |

Say: *"The same pipeline reaches opposite conclusions on the two recordings,
which is what you want — it discriminates rather than always agreeing."*

**Be ready for the obvious question — "so does it measure breathing?"**
Answer: *"Not provably. The respiratory band maps exactly onto 6–30 breaths per
minute, so any noise peak yields a plausible number. Correlation between two
independent nodes is about twice the decorrelated controls — suggestive of a
shared signal, but with no ground-truth sensor I can't claim accuracy."*

That answer is stronger than a confident number. It shows you know what your
evidence supports.

---

## Part 3 — the reference system (optional)

RuView's full UI, as the architectural target. **Synthetic data — say so
immediately.**

Start Docker Desktop first and wait for the whale icon to settle, then:

```bash
docker start ruview-demo
```

Opens at <http://localhost:3000>. If the container was removed:

```bash
docker run -d --name ruview-demo -e CSI_SOURCE=simulated -e RUVIEW_ALLOW_UNAUTHENTICATED=1 -p 127.0.0.1:3000:3000 -p 127.0.0.1:3001:3001 ruvnet/wifi-densepose:latest
```

Both environment variables are required — without them the server exits with
code 78 or 64 by design.

Say: *"This is the open-source project whose published dataset I evaluated
against. Everything on this screen is synthetic — the dashboard labels its own
source as SIMULATED. I show it to illustrate the target architecture, not as my
result."*

---

## When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` on :8000 | An earlier run is still alive | `--port 8001`, or kill it (below) |
| Dashboard shows "disconnected" | Server died | Check the terminal for a traceback |
| State stuck on CALIBRATING | Window still filling | Wait ~15 s; bar shows progress |
| Always MOTION, never CLEAR | Baseline learned during movement | Click **Recalibrate**, keep still |
| Never MOTION | Baseline learned during activity | **Recalibrate** with the area empty |
| `ModuleNotFoundError: numpy` | Wrong interpreter | Use `../.venv/Scripts/python.exe`, not bare `python` |
| Docker: `error during connect` | Docker Desktop not started | Start it, wait for the whale, retry |
| Docker exits code 78 or 64 | Missing env vars | Use the full `docker run` above |

Free port 8000:

```bash
powershell "Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
```

---

## If everything fails

You still have three fallbacks that need no running software:

1. **`output/` and `output_overnight/`** — figures and RESULTS.md, viewable in
   any image viewer or text editor
2. **`docs/architecture.svg`** — opens in any browser
3. **The GitHub repository** — renders figures and README without cloning

Regenerate the analysis at any time — no server, no Docker, ~1 min:

```bash
../.venv/Scripts/python.exe analyze_csi.py --file ../RuView/data/recordings/pretrain-1775182186.csi.jsonl --out output
```

---

## Questions you should expect

**"Did you capture this data?"**
No. The recordings come from the RuView open-source project, used as an
evaluation dataset. The processing pipeline, detector and analysis are mine.

**"Why not use your own WiFi adapter?"**
Tried and measured: the MediaTek MT7922 reports a static RSSI — 99 samples over
30 s via netsh and 236 over 12 s querying the driver directly through the WLAN
API, with zero variation in both. Driver-level smoothing removes the
fluctuation sensing depends on. That's why dedicated CSI hardware is required.

**"Why is there no skeleton like the reference system?"**
Because no trained keypoint weights exist to drive one. The reference draws its
skeleton from signal heuristics when no model is loaded. Drawing a skeleton
without a model would be showing you something that isn't a measurement.

**"What would you do next?"**
Live capture. The ESP32 source is already written against the ADR-018 frame
format and needs only hardware to verify. A Raspberry Pi 3B+/4 is an
alternative path via Nexmon CSI, which gives wider bandwidth and more
subcarriers than the ESP32.
