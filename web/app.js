/* WiFi Sensing dashboard client.
   Polls /api/state and repaints. Charts are drawn straight onto canvas — no
   charting library, so there is nothing to load from a CDN and nothing to
   break offline during a demo. */

(function () {
  "use strict";

  var POLL_MS = 250;
  var el = function (id) { return document.getElementById(id); };

  var COLORS = {
    motion: "#ff6b4a",
    idle: "#35d39a",
    calibrating: "#f0b429",
    no_signal: "#6b7686",
    accent: "#4da3ff",
    grid: "#1a2230",
    dim: "#5b6b7f"
  };

  var STATE_TEXT = {
    motion: "MOTION",
    idle: "CLEAR",
    calibrating: "CALIBRATING",
    no_signal: "NO SIGNAL"
  };

  var lastOk = 0;
  var failures = 0;

  /* ---------- charts ---------- */

  // Fit the backing store to the CSS size and device pixel ratio, so lines
  // stay crisp on high-DPI displays instead of blurring.
  function prepare(canvas) {
    var ratio = window.devicePixelRatio || 1;
    var w = canvas.clientWidth;
    var h = parseInt(canvas.getAttribute("height"), 10) || 180;
    if (canvas.width !== w * ratio || canvas.height !== h * ratio) {
      canvas.width = w * ratio;
      canvas.height = h * ratio;
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function niceBounds(values, pad) {
    var min = Infinity, max = -Infinity;
    for (var i = 0; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    if (!isFinite(min) || !isFinite(max)) { min = 0; max = 1; }
    if (min === max) { min -= 1; max += 1; }
    var span = max - min;
    return { min: min - span * pad, max: max + span * pad };
  }

  function drawLine(canvas, series, opts) {
    opts = opts || {};
    var p = prepare(canvas);
    var ctx = p.ctx, w = p.w, h = p.h;
    var padL = 44, padR = 10, padT = 12, padB = 18;
    var plotW = w - padL - padR, plotH = h - padT - padB;

    if (!series || series.length < 2) {
      ctx.fillStyle = COLORS.dim;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("waiting for data…", w / 2, h / 2);
      return;
    }

    var b = niceBounds(series.concat(opts.extraBounds || []), 0.12);
    var yOf = function (v) { return padT + plotH - ((v - b.min) / (b.max - b.min)) * plotH; };
    var xOf = function (i) { return padL + (i / (series.length - 1)) * plotW; };

    // horizontal grid + labels
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.dim;
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (var g = 0; g <= 4; g++) {
      var val = b.min + (b.max - b.min) * (g / 4);
      var y = yOf(val);
      ctx.beginPath();
      ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(val.toFixed(opts.decimals != null ? opts.decimals : 1), padL - 6, y);
    }

    // threshold markers (enter / exit)
    (opts.thresholds || []).forEach(function (t) {
      if (t.value < b.min || t.value > b.max) return;
      var ty = yOf(t.value);
      ctx.save();
      ctx.strokeStyle = t.color;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(padL, ty); ctx.lineTo(w - padR, ty); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = t.color;
      ctx.textAlign = "left";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(t.label, padL + 4, ty - 7);
      ctx.restore();
    });

    // area fill
    var stroke = opts.color || COLORS.accent;
    var grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, hexA(stroke, 0.28));
    grad.addColorStop(1, hexA(stroke, 0.01));
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(series[0]));
    for (var i = 1; i < series.length; i++) ctx.lineTo(xOf(i), yOf(series[i]));
    ctx.lineTo(xOf(series.length - 1), padT + plotH);
    ctx.lineTo(xOf(0), padT + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // line
    ctx.beginPath();
    ctx.moveTo(xOf(0), yOf(series[0]));
    for (var j = 1; j < series.length; j++) ctx.lineTo(xOf(j), yOf(series[j]));
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.stroke();

    // leading dot
    var lx = xOf(series.length - 1), ly = yOf(series[series.length - 1]);
    ctx.beginPath();
    ctx.arc(lx, ly, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = stroke;
    ctx.fill();
  }

  function drawBars(canvas, values) {
    var p = prepare(canvas);
    var ctx = p.ctx, w = p.w, h = p.h;
    var padL = 34, padR = 8, padT = 10, padB = 16;
    var plotW = w - padL - padR, plotH = h - padT - padB;

    if (!values || !values.length) return;
    var b = niceBounds(values.concat([0]), 0.05);
    var bw = plotW / values.length;

    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.dim;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (var g = 0; g <= 3; g++) {
      var val = b.min + (b.max - b.min) * (g / 3);
      var y = padT + plotH - ((val - b.min) / (b.max - b.min)) * plotH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(val.toFixed(0), padL - 5, y);
    }

    for (var i = 0; i < values.length; i++) {
      var vh = ((values[i] - b.min) / (b.max - b.min)) * plotH;
      var x = padL + i * bw;
      var grad = ctx.createLinearGradient(0, padT + plotH - vh, 0, padT + plotH);
      grad.addColorStop(0, COLORS.accent);
      grad.addColorStop(1, "#1d4f80");
      ctx.fillStyle = grad;
      ctx.fillRect(x + bw * 0.15, padT + plotH - vh, Math.max(bw * 0.7, 1), vh);
    }
  }

  function hexA(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  /* ---------- waterfall ---------- */

  // Viridis control points — the same colormap the offline matplotlib figures
  // use, so the live display and the report figures are directly comparable.
  // Perceptually uniform, and legible to viewers with colour vision deficiency.
  var VIRIDIS = [
    [68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88],
    [181, 222, 43], [253, 231, 37]
  ];

  function viridis(t) {
    if (!(t >= 0)) t = 0;
    if (t > 1) t = 1;
    var x = t * (VIRIDIS.length - 1);
    var i = Math.floor(x);
    if (i >= VIRIDIS.length - 1) return VIRIDIS[VIRIDIS.length - 1];
    var f = x - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f
    ];
  }

  var wfCols = [];        // rolling column buffer, oldest first
  var wfSeq = 0;          // last sequence number received
  var wfLo = null;        // smoothed low end of the colour scale
  var wfHi = null;
  var wfOffscreen = null;

  function wfCapacity(canvas) {
    // One column per horizontal pixel — more would be discarded, fewer would
    // waste width.
    return Math.max(120, Math.min(900, canvas.clientWidth || 600));
  }

  function pushWaterfall(columns, canvas) {
    if (!columns || !columns.length) return;
    var cap = wfCapacity(canvas);
    for (var i = 0; i < columns.length; i++) wfCols.push(columns[i]);
    if (wfCols.length > cap) wfCols = wfCols.slice(wfCols.length - cap);
  }

  function updateScale() {
    // Robust percentile scaling. Fixed min/max would let one spike wash the
    // whole display out; the guard subcarriers sit at zero and would otherwise
    // anchor the low end permanently.
    var vals = [];
    var step = Math.max(1, Math.floor(wfCols.length / 60));
    for (var i = 0; i < wfCols.length; i += step) {
      var col = wfCols[i];
      for (var j = 0; j < col.length; j++) if (col[j] > 0) vals.push(col[j]);
    }
    if (vals.length < 10) return;
    vals.sort(function (a, b) { return a - b; });
    var lo = vals[Math.floor(vals.length * 0.02)];
    var hi = vals[Math.floor(vals.length * 0.98)];
    if (hi <= lo) hi = lo + 1;
    // Ease the scale so the image does not flicker as the window slides.
    wfLo = wfLo === null ? lo : wfLo * 0.85 + lo * 0.15;
    wfHi = wfHi === null ? hi : wfHi * 0.85 + hi * 0.15;
  }

  function drawWaterfall(canvas) {
    var ratio = window.devicePixelRatio || 1;
    var w = canvas.clientWidth;
    var h = parseInt(canvas.getAttribute("height"), 10) || 200;
    if (canvas.width !== w * ratio || canvas.height !== h * ratio) {
      canvas.width = w * ratio;
      canvas.height = h * ratio;
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    if (!wfCols.length) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = COLORS.dim;
      ctx.font = (12 * ratio) + "px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("waiting for CSI frames…", canvas.width / 2, canvas.height / 2);
      return;
    }

    updateScale();

    var nCols = wfCols.length;
    var nRows = wfCols[0].length;
    var span = (wfHi - wfLo) || 1;

    // Build the image at native data resolution, then let the GPU scale it.
    // Painting one rect per cell would be tens of thousands of draw calls.
    if (!wfOffscreen) wfOffscreen = document.createElement("canvas");
    if (wfOffscreen.width !== nCols || wfOffscreen.height !== nRows) {
      wfOffscreen.width = nCols;
      wfOffscreen.height = nRows;
    }
    var octx = wfOffscreen.getContext("2d");
    var img = octx.createImageData(nCols, nRows);
    var data = img.data;

    for (var x = 0; x < nCols; x++) {
      var col = wfCols[x];
      for (var y = 0; y < nRows; y++) {
        // Flip vertically so subcarrier 0 sits at the bottom, matching the
        // matplotlib figures (origin="lower").
        var v = col[nRows - 1 - y];
        var c = viridis((v - wfLo) / span);
        var o = (y * nCols + x) * 4;
        data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(wfOffscreen, 0, 0, canvas.width, canvas.height);
  }

  function pollWaterfall(canvas, rateHz) {
    fetch("/api/waterfall?since=" + wfSeq, { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.seq != null) wfSeq = d.seq;
        pushWaterfall(d.columns, canvas);
        drawWaterfall(canvas);
        var secs = rateHz > 0 ? (wfCols.length / rateHz) : 0;
        el("wf-meta").textContent =
          wfCols.length + " frames · " + secs.toFixed(0) + "s window · " +
          (wfCols.length ? wfCols[0].length : 0) + " subcarriers" +
          (wfLo !== null ? " · scale " + wfLo.toFixed(1) + "–" + wfHi.toFixed(1) : "");
      })
      .catch(function () { /* the state poll already reports connection loss */ });
  }

  /* ---------- render ---------- */

  function render(s) {
    if (!s || !s.reading) return;

    var r = s.reading;
    var caps = s.capabilities || {};
    var state = r.state;
    var color = COLORS[state] || COLORS.no_signal;

    // headline state
    el("state-label").textContent = STATE_TEXT[state] || state.toUpperCase();
    el("state-label").style.color = color;
    el("state-conf").textContent =
      state === "calibrating" ? "learning baseline"
                              : "confidence " + Math.round(r.confidence * 100) + "%";

    // ring: fraction of the way to the enter threshold
    var ring = el("ring-fill");
    var frac = Math.max(0, Math.min(1, r.confidence));
    ring.style.strokeDashoffset = String(327 * (1 - frac));
    ring.style.stroke = color;

    el("z-value").textContent = r.z_score.toFixed(2) + " σ";
    // Source-aware: for a CSI stream the meaningful level is mean subcarrier
    // amplitude (arbitrary units), not the packet RSSI field, which for these
    // recordings is a near-constant placeholder. Only an RSSI source is
    // genuinely in dBm.
    if ((caps.source_kind) === "csi") {
      var lvl = (r.features && r.features.mean_level != null) ? r.features.mean_level : r.rssi_dbm;
      el("rssi-value").textContent = lvl.toFixed(1) + " (amp)";
    } else {
      el("rssi-value").textContent = r.rssi_dbm.toFixed(1) + " dBm";
    }

    var th = s.thresholds || {};
    el("thresholds").textContent =
      (th.enter_z != null ? th.enter_z : "—") + " / " + (th.exit_z != null ? th.exit_z : "—") + " σ";

    var bl = s.baseline || {};
    el("baseline-value").textContent =
      bl.complete ? bl.mean_std.toFixed(3) + " ± " + bl.sigma_std.toFixed(3) : "not learned";

    // Respiration: show the value only when the source and window actually
    // support it. Anything else states why, rather than showing a number.
    var resp = r.respiration || {};
    var respEl = el("resp-value");
    var w = s.windows || {};
    var respStatus = (caps.status || {}).respiration;
    if (resp.supported && resp.bpm > 0) {
      // The value is shown, but never as a validated measurement: the ablation's
      // segment test found cross-node agreement negative in 4 of 5 segments, so
      // presenting a bare number here would contradict the project's findings.
      respEl.textContent = resp.bpm.toFixed(1) + " br/min";
      if (respStatus === "unvalidated") {
        respEl.textContent += "  (unvalidated)";
        respEl.style.color = COLORS.calibrating;
      } else {
        respEl.style.color = COLORS.idle;
      }
      respEl.title = "computed, not validated — " +
                     (resp.reason || "") +
                     "; cross-node agreement did not survive segmentation";
    } else if (caps.respiration && w.respiration_filled < w.respiration_n) {
      var pct = w.respiration_n ? Math.round((w.respiration_filled / w.respiration_n) * 100) : 0;
      respEl.textContent = "filling " + w.respiration_s + "s window (" + pct + "%)";
      respEl.style.color = COLORS.calibrating;
      respEl.title = "";
    } else {
      respEl.textContent = "not available";
      respEl.style.color = COLORS.dim;
      respEl.title = resp.reason || "";
    }

    el("state-note").textContent = r.note || "";

    // calibration bar
    var wrap = el("calib-wrap");
    if (r.calibration_progress >= 1) {
      wrap.hidden = true;
    } else {
      wrap.hidden = false;
      el("calib-fill").style.width = (r.calibration_progress * 100).toFixed(1) + "%";
      el("calib-text").textContent =
        "learning ambient baseline — " + Math.round(r.calibration_progress * 100) + "%  (keep the area still)";
    }

    // source pill
    var src = s.source || {};
    el("source-name").textContent = (src.name || "—") +
      (src.verified === false ? " (unverified)" : "");

    // capabilities — the honesty panel
    renderCaps(caps);

    // charts
    drawLine(el("z-chart"), s.z_history || [], {
      color: color,
      decimals: 1,
      extraBounds: [0, th.enter_z || 3],
      thresholds: [
        { value: th.enter_z, label: "enter", color: COLORS.motion },
        { value: th.exit_z, label: "exit", color: COLORS.idle }
      ]
    });

    var isCsi = caps.source_kind === "csi";
    el("level-hint").textContent = isCsi ? "mean subcarrier amplitude" : "dBm";
    drawLine(el("level-chart"), s.level_history || [], {
      color: COLORS.accent, decimals: 1
    });

    var subCard = el("sub-card");
    var wfCard = el("wf-card");
    if (s.subcarriers && s.subcarriers.length) {
      subCard.hidden = false;
      wfCard.hidden = false;
      drawBars(el("sub-chart"), s.subcarriers);
      // Frame rate drives the window-duration label, so read it from the
      // measured rate rather than the nominal one. Uses s.stats directly --
      // the local `st` below is declared after this point.
      var sst = s.stats || {};
      pollWaterfall(el("wf-chart"), (sst.actual_rate_hz || sst.nominal_rate_hz || 0));
    } else {
      subCard.hidden = true;
      wfCard.hidden = true;
    }

    // stream stats
    var st = s.stats || {};
    var f = r.features || {};
    var stats = [
      ["samples", st.samples],
      ["rate", (st.actual_rate_hz != null ? st.actual_rate_hz.toFixed(2) : "—") + " Hz"],
      ["nominal", (st.nominal_rate_hz != null ? st.nominal_rate_hz : "—") + " Hz"],
      ["dropped", st.dropped],
      ["uptime", (st.uptime_s != null ? st.uptime_s.toFixed(0) : "—") + " s"],
      ["window σ", f.std != null ? f.std.toFixed(3) : "—"],
      ["motion pwr", f.motion_power != null ? f.motion_power.toFixed(4) : "—"],
      ["dominant", f.dominant_hz != null ? f.dominant_hz.toFixed(2) + " Hz" : "—"]
    ];
    el("stat-grid").innerHTML = stats.map(function (kv) {
      return '<div class="stat"><span class="k">' + kv[0] +
             '</span><span class="v">' + (kv[1] == null ? "—" : kv[1]) + "</span></div>";
    }).join("");

    // footer provenance
    var sm = s.sample_meta || {};
    var bits = [];
    if (sm.ssid) bits.push("SSID " + sm.ssid);
    if (sm.band) bits.push(sm.band);
    if (sm.channel) bits.push("ch " + sm.channel);
    if (sm.node_id != null) bits.push("node " + sm.node_id);
    if (sm.subcarriers) bits.push(sm.subcarriers + " subcarriers");
    if (sm.file) bits.push(sm.file);
    el("footer-note").textContent = bits.join("  ·  ");
  }

  // Three display states, mirroring detector.capabilities(). "unvalidated" is
  // distinct from both: the value is computable but the project's own evidence
  // does not support presenting it as a measurement.
  var CAP_STATE = {
    validated:   { cls: "yes",   mark: "✓" },
    unvalidated: { cls: "warn",  mark: "!" },
    unavailable: { cls: "no",    mark: "✕" }
  };

  function renderCaps(caps) {
    var status = caps.status || {};
    var why = caps.why || {};
    var order = [
      ["motion", "Motion detection"],
      ["presence", "Presence (coarse)"],
      ["respiration", "Respiration rate"],
      ["cardiac", "Heart rate"],
      ["pose", "Pose / skeleton"]
    ];

    el("caps-list").innerHTML = order.map(function (pair) {
      var key = pair[0];
      var st = CAP_STATE[status[key]] || CAP_STATE.unavailable;
      return '<li class="' + st.cls + '">' +
             '<span class="mark">' + st.mark + "</span>" +
             "<span>" + pair[1] + "</span>" +
             '<span class="why">' + shortReason(why[key]) + "</span></li>";
    }).join("");

    el("caps-foot").innerHTML =
      (caps.source_kind === "rssi"
        ? "RSSI source at " + caps.sample_rate_hz + " Hz. Link-quality magnitude supports motion and " +
          "coarse presence only — vital signs require per-subcarrier CSI phase."
        : "CSI source at " + caps.sample_rate_hz + " Hz with per-subcarrier amplitude and phase.") +
      '<br><span class="legend">' +
      '<b class="ok">✓</b> validated &nbsp; ' +
      '<b class="wn">!</b> computable but not demonstrated &nbsp; ' +
      '<b class="nk">✕</b> not available' +
      "</span>";
  }

  function shortReason(reason) {
    if (!reason) return "unsupported at this source";
    return reason.length > 52 ? reason.slice(0, 49) + "…" : reason;
  }

  /* ---------- polling ---------- */

  function setLive(cls, text) {
    var d = el("live-dot");
    d.className = "live " + cls;
    el("live-text").textContent = text;
  }

  function poll() {
    fetch("/api/state", { cache: "no-store" })
      .then(function (res) { return res.json(); })
      .then(function (s) {
        lastOk = Date.now();
        failures = 0;
        setLive("ok", "live");
        render(s);
      })
      .catch(function () {
        failures++;
        if (failures > 3) setLive("down", "disconnected");
        else setLive("stale", "reconnecting");
      });
  }

  el("calibrate-btn").addEventListener("click", function () {
    fetch("/api/calibrate", { method: "POST" }).then(poll);
  });

  /* ---------- view switching (Live / Findings) ---------- */

  function switchView(view) {
    var live = view !== "findings";
    el("view-live").hidden = !live;
    el("view-findings").hidden = live;
    el("tab-live").classList.toggle("active", live);
    el("tab-findings").classList.toggle("active", !live);
    // Repaint charts when returning to Live, since a hidden canvas has zero
    // client size and would have been skipped.
    if (live) poll();
  }

  var tabs = document.querySelectorAll(".viewtab");
  for (var i = 0; i < tabs.length; i++) {
    tabs[i].addEventListener("click", function () {
      switchView(this.getAttribute("data-view"));
    });
  }

  window.addEventListener("resize", function () { /* next poll repaints at new size */ });

  poll();
  setInterval(poll, POLL_MS);
})();
