"""Browser-side synchronized playback for LeRobot camera and signal data."""

import html
import json
import os

import numpy as np
import streamlit as st
from streamlit import runtime


SIGNAL_PALETTE = (
    "#729FE0", "#B39AD4", "#69B3A2", "#E3A16D", "#D98DA3", "#76B6CF",
    "#9EB66B", "#D9B65F", "#9AABE0", "#C58FBF", "#77B69A", "#D9997D",
    "#83AEC9", "#B59ACB", "#97B274", "#D8A671", "#B393B4", "#78B6B3",
    "#BDB36F", "#CB9B89", "#92A3C5", "#CBA2BA", "#8EBCA9", "#B5A6CD",
)


def lighter_signal_color(color, amount=0.16):
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "rgb(" + ", ".join(str(round(c + (255 - c) * amount)) for c in rgb) + ")"


def select_signal_dimensions(labels, defaults, *, key, display_func=str, previous=None):
    """Persist semantic signal names independently of episode and source topic."""
    store_key = f"{key}_signal_ids"
    if store_key not in st.session_state:
        initial = defaults if previous is None else previous
        st.session_state[store_key] = [display_func(label) for label in initial]
    available = {display_func(label): label for label in labels}
    # Only a user change overwrites the saved choices. Temporarily unavailable
    # fields can therefore reappear when returning to another episode.
    st.session_state[key] = [available[name] for name in st.session_state[store_key]
                             if name in available]

    def save_selection():
        st.session_state[store_key] = [display_func(label) for label in st.session_state[key]]

    return st.multiselect("Select state / action dims", labels, key=key,
                          format_func=display_func, on_change=save_selection)


def show_smooth_player(camera_paths, camera_offsets, timestamps, traces, start_step=0):
    """Play videos in the browser so Streamlit does not redraw every frame."""
    if not runtime.exists():
        return  # Streamlit's bare test runner has no media server.

    media = runtime.get_instance().media_file_mgr
    cameras = []
    for index, (name, path) in enumerate(camera_paths.items()):
        if not os.path.isfile(path):
            continue
        url = media.add(path, "video/mp4", f"9.{index}")
        cameras.append({"name": name, "url": url, "offset": camera_offsets[name]})

    if not cameras:
        st.caption("没有可播放的相机视频，按数据时间轴播放关节曲线。")

    payload = {
        "times": [round(float(t), 6) for t in timestamps],
        "cameras": cameras,
        "traces": [
            {
                "name": item["name"],
                "color": item["color"],
                "dashed": item["dashed"],
                "values": [None if not np.isfinite(v) else round(float(v), 5)
                           for v in item["values"]],
            }
            for item in traces
        ],
        "startStep": int(start_step),
    }
    config_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    videos_html = "".join(
        f'<div class="camera"><video muted playsinline preload="auto" '
        f'src="{html.escape(cam["url"], quote=True)}"></video>'
        f'<div>{html.escape(cam["name"])}</div></div>'
        for cam in cameras
    )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  body { margin: 0; font: 14px sans-serif; color: #31343b; }
  .videos { display: grid; grid-template-columns: repeat(CAMERA_COUNT, minmax(0, 1fr)); gap: 10px; }
  .camera video { display: block; width: 100%; height: min(25vw, 260px); min-height: 100px;
                  object-fit: contain; background: #111; border-radius: 6px; }
  .camera div { text-align: center; padding: 4px; font-size: 12px; }
  .controls { display: flex; gap: 10px; align-items: center; margin: 12px 0 8px; flex-wrap: wrap; }
  .controls label { white-space: nowrap; font-size: 12px; }
  button, select { padding: 6px 10px; font-size: 14px; cursor: pointer; }
  input[type=range] { flex: 1; min-width: 60px; }
  #stepLabel { min-width: 92px; text-align: right; }
  .chart { position: relative; height: 230px; border: 1px solid #ddd; border-radius: 6px; }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  #cursor { cursor: crosshair; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 8px 0 0; font-size: 12px; }
  .legend span { white-space: nowrap; }
  .legend b { display: inline-block; width: 18px; vertical-align: middle; border-top: 2px solid; margin-right: 4px; }
</style></head><body>
  <div class="videos">VIDEOS_HTML</div>
  <div class="controls">
    <button id="play">▶ 播放</button>
    <input id="step" type="range" min="0" value="0">
    <span id="stepLabel"></span>
    <label for="frequency">播放频率</label>
    <select id="frequency" aria-label="播放频率" title="30/60 Hz 表示每秒播放 30/60 步；可叠加倍速">
      <option value="native">原速（时间戳）</option>
      <option value="30">30 Hz（30 步/秒）</option>
      <option value="60">60 Hz（60 步/秒）</option>
    </select>
    <select id="speed" aria-label="播放速度" title="在选定播放频率基础上调整倍速">
      <option value="0.5">0.5×</option><option value="1" selected>1×</option>
      <option value="2">2×</option>
    </select>
  </div>
  <div class="chart"><canvas id="signals"></canvas><canvas id="cursor"></canvas></div>
  <div class="legend" id="legend"></div>
  <script>
    const cfg = CONFIG_JSON;
    const videos = [...document.querySelectorAll('video')];
    const stepInput = document.getElementById('step');
    const stepLabel = document.getElementById('stepLabel');
    const playButton = document.getElementById('play');
    const speedInput = document.getElementById('speed');
    const frequencyInput = document.getElementById('frequency');
    const signals = document.getElementById('signals');
    const cursor = document.getElementById('cursor');
    const count = cfg.times.length;
    const time0 = cfg.times[0] || 0;
    const endTime = (cfg.times[count - 1] || time0) - time0;
    const pad = {left: 42, right: 12, top: 12, bottom: 25};
    let step = Math.min(Math.max(cfg.startStep, 0), count - 1);
    let playing = false;
    let speed = 1;
    let anchorTime = 0;
    let anchorLocal = 0;
    let anchorStep = 0;
    let frequency = 'native';
    try {
      const saved = sessionStorage.getItem('vla-player-frequency');
      if (['native', '30', '60'].includes(saved)) frequency = saved;
    } catch (_) {}
    frequencyInput.value = frequency;
    const periods = cfg.times.slice(1).map((t, i) => t - cfg.times[i])
                       .filter(t => t > 0).sort((a, b) => a - b);
    const medianPeriod = periods.length ? periods[Math.floor(periods.length / 2)] : 1 / 30;
    let yMin = -1, yMax = 1;
    stepInput.max = String(count - 1);

    function localAt(i) { return cfg.times[i] - time0; }
    function stepAt(local) {
      let lo = 0, hi = count;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (cfg.times[mid] - time0 <= local) lo = mid + 1;
        else hi = mid;
      }
      return Math.max(0, lo - 1);
    }
    function frameAt(local) {
      const i = stepAt(local);
      if (i >= count - 1) return count - 1;
      const duration = localAt(i + 1) - localAt(i);
      return i + (duration > 0 ? Math.max(0, local - localAt(i)) / duration : 0);
    }
    function localAtFrame(frame) {
      const i = Math.min(count - 1, Math.max(0, Math.floor(frame)));
      return localAt(i) + (frame - i) * (localAt(Math.min(i + 1, count - 1)) - localAt(i));
    }
    function positionAt(now) {
      const elapsed = (now - anchorTime) / 1000 * speed;
      if (frequency === 'native') {
        const local = Math.min(endTime, anchorLocal + elapsed);
        return {local, frame: frameAt(local), ended: local >= endTime};
      }
      const frame = Math.min(count - 1, anchorStep + elapsed * Number(frequency));
      return {local: localAtFrame(frame), frame, ended: frame >= count - 1};
    }
    function reanchor() {
      const now = performance.now();
      const position = playing ? positionAt(now) : {local: localAt(step), frame: step};
      anchorLocal = position.local;
      anchorStep = position.frame;
      anchorTime = now;
    }
    function updateVideoRate() {
      const rate = speed * (frequency === 'native' ? 1 : Number(frequency) * medianPeriod);
      videos.forEach(video => video.playbackRate = Math.max(0.0625, Math.min(16, rate)));
    }
    function syncVideos(local, force = false) {
      videos.forEach((video, i) => {
        const wanted = cfg.cameras[i].offset + time0 + local;
        if (force || Math.abs(video.currentTime - wanted) > 0.18) video.currentTime = wanted;
      });
    }
    function stop() {
      playing = false;
      videos.forEach(video => video.pause());
      playButton.textContent = '▶ 播放';
    }
    function setStep(i, seek = false) {
      step = Math.max(0, Math.min(count - 1, i));
      stepInput.value = String(step);
      stepLabel.textContent = `Step ${step} / ${count - 1}`;
      if (seek) syncVideos(localAt(step), true);
      drawCursor();
    }
    playButton.addEventListener('click', async () => {
      if (playing) { stop(); return; }
      if (step >= count - 1) setStep(0, true);
      anchorLocal = localAt(step);
      anchorStep = step;
      anchorTime = performance.now();
      syncVideos(anchorLocal, true);
      updateVideoRate();
      const results = await Promise.allSettled(videos.map(video => {
        return video.play();
      }));
      if (videos.length && !results.some(result => result.status === 'fulfilled')) {
        playButton.textContent = '视频无法播放';
        return;
      }
      playing = true;
      anchorTime = performance.now();
      playButton.textContent = '⏸ 暂停';
    });
    stepInput.addEventListener('input', () => {
      stop();
      setStep(Number(stepInput.value), true);
    });
    speedInput.addEventListener('change', () => {
      reanchor();
      speed = Number(speedInput.value);
      updateVideoRate();
    });
    frequencyInput.addEventListener('change', () => {
      reanchor();
      frequency = frequencyInput.value;
      try { sessionStorage.setItem('vla-player-frequency', frequency); } catch (_) {}
      updateVideoRate();
    });

    function fitCanvas(canvas) {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(canvas.clientWidth * ratio);
      canvas.height = Math.round(canvas.clientHeight * ratio);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return ctx;
    }
    function chartX(i) {
      return pad.left + i / Math.max(count - 1, 1) * (signals.clientWidth - pad.left - pad.right);
    }
    function drawSignals() {
      const ctx = fitCanvas(signals);
      fitCanvas(cursor);
      const width = signals.clientWidth, height = signals.clientHeight;
      yMin = Infinity; yMax = -Infinity;
      for (const trace of cfg.traces) {
        for (const value of trace.values) {
          if (value !== null && Number.isFinite(value)) {
            yMin = Math.min(yMin, value); yMax = Math.max(yMax, value);
          }
        }
      }
      if (!Number.isFinite(yMin)) { yMin = -1; yMax = 1; }
      if (yMax - yMin < 1e-6) { yMin -= 1; yMax += 1; }
      const margin = (yMax - yMin) * 0.08;
      yMin -= margin; yMax += margin;
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, height - pad.bottom);
      ctx.lineTo(width - pad.right, height - pad.bottom); ctx.stroke();
      ctx.fillStyle = '#555'; ctx.font = '12px sans-serif';
      ctx.fillText(yMax.toFixed(2), 2, pad.top + 9);
      ctx.fillText(yMin.toFixed(2), 2, height - pad.bottom);
      ctx.fillText('0', pad.left, height - 5);
      ctx.fillText(String(count - 1), width - pad.right - 32, height - 5);
      for (const trace of cfg.traces) {
        ctx.beginPath(); ctx.strokeStyle = trace.color; ctx.lineWidth = 1.7;
        ctx.setLineDash(trace.dashed ? [5, 3] : []);
        let open = false;
        trace.values.forEach((value, i) => {
          if (value === null) { open = false; return; }
          const y = pad.top + (yMax - value) / (yMax - yMin) * (height - pad.top - pad.bottom);
          if (open) ctx.lineTo(chartX(i), y);
          else { ctx.moveTo(chartX(i), y); open = true; }
        });
        ctx.stroke();
      }
      ctx.setLineDash([]);
      drawCursor();
    }
    function drawCursor() {
      const ctx = cursor.getContext('2d');
      const width = cursor.clientWidth, height = cursor.clientHeight;
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = '#db3131'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(chartX(step), pad.top);
      ctx.lineTo(chartX(step), height - pad.bottom); ctx.stroke();
    }
    cursor.addEventListener('click', event => {
      const rect = cursor.getBoundingClientRect();
      const frac = (event.clientX - rect.left - pad.left) /
                   (cursor.clientWidth - pad.left - pad.right);
      stop(); setStep(Math.round(frac * (count - 1)), true);
    });
    document.getElementById('legend').innerHTML = cfg.traces.map(trace =>
      `<span><b style="border-color:${trace.color};border-style:${trace.dashed ? 'dashed' : 'solid'}"></b>${trace.name.replaceAll('&', '&amp;').replaceAll('<', '&lt;')}</span>`
    ).join('');
    new ResizeObserver(drawSignals).observe(signals);
    setStep(step, true);
    drawSignals();
    function tick() {
      if (playing) {
        const position = positionAt(performance.now());
        if (position.ended) { setStep(count - 1, true); stop(); }
        else { setStep(Math.floor(position.frame)); syncVideos(position.local); }
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  </script>
</body></html>"""
    page = (page.replace("CAMERA_COUNT", str(max(len(cameras), 1)))
                .replace("VIDEOS_HTML", videos_html)
                .replace("CONFIG_JSON", config_json))
    st.iframe(page, height="content")
