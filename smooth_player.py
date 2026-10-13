"""Browser-side synchronized playback for LeRobot camera and signal data."""

import html
import json
import os

import numpy as np
import streamlit as st
from streamlit import runtime


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
        st.warning("没有可播放的相机视频。")
        return

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
  .controls { display: flex; gap: 12px; align-items: center; margin: 12px 0 8px; }
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
    <select id="speed" aria-label="播放速度">
      <option value="0.5">0.5×</option><option value="1" selected>1× 原速</option>
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
      anchorTime = performance.now();
      syncVideos(anchorLocal, true);
      const results = await Promise.allSettled(videos.map(video => {
        video.playbackRate = speed;
        return video.play();
      }));
      if (!results.some(result => result.status === 'fulfilled')) {
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
      if (playing) {
        anchorLocal += (performance.now() - anchorTime) / 1000 * speed;
        anchorTime = performance.now();
      }
      speed = Number(speedInput.value);
      videos.forEach(video => video.playbackRate = speed);
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
        const local = anchorLocal + (performance.now() - anchorTime) / 1000 * speed;
        if (local >= endTime) { setStep(count - 1, true); stop(); }
        else { setStep(stepAt(local)); syncVideos(local); }
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  </script>
</body></html>"""
    page = (page.replace("CAMERA_COUNT", str(len(cameras)))
                .replace("VIDEOS_HTML", videos_html)
                .replace("CONFIG_JSON", config_json))
    st.iframe(page, height="content")
