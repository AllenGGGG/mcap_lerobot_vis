"""Temporary MP4 previews for the shared LeRobot browser player."""
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import cv2
import numpy as np
import streamlit as st
from streamlit import runtime

from smooth_player import show_smooth_player


def encode_preview(path, frames, timestamps):
    """Resample the aligned camera to video time, preserving real episode duration."""
    times = np.asarray(timestamps, dtype=np.float64)
    if len(frames) != len(times) or not len(times):
        raise ValueError("预览图像与时间轴长度不一致")
    periods = np.diff(times)
    periods = periods[periods > 0]
    period = float(np.median(periods)) if len(periods) else 1 / 30
    fps = min(60.0, max(1.0, 1 / period))
    count = max(1, math.ceil((times[-1] + 1 / fps) * fps))
    indices = np.searchsorted(times, np.arange(count) / fps, side="right") - 1
    indices = np.clip(indices, 0, len(frames) - 1)
    sample = next((frame for frame in frames if frame is not None), None)
    if sample is None:
        return False
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("流畅播放需要 ffmpeg（包含 libx264 编码器），请运行 install_env.sh 安装。")
    height, width = sample.shape[:2]
    missing = np.zeros_like(sample)
    cv2.putText(missing, "missing", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (230, 230, 230), 1, cv2.LINE_AA)
    path = Path(path)
    temporary = path.with_suffix(".partial.mp4")
    command = [executable, "-y", "-nostdin", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2",
               "-movflags", "+faststart", str(temporary)]
    # Capture errors in a file so a full stderr pipe cannot deadlock encoding.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            try:
                for index in indices:
                    frame = frames[index]
                    process.stdin.write((missing if frame is None else frame).tobytes())
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError("生成播放预览失败：" + errors.read().decode(errors="replace")[-1000:])
            os.replace(temporary, path)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            temporary.unlink(missing_ok=True)
    return True


class PreviewCache:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="mcap-viewer-")
        self.lock = threading.Lock()

    def prepare(self, source_key, cameras, timestamps):
        digest = hashlib.sha256(repr(source_key).encode() + np.asarray(timestamps).tobytes()).hexdigest()
        paths = {}
        with self.lock:
            for index, (name, frames) in enumerate(cameras.items()):
                path = Path(self.directory.name) / f"{digest}-{index}.mp4"
                if path.is_file() or encode_preview(path, frames, timestamps):
                    paths[name] = str(path)
        return paths


@st.cache_resource
def _preview_cache():
    return PreviewCache()


def show_mcap_smooth_player(source_key, cameras, master_times, traces, start_step=0):
    if not runtime.exists():
        return
    timestamps = (np.asarray(master_times, dtype=np.int64) - int(master_times[0])) / 1e9
    try:
        with st.spinner("准备播放视频…"):
            paths = _preview_cache().prepare(source_key, cameras, timestamps)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return
    show_smooth_player(paths, {name: 0.0 for name in paths}, timestamps, traces, start_step)
