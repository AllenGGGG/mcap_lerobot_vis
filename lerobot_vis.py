import pyarrow  # noqa: F401 — must be first
import argparse
import json
import os
import glob
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="VLA Episode Viewer (LeRobot)", layout="wide")

# =========================================================
# Constants
# =========================================================
IMAGE_RESIZE_SIZE = (224, 224)
NOMINAL_HZ_CAMERA = 30
GAP_TOLERANCE_MULTIPLIER = 1.5
MAX_RENDERED_GAPS = 80
lrv_cache_version = 2  # bump to invalidate cache


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="lerobot_data", help="LeRobot dataset root dir")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


DEFAULT_ROOT = _parse_args().data_path

# =========================================================
# Shared utility functions (mirrored from mcap_vis.py)
# =========================================================

def _missing_runs(missing_mask):
    mask = np.asarray(missing_mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [{"x0": int(s), "x1": int(e), "drop_frames": int(e - s + 1)} for s, e in zip(starts, ends)]


def _topic_stats(times_ns):
    if len(times_ns) == 0:
        return {"count": 0, "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    if len(times_ns) == 1:
        return {"count": 1, "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    arr = np.array(times_ns, dtype=np.int64)
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return {"count": len(times_ns), "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    duration_s = max((arr[-1] - arr[0]) / 1e9, 1e-9)
    return {
        "count": len(times_ns),
        "hz": (len(times_ns) - 1) / duration_s,
        "median_ms": float(np.median(diffs)) / 1e6,
        "max_gap_ms": float(np.max(diffs)) / 1e6,
    }


def _health_row(times_ns, spans, nominal_hz):
    base = _topic_stats(times_ns)
    base["drop_gaps"] = len(spans)
    base["drop_frames"] = int(sum(s["drop_frames"] for s in spans))
    base["coverage"] = (base["hz"] / nominal_hz) if nominal_hz else float("nan")
    return base


def _add_gap_lines(fig, spans, y0=0.05, y1=0.95, color="rgb(220,38,38)", label_prefix="", x_values=None):
    for span in spans:
        x = (x_values[span["x0"]] + x_values[span["x1"]]) / 2 if x_values is not None else (span["x0"] + span["x1"]) / 2
        fig.add_shape(type="line", x0=x, x1=x, y0=y0, y1=y1, yref="paper",
                      line=dict(color=color, width=2, dash="dot"))
        fig.add_annotation(x=x, y=y1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"⚠ {label_prefix}{span['drop_frames']}帧",
                           font=dict(size=10, color=color))


def labels_with_data(state, state_names, action, action_names):
    selected = []
    if getattr(state, "ndim", 0) == 2:
        for i, name in enumerate(state_names):
            if i < state.shape[1] and np.any(np.isfinite(state[:, i])):
                selected.append(f"S:{name}")
    if getattr(action, "ndim", 0) == 2:
        for i, name in enumerate(action_names):
            if i < action.shape[1] and np.any(np.isfinite(action[:, i])):
                selected.append(f"A:{name}")
    return selected


# =========================================================
# LeRobot data helpers
# =========================================================

def load_meta(root_dir):
    info_path = os.path.join(root_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path) as f:
        return json.load(f)


def _camera_keys_from_meta(meta):
    """Short camera keys (strip 'observation.images.' prefix) from info.json features."""
    features = meta.get("features", {})
    keys = []
    for feat_key, feat_val in features.items():
        if feat_key.startswith("observation.images.") and feat_val.get("dtype") == "video":
            keys.append(feat_key[len("observation.images."):])
    return tuple(keys) if keys else ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _parse_chunk_file_indices(parquet_path):
    p = Path(parquet_path)
    file_idx = int(re.search(r"file-(\d+)", p.stem).group(1))
    chunk_idx = int(re.search(r"chunk-(\d+)", p.parent.name).group(1))
    return chunk_idx, file_idx


def list_episodes(root_dir):
    """Return (display_ids, ep2info) where ep2info maps display_id -> {path, episode_idx, chunk_idx, file_idx}."""
    pattern = os.path.join(root_dir, "data", "**", "*.parquet")
    paths = sorted(glob.glob(pattern, recursive=True))
    display_ids = []
    ep2info = {}
    for p in paths:
        try:
            chunk_idx, file_idx = _parse_chunk_file_indices(p)
        except Exception:
            continue
        df = pd.read_parquet(p, columns=["episode_index"])
        for ep_idx in sorted(df["episode_index"].unique()):
            ep_idx = int(ep_idx)
            disp = f"episode_{ep_idx:06d}"
            display_ids.append(disp)
            ep2info[disp] = {"path": p, "episode_idx": ep_idx,
                             "chunk_idx": chunk_idx, "file_idx": file_idx}
    return display_ids, ep2info


@lru_cache(maxsize=1000)
def _get_frame(video_path: str, frame_idx: int):
    """Read a single frame from MP4. Returns rgb uint8 array or None."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, IMAGE_RESIZE_SIZE, interpolation=cv2.INTER_AREA)


@st.cache_data(max_entries=2, show_spinner="Loading LeRobot episode...")
def load_episode(parquet_path: str, episode_idx: int, root_dir: str,
                 chunk_idx: int, file_idx: int, camera_keys: tuple):
    _ = lrv_cache_version  # noqa: F841 — bump to invalidate cache

    df_full = pd.read_parquet(parquet_path)
    df = df_full[df_full["episode_index"] == episode_idx].reset_index(drop=True)

    meta = load_meta(root_dir)
    features = meta.get("features", {})
    nominal_fps = float(meta.get("fps", NOMINAL_HZ_CAMERA))
    nominal_period_ns = round(1e9 / nominal_fps)

    master_times_ns = (df["timestamp"].values.astype(np.float64) * 1e9).astype(np.int64)
    frame_indices = df["frame_index"].values.astype(np.int64)
    T = len(df)

    # ---- timestamp gap mask ----
    gap_tol_ns = round(nominal_period_ns * GAP_TOLERANCE_MULTIPLIER)
    if T > 1:
        ts_gap_mask = np.concatenate([[False], np.diff(master_times_ns) > gap_tol_ns])
    else:
        ts_gap_mask = np.zeros(T, dtype=bool)

    # ---- cameras ----
    video_path_tpl = meta.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    camera_video_paths = {}
    camera_gap_spans = {}
    camera_stats = {}
    master_missing = ts_gap_mask.copy()

    for cam_key in camera_keys:
        video_key = f"observation.images.{cam_key}"
        video_path = os.path.join(
            root_dir,
            video_path_tpl.format(video_key=video_key,
                                  chunk_index=chunk_idx, file_index=file_idx)
        )
        camera_video_paths[cam_key] = video_path
        if not os.path.exists(video_path):
            cam_missing = np.ones(T, dtype=bool)
        else:
            cap = cv2.VideoCapture(video_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            cam_missing = frame_indices >= total_video_frames
        master_missing |= cam_missing
        cam_spans = _missing_runs(cam_missing)
        camera_gap_spans[cam_key] = cam_spans
        camera_stats[f"camera:{cam_key}"] = _health_row(
            master_times_ns.tolist(), cam_spans, nominal_fps
        )

    gap_spans = _missing_runs(master_missing)

    # ---- state / action: NaN at gap positions ----
    signal_stats = {}
    state_matrix = np.zeros((T, 0))
    state_names = []
    if "observation.state" in df.columns:
        raw_state = np.stack(df["observation.state"].values).astype(np.float32)
        raw_state[master_missing] = np.nan
        raw_names = features.get("observation.state", {}).get(
            "names", [f"[{i}]" for i in range(raw_state.shape[1])]
        )
        state_matrix = raw_state
        state_names = list(raw_names)
        signal_stats["observation.state"] = _health_row(master_times_ns.tolist(), gap_spans, nominal_fps)

    action_matrix = np.zeros((T, 0))
    action_names = []
    if "action" in df.columns:
        raw_action = np.stack(df["action"].values).astype(np.float32)
        raw_action[master_missing] = np.nan
        raw_names = features.get("action", {}).get(
            "names", [f"[{i}]" for i in range(raw_action.shape[1])]
        )
        action_matrix = raw_action
        action_names = list(raw_names)
        signal_stats["action"] = _health_row(master_times_ns.tolist(), gap_spans, nominal_fps)

    intervention_mask = None
    if "observation.intervention" in df.columns:
        intervention_mask = df["observation.intervention"].values.astype(np.int64)

    return {
        "camera_video_paths": camera_video_paths,
        "frame_indices": frame_indices.tolist(),  # list[int], len=T
        "camera_gap_spans": camera_gap_spans,
        "master_times": master_times_ns,
        "gap_spans": gap_spans,
        "stats": {**camera_stats, **signal_stats},
        "state": state_matrix,
        "action": action_matrix,
        "state_names": state_names,
        "action_names": action_names,
        "intervention_mask": intervention_mask,
        "total_step": T,
        "nominal_fps": nominal_fps,
    }


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Data Source (LeRobot)")
root_dir = st.sidebar.text_input("dataset root dir", DEFAULT_ROOT)
meta = load_meta(root_dir)
camera_keys = _camera_keys_from_meta(meta)

display_ids, ep2info = list_episodes(root_dir)
if not display_ids:
    st.error(f"在 {root_dir}/data 下没有找到 *.parquet 文件")
    st.stop()

episode_disp = st.sidebar.selectbox("Select Episode", display_ids)
info = ep2info[episode_disp]
st.sidebar.caption(info["path"])

try:
    data = load_episode(
        info["path"], info["episode_idx"], root_dir,
        info["chunk_idx"], info["file_idx"], camera_keys
    )
except Exception as e:
    st.error(f"加载失败：{e}")
    st.stop()

state = data["state"]
action = data["action"]
state_names = data["state_names"]
action_names = data["action_names"]
T = data["total_step"]
camera_video_paths = data["camera_video_paths"]
frame_indices = data["frame_indices"]
camera_gap_spans = data["camera_gap_spans"]
master_times = data["master_times"]
gap_spans = data["gap_spans"]
stats = data["stats"]
intervention_mask = data["intervention_mask"]

# =========================================================
# Episode change detection
# =========================================================
if "lr_episode_key" not in st.session_state:
    st.session_state.lr_episode_key = episode_disp
if st.session_state.lr_episode_key != episode_disp:
    st.session_state.lr_episode_key = episode_disp
    st.session_state.t = 0
    st.session_state.playing = False
    st.rerun()

# =========================================================
# Session state
# =========================================================
for key, default in [("t", 0), ("playing", False), ("fps", 30)]:
    if key not in st.session_state:
        st.session_state[key] = default
st.session_state.t = min(st.session_state.t, T - 1) if T > 0 else 0


# =========================================================
# Playback fragment
# =========================================================
@st.fragment
def _playback_fragment():
    st.markdown("## Img / State / Action")
    toggle_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
    if st.button(toggle_label, width="stretch"):
        st.session_state.playing = not st.session_state.playing
        st.rerun(scope="fragment")
    st.session_state.fps = st.slider("FPS", 5, 60, st.session_state.fps)
    t_manual = st.slider("Step", 0, max(T - 1, 0), st.session_state.t)
    if t_manual != st.session_state.t:
        st.session_state.t = t_manual
        st.session_state.playing = False
    t = st.session_state.t
    D_state = state.shape[1] if state.ndim == 2 else 0
    D_action = action.shape[1] if action.ndim == 2 else 0

    # ---- Dim selection ----
    all_labels = [f"S:{n}" for n in state_names] + [f"A:{n}" for n in action_names]
    default_dims = labels_with_data(state, state_names, action, action_names)[:8] or all_labels[:8]

    mk = f"lr_dims_{st.session_state.lr_episode_key}"
    mk_init = f"{mk}_init"
    labels_sig = tuple(sorted(all_labels))
    if st.session_state.get("lr_labels_sig") != labels_sig:
        st.session_state.lr_labels_sig = labels_sig
        st.session_state[mk] = default_dims
        st.session_state[mk_init] = True
    elif mk not in st.session_state or not st.session_state.get(mk_init):
        st.session_state[mk] = default_dims
        st.session_state[mk_init] = True

    selected_dims = st.multiselect(
        "Select state / action dims", all_labels, key=mk,
        disabled=st.session_state.playing,
    )

    # ---- Camera grid ----
    cols = st.columns(len(camera_keys))
    for col, cam_key in zip(cols, camera_keys):
        with col:
            video_path = camera_video_paths.get(cam_key, "")
            fi = frame_indices[t] if t < len(frame_indices) else -1
            img = _get_frame(video_path, fi) if fi >= 0 and video_path else None
            if img is not None:
                st.image(img,
                         caption=f"{cam_key}  step {t + 1}/{T}  ({IMAGE_RESIZE_SIZE[0]}×{IMAGE_RESIZE_SIZE[1]})",
                         output_format="JPEG")
            else:
                st.warning(f"{cam_key} missing at step {t}")

    # ---- Intervention overlay ----
    if intervention_mask is not None and intervention_mask[t]:
        st.info(f"Step {t}: intervention=1")

    # ---- Signal plot ----
    _GAP_COLORS = ["rgb(220,38,38)", "rgb(37,99,235)", "rgb(217,119,6)", "rgb(5,150,105)"]

    fig = go.Figure()
    plot_x = np.arange(T)
    for i in range(D_state):
        label = f"S:{state_names[i]}"
        if label in selected_dims:
            fig.add_trace(go.Scatter(x=plot_x, y=state[:, i], name=label))
    for i in range(D_action):
        label = f"A:{action_names[i]}"
        if label in selected_dims:
            fig.add_trace(go.Scatter(x=plot_x, y=action[:, i], name=label, line=dict(dash="dot")))

    if gap_spans and selected_dims:
        _add_gap_lines(fig, gap_spans[:MAX_RENDERED_GAPS],
                       y0=0.05, y1=0.95, color=_GAP_COLORS[0], label_prefix="gap ")

    # Intervention shading
    if intervention_mask is not None and np.any(intervention_mask):
        iv_spans = _missing_runs(intervention_mask.astype(bool))
        for span in iv_spans:
            fig.add_vrect(x0=span["x0"], x1=span["x1"],
                          fillcolor="rgba(99,102,241,0.15)", line_width=0)

    fig.update_layout(height=350, dragmode="select", xaxis_title="Step")
    if st.session_state.playing:
        fig.add_vline(x=t, line_color="red", line_width=2)
        fig.update_layout(hovermode=False)
        fig.update_xaxes(showspikes=False)
    else:
        fig.update_layout(hovermode="x")
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    st.plotly_chart(fig, width="stretch")

    if st.session_state.playing and T > 0:
        time.sleep(1.0 / st.session_state.fps)
        st.session_state.t += 1
        if st.session_state.t >= T:
            st.session_state.t = T - 1
            st.session_state.playing = False
        st.rerun(scope="fragment")


_playback_fragment()

# =========================================================
# Data Health
# =========================================================
health_rows = []
for topic, item in sorted(stats.items()):
    median_ms = item["median_ms"]
    max_gap_ms = item["max_gap_ms"]
    health_rows.append({
        "topic": topic,
        "count": item["count"],
        "hz": round(item["hz"], 2),
        "median_ms": None if np.isnan(median_ms) else round(float(median_ms), 1),
        "max_gap_ms": None if np.isnan(max_gap_ms) else round(float(max_gap_ms), 1),
        "drop_gaps": item["drop_gaps"],
        "drop_frames": item["drop_frames"],
        "coverage": f"{item['coverage'] * 100:.0f}%" if not np.isnan(item["coverage"]) else None,
    })

image_health = [r for r in health_rows if r["topic"].startswith("camera:")]
signal_health = [r for r in health_rows if not r["topic"].startswith("camera:")]

st.markdown("#### Image Health")
st.dataframe(image_health, width="stretch", hide_index=True,
             height=min(1200, 38 + 35 * max(len(image_health), 1)))

image_gap_rows = [
    {"camera": cam, "step_range": f"{s['x0']}-{s['x1']}", "drop_frames": s["drop_frames"]}
    for cam in camera_gap_spans
    for s in sorted(camera_gap_spans[cam], key=lambda x: -x["drop_frames"])
]
if image_gap_rows:
    st.markdown("#### Image Gap 明细")
    st.dataframe(image_gap_rows, width="stretch", hide_index=True,
                 height=min(600, 38 + 35 * max(len(image_gap_rows), 1)))

st.markdown("## Signal Health")
st.dataframe(signal_health, width="stretch", hide_index=True,
             height=min(1200, 38 + 35 * max(len(signal_health), 1)))

if gap_spans:
    gap_rows = [
        {"step_range": f"{s['x0']}-{s['x1']}", "drop_frames": s["drop_frames"]}
        for s in sorted(gap_spans, key=lambda x: -x["drop_frames"])
    ]
    st.markdown("#### Signal Gap 明细")
    st.dataframe(gap_rows, width="stretch", hide_index=True,
                 height=min(600, 38 + 35 * max(len(gap_rows), 1)))
