import pyarrow  # noqa: F401 — must be first
import argparse
import json
import os
import glob
import re
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from smooth_player import (show_smooth_player, select_signal_dimensions, SIGNAL_PALETTE,
                           lighter_signal_color as _lighter_hex_color)
from signal_plot import show_signal_plot

st.set_page_config(page_title="VLA Episode Viewer (LeRobot)", layout="wide")

# =========================================================
# Constants
# =========================================================
IMAGE_RESIZE_SIZE = (224, 224)
NOMINAL_HZ_CAMERA = 30
GAP_TOLERANCE_MULTIPLIER = 1.5
MAX_RENDERED_GAPS = 80
lrv_cache_version = 6  # bump to invalidate cache


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
    return tuple(keys) if features else ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _parse_chunk_file_indices(parquet_path):
    p = Path(parquet_path)
    chunk_match = re.search(r"chunk-(\d+)", p.parent.name)
    if chunk_match is None:
        raise ValueError(f"Cannot parse chunk index from {parquet_path}")

    # LeRobot v3 packs multiple episodes into file-NNN.parquet, while v2.x
    # stores one episode per episode_NNNNNN.parquet.  For v2, use the episode
    # index as a harmless file-index fallback; its metadata video template uses
    # episode_index instead of file_index.
    file_match = re.search(r"file-(\d+)", p.stem)
    episode_match = re.search(r"episode_(\d+)", p.stem)
    if file_match is not None:
        file_idx = int(file_match.group(1))
    elif episode_match is not None:
        file_idx = int(episode_match.group(1))
    else:
        raise ValueError(f"Cannot parse file or episode index from {parquet_path}")

    chunk_idx = int(chunk_match.group(1))
    return chunk_idx, file_idx


@lru_cache(maxsize=4)
def _load_episodes_meta(root_dir):
    """Load meta/episodes/**/*.parquet (LeRobot v3+): gives, per episode, which
    chunk/file each camera's video segment lives in and its time offset within
    that file. Needed because in v3 multiple episodes are packed into shared
    video files, and each camera can be chunked into different files than the
    data parquet and than each other. Returns None if absent (older format)."""
    pattern = os.path.join(root_dir, "meta", "episodes", "**", "*.parquet")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


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


def _get_frame(video_path: str, frame_idx: int):
    """Read a frame while keeping each camera decoder open for this session."""
    if not os.path.exists(video_path):
        return None
    readers = st.session_state.setdefault("lr_video_readers", {})
    reader = readers.get(video_path)
    if reader is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
        reader = {"cap": cap, "next_frame": 0, "frame_idx": None, "frame": None}
        readers[video_path] = reader
    if reader["frame_idx"] == frame_idx:
        return reader["frame"]

    cap = reader["cap"]
    next_frame = reader["next_frame"]
    if next_frame < frame_idx <= next_frame + 10:
        for _ in range(frame_idx - next_frame):
            if not cap.grab():
                return None
    elif next_frame != frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMAGE_RESIZE_SIZE, interpolation=cv2.INTER_AREA)
    reader.update(next_frame=frame_idx + 1, frame_idx=frame_idx, frame=frame)
    return frame


def _keep_video_readers(video_paths):
    """Close decoders from the previously selected episode or dataset."""
    readers = st.session_state.get("lr_video_readers", {})
    for path in list(readers):
        if path not in video_paths:
            readers.pop(path)["cap"].release()


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
    row_timestamps_s = df["timestamp"].values.astype(np.float64)
    T = len(df)

    # ---- timestamp gap mask ----
    gap_tol_ns = round(nominal_period_ns * GAP_TOLERANCE_MULTIPLIER)
    if T > 1:
        ts_gap_mask = np.concatenate([[False], np.diff(master_times_ns) > gap_tol_ns])
    else:
        ts_gap_mask = np.zeros(T, dtype=bool)

    # ---- cameras ----
    # LeRobot v3+ packs many episodes into shared video files, and each camera can
    # land in a different chunk/file than the data parquet and than other cameras.
    # meta/episodes/**/*.parquet records, per episode per camera, the actual
    # chunk/file and the from_timestamp offset of that camera's segment — use that
    # instead of assuming every camera reuses the data file's chunk/file (which
    # silently points at the wrong episode's frames once files no longer align 1:1).
    episodes_meta = _load_episodes_meta(root_dir)
    ep_row = None
    if episodes_meta is not None:
        match = episodes_meta[episodes_meta["episode_index"] == episode_idx]
        if len(match) > 0:
            ep_row = match.iloc[0]

    video_path_tpl = meta.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    camera_video_paths = {}
    camera_video_offsets = {}
    camera_frame_indices = {}
    camera_gap_spans = {}
    camera_stats = {}

    for cam_key in camera_keys:
        video_key = f"observation.images.{cam_key}"
        cam_chunk_idx, cam_file_idx, from_ts_s = chunk_idx, file_idx, 0.0
        if ep_row is not None and f"videos/{video_key}/chunk_index" in ep_row.index:
            cam_chunk_idx = int(ep_row[f"videos/{video_key}/chunk_index"])
            cam_file_idx = int(ep_row[f"videos/{video_key}/file_index"])
            from_ts_s = float(ep_row[f"videos/{video_key}/from_timestamp"])
        video_path = os.path.join(
            root_dir,
            video_path_tpl.format(video_key=video_key,
                                  chunk_index=cam_chunk_idx,
                                  file_index=cam_file_idx,
                                  episode_chunk=cam_chunk_idx,
                                  episode_index=episode_idx)
        )
        camera_video_paths[cam_key] = video_path
        camera_video_offsets[cam_key] = from_ts_s
        cam_fps = float(features.get(video_key, {}).get("info", {}).get("video.fps", nominal_fps))
        # Video-local frame index: this row's episode-local timestamp (starts at 0)
        # plus the segment's start offset within the shared video file — not the
        # (episode-local) `frame_index` column, which resets to 0 per episode and
        # is meaningless once episodes share a file.
        frame_idx_for_cam = np.round((row_timestamps_s + from_ts_s) * cam_fps).astype(np.int64)
        camera_frame_indices[cam_key] = frame_idx_for_cam.tolist()
        if not os.path.exists(video_path):
            cam_missing = np.ones(T, dtype=bool)
        else:
            cap = cv2.VideoCapture(video_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            cam_missing = (frame_idx_for_cam < 0) | (frame_idx_for_cam >= total_video_frames)
        cam_spans = _missing_runs(cam_missing)
        camera_gap_spans[cam_key] = cam_spans
        camera_stats[f"camera:{cam_key}"] = _health_row(
            master_times_ns.tolist(), cam_spans, nominal_fps
        )

    gap_spans = _missing_runs(ts_gap_mask)

    # ---- state / action: NaN at gap positions ----
    signal_stats = {}
    state_matrix = np.zeros((T, 0))
    state_names = []
    if "observation.state" in df.columns:
        raw_state = np.stack(df["observation.state"].values).astype(np.float32)
        raw_state[ts_gap_mask] = np.nan
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
        raw_action[ts_gap_mask] = np.nan
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
        "camera_video_offsets": camera_video_offsets,
        "camera_frame_indices": camera_frame_indices,  # cam_key -> list[int], len=T
        "timestamps": row_timestamps_s,
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

episode_widget_key = "lr_episode_selector"
if st.session_state.get(episode_widget_key) not in display_ids:
    st.session_state[episode_widget_key] = display_ids[0]


def _step_episode(delta):
    current = st.session_state.get(episode_widget_key, display_ids[0])
    current_idx = display_ids.index(current) if current in display_ids else 0
    next_idx = min(max(current_idx + delta, 0), len(display_ids) - 1)
    st.session_state[episode_widget_key] = display_ids[next_idx]
    st.session_state.t = 0


episode_disp = st.sidebar.selectbox(
    "Select Episode", display_ids, key=episode_widget_key
)
episode_pos = display_ids.index(episode_disp)
episode_nav_cols = st.sidebar.columns(2)
episode_nav_cols[0].button(
    "↑ Previous",
    key="lr_previous_episode",
    on_click=_step_episode,
    args=(-1,),
    disabled=episode_pos == 0,
    width="stretch",
    shortcut="Up",
)
episode_nav_cols[1].button(
    "↓ Next",
    key="lr_next_episode",
    on_click=_step_episode,
    args=(1,),
    disabled=episode_pos == len(display_ids) - 1,
    width="stretch",
    shortcut="Down",
)
st.sidebar.caption("快捷键：↑/↓ 切换 episode")
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
camera_frame_indices = data["camera_frame_indices"]
camera_gap_spans = data["camera_gap_spans"]
master_times = data["master_times"]
gap_spans = data["gap_spans"]
stats = data["stats"]
intervention_mask = data["intervention_mask"]

# =========================================================
# Episode change detection
# =========================================================
episode_key = (root_dir, episode_disp)
if st.session_state.get("lr_episode_key") != episode_key:
    st.session_state.lr_episode_key = episode_key
    st.session_state.t = 0
    st.session_state.lr_step = 0

# =========================================================
# Playback and signal selection
# =========================================================
if "t" not in st.session_state:
    st.session_state.t = 0
st.session_state.t = min(st.session_state.t, T - 1) if T > 0 else 0

st.markdown("## 流畅播放")
all_labels = [f"S:{n}" for n in state_names] + [f"A:{n}" for n in action_names]
paired_names = [name for name in state_names
                if name in action_names and not name.startswith("pad_")]
default_dims = [label for name in paired_names[:4]
                for label in (f"S:{name}", f"A:{name}")]
default_dims = default_dims or labels_with_data(state, state_names, action, action_names)[:8]
selected_dims = select_signal_dimensions(
    all_labels, default_dims, key="lr_selected_dims",
    previous=st.session_state.get("lr_selected_dims_value"),
)
st.session_state.lr_selected_dims_value = list(selected_dims)
st.caption("同名关节用同一色系：S 实线是 state，A 浅色点线是 action（command）。")

signal_names = list(dict.fromkeys([*state_names, *action_names]))
signal_colors = {
    name: SIGNAL_PALETTE[i % len(SIGNAL_PALETTE)]
    for i, name in enumerate(signal_names)
}
state_indices = {name: i for i, name in enumerate(state_names)}
action_indices = {name: i for i, name in enumerate(action_names)}
selected_set = set(selected_dims)
player_traces = []
for name in signal_names:
    state_label, action_label = f"S:{name}", f"A:{name}"
    if name in state_indices and state_label in selected_set:
        player_traces.append({"name": state_label, "color": signal_colors[name],
                              "dashed": False, "values": state[:, state_indices[name]]})
    if name in action_indices and action_label in selected_set:
        player_traces.append({"name": action_label,
                              "color": _lighter_hex_color(signal_colors[name]),
                              "dashed": True, "values": action[:, action_indices[name]]})

show_smooth_player(camera_video_paths, data["camera_video_offsets"],
                   data["timestamps"], player_traces, st.session_state.t)


# =========================================================
# Manual frame inspection
# =========================================================
@st.fragment
def _inspection_fragment():
    st.markdown("### 逐帧查看")
    _keep_video_readers(set(camera_video_paths.values()))

    def _step_frame(delta):
        st.session_state.t = min(max(st.session_state.t + delta, 0), max(T - 1, 0))
        st.session_state.lr_step = st.session_state.t

    frame_cols = st.columns(2)
    frame_cols[0].button(
        "← Previous frame", key="lr_previous_frame", on_click=_step_frame,
        args=(-1,), disabled=st.session_state.t <= 0, width="stretch", shortcut="Left",
    )
    frame_cols[1].button(
        "Next frame →", key="lr_next_frame", on_click=_step_frame,
        args=(1,), disabled=st.session_state.t >= T - 1, width="stretch", shortcut="Right",
    )
    st.caption("上方可连续播放关节曲线；这里可以拖动 Step 或用 ←/→ 逐帧细看。")
    t_manual = st.slider("Step", 0, T - 1, key="lr_step") if T > 1 else 0
    if t_manual != st.session_state.t:
        st.session_state.t = t_manual
    t = st.session_state.t

    cols = st.columns(len(camera_keys)) if camera_keys else []
    for col, cam_key in zip(cols, camera_keys):
        with col:
            video_path = camera_video_paths.get(cam_key, "")
            cam_fi = camera_frame_indices.get(cam_key, [])
            fi = cam_fi[t] if t < len(cam_fi) else -1
            img = _get_frame(video_path, fi) if fi >= 0 and video_path else None
            if img is not None:
                st.image(img,
                         caption=f"{cam_key}  step {t + 1}/{T}  ({IMAGE_RESIZE_SIZE[0]}×{IMAGE_RESIZE_SIZE[1]})",
                         output_format="JPEG")
            else:
                st.warning(f"{cam_key} missing at step {t}")

    if intervention_mask is not None and intervention_mask[t]:
        st.info(f"Step {t}: intervention=1")

    fig = go.Figure()
    plot_x = np.arange(T)
    for name in signal_names:
        color = signal_colors[name]
        state_label, action_label = f"S:{name}", f"A:{name}"
        if name in state_indices and state_label in selected_set:
            fig.add_trace(go.Scatter(
                x=plot_x, y=state[:, state_indices[name]], name=state_label,
                mode="lines", line=dict(color=color, width=2),
            ))
        if name in action_indices and action_label in selected_set:
            fig.add_trace(go.Scatter(
                x=plot_x, y=action[:, action_indices[name]], name=action_label,
                mode="lines", line=dict(color=_lighter_hex_color(color), width=2, dash="dot"),
            ))

    if gap_spans and selected_dims:
        _add_gap_lines(fig, gap_spans[:MAX_RENDERED_GAPS],
                       y0=0.05, y1=0.95, color="rgb(220,38,38)", label_prefix="gap ")
    if intervention_mask is not None and np.any(intervention_mask):
        for span in _missing_runs(intervention_mask.astype(bool)):
            fig.add_vrect(x0=span["x0"], x1=span["x1"],
                          fillcolor="rgba(99,102,241,0.15)", line_width=0)
    fig.add_vline(x=t, line_color="red", line_width=2)
    fig.update_layout(height=350, dragmode="select", xaxis_title="Step", hovermode="x")
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    show_signal_plot(fig, key="lr_signal_plot")


_inspection_fragment()

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
