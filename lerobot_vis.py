import pyarrow  # noqa: F401 — must be first
import argparse
import json
import os
import glob
import sys
import time
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
NOMINAL_HZ_STATE = 100
GAP_TOLERANCE_MULTIPLIER = 1.5
MAX_RENDERED_GAPS = 80
DEFAULT_CAMERA_KEYS = ("head", "left_wrist", "right_wrist")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="lerobot_data", help="LeRobot dataset root dir")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


DEFAULT_ROOT = _parse_args().data_path

# =========================================================
# Shared utility functions (mirrored from mcap_vis.py)
# =========================================================

def _median_period_ns(times):
    if len(times) < 2:
        return None
    diffs = np.diff(np.array(times, dtype=np.int64))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) > 0 else None


def _alignment_tolerance_ns(source_times, target_times):
    periods = [p for p in (_median_period_ns(source_times), _median_period_ns(target_times)) if p]
    if not periods:
        return 50_000_000
    return int(max(periods) * GAP_TOLERANCE_MULTIPLIER)


def _nearest_alignment(source_times, target_times, tolerance_ns):
    target_times = np.array(target_times, dtype=np.int64)
    if len(source_times) == 0:
        n = len(target_times)
        return (np.full(n, -1, dtype=int), np.full(n, np.nan),
                np.zeros(n, dtype=bool), np.full(n, -1, dtype=int), np.full(n, np.nan))
    source_times = np.array(source_times, dtype=np.int64)
    right = np.searchsorted(source_times, target_times)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    right = np.clip(right, 0, len(source_times) - 1)
    left_delta = np.abs(target_times - source_times[left])
    right_delta = np.abs(source_times[right] - target_times)
    use_right = right_delta < left_delta
    raw_idx = np.where(use_right, right, left).astype(int)
    signed_delta = source_times[raw_idx] - target_times
    valid = np.abs(signed_delta) <= tolerance_ns
    raw_delta_ms = signed_delta.astype(float) / 1e6
    idx = raw_idx.copy()
    idx[~valid] = -1
    delta_ms = raw_delta_ms.copy()
    delta_ms[~valid] = np.nan
    return idx, delta_ms, valid, raw_idx, raw_delta_ms


def _missing_runs(missing_mask):
    mask = np.asarray(missing_mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [{"x0": int(s), "x1": int(e), "drop_frames": int(e - s + 1)} for s, e in zip(starts, ends)]


def _build_reference_timeline(head_times, other_topics_times, tolerance_ns, nominal_period_ns):
    head_arr = np.array(head_times, dtype=np.int64)
    virtual = []
    bounds = [t for times in other_topics_times if len(times) > 0 for t in (times[0], times[-1])]
    if bounds:
        t_min, t_max = min(bounds), max(bounds)
        if head_arr[0] - t_min > tolerance_ns:
            n = int((head_arr[0] - t_min) // nominal_period_ns)
            virtual.extend(int(head_arr[0] - k * nominal_period_ns) for k in range(1, n + 1))
        if t_max - head_arr[-1] > tolerance_ns:
            n = int((t_max - head_arr[-1]) // nominal_period_ns)
            virtual.extend(int(head_arr[-1] + k * nominal_period_ns) for k in range(1, n + 1))
    for t0, t1 in zip(head_arr[:-1], head_arr[1:]):
        diff = int(t1 - t0)
        if diff <= tolerance_ns:
            continue
        n_missing = max(int(round(diff / nominal_period_ns)) - 1, 0)
        if n_missing == 0:
            continue
        step = diff / (n_missing + 1)
        virtual.extend(int(round(t0 + k * step)) for k in range(1, n_missing + 1))
    virtual_arr = np.array(sorted(set(virtual)), dtype=np.int64) if virtual else np.zeros(0, dtype=np.int64)
    combined = np.concatenate([head_arr, virtual_arr])
    is_real = np.concatenate([np.ones(len(head_arr), dtype=bool), np.zeros(len(virtual_arr), dtype=bool)])
    order = np.argsort(combined, kind="stable")
    return combined[order], is_real[order]


def _topic_stats(times):
    if len(times) == 0:
        return {"count": 0, "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    if len(times) == 1:
        return {"count": 1, "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    arr = np.array(times, dtype=np.int64)
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return {"count": len(times), "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    duration_s = max((arr[-1] - arr[0]) / 1e9, 1e-9)
    return {
        "count": len(times),
        "hz": (len(times) - 1) / duration_s,
        "median_ms": float(np.median(diffs)) / 1e6,
        "max_gap_ms": float(np.max(diffs)) / 1e6,
    }


def _health_row(times, spans, nominal_hz):
    base = _topic_stats(times)
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

def list_episodes(root_dir):
    pattern = os.path.join(root_dir, "data", "**", "*.parquet")
    paths = sorted(glob.glob(pattern, recursive=True))
    episodes, ep2path = [], {}
    for p in paths:
        ep_id = Path(p).stem  # e.g. episode_000000
        episodes.append(ep_id)
        ep2path[ep_id] = p
    return episodes, ep2path


def load_meta(root_dir):
    info_path = os.path.join(root_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path) as f:
        return json.load(f)


def _load_image(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, IMAGE_RESIZE_SIZE, interpolation=cv2.INTER_AREA)


def _ts_to_ns(col_values):
    """parquet timestamp 列（float64 秒）→ int64 纳秒。"""
    return (np.array(col_values, dtype=np.float64) * 1e9).astype(np.int64)


@st.cache_data(max_entries=2, show_spinner="Loading LeRobot episode...")
def load_episode(parquet_path: str, root_dir: str, camera_keys: tuple):
    lrv_cache_version = 1  # noqa: F841 — bump to invalidate cache
    df = pd.read_parquet(parquet_path)
    episode_id = Path(parquet_path).stem

    meta = load_meta(root_dir)
    features = meta.get("features", {})
    nominal_fps = meta.get("fps", NOMINAL_HZ_CAMERA)
    nominal_period_ns = round(1e9 / nominal_fps)

    T_raw = len(df)
    frame_indices = df["frame_index"].values if "frame_index" in df.columns else np.arange(T_raw)

    # ---- 各字段时间戳（秒 → ns）----
    def get_ts(feature_key):
        col = f"{feature_key}.timestamp"
        return _ts_to_ns(df[col].values) if col in df.columns else None

    head_key = camera_keys[0]
    head_ts = get_ts(f"observation.images.{head_key}")
    if head_ts is None:
        head_ts = np.arange(T_raw, dtype=np.int64) * nominal_period_ns

    other_ts_list = []
    for ck in camera_keys[1:]:
        ts = get_ts(f"observation.images.{ck}")
        if ts is not None:
            other_ts_list.append(ts)
    state_ts = get_ts("observation.state")
    action_ts = get_ts("action")
    if state_ts is not None:
        other_ts_list.append(state_ts)
    if action_ts is not None:
        other_ts_list.append(action_ts)

    # ---- Master 时间线（以 head 相机为基准）----
    head_self_tol = _alignment_tolerance_ns(head_ts, head_ts)
    master_times, head_is_real = _build_reference_timeline(
        head_ts, other_ts_list, head_self_tol, nominal_period_ns
    )
    T = len(master_times)

    # ---- 相机图片 ----
    cameras = {}
    camera_valid = {}
    camera_counts = {}
    camera_gap_spans = {}
    camera_stats = {}

    for cam_key in camera_keys:
        cam_ts = get_ts(f"observation.images.{cam_key}")
        if cam_ts is None:
            cameras[cam_key] = [None] * T
            camera_valid[cam_key] = [False] * T
            camera_counts[cam_key] = 0
            camera_gap_spans[cam_key] = []
            camera_stats[f"camera:{cam_key}"] = _health_row([], [], nominal_fps)
            continue

        tol = _alignment_tolerance_ns(cam_ts, master_times)
        idx, _, valid, raw_idx, _ = _nearest_alignment(cam_ts, master_times, tol)
        if cam_key == head_key:
            valid = head_is_real
            idx = np.where(valid, raw_idx, -1)

        frames = []
        for is_valid, raw_i in zip(valid, idx):
            if is_valid and 0 <= raw_i < T_raw:
                fidx = int(frame_indices[raw_i])
                img_path = os.path.join(
                    root_dir, "images", cam_key, episode_id, f"frame_{fidx:06d}.jpg"
                )
                frames.append(_load_image(img_path))
            else:
                frames.append(None)

        valid_mask = np.array(valid, dtype=bool)
        spans = _missing_runs(~valid_mask)
        cameras[cam_key] = frames
        camera_valid[cam_key] = valid_mask.tolist()
        camera_counts[cam_key] = T_raw
        camera_gap_spans[cam_key] = spans
        camera_stats[f"camera:{cam_key}"] = _health_row(cam_ts.tolist(), spans, nominal_fps)

    # ---- State / Action ----
    signal_stats = {}
    signal_gap_spans = {}

    def _align_signal(ts, raw_arr, feature_key, nominal_hz):
        tol = _alignment_tolerance_ns(ts, master_times)
        s_idx, _, valid, _, _ = _nearest_alignment(ts, master_times, tol)
        aligned = np.zeros((T, raw_arr.shape[1]))
        vi = valid & (s_idx >= 0)
        aligned[vi] = raw_arr[s_idx[vi]]
        spans = _missing_runs(~valid)
        signal_stats[feature_key] = _health_row(ts.tolist(), spans, nominal_hz)
        return aligned, spans

    state_matrix = np.zeros((T, 0))
    state_names = []
    if "observation.state" in df.columns and state_ts is not None:
        raw_state = np.stack(df["observation.state"].values)
        raw_names = features.get("observation.state", {}).get(
            "names", [f"[{i}]" for i in range(raw_state.shape[1])]
        )
        aligned, spans = _align_signal(state_ts, raw_state, "observation.state", NOMINAL_HZ_STATE)
        state_matrix = aligned
        state_names = [f"observation.state.{n}" for n in raw_names]
        if spans:
            for n in state_names:
                signal_gap_spans[n] = spans

    action_matrix = np.zeros((T, 0))
    action_names = []
    if "action" in df.columns and action_ts is not None:
        raw_action = np.stack(df["action"].values)
        raw_names = features.get("action", {}).get(
            "names", [f"[{i}]" for i in range(raw_action.shape[1])]
        )
        aligned, spans = _align_signal(action_ts, raw_action, "action", NOMINAL_HZ_STATE)
        action_matrix = aligned
        action_names = [f"action.{n}" for n in raw_names]
        if spans:
            for n in action_names:
                signal_gap_spans[n] = spans

    return {
        "cameras": cameras,
        "camera_valid": camera_valid,
        "camera_counts": camera_counts,
        "camera_gap_spans": camera_gap_spans,
        "master_times": master_times,
        "stats": {**camera_stats, **signal_stats},
        "signal_gap_spans": signal_gap_spans,
        "state": state_matrix,
        "action": action_matrix,
        "state_names": state_names,
        "action_names": action_names,
        "total_step": T,
    }


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Data Source (LeRobot)")
root_dir = st.sidebar.text_input("dataset root dir", DEFAULT_ROOT)
episodes, ep2path = list_episodes(root_dir)
if not episodes:
    st.error(f"在 {root_dir}/data 下没有找到 *.parquet 文件")
    st.stop()

episode_id = st.sidebar.selectbox("Select Episode", episodes)
parquet_path = ep2path[episode_id]
st.sidebar.caption(parquet_path)

camera_keys = DEFAULT_CAMERA_KEYS
try:
    data = load_episode(parquet_path, root_dir, camera_keys)
except Exception as e:
    st.error(f"加载失败：{e}")
    st.stop()

state = data["state"]
action = data["action"]
state_names = data["state_names"]
action_names = data["action_names"]
T = data["total_step"]
cameras = data["cameras"]
camera_counts = data["camera_counts"]
camera_gap_spans = data["camera_gap_spans"]
master_times = data["master_times"]
stats = data["stats"]
signal_gap_spans = data["signal_gap_spans"]

D_state = state.shape[1] if state.ndim == 2 else 0
D_action = action.shape[1] if action.ndim == 2 else 0

# =========================================================
# Episode change detection
# =========================================================
if "lr_episode_key" not in st.session_state:
    st.session_state.lr_episode_key = episode_id
if st.session_state.lr_episode_key != episode_id:
    st.session_state.lr_episode_key = episode_id
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
    if st.button(toggle_label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing
    st.session_state.fps = st.slider("FPS", 5, 60, st.session_state.fps)
    t_manual = st.slider("Step", 0, max(T - 1, 0), st.session_state.t)
    if t_manual != st.session_state.t:
        st.session_state.t = t_manual
        st.session_state.playing = False
    t = st.session_state.t

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
            frames = cameras.get(cam_key, [])
            img = frames[t] if frames and t < len(frames) else None
            total = camera_counts.get(cam_key, 0)
            if img is not None:
                st.image(img,
                         caption=f"{cam_key}  step {t + 1}/{total}  ({IMAGE_RESIZE_SIZE[0]}×{IMAGE_RESIZE_SIZE[1]})",
                         output_format="JPEG")
            else:
                st.warning(f"{cam_key} missing at step {t}")

    # ---- Signal plot ----
    topics_in_view = []
    if any(l.startswith("S:") for l in selected_dims):
        topics_in_view.append("observation.state")
    if any(l.startswith("A:") for l in selected_dims):
        topics_in_view.append("action")

    _GAP_COLORS = ["rgb(220,38,38)", "rgb(37,99,235)", "rgb(217,119,6)", "rgb(5,150,105)"]
    n_lanes = max(len(topics_in_view), 1)
    lane_h = (0.95 - 0.05) / n_lanes
    topic_style = {
        tpc: {"y0": 0.05 + i * lane_h, "y1": 0.05 + (i + 1) * lane_h,
              "color": _GAP_COLORS[i % len(_GAP_COLORS)]}
        for i, tpc in enumerate(topics_in_view)
    }

    gap_spans_by_topic = {}
    for label in selected_dims:
        series_name = label[2:]
        tpc = "observation.state" if label.startswith("S:") else "action"
        seen = {(s["x0"], s["x1"]) for s in gap_spans_by_topic.get(tpc, [])}
        for span in signal_gap_spans.get(series_name, [])[:MAX_RENDERED_GAPS]:
            key = (span["x0"], span["x1"])
            if key not in seen:
                seen.add(key)
                gap_spans_by_topic.setdefault(tpc, []).append(span)

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
    for tpc, spans in gap_spans_by_topic.items():
        style = topic_style[tpc]
        _add_gap_lines(fig, spans[:MAX_RENDERED_GAPS],
                       y0=style["y0"], y1=style["y1"], color=style["color"],
                       label_prefix=f"{tpc} ")
    fig.update_layout(height=350, dragmode="select", xaxis_title="Step")
    if st.session_state.playing:
        fig.add_vline(x=t, line_color="red", line_width=2)
        fig.update_layout(hovermode=False)
        fig.update_xaxes(showspikes=False)
    else:
        fig.update_layout(hovermode="x")
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.playing and T > 0:
        time.sleep(1.0 / st.session_state.fps)
        st.session_state.t += 1
        if st.session_state.t >= T:
            st.session_state.t = T - 1
            st.session_state.playing = False
        st.rerun()


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
st.dataframe(image_health, use_container_width=True, hide_index=True,
             height=min(1200, 38 + 35 * max(len(image_health), 1)))

image_gap_rows = [
    {"camera": cam, "step_range": f"{s['x0']}-{s['x1']}", "drop_frames": s["drop_frames"]}
    for cam in camera_gap_spans
    for s in sorted(camera_gap_spans[cam], key=lambda x: -x["drop_frames"])
]
if image_gap_rows:
    st.markdown("#### Image Gap 明细")
    st.dataframe(image_gap_rows, use_container_width=True, hide_index=True,
                 height=min(600, 38 + 35 * max(len(image_gap_rows), 1)))

st.markdown("## Signal Health")
st.dataframe(signal_health, use_container_width=True, hide_index=True,
             height=min(1200, 38 + 35 * max(len(signal_health), 1)))

joint_gap_by_topic = {}
for field_key, spans in signal_gap_spans.items():
    topic = ".".join(field_key.split(".")[:2])  # observation.state or action
    seen = {(s["x0"], s["x1"]) for s in joint_gap_by_topic.get(topic, [])}
    for span in spans:
        key = (span["x0"], span["x1"])
        if key not in seen:
            seen.add(key)
            joint_gap_by_topic.setdefault(topic, []).append(span)
joint_gap_rows = [
    {"topic": topic, "step_range": f"{s['x0']}-{s['x1']}", "drop_frames": s["drop_frames"]}
    for topic in joint_gap_by_topic
    for s in sorted(joint_gap_by_topic[topic], key=lambda x: -x["drop_frames"])
]
if joint_gap_rows:
    st.markdown("#### Signal Gap 明细")
    st.dataframe(joint_gap_rows, use_container_width=True, hide_index=True,
                 height=min(600, 38 + 35 * max(len(joint_gap_rows), 1)))
