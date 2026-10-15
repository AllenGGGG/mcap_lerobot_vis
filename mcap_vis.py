import pyarrow  # noqa: F401 — must be first; forces full init before narwhals/plotly touch pa.Table
import argparse
import os
import glob
import struct
import sys
from array import array
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
from mcap.reader import make_reader, NonSeekingReader
from mcap.exceptions import McapError, EndOfFile
from mcap_ros2.decoder import DecoderFactory
from smooth_player import (select_signal_dimensions, SIGNAL_PALETTE,
                           lighter_signal_color as _lighter_hex_color)
from mcap_video import show_mcap_smooth_player
from signal_plot import show_signal_plot
from mcap_joint_signals import (
    COMMAND_TOPICS, NAMES_TOPIC, JOINT_TOPIC, CommandDecoder,
    joint_positions, signal_identity, display_label,
)

MCAP_READ_ERRORS = (McapError, EndOfFile, struct.error)
# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="VLA Episode Viewer (mcap)", layout="wide")


def _parse_args():
    """支持 `streamlit run app.py -- --data-path /your/data` 指定数据目录，
    不传就用默认值；用 parse_known_args 避免跟 Streamlit 自己的参数冲突。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="test", help="mcap root dir")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


# 数据目录结构假设:
#   root/20260706_111225_545157/metadata.json
#   root/20260706_111225_545157/recording/recording_0.mcap
DEFAULT_ROOT = _parse_args().data_path
DEFAULT_CAMERA_TOPICS = {
    "left_wrist": "/camera_left_wrist/color/image_raw/compressed",
    "head": "/camera_head/color/image_raw/compressed",
    "right_wrist": "/camera_right_wrist/color/image_raw/compressed",
}
DEFAULT_STATE_TOPICS = [JOINT_TOPIC]
DEFAULT_ACTION_TOPICS = list(COMMAND_TOPICS)
# 相机画面统一 resize 到的尺寸 (width, height)，VLA 模型输入常用 224x224
IMAGE_RESIZE_SIZE = (224, 224)
# 标称频率：仅用于 Data Health 表里的"整体是否达标" coverage 对比，不参与逐 step 缺帧判定
# （逐 step 判定继续用各 topic 自己的自适应容差，见 _alignment_tolerance_ns）
NOMINAL_HZ_CAMERA = 30
NOMINAL_HZ_JOINT = 500
# 缺帧容差倍数：相邻/最近邻时间差超过 (该 topic 自适应周期 × 这个倍数) 就判定为缺帧
GAP_TOLERANCE_MULTIPLIER = 1.5
# Plotly 图上最多渲染多少段丢帧竖线，避免极端情况下卡顿
MAX_RENDERED_GAPS = 80


# =========================================================
# mcap helpers
# =========================================================
def list_episodes(root_dir):
    """扫描 root_dir/*/recording/*.mcap，用时间戳文件夹名作为 episode id"""
    pattern = os.path.join(root_dir, "*", "recording", "*.mcap")
    paths = sorted(glob.glob(pattern))
    ep2path = {}
    for p in paths:
        ep_id = Path(p).parents[1].name  # .../<episode_id>/recording/xxx.mcap
        ep2path.setdefault(ep_id, []).append(p)
    return list(ep2path), {key: tuple(value) for key, value in ep2path.items()}


def _iter_mcap_messages(path, topics=None, issues=None):
    """Read indexed or unfinished files; keep partial-read failures visible."""
    with open(path, "rb") as stream:
        try:
            reader = make_reader(stream)
            if reader.get_summary() is None:
                raise EndOfFile()
        except MCAP_READ_ERRORS:
            stream.seek(0)
            reader = NonSeekingReader(stream)
            if issues is not None:
                issues.append(f"{Path(path).name} 缺少可用索引，已改为顺序扫描。")
        try:
            yield from reader.iter_messages(topics=topics, log_time_order=False)
        except MCAP_READ_ERRORS as exc:
            if issues is not None:
                issues.append(f"{Path(path).name} 读取中断，仅展示已读数据：{type(exc).__name__}")


def _flatten(msg, prefix=""):
    """递归展开一个 decode 后的 ros2 消息为 {字段路径: float} 的字典。
    跳过消息头、字符串及字节等非信号字段。"""
    out = {}
    if isinstance(msg, (bool, np.bool_)):
        return {prefix: float(msg)}
    if isinstance(msg, (int, float, np.integer, np.floating)):
        out[prefix] = float(msg)
        return out
    if isinstance(msg, (bytes, str)):
        return out
    if isinstance(msg, (list, tuple, np.ndarray, array)):
        for i, v in enumerate(msg):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    attrs = getattr(msg, "__dict__", None)
    slots = getattr(msg, "__slots__", None)
    if slots:
        attrs = {k: getattr(msg, k) for k in slots if hasattr(msg, k)}
    elif attrs is None:
        # 有些动态消息类用 __slots__，没有 __dict__，退化用 dir() 取属性
        attrs = {
            k: getattr(msg, k)
            for k in dir(msg)
            if not k.startswith("_") and not callable(getattr(msg, k, None))
        }
    for k, v in attrs.items():
        if k.startswith("_") or k == "header":
            continue
        new_prefix = f"{prefix}.{k}" if prefix else k
        out.update(_flatten(v, new_prefix))
    return out


def _decode_compressed_image(msg):
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_RESIZE_SIZE, interpolation=cv2.INTER_AREA)
    return img


def _decode_image(msg):
    if not hasattr(msg, "encoding"):
        return _decode_compressed_image(msg)
    formats = {"rgb8": (3, None), "bgr8": (3, cv2.COLOR_BGR2RGB),
               "rgba8": (4, cv2.COLOR_RGBA2RGB), "bgra8": (4, cv2.COLOR_BGRA2RGB),
               "mono8": (1, cv2.COLOR_GRAY2RGB)}
    if msg.encoding not in formats:
        raise ValueError(f"暂不支持原始图像编码 {msg.encoding}")
    channels, conversion = formats[msg.encoding]
    # ROS Image.step includes row padding.
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(int(msg.height), int(msg.step))
    img = rows[:, :int(msg.width) * channels].reshape(int(msg.height), int(msg.width), channels)
    if conversion is not None:
        img = cv2.cvtColor(img, conversion)
    return cv2.resize(img, IMAGE_RESIZE_SIZE, interpolation=cv2.INTER_AREA)


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


def _signal_identity(name):
    return signal_identity(name)


def default_selected_dims(state_names, action_names, limit=8):
    states = {_signal_identity(name): f"S:{name}" for name in state_names}
    actions = {_signal_identity(name): f"A:{name}" for name in action_names}
    pairs = [name for name in states if name in actions]
    joint_order = ["body_joint3", "body_joint4", *[f"left_joint{i}" for i in range(1, 8)],
                   *[f"right_joint{i}" for i in range(1, 8)]]
    pairs.sort(key=lambda name: (joint_order.index(name) if name in joint_order else len(joint_order), name))
    if pairs:
        return [label for name in pairs[:max(limit // 2, 1)] for label in (states[name], actions[name])]
    return list(states.values())[:max(limit // 2, 1)] + list(actions.values())[:max(limit // 2, 1)]


def ordered_signal_labels(state_names, action_names):
    """Place genuine counterparts next to each other in controls and legends."""
    groups = {}
    for prefix, names in (("S", state_names), ("A", action_names)):
        for name in names:
            groups.setdefault(_signal_identity(name), []).append(f"{prefix}:{name}")
    return [label for labels in groups.values() for label in labels]



def _median_period_ns(times):
    if len(times) < 2:
        return None
    diffs = np.diff(np.array(times, dtype=np.int64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def _alignment_tolerance_ns(source_times, target_times):
    periods = [p for p in (_median_period_ns(source_times), _median_period_ns(target_times)) if p]
    if not periods:
        return 50_000_000  # 50 ms fallback for sparse/short topics.
    return int(max(periods) * GAP_TOLERANCE_MULTIPLIER)


def _nearest_alignment(source_times, target_times, tolerance_ns):
    """返回 (idx, delta_ms, valid, raw_idx, raw_delta_ms)。
    idx/delta_ms 是超过容差就置为无效(-1/NaN)的版本，用于对齐 state/action 数值。
    raw_idx/raw_delta_ms 是不做容差过滤的"最近的那一帧"，即使超出容差(判定为 missing)
    也保留下来，用于在 img missing 时仍然显示"最近是第几帧、偏差多少 ms"这个 index 信息，
    方便判断到底是真丢帧还是频率本来就不一样。"""
    target_times = np.array(target_times, dtype=np.int64)
    if len(source_times) == 0:
        n = len(target_times)
        idx = np.full(n, -1, dtype=int)
        delta_ms = np.full(n, np.nan, dtype=float)
        valid = np.zeros(n, dtype=bool)
        raw_idx = np.full(n, -1, dtype=int)
        raw_delta_ms = np.full(n, np.nan, dtype=float)
        return idx, delta_ms, valid, raw_idx, raw_delta_ms
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


def _header_stamp_ns(msg):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    sec = getattr(stamp, "sec", None) if stamp is not None else None
    nanosec = getattr(stamp, "nanosec", None) if stamp is not None else None
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def _message_time_ns(ros_msg, message):
    return _header_stamp_ns(ros_msg) or int(message.log_time)


def _missing_runs(missing_mask):
    """在逐 step 的布尔 missing 掩码上找连续 True 区间。
    两端天然覆盖 step 0 / step T-1，无需特殊处理开头结尾。"""
    mask = np.asarray(missing_mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [{"x0": int(s), "x1": int(e), "drop_frames": int(e - s + 1)} for s, e in zip(starts, ends)]


def _build_reference_timeline(head_times, other_topics_times, tolerance_ns, nominal_period_ns):
    """补全 head 相机的时间戳，返回 (master_times, is_real)。
    1. 跟其它所有已加载 topic 的时间戳比较，找出全局最早/最晚时间；如果 head 的头/尾
       比这个范围晚开始/早结束超过 tolerance_ns，按 nominal_period_ns 往外插值补齐——
       head 自己开头/结尾丢帧，靠其它 topic 的证据发现并补上。
    2. head 自己相邻真实消息之间的间隔超过 tolerance_ns 的，同样按 nominal_period_ns
       在 gap 内部均匀插入虚拟时间点——head 自己的中段丢帧。
    3. 真实时间戳 + 两类插值时间戳一起排序，返回 master_times 和对应的 is_real 掩码。
    """
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
    combined_times = np.concatenate([head_arr, virtual_arr])
    combined_is_real = np.concatenate(
        [
            np.ones(len(head_arr), dtype=bool),
            np.zeros(len(virtual_arr), dtype=bool),
        ]
    )
    order = np.argsort(combined_times, kind="stable")
    return combined_times[order], combined_is_real[order]


def _health_row(times, spans, nominal_hz):
    """topic 自身节奏统计（_topic_stats）+ 外部传入的缺帧 spans + 相对标称频率的整体达标率。"""
    base = _topic_stats(times)
    base["drop_gaps"] = len(spans)
    base["drop_frames"] = int(sum(s["drop_frames"] for s in spans))
    base["coverage"] = (base["hz"] / nominal_hz) if nominal_hz else float("nan")
    return base


def _add_gap_lines(fig, spans, y0=0.05, y1=0.95, color="rgb(220, 38, 38)", label_prefix="", x_values=None):
    """给一批已合并的丢帧 span 各画一条纵向细线（不铺满整个 y 轴）+ 丢帧数标注。

    y0/y1 是 paper 坐标（跟曲线的实际数值 y 轴无关），配合不同的 y0/y1 可以把
    多组 span（比如不同 topic、不同相机）分到各自独立的横向"车道"里，互不遮挡。
    x_values 可选：把 span 的 x0/x1（step 索引）映射成别的 x 坐标（比如真实时间）。
    """
    for span in spans:
        if x_values is not None:
            x = (x_values[span["x0"]] + x_values[span["x1"]]) / 2
        else:
            x = (span["x0"] + span["x1"]) / 2
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=y0,
            y1=y1,
            yref="paper",
            line=dict(color=color, width=2, dash="dot"),
        )
        fig.add_annotation(
            x=x,
            y=y1,
            yref="paper",
            yanchor="bottom",
            showarrow=False,
            text=f"⚠ {label_prefix}{span['drop_frames']}帧",
            font=dict(size=10, color=color),
        )


def _topic_stats(times):
    """topic 自己的发布节奏统计；drop_gaps/drop_frames 由 _health_row 统一补充。"""
    if len(times) == 0:
        return {"count": 0, "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    if len(times) == 1:
        return {"count": 1, "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    arr = np.array(times, dtype=np.int64)
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return {"count": len(times), "hz": 0.0, "median_ms": np.nan, "max_gap_ms": np.nan}
    median_ns = float(np.median(diffs))
    duration_s = max((arr[-1] - arr[0]) / 1e9, 1e-9)
    return {
        "count": len(times),
        "hz": (len(times) - 1) / duration_s,
        "median_ms": median_ns / 1e6,
        "max_gap_ms": float(np.max(diffs)) / 1e6,
    }


@st.cache_data(max_entries=2, show_spinner="Decoding MCAP episode...")
def load_episode(file_signatures, camera_topics: dict, state_topics: tuple, action_topics: tuple):
    camera_data = {name: [] for name in camera_topics}
    camera_times = {name: [] for name in camera_topics}
    signal_series = {}
    wanted_topics = set(camera_topics.values()) | set(state_topics) | set(action_topics)
    if COMMAND_TOPICS[1] in wanted_topics:
        wanted_topics.add(NAMES_TOPIC)
    read_issues = []
    topic_results = {topic: {"failed": 0, "error": ""}
                     for topic in wanted_topics}
    camera_names = {topic: name for name, topic in camera_topics.items()}
    command_decoder = CommandDecoder()
    stale_commands = 0
    for path, _, _ in file_signatures:
        # Schema IDs are local to each file; do not share decoder caches across shards.
        factory = DecoderFactory()
        for schema, channel, message in _iter_mcap_messages(path, topics=list(wanted_topics), issues=read_issues):
            topic = channel.topic
            result = topic_results[topic]
            try:
                decoder = factory.decoder_for(channel.message_encoding, schema)
                if decoder is None:
                    raise ValueError("不支持的消息或 schema 编码")
                ros_msg = decoder(message.data)
                t_ns = _message_time_ns(ros_msg, message)
                if topic == NAMES_TOPIC or topic in COMMAND_TOPICS:
                    if topic != NAMES_TOPIC and abs(t_ns - message.log_time) > 1_000_000_000:
                        # Transient-local statistics include one old sample on subscription.
                        # These recordings contain samples 5–31 seconds before the episode.
                        stale_commands += 1
                        continue
                    for source_topic, source_time, flat in command_decoder.consume(topic, ros_msg, t_ns):
                        if flat:
                            signal_series.setdefault(source_topic, []).append((source_time, flat))
                elif topic in camera_names:
                    img = _decode_image(ros_msg)
                    if img is None:
                        raise ValueError("图像解码失败")
                    cam_name = camera_names[topic]
                    camera_data[cam_name].append(img)
                    camera_times[cam_name].append(t_ns)
                else:
                    flat = joint_positions(ros_msg) if topic == JOINT_TOPIC else _flatten(ros_msg)
                    signal_series.setdefault(topic, []).append((t_ns, flat))
            except Exception as exc:
                # A bad/custom topic must not suppress all other recorded signals.
                result["failed"] += 1
                result["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
    if command_decoder.pending:
        read_issues.append("部分关节指令缺少匹配的 StatisticsNames，未按数组下标猜测关节名："
                           + ", ".join(map(str, command_decoder.pending)))
    if stale_commands:
        read_issues.append(f"已排除 {stale_commands} 条订阅时带入的旧控制器统计消息，避免拉长录制时间轴。")
    # log_time_order=False 按物理写入顺序返回消息，正常情况下同一个 topic 内部
    # 已经是按时间递增的，但仍显式排序，供 np.searchsorted 对齐使用。
    for cam_name in list(camera_times.keys()):
        times = camera_times[cam_name]
        if len(times) == 0:
            continue
        order = np.argsort(times)
        camera_times[cam_name] = list(np.array(times)[order])
        camera_data[cam_name] = [camera_data[cam_name][i] for i in order]
    for topic in list(signal_series.keys()):
        signal_series[topic].sort(key=lambda item: item[0])
    non_empty_cams = {k: v for k, v in camera_times.items() if len(v) > 0}
    reference_camera = "head" if "head" in non_empty_cams else next(iter(non_empty_cams), None)
    if reference_camera is not None:
        head_times_raw = camera_times[reference_camera]
    else:
        non_empty_signals = {topic: series for topic, series in signal_series.items()
                             if any(flat for _, flat in series)}
        if not non_empty_signals:
            return {"total_step": 0, "read_issues": read_issues, "topic_results": topic_results}
        reference_signal = max(non_empty_signals, key=lambda topic: len(non_empty_signals[topic]))
        head_times_raw = sorted({t for t, _ in non_empty_signals[reference_signal]})
    other_times = [
        np.array(times, dtype=np.int64)
        for name, times in camera_times.items()
        if name != reference_camera and len(times) > 0
    ]
    other_times += [
        np.array([t for t, _ in series], dtype=np.int64) for series in signal_series.values() if series
    ]
    head_self_tolerance_ns = _alignment_tolerance_ns(head_times_raw, head_times_raw)
    # 插值补点用 head 自己实际的中位数周期，不用固定的标称 30Hz——否则实际帧率低于
    # 标称值的相机，每个大 gap 算出来的"丢了几帧"会被系统性高估（标称/实际 的倍数）。
    # NOMINAL_HZ_CAMERA 只留给下面 coverage 列用，不再兼职当插值密度。
    head_native_period_ns = _median_period_ns(head_times_raw) or round(1e9 / NOMINAL_HZ_CAMERA)
    master_times, head_is_real = _build_reference_timeline(
        head_times_raw, other_times, head_self_tolerance_ns, head_native_period_ns
    )
    T = len(master_times)
    # 其他相机按最近邻对齐到参考时钟；超过容差认为该 step 缺图，不复用旧帧。
    # head 自己的 valid 不用最近邻容差重新判断，直接采用 _build_reference_timeline 算出的
    # is_real——否则大 gap 内部第一个插值点可能因为离真实帧很近而被误判为"有效"。
    aligned_cameras = {}
    camera_indices = {}
    camera_delta_ms = {}
    camera_valid = {}
    camera_raw_indices = {}
    camera_raw_delta_ms = {}
    for cam_name, times in camera_times.items():
        tolerance_ns = _alignment_tolerance_ns(times, master_times)
        idx, delta_ms, valid, raw_idx, raw_delta_ms = _nearest_alignment(times, master_times, tolerance_ns)
        if cam_name == reference_camera:
            valid = head_is_real
            idx = np.where(valid, raw_idx, -1)
            delta_ms = np.where(valid, raw_delta_ms, np.nan)
        aligned_cameras[cam_name] = [
            camera_data[cam_name][i] if is_valid else None for i, is_valid in zip(idx, valid)
        ]
        camera_indices[cam_name] = [int(i) if is_valid else None for i, is_valid in zip(idx, valid)]
        camera_delta_ms[cam_name] = [float(v) if is_valid else None for v, is_valid in zip(delta_ms, valid)]
        camera_valid[cam_name] = [bool(v) for v in valid]
        # missing 时也保留"离哪一帧最近、偏差多少 ms"，方便区分是真丢帧还是本来频率就不同
        camera_raw_indices[cam_name] = [int(i) for i in raw_idx]
        camera_raw_delta_ms[cam_name] = [float(v) for v in raw_delta_ms]

    def build_matrix(topics):
        names, columns = [], []
        masks = {}
        for topic in topics:
            series = signal_series.get(topic, [])
            if not series:
                continue
            times_arr = np.array([t for t, _ in series], dtype=np.int64)
            all_keys = sorted({key for _, flat in series for key in flat.keys()})
            tolerance_ns = _alignment_tolerance_ns(times_arr, master_times)
            idx, _, valid, _, _ = _nearest_alignment(times_arr, master_times, tolerance_ns)
            masks[topic] = valid
            for key in all_keys:
                names.append(f"{topic}.{key}" if key else topic)
                raw_vals = np.array([flat.get(key, np.nan) for _, flat in series], dtype=float)
                aligned_vals = np.full(T, np.nan, dtype=float)
                valid_idx = valid & (idx >= 0)
                aligned_vals[valid_idx] = raw_vals[idx[valid_idx]]
                columns.append(aligned_vals)
        if not columns:
            return np.zeros((T, 0)), [], masks
        return np.stack(columns, axis=1), names, masks

    state, state_names, state_masks = build_matrix(state_topics)
    action, action_names, action_masks = build_matrix(action_topics)
    signal_masks = {**state_masks, **action_masks}
    camera_stats = {}
    camera_gap_spans = {}
    for name in camera_times:
        if not camera_times[name]:
            continue  # An absent camera is not a whole episode of dropped images.
        valid_mask = np.array(camera_valid[name], dtype=bool)
        spans = _missing_runs(~valid_mask)
        camera_stats[f"camera:{name}"] = _health_row(camera_times[name], spans, NOMINAL_HZ_CAMERA)
        camera_gap_spans[name] = spans
    signal_stats = {}
    signal_gap_spans = {}
    for topic, series in signal_series.items():
        times = [t for t, _ in series]
        valid_mask = signal_masks.get(topic)
        spans = _missing_runs(~valid_mask) if valid_mask is not None else []
        signal_stats[topic] = _health_row(times, spans, NOMINAL_HZ_JOINT)
        if spans:
            all_keys = sorted({key for _, flat in series for key in flat.keys()})
            if all_keys:
                for key in all_keys:
                    signal_gap_spans[f"{topic}.{key}"] = spans
            else:
                signal_gap_spans[topic] = spans
    return {
        "read_issues": read_issues,
        "topic_results": topic_results,
        "cameras": aligned_cameras,
        "camera_indices": camera_indices,
        "camera_delta_ms": camera_delta_ms,
        "camera_valid": camera_valid,
        "camera_raw_indices": camera_raw_indices,
        "camera_raw_delta_ms": camera_raw_delta_ms,
        "camera_counts": {name: len(camera_times[name]) for name in camera_times},
        "camera_gap_spans": camera_gap_spans,
        "master_times": master_times,
        "stats": {**camera_stats, **signal_stats},
        "signal_gap_spans": signal_gap_spans,
        "state": state,
        "action": action,
        "state_names": state_names,
        "action_names": action_names,
        "total_step": T,
    }


# =========================================================
# Sidebar: data source
# =========================================================
st.sidebar.title("Data Source")
root_dir = st.sidebar.text_input("mcap root dir", DEFAULT_ROOT)
episodes, ep2path = list_episodes(root_dir)
if not episodes:
    st.error(f"在 {root_dir} 下没有找到 */recording/*.mcap 文件")
    st.stop()
episode_id = st.sidebar.selectbox("Select Episode", episodes)
mcap_paths = ep2path[episode_id]
file_signatures = tuple((path, os.stat(path).st_size, os.stat(path).st_mtime_ns) for path in mcap_paths)
for path in mcap_paths:
    st.sidebar.caption(path)
with st.sidebar.expander("Topic 配置", expanded=False):
    st.caption("S 读取 /joint_states 的位置反馈；A 读取控制器统计中的 command_interface 位置指令。")
    left_wrist_topic = st.text_input("left_wrist topic", DEFAULT_CAMERA_TOPICS["left_wrist"])
    head_topic = st.text_input("head topic", DEFAULT_CAMERA_TOPICS["head"])
    right_wrist_topic = st.text_input("right_wrist topic", DEFAULT_CAMERA_TOPICS["right_wrist"])
    state_topics_str = st.text_input("state topics (逗号分隔)", ",".join(DEFAULT_STATE_TOPICS))
    action_topics_str = st.text_input("action topics (逗号分隔)", ",".join(DEFAULT_ACTION_TOPICS))
camera_topics = {name: topic.strip() for name, topic in (
    ("left_wrist", left_wrist_topic), ("head", head_topic), ("right_wrist", right_wrist_topic)
) if topic.strip()}
state_topics = tuple(dict.fromkeys(t.strip() for t in state_topics_str.split(",") if t.strip()))
action_topics = tuple(dict.fromkeys(t.strip() for t in action_topics_str.split(",") if t.strip()))
if set(state_topics) & set(action_topics):
    st.error("state 和 action 配置了相同 topic，请分别指定实际反馈与控制器指令来源。")
    st.stop()
data = load_episode(file_signatures, camera_topics, state_topics, action_topics)
for issue in dict.fromkeys(data["read_issues"]):
    st.warning(issue)
for topic, result in data["topic_results"].items():
    if result["failed"]:
        st.warning(f"{topic}：{result['failed']} 条消息解析失败，{result['error']}")
if not data["total_step"]:
    st.error("没有可回放的图像或数值信号，请检查 Topic 配置。")
    st.stop()
state = data["state"]
action = data["action"]
state_names = data["state_names"]
action_names = data["action_names"]
if not action_names:
    st.warning("没有解析到关节位置指令，请检查控制器统计 topic；不会用 state 代替 Action。")
T = data["total_step"]
cameras = data["cameras"]
camera_indices = data["camera_indices"]
camera_delta_ms = data["camera_delta_ms"]
camera_raw_indices = data.get("camera_raw_indices", {})
camera_raw_delta_ms = data.get("camera_raw_delta_ms", {})
camera_counts = data["camera_counts"]
camera_gap_spans = data.get("camera_gap_spans", {})
master_times = data["master_times"]
stats = data["stats"]
signal_gap_spans = data.get("signal_gap_spans", {})
# =========================================================
# Episode changes reset the cursor, while the saved semantic signal IDs survive.
# =========================================================
episode_key = repr((root_dir, episode_id, file_signatures))
if st.session_state.get("current_episode_key") != episode_key:
    st.session_state.current_episode_key = episode_key
    st.session_state.t = 0
    st.session_state.mcap_step = 0
st.session_state.t = min(st.session_state.get("t", 0), max(T - 1, 0))

st.markdown("## 流畅播放")
D_state = state.shape[1]
D_action = action.shape[1]
labels = ordered_signal_labels(state_names, action_names)
data_labels = labels_with_data(state, state_names, action, action_names)
default_dims = [label for label in default_selected_dims(state_names, action_names)
                if label in data_labels] or data_labels[:8]
selected_dims = select_signal_dimensions(
    labels, default_dims, key="mcap_selected_dims", display_func=display_label,
    previous=st.session_state.get("selected_dims"),
)
st.session_state.selected_dims = list(selected_dims)
st.caption(f"S 实线：实测关节位置（{D_state} 维）；A 浅色点线：控制器关节位置指令（{D_action} 维）。同名关节配对。")
identities = list(dict.fromkeys(_signal_identity(name) for name in [*state_names, *action_names]))
signal_colors = {name: SIGNAL_PALETTE[i % len(SIGNAL_PALETTE)] for i, name in enumerate(identities)}
legend_order = {label: index for index, label in enumerate(labels)}


def line_style(name, action=False):
    color = signal_colors[_signal_identity(name)]
    return dict(color=_lighter_hex_color(color) if action else color,
                dash="dot" if action else "solid", width=2)


signal_values = {f"S:{name}": state[:, i] for i, name in enumerate(state_names)}
signal_values.update({f"A:{name}": action[:, i] for i, name in enumerate(action_names)})
player_traces = [
    {"name": display_label(label), "color": line_style(label[2:], label.startswith("A:"))["color"],
     "dashed": label.startswith("A:"), "values": signal_values[label]}
    for label in labels if label in selected_dims
]
show_mcap_smooth_player((file_signatures, tuple(camera_topics.items())), cameras, master_times,
                        player_traces, st.session_state.t)


@st.fragment
def _inspection_fragment():
    st.markdown("### 逐帧查看")

    def step_frame(delta):
        st.session_state.t = min(max(st.session_state.t + delta, 0), T - 1)
        st.session_state.mcap_step = st.session_state.t

    previous, following = st.columns(2)
    previous.button("← Previous frame", key="mcap_previous_frame", on_click=step_frame,
                    args=(-1,), disabled=st.session_state.t <= 0, shortcut="Left", width="stretch")
    following.button("Next frame →", key="mcap_next_frame", on_click=step_frame,
                     args=(1,), disabled=st.session_state.t >= T - 1, shortcut="Right", width="stretch")
    st.caption("上方可连续播放关节曲线；这里可以拖动 Step 或用 ←/→ 逐帧细看。")
    if T > 1:
        st.session_state.t = st.slider("Step", 0, T - 1, key="mcap_step")
    else:
        st.caption("Step 0（仅一帧）")
    t = st.session_state.t
    camera_names = [name for name in cameras if camera_counts.get(name, 0) > 0]
    camera_columns = st.columns(min(3, len(camera_names))) if camera_names else []
    for index, cam_name in enumerate(camera_names):
        col = camera_columns[index % len(camera_columns)]
        with col:
            frames = cameras.get(cam_name, [None] * T)
            img = frames[t] if T > 0 and t < len(frames) else None
            raw_idx = camera_indices.get(cam_name, [None] * T)[t] if T > 0 else None
            total_raw = camera_counts.get(cam_name, 0)
            if img is not None and raw_idx is not None:
                st.image(
                    img,
                    caption=f"{cam_name} index {raw_idx + 1}/{total_raw} ({IMAGE_RESIZE_SIZE[0]}x{IMAGE_RESIZE_SIZE[1]})",
                    output_format="JPEG",
                )
            elif img is not None:
                st.image(
                    img,
                    caption=f"{cam_name} index unknown/{total_raw} ({IMAGE_RESIZE_SIZE[0]}x{IMAGE_RESIZE_SIZE[1]})",
                    output_format="JPEG",
                )
            else:
                # missing：即使超出对齐容差没有可用帧，也把"最近是第几帧、偏差多少 ms"标出来，
                # 这样能区分是真丢帧，还是这个相机本身采集频率就比 master 低。
                nearest_idx_list = camera_raw_indices.get(cam_name, [])
                nearest_delta_list = camera_raw_delta_ms.get(cam_name, [])
                nearest_idx = nearest_idx_list[t] if T > 0 and t < len(nearest_idx_list) else None
                nearest_delta = nearest_delta_list[t] if T > 0 and t < len(nearest_delta_list) else None
                if nearest_idx is not None and nearest_idx >= 0 and nearest_delta is not None:
                    st.warning(
                        f"{cam_name} missing at step {t} "
                        f"(nearest raw frame index {nearest_idx + 1}/{total_raw}, {nearest_delta:+.1f} ms off)"
                    )
                else:
                    st.warning(f"{cam_name} missing at step {t} (no frames recorded, total {total_raw})")

    # Plotly View (joint vis) — 一张图，但每个 topic 的丢帧竖线各占独立的颜色+横向车道，
    # 避免"哪个 topic 在这段时间丢了"被混在一条通用红线里看不出来。合并到上面
    # "img/state/action" 一个标题下，这里不再单独起标题。
    # =========================================================

    def _label_topic(label):
        return label[2:].split(".", 1)[0]

    topics_in_view = [
        tpc for tpc in (list(state_topics) + list(action_topics)) if any(_label_topic(l) == tpc for l in selected_dims)
    ]
    _GAP_COLOR_PALETTE = [
        "rgb(220, 38, 38)",
        "rgb(37, 99, 235)",
        "rgb(217, 119, 6)",
        "rgb(5, 150, 105)",
        "rgb(171, 139, 204)",
    ]
    n_lanes = max(len(topics_in_view), 1)
    lane_height = (0.95 - 0.05) / n_lanes
    topic_style = {
        tpc: {
            "y0": 0.05 + i * lane_height,
            "y1": 0.05 + (i + 1) * lane_height,
            "color": _GAP_COLOR_PALETTE[i % len(_GAP_COLOR_PALETTE)],
        }
        for i, tpc in enumerate(topics_in_view)
    }

    gap_spans_by_topic = {}
    for label in selected_dims:
        if not label.startswith(("S:", "A:")):
            continue
        series_name = label[2:]
        tpc = _label_topic(label)
        seen_in_topic = {(s["x0"], s["x1"]) for s in gap_spans_by_topic.get(tpc, [])}
        for span in signal_gap_spans.get(series_name, [])[:MAX_RENDERED_GAPS]:
            key = (span["x0"], span["x1"])
            if key in seen_in_topic:
                continue
            seen_in_topic.add(key)
            gap_spans_by_topic.setdefault(tpc, []).append(span)

    plot_x = np.arange(T)
    fig_plotly = go.Figure()
    for i in range(D_state):
        label = f"S:{state_names[i]}"
        if label in selected_dims:
            fig_plotly.add_trace(go.Scatter(x=plot_x, y=state[:, i], name=display_label(label),
                                           line=line_style(state_names[i]), connectgaps=False,
                                           legendrank=legend_order[label]))
    for i in range(D_action):
        label = f"A:{action_names[i]}"
        if label in selected_dims:
            fig_plotly.add_trace(go.Scatter(x=plot_x, y=action[:, i], name=display_label(label),
                                           line=line_style(action_names[i], action=True), connectgaps=False,
                                           legendrank=legend_order[label]))
    for tpc, spans in gap_spans_by_topic.items():
        style = topic_style[tpc]
        _add_gap_lines(
            fig_plotly,
            spans[:MAX_RENDERED_GAPS],
            y0=style["y0"],
            y1=style["y1"],
            color=style["color"],
            label_prefix=f"{tpc} ",
        )
    fig_plotly.add_vline(x=t, line_color="red", line_width=2)
    fig_plotly.update_layout(
        height=350,
        dragmode="select",
        xaxis_title="Step",
        hovermode="x",
    )
    fig_plotly.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    show_signal_plot(fig_plotly, key="mcap_signal_plot")


_inspection_fragment()

# =========================================================
# 拆分 Data Health -> Image Health / Joint Health
# =========================================================
health_rows = []
for topic, item in sorted(stats.items()):
    median_ms = item["median_ms"]
    max_gap_ms = item["max_gap_ms"]
    health_rows.append(
        {
            "topic": topic,
            "count": item["count"],
            "hz": round(item["hz"], 2),
            "median_ms": None if np.isnan(median_ms) else round(float(median_ms), 1),
            "max_gap_ms": None if np.isnan(max_gap_ms) else round(float(max_gap_ms), 1),
            "drop_gaps": item["drop_gaps"],
            "drop_frames": item["drop_frames"],
            "coverage": f"{item['coverage'] * 100:.0f}%" if not np.isnan(item["coverage"]) else None,
        }
    )
image_health_rows = [r for r in health_rows if r["topic"].startswith("camera:")]
joint_health_rows = [r for r in health_rows if not r["topic"].startswith("camera:")]

st.markdown("#### Image Health")
if image_health_rows:
    st.dataframe(
        image_health_rows,
        width="stretch",
        hide_index=True,
        height=min(1200, 38 + 35 * len(image_health_rows)),
    )
else:
    st.caption("本段没有解析到配置的相机图像。")

# 只给数字看不到位置，没法去 Playback 的 Step 滑条对着核实——补一张丢帧位置明细表，
# 不用图表（避免重蹈"图和表对不上"的观感问题），直接给 step 区间，方便拖到那一段验证。
image_gap_rows = [
    {"camera": cam_name, "step_range": f"{span['x0']}-{span['x1']}", "drop_frames": span["drop_frames"]}
    for cam_name in camera_gap_spans
    for span in sorted(camera_gap_spans[cam_name], key=lambda s: -s["drop_frames"])
]
if image_gap_rows:
    st.markdown("#### Image Gap 明细（按丢帧数从大到小；把 Playback 的 Step 拖到这个区间可核实）")
    st.dataframe(
        image_gap_rows,
        width="stretch",
        hide_index=True,
        height=min(600, 38 + 35 * max(len(image_gap_rows), 1)),
    )

# =========================================================
# Signal Health
# =========================================================
st.markdown("## Signal Health")
st.dataframe(
    joint_health_rows,
    width="stretch",
    hide_index=True,
    height=min(1200, 38 + 35 * max(len(joint_health_rows), 1)),
)

# 跟 Image Gap 明细一样：只给数字看不到位置，没法去 Playback 的 Step 滑条对着核实。
# signal_gap_spans 是按字段(x/y/z)存的，同一个 topic 下多个字段共享同一份 span，
# 这里按 topic 去重，避免同一段 gap 因为选中了 x/y/z 三个字段而重复出现三次。
joint_gap_by_topic = {}
for field_key, spans in signal_gap_spans.items():
    topic = field_key.split(".", 1)[0]
    seen_in_topic = {(s["x0"], s["x1"]) for s in joint_gap_by_topic.get(topic, [])}
    for span in spans:
        key = (span["x0"], span["x1"])
        if key in seen_in_topic:
            continue
        seen_in_topic.add(key)
        joint_gap_by_topic.setdefault(topic, []).append(span)
joint_gap_rows = [
    {"topic": topic, "step_range": f"{span['x0']}-{span['x1']}", "drop_frames": span["drop_frames"]}
    for topic in joint_gap_by_topic
    for span in sorted(joint_gap_by_topic[topic], key=lambda s: -s["drop_frames"])
]
if joint_gap_rows:
    st.markdown("#### Joint Gap 明细（按丢帧数从大到小；把 Playback 的 Step 拖到这个区间可核实）")
    st.dataframe(
        joint_gap_rows,
        width="stretch",
        hide_index=True,
        height=min(600, 38 + 35 * max(len(joint_gap_rows), 1)),
    )
