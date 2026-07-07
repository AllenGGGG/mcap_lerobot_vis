import argparse
import os
import glob
import struct
import sys
import time
from array import array
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
from mcap.reader import make_reader, NonSeekingReader
from mcap.exceptions import McapError, EndOfFile
from mcap_ros2.decoder import DecoderFactory

MCAP_READ_ERRORS = (McapError, EndOfFile, struct.error)
# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="VLA Episode Viewer (mcap)", layout="wide")


def _parse_args():
    """支持 `streamlit run app.py -- --data-path /your/data` 指定数据目录，
    不传就用默认值；用 parse_known_args 避免跟 Streamlit 自己的参数冲突。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/ssd/data", help="mcap root dir")
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
DEFAULT_STATE_TOPICS = ["/left_current_pose", "/right_current_pose"]
DEFAULT_ACTION_TOPICS = ["/left_current_target", "/right_current_target"]
# 相机画面统一 resize 到的尺寸 (width, height)，VLA 模型输入常用 224x224
IMAGE_RESIZE_SIZE = (224, 224)
# "Select state / action dims" 多选框默认选中哪些 topic 下的维度
DEFAULT_DIM_TOPIC_PREFIXES = ("/left_current_pose", "/right_current_pose")
# 标称频率：仅用于 Data Health 表里的"整体是否达标" coverage 对比，不参与逐 step 缺帧判定
# （逐 step 判定继续用各 topic 自己的自适应容差，见 _alignment_tolerance_ns）
NOMINAL_HZ_CAMERA = 30
NOMINAL_HZ_JOINT = 100
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
    episodes, ep2path = [], {}
    for p in paths:
        ep_id = Path(p).parents[1].name  # .../<episode_id>/recording/xxx.mcap
        episodes.append(ep_id)
        ep2path[ep_id] = p
    return episodes, ep2path


def _sequence_items(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, array):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _flatten_joint_state(msg, prefix=""):
    names = _sequence_items(getattr(msg, "name", None))
    if names is None:
        return None
    out = {}
    for field in ("position", "velocity", "effort"):
        values = _sequence_items(getattr(msg, field, None))
        if values is None:
            continue
        for i, value in enumerate(values):
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            joint_name = str(names[i]) if i < len(names) and names[i] else f"[{i}]"
            key = f"{field}.{joint_name}" if not joint_name.startswith("[") else f"{field}{joint_name}"
            out[f"{prefix}.{key}" if prefix else key] = float(value)
    return out or None


def _flatten(msg, prefix=""):
    """递归展开一个 decode 后的 ros2 消息为 {字段路径: float} 的字典。
    自动跳过字符串/字节等非数值字段（如 frame_id）。"""
    out = {}
    if isinstance(msg, bool):
        return out
    if isinstance(msg, (int, float, np.integer, np.floating)):
        out[prefix] = float(msg)
        return out
    if isinstance(msg, (bytes, str)):
        return out
    if isinstance(msg, (list, tuple, np.ndarray, array)):
        for i, v in enumerate(msg):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    joint_state = _flatten_joint_state(msg, prefix)
    if joint_state is not None:
        return joint_state
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
        if k.startswith("_"):
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


def _label_matches_topic_prefixes(label, prefixes):
    name = label[2:]  # 去掉 "S:"/"A:" 前缀，还原成 topic.field 形式
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def default_selected_dims(state_names, action_names, limit=8):
    labels = [f"S:{n}" for n in state_names] + [f"A:{n}" for n in action_names]
    # 优先选中 /left_current_pose、/right_current_pose 下的维度
    preferred = [
        label for label in labels if _label_matches_topic_prefixes(label, DEFAULT_DIM_TOPIC_PREFIXES)
    ]
    if preferred:
        return preferred
    # 兜底：如果这两个 topic 都没有数据（比如 topic 配置改了），退回到原来
    # "优先 joint_states.position、否则 joint_states 其它字段、否则任意 state"的逻辑，
    # 保证多选框不会因为匹配不到而完全空着。
    state_labels = [f"S:{n}" for n in state_names]
    action_labels = [f"A:{n}" for n in action_names]
    joint_positions = [label for label in state_labels if label.startswith("S:/joint_states.position")]
    joint_dims = [label for label in state_labels if label.startswith("S:/joint_states.")]
    selected = []
    for label in (joint_positions or joint_dims or state_labels)[: max(limit // 2, 1)]:
        selected.append(label)
    for label in action_labels[: max(limit - len(selected), 1)]:
        selected.append(label)
    return selected[:limit]


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


def _is_xyzw_display_field(key):
    """用于 "Select state / action dims" 多选框的过滤：只显示 x/y/z/w 分量。"""
    return key in ("x", "y", "z", "w") or key.endswith((".x", ".y", ".z", ".w"))


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


@st.cache_data(max_entries=2, show_spinner="Decoding mcap file...")
def load_episode(mcap_path, camera_topics: dict, state_topics: tuple, action_topics: tuple):
    parser_cache_version = 11
    camera_data = {name: [] for name in camera_topics}
    camera_times = {name: [] for name in camera_topics}
    signal_series = {}  # topic -> [(t_ns, flat_dict), ...]
    wanted_topics = set(camera_topics.values()) | set(state_topics) | set(action_topics)
    # 部分 mcap 文件在录制中被强制中断（进程被杀 / 容器退出），末尾缺少 Footer，
    # 用默认的 make_reader（依赖文件末尾索引）会报 RecordLengthLimitExceeded。
    # 这里先尝试正常读取，失败则自动降级为 NonSeekingReader 做线性顺序扫描
    # （不依赖 Footer，能读多少算多少，直到遇到损坏的尾部为止）。
    used_fallback = False
    f = open(mcap_path, "rb")
    try:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        reader.get_summary()  # 提前触发一次索引读取，尽早暴露截断问题
        message_iter = reader.iter_decoded_messages(topics=list(wanted_topics))
    except MCAP_READ_ERRORS:
        f.close()
        used_fallback = True
        f = open(mcap_path, "rb")
        reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
        # log_time_order=False: 不做跨 chunk 按时间排序合并（那需要读取每个
        # chunk 后面的 MessageIndex 记录）。截断文件的最后一个 chunk 往往正是
        # 卡在 MessageIndex 没写完的地方，用物理顺序读取可以绕开这个依赖。
        message_iter = reader.iter_decoded_messages(topics=list(wanted_topics), log_time_order=False)
    try:
        for schema, channel, message, ros_msg in message_iter:
            t_ns = _message_time_ns(ros_msg, message)
            topic = channel.topic
            if topic in camera_topics.values():
                cam_name = [k for k, v in camera_topics.items() if v == topic][0]
                img = _decode_compressed_image(ros_msg)
                if img is not None:
                    camera_data[cam_name].append(img)
                    camera_times[cam_name].append(t_ns)
            else:
                flat = _flatten(ros_msg)
                signal_series.setdefault(topic, []).append((t_ns, flat))
    except MCAP_READ_ERRORS:
        # 文件在扫描过程中损坏截断，忽略之后的部分，用已经读到的数据继续
        pass
    finally:
        f.close()
    if used_fallback:
        st.warning(
            "⚠️ 这个 mcap 文件末尾缺少 Footer（很可能录制被强制中断），"
            "已自动切换为线性扫描模式读取，可能会丢失文件最后一小段数据。"
            "建议用 `mcap recover` 修复原文件。"
        )
    # log_time_order=False 按物理写入顺序返回消息，正常情况下同一个 topic 内部
    # 已经是按时间递增的，但这里显式排序一次以防万一，因为后面 np.searchsorted
    # 要求输入数组严格递增
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
    if not non_empty_cams:
        raise ValueError("没有解析到任何图像帧，检查左侧 Topic 配置里的相机 topic 名称是否正确")
    if "head" not in non_empty_cams:
        raise ValueError(
            "head 相机没有解析到任何帧，无法作为参考时钟，检查左侧 Topic 配置里的 head topic 名称是否正确"
        )
    head_times_raw = camera_times["head"]
    other_times = [
        np.array(times, dtype=np.int64)
        for name, times in camera_times.items()
        if name != "head" and len(times) > 0
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
        if cam_name == "head":
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
            tolerance_ns = _alignment_tolerance_ns(times_arr, master_times)
            idx, _, valid, _, _ = _nearest_alignment(times_arr, master_times, tolerance_ns)
            masks[topic] = valid
            all_keys = sorted({key for _, flat in series for key in flat.keys()})
            if not all_keys:
                continue
            for key in all_keys:
                names.append(f"{topic}.{key}" if key else topic)
                raw_vals = np.array([flat.get(key, 0.0) for _, flat in series], dtype=float)
                aligned_vals = np.zeros(T, dtype=float)
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
                # gap 是按整个 topic 的原生时间线算的，同一 topic 下所有字段共享同一份 span。
                for key in all_keys:
                    signal_gap_spans[f"{topic}.{key}"] = spans
            else:
                signal_gap_spans[topic] = spans
    return {
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
mcap_path = ep2path[episode_id]
st.sidebar.caption(mcap_path)
with st.sidebar.expander("Topic 配置", expanded=False):
    st.caption(
        "这里把页面里的相机/state/action 槽位映射到 MCAP 里的 ROS topic。"
        "录制 topic 名变了时在这里改；没填进来的 topic 不会被读取。"
    )
    left_wrist_topic = st.text_input("left_wrist topic", DEFAULT_CAMERA_TOPICS["left_wrist"])
    head_topic = st.text_input("head topic", DEFAULT_CAMERA_TOPICS["head"])
    right_wrist_topic = st.text_input("right_wrist topic", DEFAULT_CAMERA_TOPICS["right_wrist"])
    state_topics_str = st.text_input("state topics (逗号分隔)", ",".join(DEFAULT_STATE_TOPICS))
    action_topics_str = st.text_input("action topics (逗号分隔)", ",".join(DEFAULT_ACTION_TOPICS))
camera_topics = {
    "left_wrist": left_wrist_topic,
    "head": head_topic,
    "right_wrist": right_wrist_topic,
}
state_topics = tuple(t.strip() for t in state_topics_str.split(",") if t.strip())
action_topics = tuple(t.strip() for t in action_topics_str.split(",") if t.strip())
data = load_episode(mcap_path, camera_topics, state_topics, action_topics)
state = data["state"]
action = data["action"]
state_names = data["state_names"]
action_names = data["action_names"]
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
# Episode change detection
# =========================================================
if "current_episode_key" not in st.session_state:
    st.session_state.current_episode_key = episode_id
if st.session_state.current_episode_key != episode_id:
    st.session_state.current_episode_key = episode_id
    st.session_state.t = 0
    st.session_state.playing = False
    st.session_state.selected_dims = default_selected_dims(state_names, action_names)
    st.rerun()
# =========================================================
# Session state
# =========================================================
if "t" not in st.session_state:
    st.session_state.t = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "fps" not in st.session_state:
    st.session_state.fps = 30
st.session_state.t = min(st.session_state.t, T - 1) if T > 0 else 0
t = st.session_state.t
# =========================================================
# Playback + Cameras
# =========================================================
# 整段包成一个 fragment：Play 期间只有这部分（按钮/滑条/相机图片/自增 t）重跑，
# 不会每一帧都把下面的 Plotly 图表、Data Health 表格重新构建并整份推给远端浏览器——
# 这是远程运行时 Play 卡顿（尤其是图片）的主要原因，包进 fragment 后每帧只传三张小图。
@st.fragment
def _playback_fragment():
    st.markdown("## Img/state/action")
    # 用一个按钮在 Play/Pause 之间切换，而不是同时摆两个按钮——
    # 当前状态已经决定了哪个动作有意义，没必要两个都常驻显示。
    toggle_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing
    st.session_state.fps = st.slider("FPS", 5, 60, st.session_state.fps)
    t_manual = st.slider("Step", 0, max(T - 1, 0), st.session_state.t)
    if t_manual != st.session_state.t:
        st.session_state.t = t_manual
        st.session_state.playing = False
    t = st.session_state.t
    # =========================================================
    # State / Action selection
    # =========================================================
    D_state = state.shape[1] if state.ndim == 2 else 0
    D_action = action.shape[1] if action.ndim == 2 else 0
    all_dim_labels = [f"S:{n}" for n in state_names] + [f"A:{n}" for n in action_names]
    # "Select state / action dims" 下拉列表只保留 x/y/z/w 字段(比如 position.x/y/z、
    # orientation.x/y/z/w)，其它字段(比如 joint_states 的 position/velocity/effort)
    # 不再出现在这个列表里，避免选项被挤满。
    labels = [label for label in all_dim_labels if _is_xyzw_display_field(label[2:])]
    data_labels = [
        label for label in labels_with_data(state, state_names, action, action_names) if label in labels
    ]
    # 之前是 data_labels or default_selected_dims(...)：data_labels("所有有数据的维度")
    # 几乎总是非空，导致 default_selected_dims 里"优先选 left/right_current_pose"这个逻辑
    # 实际上永远不会生效。这里反过来，优先用 default_selected_dims 的结果，
    # 只有它也选不出东西时才退回到"随便选点有数据的维度"。
    default_dims = [
        label for label in default_selected_dims(state_names, action_names) if label in labels
    ] or data_labels
    multiselect_key = f"selected_dims_{st.session_state.current_episode_key}"
    multiselect_init_key = f"{multiselect_key}_all_data_initialized"
    # 可选字段的全集(labels)如果变了——比如改了 topic 配置、或者代码升级调整了过滤
    # 规则(像这次把选项收窄成只显示 x/y/z/w 字段)——旧的选择这时候可能只剩一两个
    # 凑巧还留在新列表里的字段，不能当成"用户特意只想要这个选择"，应该整体重置成
    # 新的默认值，而不是"能留几个是几个"（后者会导致只剩 1 条线还看不出是 bug）。
    labels_signature = tuple(sorted(labels))
    labels_changed = st.session_state.get("dim_labels_signature") != labels_signature
    st.session_state.dim_labels_signature = labels_signature
    if labels_changed:
        st.session_state.selected_dims = default_dims
        st.session_state[multiselect_key] = default_dims
        st.session_state[multiselect_init_key] = True
    else:
        # 注意：这里不能"选择变空就回填默认值"——用户手动把最后一个 tag 删掉、
        # 故意清空选择时，selected_dims/multiselect_key 也会变空，那是合法状态，
        # 不是 bug，不该被强制改回去。真正需要回填默认值的场景("topic 配置变了、
        # 选项被过滤掉了")已经在上面 labels_changed 分支单独处理。
        if "selected_dims" not in st.session_state:
            st.session_state.selected_dims = default_dims
        else:
            st.session_state.selected_dims = [
                label for label in st.session_state.selected_dims if label in labels
            ]
        if multiselect_key not in st.session_state or not st.session_state.get(multiselect_init_key):
            st.session_state[multiselect_key] = st.session_state.selected_dims
            st.session_state[multiselect_init_key] = True
        else:
            st.session_state[multiselect_key] = [
                label for label in st.session_state[multiselect_key] if label in labels
            ]
    # Play/Pause/FPS/Step 和这个 multiselect 放在一起，算作一个统一的控制区域，
    # 图片和图表都在它下面，用的是这里选出来的 selected_dims。
    selected_dims = st.multiselect(
        "Select state / action dims",
        labels,
        key=multiselect_key,
        disabled=st.session_state.playing,
    )
    st.session_state.selected_dims = selected_dims


    c1, c2, c3 = st.columns(3)
    for col, cam_name in [
        (c1, "left_wrist"),
        (c2, "head"),
        (c3, "right_wrist"),
    ]:
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
        "rgb(147, 51, 234)",
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
                # 同一 topic 下多个字段(x/y/z)共享同一份 gap，只画一次线，避免重叠的线叠加浪费。
                continue
            seen_in_topic.add(key)
            gap_spans_by_topic.setdefault(tpc, []).append(span)

    plot_x = np.arange(T)
    fig_plotly = go.Figure()
    # 没有数据的 step 在 build_matrix 里已经被置成了 0，这里直接画原始值就行——
    # 曲线会在缺数据的地方自然掉到 0，是否"缺数据"完全靠下面的竖线来标注，
    # 不再额外把这段区间 NaN 掉断线。
    for i in range(D_state):
        label = f"S:{state_names[i]}"
        if label in selected_dims:
            fig_plotly.add_trace(go.Scatter(x=plot_x, y=state[:, i], name=label))
    for i in range(D_action):
        label = f"A:{action_names[i]}"
        if label in selected_dims:
            fig_plotly.add_trace(go.Scatter(x=plot_x, y=action[:, i], name=label, line=dict(dash="dot")))
    # 竖线直接用真实的 x0/x1（master step）画，和曲线上被置 0 的缺数据区域用的
    # 是完全相同的坐标（同一份 valid_mask 算出来的），保证两者永远对得上。
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
    fig_plotly.update_layout(
        height=350,
        dragmode="select",
        xaxis_title="Step",
    )
    if st.session_state.playing:
        current_x = t
        fig_plotly.add_vline(x=current_x, line_color="red", line_width=2)
        fig_plotly.update_layout(hovermode=False)
        fig_plotly.update_xaxes(showspikes=False)
    else:
        fig_plotly.update_layout(hovermode="x")
        fig_plotly.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)
    st.plotly_chart(fig_plotly, use_container_width=True)

    # Play 期间的自增+rerun 放在 fragment 内部：st.rerun() 在 fragment 里默认只重跑
    # fragment 本身，不会带动全页重新执行（下面的图表/表格不受影响）。
    if st.session_state.playing and T > 0:
        time.sleep(1.0 / st.session_state.fps)
        st.session_state.t += 1
        if st.session_state.t >= T:
            st.session_state.t = T - 1
            st.session_state.playing = False
        st.rerun()


_playback_fragment()

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
st.dataframe(
    image_health_rows,
    use_container_width=True,
    hide_index=True,
    height=min(1200, 38 + 35 * max(len(image_health_rows), 1)),
)

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
        use_container_width=True,
        hide_index=True,
        height=min(600, 38 + 35 * max(len(image_gap_rows), 1)),
    )

# =========================================================
# Joint Health
# =========================================================
st.markdown("## Joint Health")
st.dataframe(
    joint_health_rows,
    use_container_width=True,
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
        use_container_width=True,
        hide_index=True,
        height=min(600, 38 + 35 * max(len(joint_gap_rows), 1)),
    )
