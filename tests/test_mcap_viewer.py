"""Match the full and names/values command formats found in real recordings."""
import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from mcap_ros2.writer import Writer
from streamlit.testing.v1 import AppTest

from mcap_joint_signals import COMMAND_TOPICS, NAMES_TOPIC

APP = Path(__file__).resolve().parents[1] / "mcap_vis.py"
CAMERA = "/camera_head/color/image_raw/compressed"
HEADER_DEPS = """
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
FULL = """std_msgs/Header header
pal_statistics_msgs/Statistic[] statistics
================================================================================
MSG: pal_statistics_msgs/Statistic
string name
float64 value
""" + HEADER_DEPS
NAMES = "std_msgs/Header header\nstring[] names\nuint32 names_version\n" + HEADER_DEPS
VALUES = "std_msgs/Header header\nfloat64[] values\nuint32 names_version\n" + HEADER_DEPS
JOINTS = ["body_joint1", "body_joint3", "body_joint4", "left_joint1", "right_joint1", "left_hand_index_joint1"]


def header(timestamp):
    return {"stamp": {"sec": timestamp // 10**9, "nanosec": timestamp % 10**9}, "frame_id": ""}


def write_recording(path, steps, mode="full", camera=True, missing_names=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        if min(steps) > 0:
            writer.register_msgdef("std_msgs/msg/String", "string data\n")
        joint = writer.register_msgdef("sensor_msgs/msg/JointState", "string[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n")
        image = writer.register_msgdef("sensor_msgs/msg/CompressedImage", "string format\nuint8[] data\n")
        full = writer.register_msgdef("pal_statistics_msgs/msg/Statistics", FULL)
        names_schema = writer.register_msgdef("pal_statistics_msgs/msg/StatisticsNames", NAMES)
        values_schema = writer.register_msgdef("pal_statistics_msgs/msg/StatisticsValues", VALUES)
        jpeg = cv2.imencode(".jpg", np.full((4, 4, 3), 100, dtype=np.uint8))[1].tobytes()
        command_names = [f"command_interface.{name}/position" for name in JOINTS]
        command_names += ["command_interface.left_joint1/position.is_limited", "state_interface.left_joint1/position", "topic_stats.counter"]
        for step in steps:
            timestamp = 10_000_000_000 + step * 33_333_333

            def write(topic, schema, message):
                def as_message(value):
                    if isinstance(value, dict):
                        return SimpleNamespace(**{key: as_message(item) for key, item in value.items()})
                    if isinstance(value, list):
                        return [as_message(item) for item in value]
                    return value
                writer.write_message(topic, schema, as_message(message), log_time=timestamp)

            if camera:
                write(CAMERA, image, {"format": "jpeg", "data": jpeg})
            if step in (0, 1, 2, 9, 10, 11):
                write("/joint_states", joint, {"name": JOINTS,
                      "position": [float(step + i * 20) for i in range(len(JOINTS))],
                      "velocity": [999.0] * len(JOINTS), "effort": [888.0] * len(JOINTS)})
            commands = [float(step + i * 20) + 0.25 for i in range(len(JOINTS))] + [1.0, -999.0, 12345.0]
            version = 7 if step < 6 else 8
            names, values = (command_names, commands) if version == 7 else (command_names[::-1], commands[::-1])
            if mode == "full":
                message = {"header": header(timestamp), "statistics": [
                    {"name": name, "value": value} for name, value in zip(names, values)]}
                if step == min(steps):
                    # Transient-local old sample must not extend the reference by 9 seconds.
                    write(COMMAND_TOPICS[0], full, {**message, "header": header(1_000_000_000)})
                write(COMMAND_TOPICS[0], full, message)
            elif mode == "split":
                write(COMMAND_TOPICS[1], values_schema, {"header": header(timestamp), "values": values, "names_version": version})
                if step in (min(steps), 6) and not missing_names:
                    # The first values message deliberately precedes its matching names.
                    write(NAMES_TOPIC, names_schema, {"header": header(1_000_000_000), "names": names, "names_version": version})
        writer.finish()


def trace_values(trace):
    values = trace["y"]
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.array(values, dtype=float)


class McapViewerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def open_app(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.sidebar.text_input[0].set_value(str(self.root)).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        return app

    def traces(self, app):
        return {trace["name"]: trace for trace in json.loads(app.get("plotly_chart")[0].proto.spec)["data"]}

    def check_joint_pairs(self, mode):
        folder = self.root / "episode" / "recording"
        write_recording(folder / "part-0.mcap", range(6), mode=mode)
        write_recording(folder / "part-1.mcap", range(6, 12), mode=mode)
        app = self.open_app()
        self.assertEqual(app.sidebar.selectbox[0].options, ["episode"])
        options = app.multiselect[0].options
        self.assertEqual(len(options), 10)  # Five active joints, each with feedback and command.
        self.assertFalse(any(any(field in name for field in ("velocity", "effort", "header", "limited", "counter", "body_joint1")) for name in options))
        source = COMMAND_TOPICS[0 if mode == "full" else 1]
        traces = self.traces(app)
        for joint in ("body_joint3", "body_joint4", "left_joint1", "right_joint1"):
            state, action = traces[f"S:{joint}"], traces[f"A:{joint}"]
            self.assertEqual(action["legendrank"], state["legendrank"] + 1)
            self.assertEqual(state["line"]["dash"], "solid")
            self.assertEqual(action["line"]["dash"], "dot")
            s_values, a_values = trace_values(state), trace_values(action)
            self.assertEqual(len(a_values), 12)
            self.assertTrue(np.isfinite(a_values).all())
            recorded_steps = [0, 1, 2, 9, 10, 11]
            np.testing.assert_allclose(a_values[recorded_steps] - s_values[recorded_steps], 0.25)
            np.testing.assert_allclose(a_values, np.arange(12) + JOINTS.index(joint) * 20 + 0.25)
            self.assertTrue(np.isnan(s_values[5]))
        app.multiselect[0].set_value(["S:/joint_states.position.left_hand_index_joint1", f"A:{source}.position.left_hand_index_joint1"]).run()
        self.assertFalse(app.exception)
        traces = self.traces(app)
        self.assertEqual(trace_values(traces["A:left_hand_index_joint1"])[0], 100.25)
        app.multiselect[0].set_value([]).run()
        app.run()
        self.assertEqual(app.multiselect[0].value, [])
        self.assertEqual([button.label for button in app.button], ["← Previous frame", "Next frame →"])

    def test_full_commands_and_feedback(self):
        self.check_joint_pairs("full")

    def test_split_commands_use_names_version_and_late_names(self):
        self.check_joint_pairs("split")

    def test_missing_names_never_guesses_or_copies_feedback(self):
        write_recording(self.root / "episode" / "recording" / "data.mcap", range(3), mode="split", missing_names=True)
        app = self.open_app()
        self.assertFalse(any(name.startswith("A:") for name in app.multiselect[0].options))
        self.assertTrue(any("StatisticsNames" in warning.value for warning in app.warning))
        self.assertTrue(any("没有解析到关节位置指令" in warning.value for warning in app.warning))

    def test_signal_only_single_frame(self):
        write_recording(self.root / "single" / "recording" / "data.mcap", [0], camera=False)
        app = self.open_app()
        self.assertTrue(app.multiselect[0].options)
        self.assertFalse(app.error)
        self.assertFalse(any("no frames recorded" in item.value for item in app.warning))
        self.assertFalse(any("Image Gap" in item.value for item in app.markdown))
        self.assertFalse(any(widget.label == "Step" for widget in app.slider))

    def test_signal_only_episode_switch_keeps_independent_commands(self):
        for episode in ("one", "two"):
            write_recording(self.root / episode / "recording" / "data.mcap",
                            range(3), mode="split", camera=False)
        app = self.open_app()
        app.multiselect[0].set_value([
            "S:/joint_states.position.left_hand_index_joint1",
            f"A:{COMMAND_TOPICS[1]}.position.left_hand_index_joint1",
        ]).run()
        app.slider[0].set_value(2).run()
        app.sidebar.selectbox[0].set_value("two").run()
        self.assertEqual(app.session_state.t, 0)
        self.assertEqual(len(app.multiselect[0].value), 2)
        traces = self.traces(app)
        np.testing.assert_allclose(
            trace_values(traces["A:left_hand_index_joint1"])
            - trace_values(traces["S:left_hand_index_joint1"]), .25)
        self.assertFalse(app.warning)

    def test_truncated_file(self):
        path = self.root / "episode" / "recording" / "data.mcap"
        write_recording(path, range(3), mode="split")
        path.write_bytes(path.read_bytes()[:-37])
        app = self.open_app()
        self.assertEqual(len(trace_values(self.traces(app)["A:left_joint1"])), 3)
        self.assertTrue(any("顺序扫描" in warning.value for warning in app.warning))

    def test_episode_switch_keeps_joint_selection_across_command_formats(self):
        write_recording(self.root / "old" / "recording" / "data.mcap", range(12), mode="full")
        write_recording(self.root / "new" / "recording" / "data.mcap", range(12), mode="split")
        write_recording(self.root / "missing" / "recording" / "data.mcap", range(3), mode="split", missing_names=True)
        app = self.open_app()
        app.sidebar.selectbox[0].set_value("old").run()
        app.multiselect[0].set_value(["S:/joint_states.position.left_hand_index_joint1",
                                     f"A:{COMMAND_TOPICS[0]}.position.left_hand_index_joint1"]).run()
        for episode, source in (("new", COMMAND_TOPICS[1]), ("old", COMMAND_TOPICS[0])):
            app.sidebar.selectbox[0].set_value(episode).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.multiselect[0].value,
                             ["S:/joint_states.position.left_hand_index_joint1",
                              f"A:{source}.position.left_hand_index_joint1"])
            self.assertEqual(set(self.traces(app)), {"S:left_hand_index_joint1", "A:left_hand_index_joint1"})
        app.sidebar.selectbox[0].set_value("missing").run()
        self.assertEqual(app.multiselect[0].value, ["S:/joint_states.position.left_hand_index_joint1"])
        app.sidebar.selectbox[0].set_value("new").run()
        self.assertEqual(len(app.multiselect[0].value), 2)
        app.multiselect[0].set_value([]).run()
        app.sidebar.selectbox[0].set_value("old").run()
        self.assertEqual(app.multiselect[0].value, [])

    def test_manual_frame_buttons_reset_cursor_but_keep_selection(self):
        for episode in ("one", "two"):
            write_recording(self.root / episode / "recording" / "data.mcap", range(12))
        app = self.open_app()
        app.slider[0].set_value(5).run()
        self.assertEqual(app.session_state.t, 5)
        app.button(key="mcap_next_frame").click().run()
        self.assertEqual(app.session_state.t, 6)
        app.button(key="mcap_previous_frame").click().run()
        self.assertEqual(app.session_state.t, 5)
        app.sidebar.selectbox[0].set_value("two").run()
        self.assertEqual(app.session_state.t, 0)


if __name__ == "__main__":
    unittest.main()
