"""Check VR commands and robot feedback stay independent in the real UI."""
import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from mcap_ros2.writer import Writer
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "mcap_vis.py"
CAMERA = "/camera_head/color/image_raw/compressed"
IMAGE = "string format\nuint8[] data\n"
JOINT = "string[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n"
POSE = """std_msgs/Header header
geometry_msgs/Pose pose
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""


def write_recording(path, steps, camera=True, side="both", targets=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        if min(steps) > 0:
            writer.register_msgdef("std_msgs/msg/String", "string data\n")
        joint = writer.register_msgdef("sensor_msgs/msg/JointState", JOINT)
        image = writer.register_msgdef("sensor_msgs/msg/CompressedImage", IMAGE)
        pose = writer.register_msgdef("geometry_msgs/msg/PoseStamped", POSE)
        jpeg = cv2.imencode(".jpg", np.full((4, 4, 3), 100, dtype=np.uint8))[1].tobytes()
        for step in steps:
            timestamp = 1_000_000_000 + step * 33_333_333

            def write(topic, schema, value):
                writer.write_message(topic, schema, value, log_time=timestamp)

            if camera:
                write(CAMERA, image, {"format": "jpeg", "data": jpeg})
            if step in (0, 1, 2, 9, 10, 11):
                write("/joint_states", joint, {"name": ["hip"], "position": [float(step + 1)],
                                              "velocity": [2.0], "effort": [3.0]})
            for arm, offset in (("left", 0.0), ("right", 100.0)):
                if side != "both" and side != arm:
                    continue
                for suffix, delta in (("pose", 0.0), ("target", 0.5)):
                    if suffix == "target" and not targets:
                        continue
                    write(f"/{arm}_current_{suffix}", pose, {
                        "header": {"stamp": {"sec": timestamp // 1_000_000_000,
                                             "nanosec": timestamp % 1_000_000_000}, "frame_id": "base_link"},
                        "pose": {"position": {"x": offset + step + delta,
                                              "y": offset + 2 + delta, "z": offset + 3 + delta},
                                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}})
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
        return json.loads(app.get("plotly_chart")[0].proto.spec)["data"]

    def test_vr_feedback_pairs_keep_distinct_values_and_both_arms(self):
        folder = self.root / "episode" / "recording"
        write_recording(folder / "part-0.mcap", range(6))
        write_recording(folder / "part-1.mcap", range(6, 12))
        app = self.open_app()
        self.assertEqual(app.sidebar.selectbox[0].options, ["episode"])
        selected = app.multiselect[0].value
        self.assertEqual(len(selected), 12)
        self.assertFalse(any("header" in name for name in app.multiselect[0].options))
        for field in ("position", "velocity", "effort"):
            self.assertIn(f"S:/joint_states.{field}.hip", app.multiselect[0].options)
        traces = {trace["name"]: trace for trace in self.traces(app)}
        options = app.multiselect[0].options
        for arm in ("left", "right"):
            for axis in "xyz":
                state = f"S:/{arm}_current_pose.pose.position.{axis}"
                action = f"A:/{arm}_current_target.pose.position.{axis}"
                self.assertIn(state, selected)
                self.assertIn(action, selected)
                self.assertEqual(options.index(action), options.index(state) + 1)
                self.assertEqual(traces[action]["legendrank"], traces[state]["legendrank"] + 1)
                self.assertEqual(traces[state]["line"]["dash"], "solid")
                self.assertEqual(traces[action]["line"]["dash"], "dot")
                np.testing.assert_allclose(trace_values(traces[action]) - trace_values(traces[state]), 0.5)
                self.assertEqual(len(trace_values(traces[state])), 12)
        np.testing.assert_allclose(trace_values(traces["S:/right_current_pose.pose.position.x"])
                                   - trace_values(traces["S:/left_current_pose.pose.position.x"]), 100.0)
        # A selected feedback joint must not invent a corresponding VR joint command.
        joint = "S:/joint_states.position.hip"
        self.assertNotIn("A:/joint_states.position.hip", options)
        app.multiselect[0].set_value([joint]).run()
        values = trace_values(self.traces(app)[0])
        self.assertTrue(np.isnan(values[5]))
        self.assertEqual(values[0], 1.0)
        app.multiselect[0].set_value([]).run()
        app.run()
        self.assertEqual(app.multiselect[0].value, [])
        self.assertFalse(app.exception)
        self.assertFalse(app.sidebar.checkbox)
        self.assertEqual([button.label for button in app.button], ["▶ Play"])

    def test_missing_target_does_not_copy_state_to_action(self):
        write_recording(self.root / "episode" / "recording" / "data.mcap", range(3), targets=False)
        app = self.open_app()
        self.assertFalse(any(name.startswith("A:") for name in app.multiselect[0].options))
        self.assertTrue(any(name.startswith("S:") for name in app.multiselect[0].options))

    def test_signal_only_single_frame(self):
        write_recording(self.root / "single" / "recording" / "data.mcap", [0], camera=False)
        app = self.open_app()
        self.assertTrue(app.multiselect[0].options)
        self.assertFalse(app.error)
        self.assertFalse(any(widget.label == "Step" for widget in app.slider))

    def test_single_arm_selects_complete_xyz_pairs(self):
        write_recording(self.root / "episode" / "recording" / "data.mcap", range(3), side="right")
        app = self.open_app()
        self.assertEqual(len(app.multiselect[0].value), 6)
        self.assertTrue(all("/right_current_" in name for name in app.multiselect[0].value))

    def test_truncated_file_is_scanned_and_reported(self):
        path = self.root / "episode" / "recording" / "data.mcap"
        write_recording(path, range(3))
        path.write_bytes(path.read_bytes()[:-37])
        app = self.open_app()
        self.assertEqual(len(trace_values(self.traces(app)[0])), 3)
        self.assertTrue(any("顺序扫描" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
