"""Exercise the real decoder and Streamlit UI with small synthetic recordings."""
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
POSE = "float64 x\nfloat64 y\nfloat64 z\n"


def write_recording(path, steps, camera=True, extra=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        # Different registration orders catch cross-file schema-ID cache reuse.
        if min(steps) > 0:
            writer.register_msgdef("std_msgs/msg/String", "string data\n")
        joint = writer.register_msgdef("sensor_msgs/msg/JointState", JOINT)
        image = writer.register_msgdef("sensor_msgs/msg/CompressedImage", IMAGE)
        pose = writer.register_msgdef("example/msg/Pose", POSE)
        diagnostic = writer.register_msgdef("example/msg/Diagnostic", "float64 temperature\nbool valid\n")
        text = writer.register_msgdef("std_msgs/msg/String", "string data\n")
        raw = writer.register_msgdef("sensor_msgs/msg/Image", "uint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n")
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
            write("/left_current_pose", pose, {"x": float(step), "y": 2.0, "z": 3.0})
            write("/left_current_target", pose, {"x": float(step) + 0.5, "y": 2.0, "z": 3.0})
            if extra:
                write("/diagnostics", diagnostic, {"temperature": 42.0, "valid": True})
                write("/notes", text, {"data": "recorded text"})
                write("/extra_camera", image, {"format": "jpeg", "data": jpeg})
                write("/raw_camera", raw, {"height": 2, "width": 2, "encoding": "rgb8",
                                          "is_bigendian": 0, "step": 8,
                                          "data": bytes([255, 0, 0, 0, 255, 0, 0, 0] * 2)})
                write("/broken_camera", image, {"format": "jpeg", "data": b"bad image"})
        if extra:
            channel = writer._writer.register_channel(topic="/unsupported", message_encoding="json", schema_id=0)
            writer._writer.add_message(channel_id=channel, log_time=timestamp, publish_time=timestamp, data=b"{}")
        writer.finish()


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

    def coverage(self, app):
        return app.dataframe[0].value.set_index("topic")

    def test_shards_all_fields_coverage_and_missing_values(self):
        folder = self.root / "episode" / "recording"
        write_recording(folder / "part-0.mcap", range(6), extra=True)
        write_recording(folder / "part-1.mcap", range(6, 12), extra=True)
        app = self.open_app()
        self.assertEqual(app.sidebar.selectbox[0].options, ["episode"])
        coverage = self.coverage(app)
        self.assertEqual(coverage.loc[CAMERA, "recorded"], 12)
        self.assertEqual(coverage.loc[CAMERA, "decoded"], 12)
        self.assertEqual(coverage.loc["/raw_camera", "status"], "可展示")
        self.assertEqual(coverage.loc["/extra_camera", "role"], "camera")
        self.assertEqual(coverage.loc["/unsupported", "status"], "不支持的编码")
        self.assertEqual(coverage.loc["/notes", "status"], "已解析，无数值字段")
        self.assertEqual(coverage.loc["/broken_camera", "failed"], 12)
        options = app.multiselect[0].options
        for field in ("position", "velocity", "effort"):
            self.assertIn(f"S:/joint_states.{field}.hip", options)
        self.assertIn("S:/diagnostics.valid", options)
        self.assertIn("S:/left_current_pose.x", app.multiselect[0].value)
        self.assertIn("A:/left_current_target.x", app.multiselect[0].value)
        step_slider = next(widget for widget in app.slider if widget.label == "Step")
        step_slider.set_value(5).run()
        values = app.dataframe[1].value.set_index("field")
        self.assertTrue(np.isnan(values.loc["/joint_states.position.hip", "value"]))
        self.assertEqual(values.loc["/diagnostics.valid", "value"], 1.0)
        next(button for button in app.button if button.label == "选择全部维度").click().run()
        self.assertEqual(set(app.multiselect[0].value), set(options))
        next(button for button in app.button if button.label == "清空选择").click().run()
        self.assertEqual(app.multiselect[0].value, [])
        app.run()
        self.assertEqual(app.multiselect[0].value, [])
        next(button for button in app.button if button.label == "恢复默认配对").click().run()
        self.assertIn("A:/left_current_target.x", app.multiselect[0].value)
        self.assertFalse(app.exception)

    def test_signal_only_single_frame(self):
        write_recording(self.root / "single" / "recording" / "data.mcap", [0], camera=False)
        app = self.open_app()
        self.assertTrue(app.multiselect[0].options)
        self.assertFalse(app.error)
        self.assertFalse(any(widget.label == "Step" for widget in app.slider))

    def test_no_head_and_manual_topic_coverage(self):
        write_recording(self.root / "episode" / "recording" / "data.mcap", range(3), camera=False, extra=True)
        app = self.open_app()
        self.assertEqual(self.coverage(app).loc["/extra_camera", "status"], "可展示")
        app.sidebar.checkbox[0].uncheck().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.coverage(app).loc["/joint_states", "status"], "未加载")
        state_input = next(widget for widget in app.sidebar.text_input if widget.label.startswith("state topics"))
        state_input.set_value("/joint_states").run()
        self.assertIn("S:/joint_states.position.hip", app.multiselect[0].options)

    def test_truncated_file_is_scanned_and_reported(self):
        path = self.root / "episode" / "recording" / "data.mcap"
        write_recording(path, range(3))
        # Keep the message chunks but remove the Footer and trailing magic.
        path.write_bytes(path.read_bytes()[:-37])
        app = self.open_app()
        self.assertEqual(self.coverage(app).loc[CAMERA, "decoded"], 3)
        self.assertTrue(any("顺序扫描" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
