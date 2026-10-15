"""Camera availability must not hide the state/action stored in Parquet."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from test_mcap_viewer import trace_values, plot_traces


APP = Path(__file__).resolve().parents[1] / "lerobot_vis.py"


def write_dataset(root, steps=3, camera=False):
    (root / "meta").mkdir(parents=True)
    folder = root / "data" / "chunk-000"
    folder.mkdir(parents=True)
    features = {key: {"dtype": "float32", "shape": [1], "names": ["left_joint1"]}
                for key in ("observation.state", "action")}
    if camera:
        features["observation.images.head"] = {"dtype": "video", "shape": [32, 32, 3]}
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "features": features}))
    pd.DataFrame([
        {"episode_index": episode, "timestamp": i / 30,
         "observation.state": [float(episode * 100 + i)],
         "action": [float(episode * 100 + i) + .25]}
        for episode in (0, 1) for i in range(steps)
    ]).to_parquet(folder / "file-000.parquet")


class LeRobotSignalTest(unittest.TestCase):
    def check_dataset(self, camera=False, steps=3):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, steps, camera)
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.sidebar.text_input[0].set_value(str(root)).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            traces = plot_traces(app)
            np.testing.assert_allclose(trace_values(traces["S:left_joint1"]), np.arange(steps))
            np.testing.assert_allclose(trace_values(traces["A:left_joint1"]), np.arange(steps) + .25)
            if steps > 1:
                app.slider[0].set_value(steps - 1).run()
                app.sidebar.selectbox[0].set_value("episode_000001").run()
                self.assertEqual(app.session_state.t, 0)
                self.assertEqual(len(app.multiselect[0].value), 2)
                app.button(key="lr_next_frame").click().run()
                self.assertEqual(app.session_state.t, 1)
            self.assertFalse(app.exception)

    def test_joint_only_dataset(self):
        self.check_dataset()

    def test_missing_video_keeps_recorded_joint_values(self):
        self.check_dataset(camera=True)

    def test_single_frame_joint_only_dataset(self):
        self.check_dataset(steps=1)


if __name__ == "__main__":
    unittest.main()
