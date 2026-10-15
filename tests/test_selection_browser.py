"""Browser regression for both the dimension picker and Plotly legend choices.

Requires Playwright and Chrome; skips when browser test tools are unavailable.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

import pandas as pd

from test_mcap_viewer import write_recording

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    sync_playwright = None


REPO = Path(__file__).resolve().parents[1]
CHROME = shutil.which("google-chrome")


@unittest.skipUnless(sync_playwright and CHROME, "Playwright and Chrome required")
class JointSelectionBrowserTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="joint-selection-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        playwright = sync_playwright().start()
        self.addCleanup(playwright.stop)
        self.browser = playwright.chromium.launch(
            executable_path=CHROME, headless=True, args=["--no-sandbox"])
        self.addCleanup(self.browser.close)

    def open_app(self, script, root):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = (self.root / "server.log").open("w+")
        self.addCleanup(log.close)
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(REPO / script),
            "--server.address", "127.0.0.1", "--server.port", str(port),
            "--server.headless", "true", "--", "--data-path", str(root),
        ], cwd=REPO, stdout=log, stderr=log,
            env=dict(os.environ, PYTHONNOUSERSITE="1"))

        def stop_server():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.addCleanup(stop_server)
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            if process.poll() is not None:
                log.seek(0)
                self.fail(log.read())
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1).close()
                break
            except OSError:
                time.sleep(.1)
        page = self.browser.new_page(viewport={"width": 1600, "height": 1400})
        page.set_default_timeout(10000)
        page.goto(url)
        self.chart(page).wait_for()
        return page

    def chart(self, page):
        for _ in range(100):
            for frame in page.frames:
                try:
                    chart = frame.locator(".js-plotly-plot").first
                    if chart.count():
                        return chart
                except Exception:
                    pass
            page.wait_for_timeout(100)
        self.fail("Signal chart did not load")

    def choose_dimensions(self, page, labels):
        picker = page.get_by_test_id("stMultiSelect")
        picker.get_by_role("button", name="Clear all").click()
        for label in labels:
            picker.get_by_role("combobox").fill(label)
            page.get_by_role("option", name=label, exact=True).click()
        page.keyboard.press("Escape")
        self.assert_picker(page, labels)

    def assert_picker(self, page, labels):
        expect(page.get_by_test_id("stMultiSelect").locator(
            '[data-baseweb="tag"] > span[title]')).to_have_text(labels)

    def visible_state(self, page):
        return self.chart(page).evaluate(
            "e => Object.fromEntries(e.data.map(t => [t.name, t.visible !== 'legendonly' && t.visible !== false]))")

    def wait_visible_state(self, page, expected):
        for _ in range(100):
            try:
                if self.visible_state(page) == expected:
                    return
            except Exception:
                pass
            page.wait_for_timeout(100)
        self.assertEqual(self.visible_state(page), expected)

    def switch_episode(self, page, episode, expected_labels, expected_visibility):
        page.get_by_test_id("stSelectbox").first.click()
        page.get_by_role("option", name=episode, exact=True).click()
        page.wait_for_timeout(150)
        page.get_by_test_id("stStatusWidget").wait_for(state="hidden")
        # Wait for the new episode's actual chart update, not just the selectbox.
        expect(page.get_by_test_id("stSlider").get_by_role("slider")).to_have_value("0")
        self.wait_visible_state(page, expected_visibility)
        self.assert_picker(page, expected_labels)
        expect(page.get_by_test_id("stException")).to_have_count(0)

    def check_choices(self, page, episodes, labels):
        self.choose_dimensions(page, labels)
        expected = dict.fromkeys(labels, True)
        self.wait_visible_state(page, expected)
        chart = self.chart(page)
        chart.locator(".legend .traces").filter(has_text=labels[1]).locator(".legendtoggle").click()
        expected[labels[1]] = False
        self.wait_visible_state(page, expected)
        for episode in episodes:
            self.switch_episode(page, episode, labels, expected)
            page.get_by_test_id("stButton").filter(has_text="Next frame →").get_by_role("button").last.click()
            expect(page.get_by_test_id("stSlider").get_by_role("slider")).to_have_value("1")
            self.wait_visible_state(page, expected)
        # Double-click isolation must also survive episode changes.
        self.chart(page).locator(".legend .traces").filter(has_text=labels[1]).locator(".legendtoggle").click()
        self.wait_visible_state(page, dict.fromkeys(labels, True))
        self.chart(page).locator(".legend .traces").filter(has_text=labels[1]).locator(".legendtoggle").dblclick()
        expected = {labels[0]: False, labels[1]: True}
        self.wait_visible_state(page, expected)
        self.switch_episode(page, episodes[0], labels, expected)
        # A user's explicit empty choice must stay empty.
        page.get_by_test_id("stMultiSelect").get_by_role("button", name="Clear all").click()
        self.wait_visible_state(page, {})
        self.switch_episode(page, episodes[1], [], {})

    def test_mcap_picker_and_legend_survive_episode_and_format_changes(self):
        root = self.root / "mcap"
        for episode, mode, steps in (("one", "full", range(12)),
                                     ("two", "split", range(9, 21)),
                                     ("three", "split", range(24))):
            write_recording(root / episode / "recording" / "data.mcap", steps, mode=mode)
        page = self.open_app("mcap_vis.py", root)
        self.check_choices(page, ["two", "three", "one"],
                           ["S:left_hand_index_joint1", "A:left_hand_index_joint1"])

    def test_lerobot_picker_and_legend_survive_episode_changes(self):
        root = self.root / "lerobot"
        (root / "meta").mkdir(parents=True)
        data = root / "data" / "chunk-000"
        data.mkdir(parents=True)
        features = {key: {"dtype": "float32", "shape": [2],
                          "names": ["left_joint1", "right_joint1"]}
                    for key in ("observation.state", "action")}
        (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "features": features}))
        pd.DataFrame([
            {"episode_index": ep, "timestamp": i / 30,
             "observation.state": [ep * 100 + i, ep * 100 + i + 10],
             "action": [ep * 100 + i + .25, ep * 100 + i + 10.25]}
            for ep in range(3) for i in range(12)
        ]).to_parquet(data / "file-000.parquet")
        page = self.open_app("lerobot_vis.py", root)
        self.check_choices(page, ["episode_000001", "episode_000002", "episode_000000"],
                           ["S:right_joint1", "A:right_joint1"])


if __name__ == "__main__":
    unittest.main()
