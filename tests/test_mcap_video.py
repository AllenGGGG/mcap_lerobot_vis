import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from mcap_video import PreviewCache, encode_preview


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
class McapVideoTest(unittest.TestCase):
    def test_real_duration_missing_frames_and_browser_codec(self):
        red = np.full((32, 32, 3), [255, 0, 0], dtype=np.uint8)
        blue = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "video.mp4"
            self.assertTrue(encode_preview(path, [red, red, None, blue, white], [0, .1, .2, 1, 1.1]))
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
            ]))
            self.assertEqual(probe["streams"][0]["codec_name"], "h264")
            self.assertEqual(probe["streams"][0]["pix_fmt"], "yuv420p")
            self.assertAlmostEqual(float(probe["format"]["duration"]), 1.2, delta=.11)
            reader = cv2.VideoCapture(str(path))
            self.addCleanup(reader.release)
            ok, first = reader.read()
            self.assertTrue(ok)
            self.assertGreater(int(first[0, 0, 2]), 200)
            self.assertLess(int(first[0, 0, 0]), 30)
            reader.set(cv2.CAP_PROP_POS_MSEC, 500)
            ok, gap = reader.read()
            self.assertTrue(ok)
            self.assertLess(int(gap[0, 0].max()), 20)

    def test_preview_reused_and_source_change_invalidates_cache(self):
        cache = PreviewCache()
        self.addCleanup(cache.directory.cleanup)
        frames = {"head": [np.zeros((32, 32, 3), dtype=np.uint8)]}
        first = cache.prepare(("episode", 1), frames, [0.0])
        modified = Path(first["head"]).stat().st_mtime_ns
        self.assertEqual(first, cache.prepare(("episode", 1), frames, [0.0]))
        self.assertEqual(Path(first["head"]).stat().st_mtime_ns, modified)
        self.assertNotEqual(first, cache.prepare(("episode", 2), frames, [0.0]))


if __name__ == "__main__":
    unittest.main()
