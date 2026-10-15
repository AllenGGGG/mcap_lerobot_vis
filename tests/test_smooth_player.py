"""The shared player must support recordings without any camera media."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import smooth_player


class SignalPlayerTest(unittest.TestCase):
    def test_missing_camera_still_renders_playable_signals(self):
        with patch.object(smooth_player.runtime, "exists", return_value=True), \
             patch.object(smooth_player.runtime, "get_instance",
                          return_value=SimpleNamespace(media_file_mgr=None)), \
             patch.object(smooth_player.st, "caption"), \
             patch.object(smooth_player.st, "iframe") as iframe:
            smooth_player.show_smooth_player({}, {}, [0., .1], [
                {"name": "S:left_joint1", "color": "#729FE0",
                 "dashed": False, "values": [1., 2.]},
                {"name": "A:left_joint1", "color": "#729FE0",
                 "dashed": True, "values": [1.1, 2.1]},
            ])
        iframe.assert_called_once()
        page = iframe.call_args.args[0]
        self.assertIn('id="play"', page)
        self.assertIn('"cameras":[]', page)
        self.assertIn('"name":"A:left_joint1"', page)


if __name__ == "__main__":
    unittest.main()
