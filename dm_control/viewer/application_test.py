# Copyright 2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests of the application.py module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.viewer import application
import dm_env
from dm_env import specs
import mock
import numpy as np


class ApplicationTest(parameterized.TestCase):

  def setUp(self):
    super(ApplicationTest, self).setUp()
    with mock.patch(application.__name__ + '.gui'):
      self.app = application.Application()

    self.app._viewer = mock.MagicMock()
    self.app._keyboard_action = mock.MagicMock()

    self.environment = mock.MagicMock(spec=dm_env.Environment)
    self.environment.action_spec.return_value = specs.BoundedArray(
        (1,), np.float64, -1, 1)
    self.environment.physics = mock.MagicMock()
    self.app._environment = self.environment
    self.agent = mock.MagicMock()
    self.loader = lambda: self.environment

  def test_on_reload_defers_viewer_initialization_until_tick(self):
    self.app._on_reload(zoom_to_scene=True)
    self.assertEqual(0, self.app._viewer.initialize.call_count)
    self.assertIsNotNone(self.app._deferred_reload_request)

  def test_deferred_on_reload_parameters(self):
    self.app._on_reload(zoom_to_scene=True)
    self.assertTrue(self.app._deferred_reload_request.zoom_to_scene)
    self.app._on_reload(zoom_to_scene=False)
    self.assertFalse(self.app._deferred_reload_request.zoom_to_scene)

  def test_executing_deferred_initialization(self):
    self.app._deferred_reload_request = application.ReloadParams(False)
    self.app._tick()
    self.app._viewer.initialize.assert_called_once()

  def test_processing_zoom_to_scene_request(self):
    self.app._perform_deferred_reload(application.ReloadParams(True))
    self.app._viewer.zoom_to_scene.assert_called_once()

  def test_skipping_zoom_to_scene(self):
    self.app._perform_deferred_reload(application.ReloadParams(False))
    self.app._viewer.zoom_to_scene.assert_not_called()

  def test_on_reload_deinitializes_viewer_instantly(self):
    self.app._on_reload()
    self.app._viewer.deinitialize.assert_called_once()

  def test_zoom_to_scene_after_launch(self):
    self.app.launch(self.loader, self.agent)
    self.app._viewer.zoom_to_scene()

  def test_tick_runtime(self):
    self.app._runtime = mock.MagicMock()
    self.app._pause_subject.value = False
    self.app._tick()
    self.app._runtime.tick.assert_called_once()

  def test_restart_runtime(self):
    self.app._load_environment = mock.MagicMock()
    self.app._runtime = mock.MagicMock()
    self.app._restart_runtime()
    self.app._runtime.stop.assert_called_once()
    self.app._load_environment.assert_called_once()


if __name__ == '__main__':
  absltest.main()
