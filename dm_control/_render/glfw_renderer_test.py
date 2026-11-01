# Copyright 2017 The dm_control Authors.
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

"""Tests for GLFWContext."""

import os
import re
import subprocess
import sys
import textwrap
import unittest
from absl.testing import absltest
from dm_control import _render
from dm_control.mujoco import wrapper
from dm_control.mujoco.testing import decorators

import mock  # pylint: disable=g-import-not-at-top

MAX_WIDTH = 1024
MAX_HEIGHT = 1024

CONTEXT_PATH = _render.__name__ + '.glfw_renderer.glfw'


@unittest.skipUnless(
    _render.BACKEND == _render.constants.GLFW[0],
    reason='GLFW beckend not selected.')
class GLFWContextTest(absltest.TestCase):

  def test_init(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
    self.assertIs(renderer._context, mock_context)

  def test_make_current(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
      with renderer.make_current():
        pass
    mock_glfw.make_context_current.assert_called_once_with(mock_context)

  def test_freeing(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
      renderer.free()
    mock_glfw.destroy_window.assert_called_once_with(mock_context)
    self.assertIsNone(renderer._context)

  @decorators.run_threaded(num_threads=1, calls_per_thread=20)
  def test_repeatedly_create_and_destroy_context(self):
    renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
    wrapper.MjrContext(wrapper.MjModel.from_xml_string('<mujoco/>'), renderer)


_COUNT_INIT_CALLS_SCRIPT = textwrap.dedent("""
    import sys
    import types
    sys.platform = {platform}
    stub = types.ModuleType('glfw')
    stub.GLFWError = type('GLFWError', (Exception,), {{}})
    stub.VISIBLE = 1
    stub.DOUBLEBUFFER = 2
    stub._init_calls = []
    stub.init = lambda: stub._init_calls.append(1)
    stub.window_hint = lambda *args, **kwargs: None
    stub.create_window = lambda **kwargs: object()
    stub.make_context_current = lambda window: None
    stub.get_current_context = lambda: None
    stub.destroy_window = lambda window: None
    sys.modules['glfw'] = stub
    import dm_control._render.glfw_renderer as glfw_renderer
    print('INIT_AT_IMPORT', len(stub._init_calls))
    glfw_renderer.GLFWContext(max_width=64, max_height=64)
    print('INIT_AFTER_CONTEXT', len(stub._init_calls))
""")


class GlfwInitAtImportTest(absltest.TestCase):
  """Tests when `glfw.init()` runs relative to importing `glfw_renderer`.

  Initializing GLFW at import time interferes with other GUI toolkits that
  share the process's NSApplication on macOS, most notably matplotlib's
  default 'MacOSX' backend (https://github.com/google-deepmind/dm_control/
  issues/377), so on macOS the initialization must be deferred until a
  `GLFWContext` is actually created. On other platforms it must still happen
  at import time so that the automatic backend selection in
  `dm_control/_render/__init__.py` can fall back to EGL/OSMesa when GLFW
  cannot create a context (e.g. on headless Linux machines).
  """

  def _run_with_stubbed_glfw(self, forced_platform):
    script = _COUNT_INIT_CALLS_SCRIPT.format(platform=repr(forced_platform))
    env = dict(os.environ, MUJOCO_GL='glfw')
    completed_process = subprocess.run(
        args=[sys.executable, '-c', script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # The stub module injected here only defines the attributes that
        # `glfw_renderer` accesses at import time and at initialization time.
        timeout=120,
        check=True,
        text=True)
    init_at_import = re.search(r'^INIT_AT_IMPORT (\d+)$',
                               completed_process.stdout, re.M)
    init_after_context = re.search(r'^INIT_AFTER_CONTEXT (\d+)$',
                                   completed_process.stdout, re.M)
    assert init_at_import and init_after_context, (
        completed_process.stdout + completed_process.stderr)
    return int(init_at_import.group(1)), int(init_after_context.group(1))

  def test_defers_initialization_on_macos(self):
    init_at_import, init_after_context = self._run_with_stubbed_glfw('darwin')
    self.assertEqual(init_at_import, 0)
    self.assertEqual(init_after_context, 1)

  def test_initializes_at_import_on_other_platforms(self):
    init_at_import, init_after_context = self._run_with_stubbed_glfw('linux')
    self.assertEqual(init_at_import, 1)
    self.assertEqual(init_after_context, 2)


if __name__ == '__main__':
  absltest.main()
