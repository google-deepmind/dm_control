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
"""Viewer application module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import _render
from dm_control.viewer import gui
from dm_control.viewer import renderer
from dm_control.viewer import runtime
from dm_control.viewer import user_input
from dm_control.viewer import util
from dm_control.viewer import viewer
from dm_control.viewer import views

_DOUBLE_BUFFERING = (user_input.KEY_F5)
_PAUSE = user_input.KEY_SPACE
_RESTART = user_input.KEY_BACKSPACE
_ADVANCE_SIMULATION = user_input.KEY_RIGHT
_SPEED_UP_TIME = user_input.KEY_EQUAL
_SLOW_DOWN_TIME = user_input.KEY_MINUS
_HELP = user_input.KEY_F1
_STATUS = user_input.KEY_F2
_RELOAD = user_input.KEY_F3

_MAX_FRONTBUFFER_SIZE = 2048
_MISSING_STATUS_ENTRY = '--'
_RUNTIME_STOPPED_LABEL = 'EPISODE TERMINATED - hit backspace to restart'
_STATUS_LABEL = 'Status'
_TIME_LABEL = 'Time'
_CPU_LABEL = 'CPU'
_FPS_LABEL = 'FPS'
_CAMERA_LABEL = 'Camera'
_PAUSED_LABEL = 'Paused'
_ERROR_LABEL = 'Error'


class Help(views.ColumnTextModel):
  """Contains the description of input map employed in the application."""

  def __init__(self):
    """Instance initializer."""
    self._value = [
        ['Help', 'F1'],
        ['Info', 'F2'],
        ['Stereo', 'F5'],
        ['Frame', 'F6'],
        ['Label', 'F7'],
        ['--------------', ''],
        ['Pause', 'Space'],
        ['Reset', 'BackSpace'],
        ['Autoscale', 'Ctrl A'],
        ['Geoms', '0 - 4'],
        ['Sites', 'Shift 0 - 4'],
        ['Speed Up', '='],
        ['Slow Down', '-'],
        ['Switch Cam', '[ ]'],
        ['--------------', ''],
        ['Translate', 'R drag'],
        ['Rotate', 'L drag'],
        ['Zoom', 'Scroll'],
        ['Select', 'L dblclick'],
        ['Center', 'R dblclick'],
        ['Track', 'Ctrl R dblclick / Esc'],
        ['Perturb', 'Ctrl [Shift] L/R drag'],
    ]

  def get_columns(self):
    """Returns the text to display in two columns."""
    return self._value


class Status(views.ColumnTextModel):
  """Monitors and returns the status of the application."""

  def __init__(self, time_multiplier, pause, frame_timer):
    """Instance initializer.

    Args:
      time_multiplier: Instance of util.TimeMultiplier.
      pause: An observable pause subject, instance of util.ObservableFlag.
      frame_timer: A Timer instance counting duration of frames.
    """
    self._runtime = None
    self._time_multiplier = time_multiplier
    self._camera = None
    self._pause = pause
    self._frame_timer = frame_timer
    self._fps_counter = util.Integrator()
    self._cpu_counter = util.Integrator()

    self._value = collections.OrderedDict([
        (_STATUS_LABEL, _MISSING_STATUS_ENTRY),
        (_TIME_LABEL, _MISSING_STATUS_ENTRY),
        (_CPU_LABEL, _MISSING_STATUS_ENTRY),
        (_FPS_LABEL, _MISSING_STATUS_ENTRY),
        (_CAMERA_LABEL, _MISSING_STATUS_ENTRY),
        (_PAUSED_LABEL, _MISSING_STATUS_ENTRY),
        (_ERROR_LABEL, _MISSING_STATUS_ENTRY),
    ])

  def set_camera(self, camera):
    """Updates the active camera instance.

    Args:
      camera: Instance of renderer.SceneCamera.
    """
    self._camera = camera

  def set_runtime(self, instance):
    """Updates the active runtime instance.

    Args:
      instance: Instance of runtime.Base.
    """
    if self._runtime:
      self._runtime.on_error -= self._on_error
      self._runtime.on_episode_begin -= self._clear_error
    self._runtime = instance
    if self._runtime:
      self._runtime.on_error += self._on_error
      self._runtime.on_episode_begin += self._clear_error

  def get_columns(self):
    """Returns the text to display in two columns."""
    if self._frame_timer.measured_time > 0:
      self._fps_counter.value = 1. / self._frame_timer.measured_time
    self._value[_FPS_LABEL] = '{0:.1f}'.format(self._fps_counter.value)

    if self._runtime:
      if self._runtime.state == runtime.State.STOPPED:
        self._value[_STATUS_LABEL] = _RUNTIME_STOPPED_LABEL
      else:
        self._value[_STATUS_LABEL] = str(self._runtime.state)

      self._cpu_counter.value = self._runtime.simulation_time

      self._value[_TIME_LABEL] = '{0:.1f} ({1}x)'.format(
          self._runtime.get_time(), str(self._time_multiplier))
      self._value[_CPU_LABEL] = '{0:.2f}ms'.format(
          self._cpu_counter.value * 1000.0)
    else:
      self._value[_STATUS_LABEL] = _MISSING_STATUS_ENTRY
      self._value[_TIME_LABEL] = _MISSING_STATUS_ENTRY
      self._value[_CPU_LABEL] = _MISSING_STATUS_ENTRY

    if self._camera:
      self._value[_CAMERA_LABEL] = self._camera.name
    else:
      self._value[_CAMERA_LABEL] = _MISSING_STATUS_ENTRY

    self._value[_PAUSED_LABEL] = str(self._pause.value)

    return list(self._value.items())  # For Python 2/3 compatibility.

  def _clear_error(self):
    self._value[_ERROR_LABEL] = _MISSING_STATUS_ENTRY

  def _on_error(self, error_msg):
    self._value[_ERROR_LABEL] = error_msg


class ReloadParams(collections.namedtuple(
    'RefreshParams', ['zoom_to_scene'])):
  """Parameters of a reload request."""


class Application(object):
  """Viewer application."""

  def __init__(self, title='Explorer', width=1024, height=768):
    """Instance initializer."""
    self._render_surface = None
    self._renderer = renderer.NullRenderer()
    self._viewport = renderer.Viewport(width, height)
    self._window = gui.RenderWindow(width, height, title)

    self._pause_subject = util.ObservableFlag(True)
    self._time_multiplier = util.TimeMultiplier(1.)
    self._frame_timer = util.Timer()
    self._viewer = viewer.Viewer(
        self._viewport, self._window.mouse, self._window.keyboard)
    self._viewer_layout = views.ViewportLayout()
    self._status = Status(
        self._time_multiplier, self._pause_subject, self._frame_timer)

    self._runtime = None
    self._environment_loader = None
    self._environment = None
    self._policy = None
    self._deferred_reload_request = None

    status_view_toggle = self._build_view_toggle(
        views.ColumnTextView(self._status), views.PanelLocation.BOTTOM_LEFT)
    help_view_toggle = self._build_view_toggle(
        views.ColumnTextView(Help()), views.PanelLocation.TOP_RIGHT)
    status_view_toggle()

    self._input_map = user_input.InputMap(
        self._window.mouse, self._window.keyboard)
    self._input_map.bind(self._pause_subject.toggle, _PAUSE)
    self._input_map.bind(self._time_multiplier.increase, _SPEED_UP_TIME)
    self._input_map.bind(self._time_multiplier.decrease, _SLOW_DOWN_TIME)
    self._input_map.bind(self._advance_simulation, _ADVANCE_SIMULATION)
    self._input_map.bind(self._restart_runtime, _RESTART)
    self._input_map.bind(self._on_reload, _RELOAD)
    self._input_map.bind(help_view_toggle, _HELP)
    self._input_map.bind(status_view_toggle, _STATUS)

  def _on_reload(self, zoom_to_scene=False):
    """Perform initialization related to Physics reload.

    Reset the components that depend on a specific Physics class instance.

    Args:
      zoom_to_scene: Should the camera zoom to show the entire scene after the
        reload is complete.
    """
    self._deferred_reload_request = ReloadParams(zoom_to_scene)
    self._viewer.deinitialize()
    self._status.set_camera(None)

  def _perform_deferred_reload(self, params):
    """Performs the deferred part of initialization related to Physics reload.

    Args:
      params: Deferred reload parameters, an instance of ReloadParams.
    """
    if self._render_surface:
      self._render_surface.free()
    if self._renderer:
      self._renderer.release()
    self._render_surface = _render.Renderer(
        max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE)
    self._renderer = renderer.OffScreenRenderer(
        self._environment.physics.model, self._render_surface)
    self._renderer.components += self._viewer_layout
    self._viewer.initialize(
        self._environment.physics, self._renderer, touchpad=False)
    self._status.set_camera(self._viewer.camera)
    if params.zoom_to_scene:
      self._viewer.zoom_to_scene()

  def _build_view_toggle(self, view, location):
    def toggle():
      if view in self._viewer_layout:
        self._viewer_layout.remove(view)
      else:
        self._viewer_layout.add(view, location)
    return toggle

  def _tick(self):
    """Handle GUI events until the main window is closed."""
    if self._deferred_reload_request:
      self._perform_deferred_reload(self._deferred_reload_request)
      self._deferred_reload_request = None
    time_elapsed = self._frame_timer.tick() * self._time_multiplier.get()
    if self._runtime:
      with self._viewer.perturbation.apply(self._pause_subject.value):
        self._runtime.tick(time_elapsed, self._pause_subject.value)
    self._viewer.render()

  def _load_environment(self, zoom_to_scene):
    """Loads a new environment."""
    if self._runtime:
      del self._runtime
      self._runtime = None
    self._environment = None
    environment_instance = None
    if self._environment_loader:
      environment_instance = self._environment_loader()
    if environment_instance:
      self._environment = environment_instance
      self._runtime = runtime.Runtime(
          environment=self._environment, policy=self._policy)
      self._runtime.on_physics_changed += lambda: self._on_reload(False)
    self._status.set_runtime(self._runtime)
    self._on_reload(zoom_to_scene=zoom_to_scene)

  def _restart_runtime(self):
    """Restarts the episode, resetting environment, model, and data."""
    if self._runtime:
      self._runtime.stop()
    self._load_environment(zoom_to_scene=False)

  def _advance_simulation(self):
    if self._runtime:
      self._runtime.single_step()

  def launch(self, environment_loader, policy=None):
    """Starts the viewer with the specified policy and environment.

    Args:
      environment_loader: Either a callable that takes no arguments and returns
        an instance of dm_control.rl.control.Environment, or an instance of
        dm_control.rl.control.Environment.
      policy: An optional callable corresponding to a policy to execute
        within the environment. It should accept a `TimeStep` and return
        a numpy array of actions conforming to the output of
        `environment.action_spec()`.

    Raises:
      ValueError: If `environment_loader` is None.
    """
    if environment_loader is None:
      raise ValueError('"environment_loader" argument is required.')
    if callable(environment_loader):
      self._environment_loader = environment_loader
    else:
      self._environment_loader = lambda: environment_loader
    self._policy = policy
    self._load_environment(zoom_to_scene=True)
    def tick():
      self._viewport.set_size(*self._window.shape)
      self._tick()
      return self._renderer.pixels
    self._window.event_loop(tick_func=tick)
    self._window.close()
