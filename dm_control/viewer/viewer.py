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
"""Mujoco Physics viewer, with custom input controllers."""


from dm_control.mujoco.wrapper import mjbindings
from dm_control.viewer import renderer
from dm_control.viewer import user_input
from dm_control.viewer import util
import mujoco

functions = mjbindings.functions

_NUM_GROUP_KEYS = 10

_PAN_CAMERA_VERTICAL_MOUSE = user_input.Exclusive(
    user_input.MOUSE_BUTTON_RIGHT)
_PAN_CAMERA_HORIZONTAL_MOUSE = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_RIGHT, user_input.MOD_SHIFT))
_ROTATE_OBJECT_MOUSE = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_CONTROL))
_MOVE_OBJECT_VERTICAL_MOUSE = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_RIGHT, user_input.MOD_CONTROL))
_MOVE_OBJECT_HORIZONTAL_MOUSE = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_RIGHT, user_input.MOD_SHIFT_CONTROL))

_PAN_CAMERA_VERTICAL_TOUCHPAD = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_ALT))
_PAN_CAMERA_HORIZONTAL_TOUCHPAD = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_RIGHT, user_input.MOD_ALT))
_ROTATE_OBJECT_TOUCHPAD = user_input.Exclusive(
    user_input.MOUSE_BUTTON_RIGHT)
_MOVE_OBJECT_VERTICAL_TOUCHPAD = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_CONTROL))
_MOVE_OBJECT_HORIZONTAL_TOUCHPAD = user_input.Exclusive(
    (user_input.MOUSE_BUTTON_LEFT, user_input.MOD_SHIFT_CONTROL))

_ROTATE_CAMERA = user_input.Exclusive(user_input.MOUSE_BUTTON_LEFT)
_CENTER_CAMERA = user_input.DoubleClick(user_input.MOUSE_BUTTON_RIGHT)
_SELECT_OBJECT = user_input.DoubleClick(user_input.MOUSE_BUTTON_LEFT)
_TRACK_OBJECT = user_input.DoubleClick(
    (user_input.MOUSE_BUTTON_RIGHT, user_input.MOD_CONTROL))
_FREE_LOOK = user_input.KEY_ESCAPE
_NEXT_CAMERA = user_input.KEY_RIGHT_BRACKET
_PREVIOUS_CAMERA = user_input.KEY_LEFT_BRACKET
_ZOOM_TO_SCENE = (user_input.KEY_A, user_input.MOD_CONTROL)
_DOUBLE_BUFFERING = user_input.KEY_F5
_PREV_RENDERING_MODE = (user_input.KEY_F6, user_input.MOD_SHIFT)
_NEXT_RENDERING_MODE = user_input.KEY_F6
_PREV_LABELING_MODE = (user_input.KEY_F7, user_input.MOD_SHIFT)
_NEXT_LABELING_MODE = user_input.KEY_F7
_PRINT_CAMERA = user_input.KEY_F11
_VISUALIZATION_FLAGS = user_input.Range([
    ord(functions.mjVISSTRING[i][2])
    for i in range(0, mujoco.mjtVisFlag.mjNVISFLAG)
])
_GEOM_GROUPS = user_input.Range(
    [i + ord('0') for i in range(min(_NUM_GROUP_KEYS, mujoco.mjNGROUP))])
_SITE_GROUPS = user_input.Range([
    (i + ord('0'), user_input.MOD_SHIFT)
    for i in range(min(_NUM_GROUP_KEYS, mujoco.mjNGROUP))
])
_RENDERING_FLAGS = user_input.Range([
    ord(functions.mjRNDSTRING[i][2]) if functions.mjRNDSTRING[i][2] else 0
    for i in range(0, mujoco.mjtRndFlag.mjNRNDFLAG)
])

_CAMERA_MOVEMENT_ACTIONS = [
    mujoco.mjtMouse.mjMOUSE_MOVE_V, mujoco.mjtMouse.mjMOUSE_ROTATE_H
]

# Translates mouse wheel rotations to zoom speed.
_SCROLL_SPEED_FACTOR = 0.05

# Distance, in meters, at which to focus on the clicked object.
_LOOK_AT_DISTANCE = 1.5

# Zoom factor used when zooming in on the entire scene.
_FULL_SCENE_ZOOM_FACTOR = 1.5


class Viewer:
  """Viewport displaying the contents of a physics world."""

  def __init__(self, viewport, mouse, keyboard, camera_settings=None,
               zoom_factor=_FULL_SCENE_ZOOM_FACTOR, scene_callback=None):
    """Instance initializer.

    Args:
      viewport: Render viewport, instance of renderer.Viewport.
      mouse: A mouse device.
      keyboard: A keyboard device.
      camera_settings: Properties of the scene MjvCamera.
      zoom_factor: Initial scale factor for zooming into the scene.
      scene_callback: Scene callback.
        This is a callable of the form: `my_callable(MjModel, MjData, MjvScene)`
        that gets applied to every rendered scene.
    """
    self._viewport = viewport
    self._mouse = mouse

    self._null_perturbation = renderer.NullPerturbation()
    self._render_settings = renderer.RenderSettings()
    self._input_map = user_input.InputMap(mouse, keyboard)

    self._camera = None
    self._camera_settings = camera_settings
    self._renderer = None
    self._manipulator = None
    self._free_camera = None
    self._camera_select = None
    self._zoom_factor = zoom_factor
    self._scene_callback = scene_callback

  def __del__(self):
    del self._camera
    del self._renderer
    del self._manipulator
    del self._free_camera
    del self._camera_select

  def initialize(self, physics, renderer_instance, touchpad):
    """Initialize the viewer.

    Args:
      physics: Physics instance.
      renderer_instance: A renderer.Base instance.
      touchpad: A boolean, use input dedicated to touchpad.
    """
    self._camera = renderer.SceneCamera(
        physics.model,
        physics.data,
        self._render_settings,
        settings=self._camera_settings,
        zoom_factor=self._zoom_factor,
        scene_callback=self._scene_callback)

    self._manipulator = ManipulationController(
        self._viewport, self._camera, self._mouse)

    self._free_camera = FreeCameraController(
        self._viewport, self._camera, self._mouse, self._manipulator)

    self._camera_select = CameraSelector(
        physics.model, self._camera, self._free_camera)

    self._renderer = renderer_instance

    self._input_map.clear_bindings()

    if touchpad:
      self._input_map.bind(
          self._manipulator.set_move_vertical_mode,
          _MOVE_OBJECT_VERTICAL_TOUCHPAD)
      self._input_map.bind(
          self._manipulator.set_move_horizontal_mode,
          _MOVE_OBJECT_HORIZONTAL_TOUCHPAD)
      self._input_map.bind(
          self._manipulator.set_rotate_mode, _ROTATE_OBJECT_TOUCHPAD)
      self._input_map.bind(
          self._free_camera.set_pan_vertical_mode,
          _PAN_CAMERA_VERTICAL_TOUCHPAD)
      self._input_map.bind(
          self._free_camera.set_pan_horizontal_mode,
          _PAN_CAMERA_HORIZONTAL_TOUCHPAD)
    else:
      self._input_map.bind(
          self._manipulator.set_move_vertical_mode, _MOVE_OBJECT_VERTICAL_MOUSE)
      self._input_map.bind(
          self._manipulator.set_move_horizontal_mode,
          _MOVE_OBJECT_HORIZONTAL_MOUSE)
      self._input_map.bind(
          self._manipulator.set_rotate_mode, _ROTATE_OBJECT_MOUSE)
      self._input_map.bind(
          self._free_camera.set_pan_vertical_mode, _PAN_CAMERA_VERTICAL_MOUSE)
      self._input_map.bind(
          self._free_camera.set_pan_horizontal_mode,
          _PAN_CAMERA_HORIZONTAL_MOUSE)

    self._input_map.bind(self._print_camera_transform, _PRINT_CAMERA)
    self._input_map.bind(
        self._render_settings.select_prev_rendering_mode, _PREV_RENDERING_MODE)
    self._input_map.bind(
        self._render_settings.select_next_rendering_mode, _NEXT_RENDERING_MODE)
    self._input_map.bind(
        self._render_settings.select_prev_labeling_mode, _PREV_LABELING_MODE)
    self._input_map.bind(
        self._render_settings.select_next_labeling_mode, _NEXT_LABELING_MODE)
    self._input_map.bind(
        self._render_settings.select_prev_labeling_mode, _PREV_LABELING_MODE)
    self._input_map.bind(
        self._render_settings.toggle_stereo_buffering, _DOUBLE_BUFFERING)
    self._input_map.bind(
        self._render_settings.toggle_visualization_flag, _VISUALIZATION_FLAGS)
    self._input_map.bind(
        self._render_settings.toggle_site_group, _SITE_GROUPS)
    self._input_map.bind(
        self._render_settings.toggle_geom_group, _GEOM_GROUPS)
    self._input_map.bind(
        self._render_settings.toggle_rendering_flag, _RENDERING_FLAGS)

    self._input_map.bind(self._camera.zoom_to_scene, _ZOOM_TO_SCENE)
    self._input_map.bind(self._camera_select.select_next, _NEXT_CAMERA)
    self._input_map.bind(self._camera_select.select_previous, _PREVIOUS_CAMERA)
    self._input_map.bind_z_axis(self._free_camera.zoom)
    self._input_map.bind_plane(self._free_camera.on_move)
    self._input_map.bind(self._free_camera.set_rotate_mode, _ROTATE_CAMERA)
    self._input_map.bind(self._free_camera.center, _CENTER_CAMERA)
    self._input_map.bind(self._free_camera.track, _TRACK_OBJECT)
    self._input_map.bind(self._camera_select.escape, _FREE_LOOK)
    self._input_map.bind(self._manipulator.select, _SELECT_OBJECT)
    self._input_map.bind_plane(self._manipulator.on_move)

  def deinitialize(self):
    """Deinitializes the viewer instance."""
    self._input_map.clear_bindings()
    self._camera_settings = self._camera.settings if self._camera else None
    del self._camera
    del self._renderer
    del self._manipulator
    del self._free_camera
    del self._camera_select
    self._camera = None
    self._renderer = None
    self._manipulator = None
    self._free_camera = None
    self._camera_select = None

  def render(self):
    """Renders the visualized scene."""
    if self._camera and self._renderer:  # Can be None during env reload.
      scene = self._camera.render(self.perturbation)
      self._render_settings.apply_settings(scene)
      self._renderer.render(self._viewport, scene)

  def zoom_to_scene(self):
    """Utility method that set the camera to embrace the entire scene."""
    if self._camera:
      self._camera.zoom_to_scene()

  def _print_camera_transform(self):
    if self._camera:
      rotation_mtx, position = self._camera.transform
      right, up, _ = rotation_mtx
      print('<camera pos="%.3f %.3f %.3f" '
            'xyaxes="%.3f %.3f %.3f %.3f %.3f %.3f"/>' % (
                position[0], position[1], position[2], right[0], right[1],
                right[2], up[0], up[1], up[2]))

  @property
  def perturbation(self):
    """Returns an active renderer.Perturbation object."""
    if self._manipulator and self._manipulator.perturbation:
      return self._manipulator.perturbation
    else:
      return self._null_perturbation

  @property
  def camera(self):
    """Returns an active renderer.SceneCamera instance."""
    return self._camera

  @property
  def render_settings(self):
    """Returns renderer.RenderSettings used by this viewer."""
    return self._render_settings


class CameraSelector:
  """Binds camera behavior to user input."""

  def __init__(self, model, camera, free_camera, **unused):
    """Instance initializer.

    Args:
      model: Instance of MjModel.
      camera: Instance of SceneCamera.
      free_camera: Instance of FreeCameraController.
      **unused: Other arguments, not used by this class.
    """
    del unused  # Unused.
    self._model = model
    self._camera = camera
    self._free_ctrl = free_camera

    self._camera_idx = -1
    self._active_ctrl = self._free_ctrl

  def select_previous(self):
    """Cycles to the previous scene camera."""
    self._camera_idx -= 1
    if not self._model.ncam or self._camera_idx < -1:
      self._camera_idx = self._model.ncam - 1
    self._commit_selection()

  def select_next(self):
    """Cycles to the next scene camera."""
    self._camera_idx += 1
    if not self._model.ncam or self._camera_idx >= self._model.ncam:
      self._camera_idx = -1
    self._commit_selection()

  def escape(self) -> None:
    """Unconditionally switches to the free camera."""
    self._camera_idx = -1
    self._commit_selection()

  def _commit_selection(self):
    """Selects a controller that should go with the selected camera."""
    if self._camera_idx < 0:
      self._activate(self._free_ctrl)
    else:
      self._camera.set_fixed_mode(self._camera_idx)
      self._activate(None)

  def _activate(self, controller):
    """Activates a sub-controller."""
    if controller == self._active_ctrl:
      return

    if self._active_ctrl is not None:
      self._active_ctrl.deactivate()
    self._active_ctrl = controller
    if self._active_ctrl is not None:
      self._active_ctrl.activate()


class FreeCameraController:
  """Implements the free camera behavior."""

  def __init__(self, viewport, camera, pointer, selection_service, **unused):
    """Instance initializer.

    Args:
      viewport: Instance of mujoco_viewer.Viewport.
      camera: Instance of mujoco_viewer.SceneCamera.
      pointer: A pointer that moves around the screen and is used to point at
        bodies. Implements a single attribute - 'position' - that returns a
        2-component vector of pointer's screen space position.
      selection_service: An instance of a class implementing a
        'selected_body_id' property.
      **unused: Other optional parameters not used by this class.
    """
    del unused  # Unused.
    self._viewport = viewport
    self._camera = camera
    self._pointer = pointer
    self._selection_service = selection_service
    self._active = True
    self._tracked_body_idx = -1
    self._action = util.AtomicAction()

  def activate(self):
    """Activates the controller."""
    self._active = True
    self._update_camera_mode()

  def deactivate(self):
    """Deactivates the controller."""
    self._active = False
    self._action = util.AtomicAction()

  def set_pan_vertical_mode(self, enable):
    """Starts/ends the camera panning action along the vertical plane.

    Args:
      enable: A boolean flag, True to start the action, False to end it.
    """
    if self._active:
      if enable:
        self._action.begin(mujoco.mjtMouse.mjMOUSE_MOVE_V)
      else:
        self._action.end(mujoco.mjtMouse.mjMOUSE_MOVE_V)

  def set_pan_horizontal_mode(self, enable):
    """Starts/ends the camera panning action along the horizontal plane.

    Args:
      enable: A boolean flag, True to start the action, False to end it.
    """
    if self._active:
      if enable:
        self._action.begin(mujoco.mjtMouse.mjMOUSE_MOVE_H)
      else:
        self._action.end(mujoco.mjtMouse.mjMOUSE_MOVE_H)

  def set_rotate_mode(self, enable):
    """Starts/ends the camera rotation action.

    Args:
      enable: A boolean flag, True to start the action, False to end it.
    """
    if self._active:
      if enable:
        self._action.begin(mujoco.mjtMouse.mjMOUSE_ROTATE_H)
      else:
        self._action.end(mujoco.mjtMouse.mjMOUSE_ROTATE_H)

  def center(self):
    """Focuses camera on the object the pointer is currently pointing at."""
    if self._active:
      body_id, world_pos = self._camera.raycast(self._viewport,
                                                self._pointer.position)
      if body_id >= 0:
        self._camera.look_at(world_pos, _LOOK_AT_DISTANCE)

  def on_move(self, position, translation):
    """Translates mouse moves onto camera movements."""
    del position
    if self._action.in_progress:
      viewport_offset = self._viewport.screen_to_viewport(translation)
      self._camera.move(self._action.watermark, viewport_offset)

  def zoom(self, zoom_factor):
    """Zooms the camera in/out.

    Args:
      zoom_factor: A floating point value, by how much to zoom the camera.
        Positive values zoom the camera in, negative values zoom it out.
    """
    if self._active:
      offset = [0, _SCROLL_SPEED_FACTOR * zoom_factor * -1.]
      self._camera.move(mujoco.mjtMouse.mjMOUSE_ZOOM, offset)

  def track(self):
    """Makes the camera track the currently selected object.

    The selection is managed by the selection service.
    """
    if self._active and self._tracked_body_idx < 0:
      self._tracked_body_idx = self._selection_service.selected_body_id
      self._update_camera_mode()

  def free_look(self):
    """Switches the camera to a free-look mode."""
    if self._active:
      self._tracked_body_idx = -1
      self._update_camera_mode()

  def _update_camera_mode(self):
    """Sets the camera into a tracking or a free-look mode."""
    if self._tracked_body_idx >= 0:
      self._camera.set_tracking_mode(self._tracked_body_idx)
    else:
      self._camera.set_freelook_mode()


class ManipulationController:
  """Binds control over scene objects to user input."""

  def __init__(self, viewport, camera, pointer, **unused):
    """Instance initializer.

    Args:
      viewport: Instance of mujoco_viewer.Viewport.
      camera: Instance of mujoco_viewer.SceneCamera.
      pointer: A pointer that moves around the screen and is used to point at
        bodies. Implements a single attribute - 'position' - that returns a
        2-component vector of pointer's screen space position.
      **unused: Other arguments, unused by this class.
    """
    del unused  # Unused.
    self._viewport = viewport
    self._camera = camera
    self._pointer = pointer
    self._action = util.AtomicAction(self._update_action)
    self._perturb = None

  def select(self):
    """Translates mouse double-clicks to object selection action."""
    body_id, _ = self._camera.raycast(self._viewport, self._pointer.position)
    if body_id >= 0:
      self._perturb = self._camera.new_perturbation(body_id)
    else:
      self._perturb = None

  def set_move_vertical_mode(self, enable):
    """Begins/ends an object translation action along the vertical plane.

    Args:
      enable: A boolean flag, True begins the action, False ends it.
    """
    if enable:
      self._action.begin(mujoco.mjtMouse.mjMOUSE_MOVE_V)
    else:
      self._action.end(mujoco.mjtMouse.mjMOUSE_MOVE_V)

  def set_move_horizontal_mode(self, enable):
    """Begins/ends an object translation action along the horizontal plane.

    Args:
      enable: A boolean flag, True begins the action, False ends it.
    """
    if enable:
      self._action.begin(mujoco.mjtMouse.mjMOUSE_MOVE_H)
    else:
      self._action.end(mujoco.mjtMouse.mjMOUSE_MOVE_H)

  def set_rotate_mode(self, enable):
    """Begins/ends an object rotation action.

    Args:
      enable: A boolean flag, True begins the action, False ends it.
    """
    if enable:
      self._action.begin(mujoco.mjtMouse.mjMOUSE_ROTATE_H)
    else:
      self._action.end(mujoco.mjtMouse.mjMOUSE_ROTATE_H)

  def _update_action(self, action):
    if self._perturb is not None:
      if action is not None:
        _, grab_pos = self._camera.raycast(self._viewport,
                                           self._pointer.position)
        self._perturb.start_move(action, grab_pos)
      else:
        self._perturb.end_move()

  def on_move(self, position, translation):
    """Translates mouse moves to selected object movements."""
    del position
    if self._perturb is not None and self._action.in_progress:
      viewport_offset = self._viewport.screen_to_viewport(translation)
      self._perturb.tick_move(viewport_offset)

  @property
  def perturbation(self):
    """Returns the Perturbation object that represents the manipulated body."""
    return self._perturb

  @property
  def selected_body_id(self):
    """Returns the id of the selected body, or -1 if none is selected."""
    return self._perturb.body_id if self._perturb is not None else -1
