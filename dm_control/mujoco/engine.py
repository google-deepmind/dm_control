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

"""Mujoco `Physics` implementation and helper classes.

The `Physics` class provides the main Python interface to MuJoCo.

MuJoCo models are defined using the MJCF XML format. The `Physics` class
can load a model from a path to an XML file, an XML string, or from a serialized
MJB binary format. See the named constructors for each of these cases.

Each `Physics` instance defines a simulated world. To step forward the
simulation, use the `step` method. To set a control or actuation signal, use the
`set_control` method, which will apply the provided signal to the actuators in
subsequent calls to `step`.

Use the `Camera` class to create RGB or depth images. A `Camera` can render its
viewport to an array using the `render` method, and can query for objects
visible at specific positions using the `select` method. The `Physics` class
also provides a `render` method that returns a pixel array directly.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

from absl import logging

from dm_control import render
from dm_control.mujoco import index
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.mujoco.wrapper.mjbindings import types
from dm_control.rl import control as _control

import numpy as np
import six

from dm_control.rl import specs

_FONT_STYLES = {
    'normal': enums.mjtFont.mjFONT_NORMAL,
    'shadow': enums.mjtFont.mjFONT_SHADOW,
    'big': enums.mjtFont.mjFONT_BIG,
}
_GRID_POSITIONS = {
    'top left': enums.mjtGridPos.mjGRID_TOPLEFT,
    'top right': enums.mjtGridPos.mjGRID_TOPRIGHT,
    'bottom left': enums.mjtGridPos.mjGRID_BOTTOMLEFT,
    'bottom right': enums.mjtGridPos.mjGRID_BOTTOMRIGHT,
}

Contexts = collections.namedtuple('Contexts', ['gl', 'mujoco'])
Selected = collections.namedtuple(
    'Selected', ['body', 'geom', 'world_position'])
NamedIndexStructs = collections.namedtuple(
    'NamedIndexStructs', ['model', 'data'])
Pose = collections.namedtuple(
    'Pose', ['lookat', 'distance', 'azimuth', 'elevation'])

_INVALID_PHYSICS_STATE = (
    'Physics state is invalid. Warning(s) raised: {warning_names}')


class Physics(_control.Physics):
  """Encapsulates a MuJoCo model.

  A MuJoCo model is typically defined by an MJCF XML file [0]

  ```python
  physics = Physics.from_xml_path('/path/to/model.xml')

  with physics.reset_context():
    physics.named.data.qpos['hinge'] = np.random.rand()

  # Apply controls and advance the simulation state.
  physics.set_control(np.random.random_sample(size=N_ACTUATORS))
  physics.step()

  # Render a camera defined in the XML file to a NumPy array.
  rgb = physics.render(height=240, width=320, id=0)
  ```

  [0] http://www.mujoco.org/book/modeling.html
  """

  _contexts = None

  def __init__(self, data):
    """Initializes a new `Physics` instance.

    Args:
      data: Instance of `wrapper.MjData`.
    """
    self._reload_from_data(data)

  def set_control(self, control):
    """Sets the control signal for the actuators.

    Args:
      control: NumPy array or array-like actuation values.
    """
    self.data.ctrl[:] = np.asarray(control)

  def step(self):
    """Advances physics with up-to-date position and velocity dependent fields.

    The actuation can be updated by calling the `set_control` function first.
    """
    # In the case of Euler integration we assume mj_step1 has already been
    # called for this state, finish the step with mj_step2 and then update all
    # position and velocity related fields with mj_step1. This ensures that
    # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
    # integrators (e.g. RK4) an additional mj_step1 must be called after the
    # last mj_step to ensure mjData syncing.
    with self.check_invalid_state():
      if self.model.opt.integrator == enums.mjtIntegrator.mjINT_EULER:
        mjlib.mj_step2(self.model.ptr, self.data.ptr)
      else:
        mjlib.mj_step(self.model.ptr, self.data.ptr)

      mjlib.mj_step1(self.model.ptr, self.data.ptr)

  def render(self, height=240, width=320, camera_id=-1, overlays=(),
             depth=False, scene_option=None):
    """Returns a camera view as a NumPy array of pixel values.

    Args:
      height: Viewport height (number of pixels). Optional, defaults to 240.
      width: Viewport width (number of pixels). Optional, defaults to 320.
      camera_id: Optional camera name or index. Defaults to -1, the free
        camera, which is always defined. A nonnegative integer or string
        corresponds to a fixed camera, which must be defined in the model XML.
        If `camera_id` is a string then the camera must also be named.
      overlays: An optional sequence of `TextOverlay` instances to draw. Only
        supported if `depth` is False.
      depth: If `True`, this method returns a NumPy float array of depth values
        (in meters). Defaults to `False`, which results in an RGB image.
      scene_option: An optional `wrapper.MjvOption` instance that can be used to
        render the scene with custom visualization options. If None then the
        default options will be used.

    Returns:
      The rendered RGB or depth image.
    """
    camera = Camera(
        physics=self, height=height, width=width, camera_id=camera_id)
    image = camera.render(
        overlays=overlays, depth=depth, scene_option=scene_option)
    camera._scene.free()  # pylint: disable=protected-access
    return image

  def get_state(self):
    """Returns the physics state.

    Returns:
      NumPy array containing full physics simulation state.
    """
    return np.concatenate(self._physics_state_items())

  def set_state(self, physics_state):
    """Sets the physics state.

    Args:
      physics_state: NumPy array containing the full physics simulation state.

    Raises:
      ValueError: If `physics_state` has invalid size.
    """
    state_items = self._physics_state_items()

    expected_shape = (sum(item.size for item in state_items),)
    if expected_shape != physics_state.shape:
      raise ValueError('Input physics state has shape {}. Expected {}.'.format(
          physics_state.shape, expected_shape))

    start = 0
    for state_item in state_items:
      size = state_item.size
      state_item[:] = physics_state[start:start + size]
      start += size

  def copy(self, share_model=False):
    """Creates a copy of this `Physics` instance.

    Args:
      share_model: If True, the copy and the original will share a common
        MjModel instance. By default, both model and data will both be copied.

    Returns:
      A `Physics` instance.
    """
    if not share_model:
      new_model = self.model.copy()
    else:
      new_model = self.model
    new_data = wrapper.MjData(new_model)
    mjlib.mj_copyData(new_data.ptr, new_data.model.ptr, self.data.ptr)
    cls = self.__class__
    new_obj = cls.__new__(cls)
    new_obj._reload_from_data(new_data)  # pylint: disable=protected-access
    return new_obj

  def reset(self):
    """Resets internal variables of the physics simulation."""
    mjlib.mj_resetData(self.model.ptr, self.data.ptr)
    # Disable actuation since we don't yet have meaningful control inputs.
    with self.model.disable('actuation'):
      self.forward()

  def after_reset(self):
    """Runs after resetting internal variables of the physics simulation."""
    # Disable actuation since we don't yet have meaningful control inputs.
    with self.model.disable('actuation'):
      self.forward()

  def forward(self):
    """Recomputes the forward dynamics without advancing the simulation."""
    # Note: `mj_forward` differs from `mj_step1` in that it also recomputes
    # quantities that depend on acceleration (and therefore on the state of the
    # controls). For example `mj_forward` updates accelerometer and gyro
    # readings, whereas `mj_step1` does not.
    # http://www.mujoco.org/book/programming.html#siForward
    with self.check_invalid_state():
      mjlib.mj_forward(self.model.ptr, self.data.ptr)

  @contextlib.contextmanager
  def check_invalid_state(self):
    """Raises a `base.PhysicsError` if the simulation state is invalid."""
    warning_counts_before = [warning.number for warning in self.data.warning]
    yield
    warnings_raised = []
    for i, old_warning_count in enumerate(warning_counts_before):
      if self.data.warning[i].number > old_warning_count:
        warnings_raised.append(i)
    if warnings_raised:
      warning_names = [enums.mjtWarning._fields[i] for i in warnings_raised]
      raise _control.PhysicsError(
          _INVALID_PHYSICS_STATE.format(warning_names=', '.join(warning_names)))

  def __getstate__(self):
    return self.data  # All state is assumed to reside within `self.data`.

  def __setstate__(self, data):
    self._reload_from_data(data)

  def _reload_from_model(self, model):
    """Initializes a new or existing `Physics` from a `wrapper.MjModel`.

    Creates a new `wrapper.MjData` instance, then delegates to
    `_reload_from_data`.

    Args:
      model: Instance of `wrapper.MjModel`.
    """
    data = wrapper.MjData(model)
    self._reload_from_data(data)

  def _reload_from_data(self, data):
    """Initializes a new or existing `Physics` instance from a `wrapper.MjData`.

    Assigns all attributes, sets up named indexing, and creates rendering
    contexts if rendering is enabled.

    The default constructor as well as the other `reload_from` methods should
    delegate to this method.

    Args:
      data: Instance of `wrapper.MjData`.
    """
    self._data = data

    if not render.DISABLED:
      self._make_rendering_contexts()

    # Call kinematics update to enable rendering.
    try:
      self.after_reset()
    except _control.PhysicsError as e:
      logging.warn(str(e))

    # Set up named indexing.
    axis_indexers = index.make_axis_indexers(self.model)
    self._named = NamedIndexStructs(
        model=index.struct_indexer(self.model, 'mjmodel', axis_indexers),
        data=index.struct_indexer(self.data, 'mjdata', axis_indexers),)

  def free(self):
    """Frees the native MuJoCo data structures held by this `Physics` instance.

    This is an advanced feature for use when manual memory management is
    necessary. This `Physics` object MUST NOT be used after this function has
    been called.
    """
    self.data.free()
    self.model.free()
    if self._contexts:
      self._contexts.mujoco.free()
      self._contexts.gl.free()
      self._contexts = None

  @classmethod
  def from_model(cls, model):
    """A named constructor from a `wrapper.MjModel` instance."""
    data = wrapper.MjData(model)
    return cls(data)

  @classmethod
  def from_xml_string(cls, xml_string, assets=None):
    """A named constructor from a string containing an MJCF XML file.

    Args:
      xml_string: XML string containing an MJCF model description.
      assets: Optional dict containing external assets referenced by the model
        (such as additional XML files, textures, meshes etc.), in the form of
        `{filename: contents_string}` pairs. The keys should correspond to the
        filenames specified in the model XML.

    Returns:
      A new `Physics` instance.
    """
    model = wrapper.MjModel.from_xml_string(xml_string, assets=assets)
    return cls.from_model(model)

  @classmethod
  def from_byte_string(cls, byte_string):
    """A named constructor from a model binary as a byte string."""
    model = wrapper.MjModel.from_byte_string(byte_string)
    return cls.from_model(model)

  @classmethod
  def from_xml_path(cls, file_path):
    """A named constructor from a path to an MJCF XML file.

    Args:
      file_path: String containing path to model definition file.

    Returns:
      A new `Physics` instance.
    """
    model = wrapper.MjModel.from_xml_path(file_path)
    return cls.from_model(model)

  @classmethod
  def from_binary_path(cls, file_path):
    """A named constructor from a path to an MJB model binary file.

    Args:
      file_path: String containing path to model definition file.

    Returns:
      A new `Physics` instance.
    """
    model = wrapper.MjModel.from_binary_path(file_path)
    return cls.from_model(model)

  def reload_from_xml_string(self, xml_string, assets=None):
    """Reloads the `Physics` instance from a string containing an MJCF XML file.

    After calling this method, the state of the `Physics` instance is the same
    as a new `Physics` instance created with the `from_xml_string` named
    constructor.

    Args:
      xml_string: XML string containing an MJCF model description.
      assets: Optional dict containing external assets referenced by the model
        (such as additional XML files, textures, meshes etc.), in the form of
        `{filename: contents_string}` pairs. The keys should correspond to the
        filenames specified in the model XML.
    """
    new_model = wrapper.MjModel.from_xml_string(xml_string, assets=assets)
    self._reload_from_model(new_model)

  def reload_from_xml_path(self, file_path):
    """Reloads the `Physics` instance from a path to an MJCF XML file.

    After calling this method, the state of the `Physics` instance is the same
    as a new `Physics` instance created with the `from_xml_path`
    named constructor.

    Args:
      file_path: String containing path to model definition file.
    """
    self._reload_from_model(wrapper.MjModel.from_xml_path(file_path))

  @property
  def named(self):
    return self._named

  def _make_rendering_contexts(self):
    """Creates the OpenGL and MuJoCo rendering contexts."""
    # Forcibly clear the previous GL context to avoid problems with GL
    # implementations which do not support multiple contexts on a given device.
    if self._contexts:
      self._contexts.mujoco.free()
      self._contexts.gl.free()
    # Get the offscreen framebuffer size, as specified in the model XML.
    max_width = self.model.vis.global_.offwidth
    max_height = self.model.vis.global_.offheight
    # Create the OpenGL context.
    render_context = render.Renderer(max_width=max_width, max_height=max_height)
    # Create the MuJoCo context.
    mujoco_context = wrapper.MjrContext(self.model, render_context)
    self._contexts = Contexts(gl=render_context, mujoco=mujoco_context)

  @property
  def contexts(self):
    """Returns a `Contexts` namedtuple, used in `Camera`s and rendering code."""
    if render.DISABLED:
      raise RuntimeError(render.DISABLED_MESSAGE)
    return self._contexts

  @property
  def model(self):
    return self._data.model

  @property
  def data(self):
    return self._data

  def _physics_state_items(self):
    """Returns list of arrays making up internal physics simulation state.

    The physics state consists of the state variables, their derivatives and
    actuation activations.

    Returns:
      List of NumPy arrays containing full physics simulation state.
    """
    return [self.data.qpos[:], self.data.qvel[:], self.data.act[:]]

  # Named views of simulation data.

  def control(self):
    """Returns MuJoCo actuation vector as defined in the model."""
    return self.data.ctrl[:]

  def activation(self):
    """Returns the internal states of 'filter' or 'integrator' actuators.

    For details, please refer to
    http://www.mujoco.org/book/computation.html#geActuation

    Returns:
      Activations in a numpy array.
    """
    return self.data.act[:]

  def state(self):
    """Returns the full physics state. Alias for `get_physics_state`."""
    return np.concatenate(self._physics_state_items())

  def position(self):
    """Returns generalized positions (system configuration)."""
    return self.data.qpos[:]

  def velocity(self):
    """Returns generalized velocities."""
    return self.data.qvel[:]

  def timestep(self):
    """Returns the simulation timestep."""
    return self.model.opt.timestep

  def time(self):
    """Returns episode time in seconds."""
    return self.data.time


class Camera(object):
  """Mujoco scene camera.

  Holds rendering properties such as the width and height of the viewport. The
  camera position and rotation is defined by the Mujoco camera corresponding to
  the `camera_id`. Multiple `Camera` instances may exist for a single
  `camera_id`, for example to render the same view at different resolutions.
  """

  def __init__(self, physics, height=240, width=320, camera_id=-1):
    """Initializes a new `Camera`.

    Args:
      physics: Instance of `Physics`.
      height: Optional image height. Defaults to 240.
      width: Optional image width. Defaults to 320.
      camera_id: Optional camera name or index. Defaults to -1, the free
        camera, which is always defined. A nonnegative integer or string
        corresponds to a fixed camera, which must be defined in the model XML.
        If `camera_id` is a string then the camera must also be named.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
    buffer_width = physics.model.vis.global_.offwidth
    buffer_height = physics.model.vis.global_.offheight
    if width > buffer_width:
      raise ValueError('Image width {} > framebuffer width {}. Either reduce '
                       'the image width or specify a larger offscreen '
                       'framebuffer in the model XML using the clause\n'
                       '<visual>\n'
                       '  <global offwidth="my_width"/>\n'
                       '</visual>'.format(width, buffer_width))
    if height > buffer_height:
      raise ValueError('Image height {} > framebuffer height {}. Either reduce '
                       'the image height or specify a larger offscreen '
                       'framebuffer in the model XML using the clause\n'
                       '<visual>\n'
                       '  <global offheight="my_height"/>\n'
                       '</visual>'.format(height, buffer_height))
    if isinstance(camera_id, six.string_types):
      camera_id = physics.model.name2id(camera_id, 'camera')
    if camera_id < -1:
      raise ValueError('camera_id cannot be smaller than -1.')
    if camera_id >= physics.model.ncam:
      raise ValueError('model has {} fixed cameras. camera_id={} is invalid.'.
                       format(physics.model.ncam, camera_id))

    self._width = width
    self._height = height
    self._physics = physics

    # Variables corresponding to structs needed by Mujoco's rendering functions.
    self._scene = wrapper.MjvScene()
    self._scene_option = wrapper.MjvOption()

    self._perturb = wrapper.MjvPerturb()
    self._perturb.active = 0
    self._perturb.select = 0

    self._rect = types.MJRRECT(0, 0, self._width, self._height)

    self._render_camera = wrapper.MjvCamera()
    self._render_camera.fixedcamid = camera_id

    if camera_id == -1:
      self._render_camera.type_ = enums.mjtCamera.mjCAMERA_FREE
    else:
      # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
      # camera explicitly defined in the model.
      self._render_camera.type_ = enums.mjtCamera.mjCAMERA_FIXED

    # Internal buffers.
    self._rgb_buffer = np.empty((self._height, self._width, 3), dtype=np.uint8)
    self._depth_buffer = np.empty((self._height, self._width), dtype=np.float32)

    if self._physics.contexts.mujoco is not None:
      with self._physics.contexts.gl.make_current() as ctx:
        ctx.call(mjlib.mjr_setBuffer,
                 enums.mjtFramebuffer.mjFB_OFFSCREEN,
                 self._physics.contexts.mujoco.ptr)

  @property
  def width(self):
    """Returns the image width (number of pixels)."""
    return self._width

  @property
  def height(self):
    """Returns the image height (number of pixels)."""
    return self._height

  @property
  def option(self):
    """Returns the camera's visualization options."""
    return self._scene_option

  def update(self, scene_option=None):
    """Updates geometry used for rendering.

    Args:
      scene_option: A custom `wrapper.MjvOption` instance to use to render
        the scene instead of the default.  If None, will use the default.
    """
    scene_option = scene_option or self._scene_option
    mjlib.mjv_updateScene(self._physics.model.ptr, self._physics.data.ptr,
                          scene_option.ptr, self._perturb.ptr,
                          self._render_camera.ptr, enums.mjtCatBit.mjCAT_ALL,
                          self._scene.ptr)

  def _render_on_gl_thread(self, overlays, depth, scene_option):
    """Call mjr_render(), process depth and overlays."""
    self.update(scene_option=scene_option)
    mjlib.mjr_render(self._rect, self._scene.ptr,
                     self._physics.contexts.mujoco.ptr)

    if depth:
      mjlib.mjr_readPixels(None, self._depth_buffer, self._rect,
                           self._physics.contexts.mujoco.ptr)

      # Get distance of near and far clipping planes.
      extent = self._physics.model.stat.extent
      near = self._physics.model.vis.map_.znear * extent
      far = self._physics.model.vis.map_.zfar * extent

      # Convert from [0 1] to depth in meters, see links below.
      # http://stackoverflow.com/a/6657284/1461210
      # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
      self._depth_buffer = near / (1 - self._depth_buffer * (1 - near / far))

    else:
      for overlay in overlays:
        overlay.draw(self._physics.contexts.mujoco.ptr, self._rect)

      mjlib.mjr_readPixels(self._rgb_buffer, None, self._rect,
                           self._physics.contexts.mujoco.ptr)

  def render(self, overlays=(), depth=False, scene_option=None):
    """Renders the camera view as a numpy array of pixel values.

    Args:
      overlays: An optional sequence of `TextOverlay` instances to draw. Only
        supported if `depth` is False.
      depth: An optional boolean. If True make the camera measure depth
      scene_option: A custom `wrapper.MjvOption` instance to use to render
        the scene instead of the default.  If None, will use the default.

    Returns:
      The rendered scene. If `depth` is False this is a NumPy uint8 array of RGB
        values, otherwise it is a float NumPy array of depth values (in meters).

    Raises:
      ValueError: If overlays are requested with depth rendering.
    """

    if depth and overlays:
      raise ValueError('Overlays are not supported with depth rendering.')

    with self._physics.contexts.gl.make_current() as ctx:
      ctx.call(self._render_on_gl_thread, overlays, depth, scene_option)
    return np.flipud(self._depth_buffer if depth else self._rgb_buffer)

  def select(self, cursor_position):
    """Returns bodies and geoms visible at given coordinates in the frame.

    Args:
      cursor_position:  A `tuple` containing x and y coordinates, normalized to
        between 0 and 1, and where (0, 0) is bottom-left.

    Returns:
      A `Selected` namedtuple. Fields are None if nothing is selected.
    """
    self.update()
    aspect_ratio = self._width / self._height
    cursor_x, cursor_y = cursor_position
    pos = np.empty(3, np.double)
    selected_geom = mjlib.mjv_select(
        self._physics.model.ptr,
        self._physics.data.ptr,
        self._scene_option.ptr,
        aspect_ratio,
        cursor_x,
        cursor_y,
        self._scene.ptr,
        pos)

    if selected_geom == -1:  # Nothing was selected.
      return Selected(body=None, geom=None, world_position=None)
    else:
      assert 0 <= selected_geom < self._physics.model.ngeom
      selected_body = self._physics.model.geom_bodyid[selected_geom]
      assert 0 <= selected_body < self._physics.model.nbody
      return Selected(
          body=selected_body, geom=selected_geom, world_position=pos)


class MovableCamera(Camera):
  """Subclass of `Camera` that can be moved by changing its pose.

  A `MovableCamera` always corresponds to a MuJoCo free camera with id -1.
  """

  def __init__(self, physics, height=240, width=320):
    """Initializes a new `MovableCamera`.

    Args:
      physics: Instance of `Physics`.
      height: Optional image height. Defaults to 240.
      width: Optional image width. Defaults to 320.
    """
    super(MovableCamera, self).__init__(
        physics=physics, height=height, width=width, camera_id=-1)

  def get_pose(self):
    """Returns the pose of the camera.

    Returns:
      A `Pose` named tuple with fields:
        lookat: NumPy array specifying lookat point.
        distance: Float specifying distance to `lookat`.
        azimuth: Azimuth in degrees.
        elevation: Elevation in degrees.
    """
    return Pose(self._render_camera.lookat, self._render_camera.distance,
                self._render_camera.azimuth, self._render_camera.elevation)

  def set_pose(self, lookat, distance, azimuth, elevation):
    """Sets the pose of the camera.

    Args:
      lookat: NumPy array or list specifying lookat point.
      distance: Float specifying distance to `lookat`.
      azimuth: Azimuth in degrees.
      elevation: Elevation in degrees.
    """
    self._render_camera.lookat[:] = lookat
    self._render_camera.distance = distance
    self._render_camera.azimuth = azimuth
    self._render_camera.elevation = elevation


class TextOverlay(object):
  """A text overlay that can be drawn on top of a camera view."""

  __slots__ = ('title', 'body', 'style', 'position')

  def __init__(self, title='', body='', style='normal', position='top left'):
    """Initializes a new TextOverlay instance.

    Args:
      title: Title text.
      body: Body text.
      style: The font style. Can be either "normal", "shadow", or "big".
      position: The grid position of the overlay. Can be either "top left",
        "top right", "bottom left", or "bottom right".
    """
    self.title = title
    self.body = body
    self.style = _FONT_STYLES[style]
    self.position = _GRID_POSITIONS[position]

  def draw(self, context, rect):
    """Draws the overlay.

    Args:
      context: A `types.MJRCONTEXT` pointer.
      rect: A `types.MJRRECT`.
    """
    mjlib.mjr_overlay(self.style,
                      self.position,
                      rect,
                      util.to_binary_string(self.title),
                      util.to_binary_string(self.body),
                      context)


def action_spec(physics):
  """Returns a `BoundedArraySpec` matching the `physics` actuators."""
  num_actions = physics.model.nu
  is_limited = physics.model.actuator_ctrllimited.ravel().astype(np.bool)
  control_range = physics.model.actuator_ctrlrange
  minima = np.full(num_actions, fill_value=-np.inf, dtype=np.float)
  maxima = np.full(num_actions, fill_value=np.inf, dtype=np.float)
  minima[is_limited], maxima[is_limited] = control_range[is_limited].T

  return specs.BoundedArraySpec(
      shape=(num_actions,), dtype=np.float, minimum=minima, maximum=maxima)
