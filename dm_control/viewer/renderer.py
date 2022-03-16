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
"""Renderer module."""

import abc
import contextlib

from dm_control.mujoco import wrapper
from dm_control.viewer import util
import mujoco
import numpy as np


# Fixed camera -1 is the free (unfixed) camera, and each fixed camera has
# a positive index in range (0, self._model.ncam).
_FREE_CAMERA_INDEX = -1

# Index used to distinguish when a camera isn't tracking any particular body.
_NO_BODY_TRACKED_INDEX = -1

# Index used to distinguish a non-existing or an invalid body.
_INVALID_BODY_INDEX = -1

# Zoom factor used when zooming in on the entire scene.
_FULL_SCENE_ZOOM_FACTOR = 1.5

# Default values for `MjvScene.flags`. These are the same as the defaults set by
# `mjv_defaultScene`, except that we disable `mjRND_HAZE`.
_DEFAULT_RENDER_FLAGS = np.zeros(mujoco.mjtRndFlag.mjNRNDFLAG, dtype=np.ubyte)
_DEFAULT_RENDER_FLAGS[mujoco.mjtRndFlag.mjRND_SHADOW.value] = 1
_DEFAULT_RENDER_FLAGS[mujoco.mjtRndFlag.mjRND_REFLECTION.value] = 1
_DEFAULT_RENDER_FLAGS[mujoco.mjtRndFlag.mjRND_SKYBOX.value] = 1


class BaseRenderer(metaclass=abc.ABCMeta):
  """A base class for component-based Mujoco Renderers implementations.

  Attributes:
    components: A set of RendererComponent the renderer will render in addition
      to rendering the physics scene. Being a QuietSet instance, it supports
      adding and removing of components using += and -= operators.
    screen_capture_components: Components that perform screen capture and need
      a guarantee to be called when all other elements have been rendered.
  """

  def __init__(self):
    """Instance initializer."""
    self.components = util.QuietSet()
    self.screen_capture_components = util.QuietSet()

  def _render_components(self, context, viewport):
    """Renders the components.

    Args:
      context: MjrContext instance.
      viewport: Viewport instance.
    """
    for component in self.components:
      component.render(context, viewport)
    for component in self.screen_capture_components:
      component.render(context, viewport)


class Component(metaclass=abc.ABCMeta):
  """Components are a way to introduce extra rendering content.

  They are invoked after the main rendering pass, allowing to draw extra images
  into the render buffer, such as overlays.
  """

  @abc.abstractmethod
  def render(self, context, viewport):
    """Renders the component.

    Args:
      context: MjrContext instance.
      viewport: Viewport instance.
    """
    pass


class NullRenderer:
  """A stub off-screen renderer used when no other renderer is available."""

  def __init__(self):
    """Instance initializer."""
    self._black = np.zeros((1, 1, 3), dtype=np.uint8)

  def release(self):
    pass

  @property
  def pixels(self):
    """Returns a black pixel map."""
    return self._black


class OffScreenRenderer(BaseRenderer):
  """A Mujoco renderer that renders to an off-screen surface."""

  def __init__(self, model, surface):
    """Instance initializer.

    Args:
      model: instance of MjModel.
      surface: instance of dm_control.render.BaseContext.
    """
    super().__init__()
    self._surface = surface
    self._surface.increment_refcount()
    self._model = model
    self._mujoco_context = None

    self._prev_viewport = np.ones(2)
    self._rgb_buffer = np.empty((1, 1, 3), dtype=np.uint8)
    self._pixels = np.zeros((1, 1, 3), dtype=np.uint8)

  def render(self, viewport, scene):
    """Renders the scene to the specified viewport.

    Args:
      viewport: Instance of Viewport.
      scene: Instance of MjvScene.
    Returns:
      A 3-dimensional array of shape (viewport.width, viewport.height, 3),
      with the contents of the front buffer.
    """
    if not np.array_equal(self._prev_viewport, viewport.dimensions):
      self._prev_viewport = viewport.dimensions
      if self._mujoco_context:
        self._mujoco_context.free()
      self._mujoco_context = None
    if not self._mujoco_context:
      # Ensure that MuJoCo's offscreen framebuffer is large enough to
      # accommodate the viewport.
      new_offwidth = max(self._model.vis.global_.offwidth, viewport.width)
      new_offheight = max(self._model.vis.global_.offheight, viewport.height)
      self._model.vis.global_.offwidth = new_offwidth
      self._model.vis.global_.offheight = new_offheight
      self._mujoco_context = wrapper.MjrContext(
          model=self._model,
          gl_context=self._surface,
          font_scale=mujoco.mjtFontScale.mjFONTSCALE_100)
      self._rgb_buffer = np.empty(
          (viewport.height, viewport.width, 3), dtype=np.uint8)

    with self._surface.make_current() as ctx:
      ctx.call(self._render_on_gl_thread, viewport, scene)
    self._pixels = np.flipud(self._rgb_buffer)

  def _render_on_gl_thread(self, viewport, scene):
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
                         self._mujoco_context.ptr)
    mujoco.mjr_render(viewport.mujoco_rect, scene.ptr, self._mujoco_context.ptr)
    self._render_components(self._mujoco_context, viewport)
    mujoco.mjr_readPixels(self._rgb_buffer, None, viewport.mujoco_rect,
                          self._mujoco_context.ptr)

  def release(self):
    """Releases the render context and related resources."""
    if self._mujoco_context:
      self._mujoco_context.free()
      self._mujoco_context = None
      self._surface.decrement_refcount()
      self._surface.free()

  @property
  def pixels(self):
    """Returns the rendered image."""
    return self._pixels


class Perturbation:
  """A proxy that allows to move a scene object."""

  def __init__(self, body_id, model, data, scene):
    """Instance initializer.

    Args:
      body_id: A positive integer, ID of the body to manipulate.
      model: MjModel instance.
      data: MjData instance.
      scene: MjvScene instance.
    """
    self._body_id = body_id
    self._model = model
    self._data = data
    self._scene = scene
    self._action = mujoco.mjtMouse.mjMOUSE_NONE

    self._perturb = wrapper.MjvPerturb()
    self._perturb.select = self._body_id
    self._perturb.active = 0

    mujoco.mjv_initPerturb(self._model.ptr, self._data.ptr, self._scene.ptr,
                           self._perturb.ptr)

  def start_move(self, action, grab_pos):
    """Starts a movement action."""
    if action is None or grab_pos is None:
      return

    mujoco.mjv_initPerturb(self._model.ptr, self._data.ptr, self._scene.ptr,
                           self._perturb.ptr)
    self._action = action
    move_actions = np.array(
        [mujoco.mjtMouse.mjMOUSE_MOVE_V, mujoco.mjtMouse.mjMOUSE_MOVE_H])
    if any(move_actions == action):
      self._perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
    else:
      self._perturb.active = mujoco.mjtPertBit.mjPERT_ROTATE

    body_pos = self._data.xpos[self._body_id]
    body_mat = self._data.xmat[self._body_id].reshape(3, 3)
    grab_local_pos = body_mat.T.dot(grab_pos - body_pos)
    self._perturb.localpos[:] = grab_local_pos

  def tick_move(self, viewport_offset):
    """Transforms object's location/rotation by the specified amount."""
    if self._action:
      mujoco.mjv_movePerturb(self._model.ptr, self._data.ptr, self._action,
                             viewport_offset[0], viewport_offset[1],
                             self._scene.ptr, self._perturb.ptr)

  def end_move(self):
    """Ends a movement operation."""
    self._action = mujoco.mjtMouse.mjMOUSE_NONE
    self._perturb.active = 0

  @contextlib.contextmanager
  def apply(self, paused):
    """Applies the modifications introduced by performing the move operation."""
    mujoco.mjv_applyPerturbPose(self._model.ptr, self._data.ptr,
                                self._perturb.ptr, int(paused))
    if not paused:
      mujoco.mjv_applyPerturbForce(self._model.ptr, self._data.ptr,
                                   self._perturb.ptr)
    yield
    self._data.xfrc_applied[self._perturb.select] = 0

  @property
  def ptr(self):
    """Returns the underlying Mujoco Perturbation object."""
    return self._perturb.ptr

  @property
  def body_id(self):
    """A positive integer, ID of the manipulated body."""
    return self._body_id


class NullPerturbation:
  """An empty perturbation.

  A null-object pattern, used to avoid cumbersome if clauses.
  """

  @contextlib.contextmanager
  def apply(self, paused):
    """Activates/deactivates the null context."""
    del paused
    yield

  @property
  def ptr(self):
    """Returns None, because this class represents an empty perturbation."""
    return None


class RenderSettings:
  """Renderer settings."""

  def __init__(self):
    self._visualization_options = wrapper.MjvOption()
    self._stereo_mode = mujoco.mjtStereo.mjSTEREO_NONE
    self._render_flags = _DEFAULT_RENDER_FLAGS

  @property
  def visualization(self):
    """Returns scene visualization options."""
    return self._visualization_options

  @property
  def render_flags(self):
    """Returns the render flags."""
    return self._render_flags

  @property
  def visualization_flags(self):
    """Returns scene visualization flags."""
    return self._visualization_options.flags

  @property
  def geom_groups(self):
    """Returns geom groups visibility flags."""
    return self._visualization_options.geomgroup

  @property
  def site_groups(self):
    """Returns site groups visibility flags."""
    return self._visualization_options.sitegroup

  def apply_settings(self, scene):
    """Applies settings to the specified scene.

    Args:
      scene: Instance of MjvScene.
    """
    scene.stereo = self._stereo_mode
    scene.flags[:] = self._render_flags[:]

  def toggle_rendering_flag(self, flag_index):
    """Toggles the specified rendering flag."""
    self._render_flags[flag_index] = not self._render_flags[flag_index]

  def toggle_visualization_flag(self, flag_index):
    """Toggles the specified visualization flag."""
    self._visualization_options.flags[flag_index] = (
        not self._visualization_options.flags[flag_index])

  def toggle_geom_group(self, group_index):
    """Toggles the specified geom group visible or not."""
    self._visualization_options.geomgroup[group_index] = (
        not self._visualization_options.geomgroup[group_index])

  def toggle_site_group(self, group_index):
    """Toggles the specified site group visible or not."""
    self._visualization_options.sitegroup[group_index] = (
        not self._visualization_options.sitegroup[group_index])

  def toggle_stereo_buffering(self):
    """Toggles the double buffering mode on/off."""
    if self._stereo_mode == mujoco.mjtStereo.mjSTEREO_NONE:
      self._stereo_mode = mujoco.mjtStereo.mjSTEREO_QUADBUFFERED
    else:
      self._stereo_mode = mujoco.mjtStereo.mjSTEREO_NONE

  def select_next_rendering_mode(self):
    """Cycles to the next rendering mode."""
    self._visualization_options.frame = (
        (self._visualization_options.frame + 1) % mujoco.mjtFrame.mjNFRAME)

  def select_prev_rendering_mode(self):
    """Cycles to the previous rendering mode."""
    self._visualization_options.frame = (
        (self._visualization_options.frame - 1) % mujoco.mjtFrame.mjNFRAME)

  def select_next_labeling_mode(self):
    """Cycles to the next scene object labeling mode."""
    self._visualization_options.label = (
        (self._visualization_options.label + 1) % mujoco.mjtLabel.mjNLABEL)

  def select_prev_labeling_mode(self):
    """Cycles to the previous scene object labeling mode."""
    self._visualization_options.label = (
        (self._visualization_options.label - 1) % mujoco.mjtLabel.mjNLABEL)


class SceneCamera:
  """A camera used to navigate around and render the scene."""

  def __init__(self,
               model,
               data,
               options,
               settings=None,
               zoom_factor=_FULL_SCENE_ZOOM_FACTOR):
    """Instance initializer.

    Args:
      model: MjModel instance.
      data: MjData instance.
      options: RenderSettings instance.
      settings: Optional, internal camera settings obtained from another
        SceneCamera instance using 'settings' property.
      zoom_factor: The initial zoom factor for zooming into the scene.
    """
    # Design notes:
    # We need to recreate the camera for each new model, because each model
    # defines different fixed cameras and objects to track, and therefore
    # severely the parameters of this class.
    self._scene = wrapper.MjvScene(model)
    self._data = data
    self._model = model
    self._options = options

    self._camera = wrapper.MjvCamera()
    self._camera.trackbodyid = _NO_BODY_TRACKED_INDEX
    self._camera.fixedcamid = _FREE_CAMERA_INDEX
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_FREE
    self._zoom_factor = zoom_factor

    if settings is not None:
      self._settings = settings
      self.settings = settings
    else:
      self._settings = self._camera

  def set_freelook_mode(self):
    """Enables 6 degrees of freedom of movement for the camera."""
    self._camera.trackbodyid = _NO_BODY_TRACKED_INDEX
    self._camera.fixedcamid = _FREE_CAMERA_INDEX
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_FREE

  def set_tracking_mode(self, body_id):
    """Latches the camera onto the specified body.

    Leaves the user only 3 degrees of freedom to rotate the camera.

    Args:
      body_id: A positive integer, ID of the body to track.
    """
    if body_id < 0:
      return
    self._camera.trackbodyid = body_id
    self._camera.fixedcamid = _FREE_CAMERA_INDEX
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_TRACKING

  def set_fixed_mode(self, fixed_camera_id):
    """Fixes the camera in a pre-defined position, taking away all DOF.

    Args:
      fixed_camera_id: A positive integer, Id of a fixed camera defined in the
        scene.
    """
    if fixed_camera_id < 0:
      return
    self._camera.trackbodyid = _NO_BODY_TRACKED_INDEX
    self._camera.fixedcamid = fixed_camera_id
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_FIXED

  def look_at(self, position, distance):
    """Positions the camera so that it's focused on the specified point."""
    self._camera.lookat[:] = position
    self._camera.distance = distance

  def move(self, action, viewport_offset):
    """Moves the camera around the scene."""
    # Not checking the validity of arguments on purpose. This method is designed
    # to be called very often, so in order to avoid the overhead, all arguments
    # are assumed to be valid.
    mujoco.mjv_moveCamera(self._model.ptr, action, viewport_offset[0],
                          viewport_offset[1], self._scene.ptr, self._camera.ptr)

  def new_perturbation(self, body_id):
    """Creates a proxy that allows to manipulate the specified object."""
    return Perturbation(body_id, self._model, self._data, self._scene)

  def raycast(self, viewport, screen_pos):
    """Shoots a ray from the specified viewport position into the scene."""
    if not self.is_initialized:
      return -1, None
    viewport_pos = viewport.screen_to_inverse_viewport(screen_pos)
    grab_world_pos = np.empty(3, dtype=np.double)
    selected_geom_id_arr = np.intc([-1])
    selected_skin_id_arr = np.intc([-1])
    selected_body_id = mujoco.mjv_select(
        self._model.ptr,
        self._data.ptr,
        self._options.visualization.ptr,
        viewport.aspect_ratio,
        viewport_pos[0],
        viewport_pos[1],
        self._scene.ptr,
        grab_world_pos,
        selected_geom_id_arr,
        selected_skin_id_arr,
    )
    del selected_geom_id_arr, selected_skin_id_arr  # Unused.
    if selected_body_id < 0:
      selected_body_id = _INVALID_BODY_INDEX
      grab_world_pos = None
    return selected_body_id, grab_world_pos

  def render(self, perturbation=None):
    """Renders the scene form this camera's perspective.

    Args:
      perturbation: (Optional), instance of Perturbation.
    Returns:
      Rendered scene, instance of MjvScene.
    """
    perturb_to_render = perturbation.ptr if perturbation else None
    mujoco.mjv_updateScene(self._model.ptr, self._data.ptr,
                           self._options.visualization.ptr, perturb_to_render,
                           self._camera.ptr, mujoco.mjtCatBit.mjCAT_ALL,
                           self._scene.ptr)
    return self._scene

  def zoom_to_scene(self):
    """Zooms in on the entire scene."""
    self.look_at(self._model.stat.center[:],
                 self._zoom_factor * self._model.stat.extent)

    self.settings = self._settings

  @property
  def transform(self):
    """Returns a tuple with camera transform.

    The transform comes in form: (3x3 rotation mtx, 3-component position).
    """
    pos = np.zeros(3)
    forward = np.zeros(3)
    up = np.zeros(3)
    for i in range(3):
      forward[i] = self._scene.camera[0].forward[i]
      up[i] = self._scene.camera[0].up[i]
      pos[i] = (self._scene.camera[0].pos[i] + self._scene.camera[1].pos[i]) / 2
    right = np.cross(forward, up)
    return np.array([right, up, forward]), pos

  @property
  def settings(self):
    """Returns internal camera settings."""
    return self._camera

  @settings.setter
  def settings(self, value):
    """Restores the camera settings."""
    self._camera.type_ = value.type_
    self._camera.fixedcamid = value.fixedcamid
    self._camera.trackbodyid = value.trackbodyid
    self._camera.lookat[:] = value.lookat[:]
    self._camera.distance = value.distance
    self._camera.azimuth = value.azimuth
    self._camera.elevation = value.elevation

  @property
  def name(self):
    """Name of the active camera."""
    if self._camera.type_ == mujoco.mjtCamera.mjCAMERA_TRACKING:
      body_name = self._model.id2name(self._camera.trackbodyid, 'body')
      if body_name:
        return 'Tracking body "%s"' % body_name
      else:
        return 'Tracking body id %d' % self._camera.trackbodyid
    elif self._camera.type_ == mujoco.mjtCamera.mjCAMERA_FIXED:
      camera_name = self._model.id2name(self._camera.fixedcamid, 'camera')
      if camera_name:
        return str(camera_name)
      else:
        return str(self._camera.fixedcamid)
    else:
      return 'Free'

  @property
  def mode(self):
    """Index of the mode the camera is currently in."""
    return self._camera.type_

  @property
  def is_initialized(self):
    """Returns True if camera is properly initialized."""
    if not self._scene:
      return False
    frustum_near = self._scene.camera[0].frustum_near
    frustum_far = self._scene.camera[0].frustum_far
    return frustum_near > 0 and frustum_near < frustum_far


class Viewport:
  """Render viewport."""

  def __init__(self, width=1, height=1):
    """Instance initializer.

    Args:
      width: Viewport width, in pixels.
      height: Viewport height, in pixels.
    """
    self._screen_size = mujoco.MjrRect(0, 0, width, height)

  def set_size(self, width, height):
    """Changes the viewport size.

    Args:
      width: Viewport width, in pixels.
      height: Viewport height, in pixels.
    """
    self._screen_size.width = width
    self._screen_size.height = height

  def screen_to_viewport(self, screen_coordinates):
    """Converts screen coordinates to viewport coordinates.

    Args:
      screen_coordinates: 2-component tuple, with components being integral
        numbers in range defined by the screen/window resolution.
    Returns:
      A 2-component tuple, with components being floating point values in range
      [0, 1].
    """
    x = screen_coordinates[0] / self._screen_size.width
    y = screen_coordinates[1] / self._screen_size.height
    return np.array([x, y], np.float32)

  def screen_to_inverse_viewport(self, screen_coordinates):
    """Converts screen coordinates to viewport coordinates flipped vertically.

    Args:
      screen_coordinates: 2-component tuple, with components being integral
        numbers in range defined by the screen/window resolution.
    Returns:
      A 2-component tuple, with components being floating point values in range
      [0, 1]. The height component value will be flipped, with 1 at the top, and
      0 at the bottom of the viewport.
    """
    x = screen_coordinates[0] / self._screen_size.width
    y = 1. - (screen_coordinates[1] / self._screen_size.height)
    return np.array([x, y], np.float32)

  @property
  def aspect_ratio(self):
    return self._screen_size.width / self._screen_size.height

  @property
  def mujoco_rect(self):
    """Instance of MJRRECT with viewport dimensions."""
    return self._screen_size

  @property
  def dimensions(self):
    """Viewport dimensions in form of a 2-component vector."""
    return np.asarray([self._screen_size.width, self._screen_size.height])

  @property
  def width(self):
    """Viewport width."""
    return self._screen_size.width

  @property
  def height(self):
    """Viewport height."""
    return self._screen_size.height
