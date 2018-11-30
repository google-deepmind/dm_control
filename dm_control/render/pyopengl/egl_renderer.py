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

"""An OpenGL renderer backed by EGL, provided through PyOpenGL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import ctypes
import os

from dm_control.render import base
from dm_control.render import constants
from dm_control.render import executor

PYOPENGL_PLATFORM = os.environ.get(constants.PYOPENGL_PLATFORM)

if not PYOPENGL_PLATFORM:
  os.environ[constants.PYOPENGL_PLATFORM] = constants.EGL
elif PYOPENGL_PLATFORM != constants.EGL:
  raise ImportError(
      'Cannot use EGL rendering platform. '
      'The PYOPENGL_PLATFORM environment variable is set to {!r} '
      '(should be either unset or {!r}).'
      .format(PYOPENGL_PLATFORM, constants.EGL))


# pylint: disable=g-import-not-at-top
from dm_control.render.pyopengl import egl_ext as EGL


def create_initialized_headless_egl_display():
  """Creates an initialized EGL display directly on a device."""
  display = EGL.EGL_NO_DISPLAY
  devices = EGL.eglQueryDevicesEXT()
  for device in devices:
    display = EGL.eglGetPlatformDisplayEXT(
        EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
    if display and EGL.eglGetError() == EGL.EGL_SUCCESS:
      initialized = EGL.eglInitialize(display, None, None)
      if EGL.eglGetError() == EGL.EGL_SUCCESS and initialized == EGL.EGL_TRUE:
        break
      else:
        display = EGL.EGL_NO_DISPLAY
  return display


EGL_DISPLAY = create_initialized_headless_egl_display()
if EGL_DISPLAY == EGL.EGL_NO_DISPLAY:
  raise ImportError('Cannot initialize a headless EGL display.')
atexit.register(EGL.eglTerminate, EGL_DISPLAY)


EGL_ATTRIBUTES = (
    EGL.EGL_RED_SIZE, 8,
    EGL.EGL_GREEN_SIZE, 8,
    EGL.EGL_BLUE_SIZE, 8,
    EGL.EGL_ALPHA_SIZE, 8,
    EGL.EGL_DEPTH_SIZE, 24,
    EGL.EGL_STENCIL_SIZE, 8,
    EGL.EGL_COLOR_BUFFER_TYPE, EGL.EGL_RGB_BUFFER,
    EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
    EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
    EGL.EGL_NONE
)


class EGLContext(base.ContextBase):
  """An OpenGL context backed by EGL."""

  def __init__(self, max_width, max_height):
    # EGLContext currently only works with `PassthroughRenderExecutor`.
    # TODO(b/110927854) Make this work with the offloading executor.
    super(EGLContext, self).__init__(max_width, max_height,
                                     executor.PassthroughRenderExecutor)

  def _platform_init(self, unused_max_width, unused_max_height):
    """Initialization this EGL context."""
    num_configs = ctypes.c_long()
    config_size = 1
    config = EGL.EGLConfig()
    EGL.eglReleaseThread()
    EGL.eglChooseConfig(
        EGL_DISPLAY,
        EGL_ATTRIBUTES,
        ctypes.byref(config),
        config_size,
        num_configs)
    if num_configs.value < 1:
      raise RuntimeError(
          'EGL failed to find a framebuffer configuration that matches the '
          'desired attributes: {}'.format(EGL_ATTRIBUTES))
    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    self._context = EGL.eglCreateContext(
        EGL_DISPLAY, config, EGL.EGL_NO_CONTEXT, None)
    if not self._context:
      raise RuntimeError('Cannot create an EGL context.')

  def _platform_make_current(self):
    if self._context:
      success = EGL.eglMakeCurrent(
          EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, self._context)
      if not success:
        raise RuntimeError('Failed to make the EGL context current.')

  def _platform_free(self):
    """Frees resources associated with this context."""
    if self._context:
      current_context = EGL.eglGetCurrentContext()
      if current_context and self._context.address == current_context.address:
        EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE,
                           EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
      EGL.eglDestroyContext(EGL_DISPLAY, self._context)
    self._context = None
