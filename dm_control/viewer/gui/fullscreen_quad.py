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

"""OpenGL utility for rendering numpy arrays as images on a quad surface."""

import ctypes
import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

# This array contains packed position and texture cooridnates of a fullscreen
# quad.
# It contains definition of 4 vertices that will be rendered as a triangle
# strip. Each vertex is described by a tuple:
# (VertexPosition.X, VertexPosition.Y, TextureCoord.U, TextureCoord.V)
_FULLSCREEN_QUAD_VERTEX_POSITONS_AND_TEXTURE_COORDS = np.array([
    -1, -1, 0, 1,
    -1, 1, 0, 0,
    1, -1, 1, 1,
    1, 1, 1, 0], dtype=np.float32)
_FLOATS_PER_XY = 2
_FLOATS_PER_VERTEX = 4
_SIZE_OF_FLOAT = ctypes.sizeof(ctypes.c_float)

_VERTEX_SHADER = """
#version 120
attribute vec2 position;
attribute vec2 uv;
void main() {
  gl_Position = vec4(position, 0, 1);
  gl_TexCoord[0].st = uv;
}
"""
_FRAGMENT_SHADER = """
#version 120
uniform sampler2D tex;
void main() {
  gl_FragColor = texture2D(tex, gl_TexCoord[0].st);
}
"""
_VAR_POSITION = 'position'
_VAR_UV = 'uv'
_VAR_TEXTURE_SAMPLER = 'tex'


class FullscreenQuadRenderer:
  """Renders pixmaps on a fullscreen quad using OpenGL."""

  def __init__(self):
    """Initializes the fullscreen quad renderer."""
    GL.glClearColor(0, 0, 0, 0)
    self._init_geometry()
    self._init_texture()
    self._init_shaders()

  def _init_geometry(self):
    """Initializes the fullscreen quad geometry."""
    vertex_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer)
    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        _FULLSCREEN_QUAD_VERTEX_POSITONS_AND_TEXTURE_COORDS.nbytes,
        _FULLSCREEN_QUAD_VERTEX_POSITONS_AND_TEXTURE_COORDS, GL.GL_STATIC_DRAW)

  def _init_texture(self):
    """Initializes the texture storage."""
    self._texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)

  def _init_shaders(self):
    """Initializes the shaders used to render the textures fullscreen quad."""
    vs = shaders.compileShader(_VERTEX_SHADER, GL.GL_VERTEX_SHADER)
    fs = shaders.compileShader(_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
    self._shader = shaders.compileProgram(vs, fs)

    stride = _FLOATS_PER_VERTEX * _SIZE_OF_FLOAT
    var_position = GL.glGetAttribLocation(self._shader, _VAR_POSITION)
    GL.glVertexAttribPointer(
        var_position, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
    GL.glEnableVertexAttribArray(var_position)

    var_uv = GL.glGetAttribLocation(self._shader, _VAR_UV)
    uv_offset = ctypes.c_void_p(_FLOATS_PER_XY * _SIZE_OF_FLOAT)
    GL.glVertexAttribPointer(
        var_uv, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, uv_offset)
    GL.glEnableVertexAttribArray(var_uv)

    self._var_texture_sampler = GL.glGetUniformLocation(
        self._shader, _VAR_TEXTURE_SAMPLER)

  def render(self, pixmap, viewport_shape):
    """Renders the pixmap on a fullscreen quad.

    Args:
      pixmap: A 3D numpy array of bytes (np.uint8), with dimensions
        (width, height, 3).
      viewport_shape: A tuple of two elements, (width, height).
    """
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glViewport(0, 0, *viewport_shape)
    GL.glUseProgram(self._shader)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, pixmap.shape[1],
                    pixmap.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                    pixmap)
    GL.glUniform1i(self._var_texture_sampler, 0)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
