# Copyright 2020 The dm_control Authors.
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
"""Bowl arena with bumps."""


from dm_control import composer
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
from dm_control.mujoco.wrapper import mjbindings
import numpy as np
from scipy import ndimage

mjlib = mjbindings.mjlib

_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1

# Constants related to terrain generation.
_TERRAIN_SMOOTHNESS = .5  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = .2  # Spatial scale of terrain bumps (in meters).


class Bowl(composer.Arena):
  """A bowl arena with sinusoidal bumps."""

  def _build(self, size=(10, 10), aesthetic='default', name='bowl'):
    super()._build(name=name)

    self._hfield = self._mjcf_root.asset.add(
        'hfield',
        name='terrain',
        nrow=201,
        ncol=201,
        size=(6, 6, 0.5, 0.1))

    if aesthetic != 'default':
      ground_info = locomotion_arenas_assets.get_ground_texture_info(aesthetic)
      sky_info = locomotion_arenas_assets.get_sky_texture_info(aesthetic)
      texturedir = locomotion_arenas_assets.get_texturedir(aesthetic)
      self._mjcf_root.compiler.texturedir = texturedir

      self._texture = self._mjcf_root.asset.add(
          'texture', name='aesthetic_texture', file=ground_info.file,
          type=ground_info.type)
      self._material = self._mjcf_root.asset.add(
          'material', name='aesthetic_material', texture=self._texture,
          texuniform='true')
      self._skybox = self._mjcf_root.asset.add(
          'texture', name='aesthetic_skybox', file=sky_info.file,
          type='skybox', gridsize=sky_info.gridsize,
          gridlayout=sky_info.gridlayout)
      self._terrain_geom = self._mjcf_root.worldbody.add(
          'geom',
          name='terrain',
          type='hfield',
          pos=(0, 0, -0.01),
          hfield='terrain',
          material=self._material)
      self._ground_geom = self._mjcf_root.worldbody.add(
          'geom',
          type='plane',
          name='groundplane',
          size=list(size) + [0.5],
          material=self._material)
    else:
      self._terrain_geom = self._mjcf_root.worldbody.add(
          'geom',
          name='terrain',
          type='hfield',
          rgba=(0.2, 0.3, 0.4, 1),
          pos=(0, 0, -0.01),
          hfield='terrain')
      self._ground_geom = self._mjcf_root.worldbody.add(
          'geom',
          type='plane',
          name='groundplane',
          rgba=(0.2, 0.3, 0.4, 1),
          size=list(size) + [0.5])

    self._mjcf_root.visual.headlight.set_attributes(
        ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

    self._regenerate = True

  def regenerate(self, random_state):
    # regeneration of the bowl requires physics, so postponed to initialization.
    self._regenerate = True

  def initialize_episode(self, physics, random_state):
    if self._regenerate:
      self._regenerate = False

      # Get heightfield resolution, assert that it is square.
      res = physics.bind(self._hfield).nrow
      assert res == physics.bind(self._hfield).ncol

      # Sinusoidal bowl shape.
      row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
      radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), .1, 1)
      bowl_shape = .5 - np.cos(2*np.pi*radius)/2

      # Random smooth bumps.
      terrain_size = 2 * physics.bind(self._hfield).size[0]
      bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
      bumps = random_state.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
      smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))

      # Terrain is elementwise product.
      terrain = bowl_shape * smooth_bumps
      start_idx = physics.bind(self._hfield).adr
      physics.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()

      # If we have a rendering context, we need to re-upload the modified
      # heightfield data.
      if physics.contexts:
        with physics.contexts.gl.make_current() as ctx:
          ctx.call(mjlib.mjr_uploadHField,
                   physics.model.ptr,
                   physics.contexts.mujoco.ptr,
                   physics.bind(self._hfield).element_id)

  @property
  def ground_geoms(self):
    return (self._terrain_geom, self._ground_geom)
