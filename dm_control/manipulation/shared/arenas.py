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

"""Suite-specific arena class."""


from dm_control import composer


class Standard(composer.Arena):
  """Suite-specific subclass of the standard Composer arena."""

  def _build(self, name=None):
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    super(Standard, self)._build(name=name)

    # Add visual assets.
    self.mjcf_model.asset.add(
        'texture',
        type='skybox',
        builtin='gradient',
        rgb1=(0.4, 0.6, 0.8),
        rgb2=(0., 0., 0.),
        width=100,
        height=100)
    groundplane_texture = self.mjcf_model.asset.add(
        'texture',
        name='groundplane',
        type='2d',
        builtin='checker',
        rgb1=(0.2, 0.3, 0.4),
        rgb2=(0.1, 0.2, 0.3),
        width=300,
        height=300,
        mark='edge',
        markrgb=(.8, .8, .8))
    groundplane_material = self.mjcf_model.asset.add(
        'material',
        name='groundplane',
        texture=groundplane_texture,
        texrepeat=(5, 5),
        texuniform='true',
        reflectance=0.2)

    # Add ground plane.
    self.mjcf_model.worldbody.add(
        'geom',
        name='ground',
        type='plane',
        material=groundplane_material,
        size=(1, 1, 0.1),
        friction=(0.4,),
        solimp=(0.95, 0.99, 0.001),
        solref=(0.002, 1))

    # Add lighting
    self.mjcf_model.worldbody.add(
        'light',
        pos=(0, 0, 1.5),
        dir=(0, 0, -1),
        diffuse=(0.7, 0.7, 0.7),
        specular=(.3, .3, .3),
        directional='false',
        castshadow='true')

    # Always initialize the free camera so that it points at the origin.
    self.mjcf_model.statistic.center = (0., 0., 0.)

  def attach_offset(self, entity, offset, attach_site=None):
    """Attaches another entity at a position offset from the attachment site.

    Args:
      entity: The `Entity` to attach.
      offset: A length 3 array-like object representing the XYZ offset.
      attach_site: (optional) The site to which to attach the entity's model.
        If not set, defaults to self.attachment_site.
    Returns:
      The frame of the attached model.
    """
    frame = self.attach(entity, attach_site=attach_site)
    frame.pos = offset
    return frame
