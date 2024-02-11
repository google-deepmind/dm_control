# Copyright 2023 The dm_control Authors.
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

"""Testing utilities."""
from typing import Any, Optional, Sequence
from unittest import mock

from dm_control.blender.fake_core import bpy


class FakePropCollection:
  """Collection that simulates bpy_prop_collection.

  Armature's bone collection is modeled using it, and the tested code depends
  on some of its features - namely the conversion to list produces dict
  values instead of its keys, as well as the collection behaving like a dict
  otherwise.

  @see
  blender_scene.py : map_blender_tree - iterating over armature.data.bones
  blender_scene.py : ObjectRef.parent.object_parent - dereferencing bone
    by its name armature.data.bones[obj.parent_bone]
  """

  def __init__(self, objects_with_name):
    if objects_with_name:
      self._keys = [i.name for i in objects_with_name]
    else:
      self._keys = []
    self._values = objects_with_name or []

  def __len__(self):
    return len(self._values)

  def __getitem__(self, idx):
    if isinstance(idx, str):
      idx = self._keys.index(idx)
    return self._values[idx]

  def __iter__(self):
    return iter(self._values)

  def __contains__(self, key):
    return key in self._keys

  def keys(self):
    return self._keys

  def values(self):
    return self._values

  def items(self):
    return list(zip(self._keys, self._values))


def build_armature(
    name: str,
    parent: Optional[Any] = None,
    parent_bone: Optional[str] = None,
    bones: Optional[Any] = None,
) -> ...:
  """TBD."""
  obj = mock.MagicMock(spec=bpy.types.Object, type='ARMATURE')
  obj.name = name
  obj.parent = parent
  obj.parent_bone = parent_bone
  obj.data = mock.MagicMock()
  obj.data.bones = FakePropCollection(bones)
  obj.pose = mock.MagicMock()
  obj.pose.bones = FakePropCollection(bones)
  return obj


def build_bone(name, parent=None, constraint=None):
  """Builds a mock bone."""
  bone = mock.MagicMock(spec=bpy.types.Bone)
  bone.name = name
  bone.parent = parent
  bone.is_in_ik_chain = constraint is not None
  bone.lock_ik_x = constraint.lock_x if constraint else False
  bone.use_ik_limit_x = constraint.use_limit_x if constraint else False
  bone.ik_min_x = constraint.min_x if constraint else 0.0
  bone.ik_max_x = constraint.max_x if constraint else 0.0
  bone.lock_ik_y = constraint.lock_y if constraint else False
  bone.use_ik_limit_y = constraint.use_limit_y if constraint else False
  bone.ik_min_y = constraint.min_y if constraint else 0.0
  bone.ik_max_y = constraint.max_y if constraint else 0.0
  bone.lock_ik_z = constraint.lock_z if constraint else False
  bone.use_ik_limit_z = constraint.use_limit_z if constraint else False
  bone.ik_min_z = constraint.min_z if constraint else 0.0
  bone.ik_max_z = constraint.max_z if constraint else 0.0
  return bone


def build_rotation_constraint(
    lock_x=False, use_limit_x=False, min_x=0, max_x=1,
    lock_y=False, use_limit_y=False, min_y=0, max_y=1,
    lock_z=False, use_limit_z=False, min_z=0, max_z=1):
  """Builds a mock rotation constraint."""
  c = mock.MagicMock()
  c.lock_x = lock_x
  c.use_limit_x = use_limit_x
  c.min_x = min_x
  c.max_x = max_x
  c.lock_y = lock_y
  c.use_limit_y = use_limit_y
  c.min_y = min_y
  c.max_y = max_y
  c.lock_z = lock_z
  c.use_limit_z = use_limit_z
  c.min_z = min_z
  c.max_z = max_z
  return c


def build_mesh(
    name, materials=None, faces=None, vert_co=None, normals=None, mat_ids=None
):
  """Builds a mock triangle mesh."""
  materials = materials or []
  faces = faces or [[0, 1, 2]]
  vert_co = vert_co or [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]
  normals = normals or vert_co
  mat_ids = mat_ids or [0] * len(faces)

  obj = mock.MagicMock(spec=bpy.types.Mesh)
  obj.name = name
  obj.materials = materials
  obj.uv_layers = mock.MagicMock()
  obj.loop_triangles = [
      mock.MagicMock(vertices=f, material_index=i)
      for f, i in zip(faces, mat_ids)
  ]
  obj.vertices = [
      mock.MagicMock(co=c, normal=n) for c, n in zip(vert_co, normals)
  ]

  return obj


def build_mesh_object(
    name: str,
    mesh: Optional[Any] = None,
    parent: Optional[Any] = None,
    parent_bone: Optional[str] = None) -> ...:
  """Builds a mock object with a mesh assigned to it."""
  obj = mock.MagicMock(spec=bpy.types.Object, type='MESH')
  obj.name = name
  obj.parent = parent
  obj.parent_bone = parent_bone
  obj.data = mesh or build_mesh(name)
  return obj


def build_light(name, light_type, lin_att, quad_att, shadow):
  """Builds a mock light."""
  obj = mock.MagicMock(spec=bpy.types.Object, type='LIGHT')
  obj.name = name
  obj.parent = None
  obj.parent_bone = None
  obj.data = mock.MagicMock(spec=bpy.types.Light)
  obj.data.type = light_type
  obj.data.linear_attenuation = lin_att
  obj.data.quadratic_attenuation = quad_att
  obj.data.use_shadow = shadow
  return obj


def build_material(
    name: str,
    color: Sequence[int] = (1, 1, 1, 1),
    specular: float = 0.5,
    metallic: float = 0.0,
    roughness: float = 0.5,
    use_backface_culling: bool = False,
):
  """Builds a mock material definition."""
  obj = mock.MagicMock(spec=bpy.types.Material)
  obj.name = name
  obj.diffuse_color = color
  obj.specular_intensity = specular
  obj.metallic = metallic
  obj.roughness = roughness
  obj.use_backface_culling = use_backface_culling
  return obj
