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

"""Blender scene parsers."""

# disable: pytype=strict-none
import dataclasses
import math
from typing import Any, Callable, Sequence, Tuple, cast

from dm_control.blender.fake_core import bpy
from dm_control.blender.fake_core import mathutils

_ARMATURE = 'ARMATURE'
_CAMERA = 'CAMERA'
_EMPTY = 'EMPTY'
_LIGHT = 'LIGHT'
_MESH = 'MESH'
_VEC_ZERO = mathutils.Vector((0, 0, 0))
_OX = mathutils.Vector((1, 0, 0))
_OY = mathutils.Vector((0, 1, 0))
_OZ = mathutils.Vector((0, 0, 1))


def _check_that_parent_bone_exists(obj: bpy.types.Object) -> None:
  """Checks if the bone the object was supposed to be parented exists."""
  assert obj is not None
  if obj.parent is None:
    raise ValueError(
        'Armature object "{}" was parented to does not exist'.format(obj.name)
    )

  armature = obj.parent
  if armature.type != _ARMATURE:
    raise ValueError(
        'Parent of object "{}" - "{}", is not an armature, but a "{}"'.format(
            obj.name, armature.name, armature.type
        )
    )

  if obj.parent_bone not in armature.data.bones:
    raise ValueError(
        'Object "{}" is parented to a non-existing bone "{}" from '
        'armature "{}"'.format(obj.name, obj.parent_bone, armature.name)
    )


def _check_constraint_in_local_space(
    constraint: bpy.types.Constraint, owner: 'ObjectRef'
) -> None:
  if constraint and constraint.owner_space != 'LOCAL':
    raise ValueError(
        'Constraint "{}" (bone "{}", armature "{}") uses an unsupported '
        'owner_mode "{}". Only "LOCAL" mode is supported at the '
        'moment'.format(
            type(constraint),
            owner.bone_name(),
            owner.obj_name(),
            constraint.owner_space,
        )
    )


def _angle_distance(lhs_deg: float, rhs_deg: float) -> float:
  """Calculates distance between two angles, in degrees."""
  x_2 = math.cos(math.radians(lhs_deg)) * math.cos(math.radians(rhs_deg))
  y_2 = math.sin(math.radians(lhs_deg)) * math.sin(math.radians(rhs_deg))
  return math.degrees(math.acos(x_2 + y_2))


@dataclasses.dataclass(frozen=True)
class AffineTransform:
  pos: mathutils.Vector
  rot: mathutils.Quaternion


@dataclasses.dataclass(frozen=True)
class Dof:
  """Degree of freedom description."""

  name: str
  axis: mathutils.Vector
  limited: bool = False
  limits: Tuple[float, float] = (0, 0)


@dataclasses.dataclass(frozen=True)
class ObjectRef:
  """References a Blender object, be that a scene object or a bone.

  Object reference is hashable and comparable. Equality is based on the names
  of the underlying objects. Blender's API guarantees that a combination of
  object/bone name will be unique across the scene. We're leveraging that rule.
  """

  native_obj: bpy.types.Object | None
  native_bone: bpy.types.Bone | None = None

  def __hash__(self) -> int:
    return hash(self.name)

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, ObjectRef):
      return False
    return self.name == rhs.name

  @property
  def is_none(self) -> bool:
    return self.native_bone is None

  @property
  def is_armature(self) -> bool:
    return (
        self.native_obj
        and self.native_obj.type == _ARMATURE
        and not self.native_bone
    )

  @property
  def is_bone(self) -> bool:
    return bool(
        self.native_obj
        and self.native_obj.type == _ARMATURE
        and self.native_bone
    )

  @property
  def is_mesh(self) -> bool:
    return self.native_obj and self.native_obj.type == _MESH

  @property
  def is_light(self) -> bool:
    return self.native_obj and self.native_obj.type == _LIGHT

  @property
  def is_camera(self) -> bool:
    return self.native_obj and self.native_obj.type == _CAMERA

  @property
  def is_empty(self) -> bool:
    return self.native_obj and self.native_obj.type == _EMPTY

  @property
  def name(self) -> str:
    if not self.native_obj:
      return ''
    if self.native_bone:
      return '{}_{}'.format(self.native_bone.name, self.native_obj.name)
    else:
      return self.native_obj.name

  def as_light(self) -> bpy.types.Light:
    return cast(bpy.types.Light, self.obj_data())

  def get_local_transform(self) -> AffineTransform:
    """Returns a transform wrt. the local reference frame."""

    def get_bone_local_mtx(bone: bpy.types.Bone) -> mathutils.Matrix:
      """Derives a local matrix of an armature bone."""
      assert isinstance(bone, bpy.types.Bone)
      if bone.parent:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local
      else:
        return bone.matrix_local

    def get_object_local_mtx(obj: bpy.types.Object) -> mathutils.Matrix:
      """Derives a local matrix of an object, such as a mesh or armature."""
      assert isinstance(obj, bpy.types.Object)
      if obj.parent:
        local_mtx = obj.parent.matrix_world.inverted() @ obj.matrix_world
        if obj.parent_bone:
          assert obj.parent.type == _ARMATURE
          armature = obj.parent
          bone = armature.data.bones[obj.parent_bone]
          return bone.matrix_local.inverted() @ local_mtx
        else:
          return local_mtx
      else:
        return obj.matrix_world

    if self.is_bone:
      local_mtx = get_bone_local_mtx(self.native_bone)
    else:
      local_mtx = get_object_local_mtx(self.native_obj)

    rot_quat = local_mtx.to_quaternion()
    pos = local_mtx.translation
    return AffineTransform(pos, rot_quat)

  @property
  def is_visible(self) -> bool:
    """Checks if an object is visible."""
    if not self.native_obj:
      return False
    if hasattr(self.native_obj, 'visible_get'):
      return self.native_obj.visible_get()
    else:
      return True

  @property
  def parent(self) -> 'ObjectRef':
    """Returns a reference to the parent of this object."""
    if self.native_obj is None:
      return ObjectRef(None)

    def bone_parent(
        armature: bpy.types.Object, bone: bpy.types.Bone
    ) -> 'ObjectRef':
      if bone.parent:
        # Parent of a child bone is its parent bone in the same armature
        assert isinstance(bone.parent, bpy.types.Bone)
        return ObjectRef.new_bone(armature, bone.parent)
      else:
        # Parent of the root bone is the armature.
        return ObjectRef.new_object(armature)

    def object_parent(obj: bpy.types.Object) -> 'ObjectRef':
      if obj.parent_bone:
        # The object is parented to an armature, and the .parent field must
        # contain a reference the armature.
        assert obj.parent and obj.parent.type == _ARMATURE
        _check_that_parent_bone_exists(obj)
        armature = obj.parent
        parent_bone = armature.data.bones[obj.parent_bone]
        return ObjectRef.new_bone(armature, parent_bone)
      elif obj.parent:
        # The object is parented to another object.
        return ObjectRef.new_object(obj.parent)
      else:
        # This is a root object
        return ObjectRef(None)

    if self.native_bone:
      return bone_parent(self.native_obj, self.native_bone)
    else:
      return object_parent(self.native_obj)

  def get_rotation_dofs(self) -> Sequence[Dof]:
    """Returns the rotation degrees of freedom in the subtractive mode.

    The method returns the degrees of freedom present from the point of view
    of Blender. These are modeled such that in absence of constraints, all
    degrees of freedom are present.

    Returns:
      A sequence of degree of freedom definitions.
    """
    if not self.is_bone:
      raise ValueError(
          'Rotation degrees of freedom are defined only for bones.'
      )
    assert self.native_obj
    assert self.native_obj.type == _ARMATURE
    armature = self.native_obj
    bone = armature.pose.bones[self.native_bone.name]
    if not bone.is_in_ik_chain:
      # Bones not in an IK chain don't receive any degrees of freedom.
      return []

    is_locked = [bone.lock_ik_x, bone.lock_ik_y, bone.lock_ik_z]
    use_limits = [bone.use_ik_limit_x, bone.use_ik_limit_y, bone.use_ik_limit_z]
    limits = [
        (math.degrees(bone.ik_min_x), math.degrees(bone.ik_max_x)),
        (math.degrees(bone.ik_min_y), math.degrees(bone.ik_max_y)),
        (math.degrees(bone.ik_min_z), math.degrees(bone.ik_max_z)),
    ]
    axes = [_OX, _OY, _OZ]
    names = [
        'rx_{}'.format(self.name),
        'ry_{}'.format(self.name),
        'rz_{}'.format(self.name),
    ]
    axis_names = ['X', 'Y', 'Z']

    def build_dof(idx):
      if limits[idx][0] >= limits[idx][1]:
        raise ValueError(
            'Bone "{}" uses incorrect IK limits for {} axis. '
            '{} < {} is violated'.format(
                self.name, axis_names[idx], limits[idx][0], limits[idx][1]
            )
        )
      return Dof(
          name=names[idx],
          axis=axes[idx],
          limited=use_limits[idx],
          limits=limits[idx],
      )

    return [build_dof(i) for i in range(3) if not is_locked[i]]

  def obj_data(self) -> bpy.types.Object | None:
    if self.native_obj is not None:
      return self.native_obj.data
    else:
      return None

  def obj_name(self) -> str | None:
    if self.native_obj is not None:
      return self.native_obj.name
    else:
      return None

  def bone_name(self) -> str | None:
    if self.native_bone is not None:
      return self.native_bone.name
    else:
      return None

  @property
  def mesh(self) -> bpy.types.Mesh:
    """Returns the mesh associated with this object."""
    assert self.is_mesh
    return cast(bpy.types.Mesh, self.obj_data())

  def get_modified_mesh(self) -> bpy.types.Mesh | None:
    """Returns a mesh with modifiers applied to it."""
    assert self.is_mesh
    assert self.native_obj is not None
    if self.native_obj.mode == 'EDIT':
      self.native_obj.update_from_editmode()

    # get the modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_owner = self.native_obj.evaluated_get(depsgraph)

    return mesh_owner.to_mesh()

  @property
  def materials(self) -> Sequence[bpy.types.Material]:
    """Returns the materials assigned to this object."""
    data = self.obj_data()
    if hasattr(data, 'materials'):
      return data.materials
    else:
      return []

  @classmethod
  def new_object(cls, obj: bpy.types.Object) -> 'ObjectRef':
    assert isinstance(obj, bpy.types.Object) or obj is None
    return cls(obj)

  @classmethod
  def new_bone(
      cls,
      armature: bpy.types.Object,
      bone: bpy.types.Bone | None,
  ) -> 'ObjectRef':
    assert isinstance(armature, bpy.types.Object)
    assert armature.type == _ARMATURE
    assert isinstance(bone, bpy.types.Bone) or bone is None
    return cls(armature, bone)


NoneRef = ObjectRef(None, None)


def map_blender_tree(
    context: bpy.types.Context, callback: Callable[[ObjectRef], Any]
) -> Sequence[Any]:
  """Returns a list of scene objects in the breadth-first order."""
  # Collect all nodes to explore - objects and bones alike
  assert context.scene is not None
  to_explore = [ObjectRef.new_object(o) for o in context.scene.objects]
  armatures = [o for o in context.scene.objects if o.type == 'ARMATURE']
  for armature in armatures:
    for bone in armature.data.bones:
      to_explore.append(ObjectRef.new_bone(armature, bone))

  explored = set()
  explored.add(NoneRef)
  result = []

  while to_explore:
    obj_ref: ObjectRef = to_explore[0]
    to_explore = to_explore[1:]

    if obj_ref.parent in explored:
      explored.add(obj_ref)
      result.append(callback(obj_ref))
    else:
      to_explore.append(obj_ref)

  return result


def get_material_mesh_pair_name(mesh_name: str, mat_name: str) -> str:
  """Build the name for a mesh-material pair."""
  return '{}_{}'.format(mesh_name, mat_name) if mat_name else mesh_name


def is_material_mesh_pair_valid(mesh: bpy.types.Mesh, mat_idx: int) -> bool:
  """Tests the mesh-material pair whether it contains any geometry."""
  mesh.calc_loop_triangles()
  faces = [f for f in mesh.loop_triangles if f.material_index == mat_idx]
  return bool(faces)


def map_materials(
    func: Callable[[bpy.types.Material], Any],
    materials: Sequence[bpy.types.Material],
) -> Sequence[Any]:
  """Maps a collection of materials, adjusting for empty collections.

  In case of an empty collection, a substitute for a default material is passed
  to the mapping callback.

  Args:
    func: Mapping callback.
    materials: Collection of materials.

  Returns:
    An arbitrary collection of data mapped out of the materials collection.
  """
  materials = materials or []
  return [func(material) for material in materials]
