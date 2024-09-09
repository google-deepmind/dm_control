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

"""Tests for blender_scene.py."""

import math
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.blender.fake_core import bpy
from dm_control.blender.fake_core import mathutils
from dm_control.blender.mujoco_exporter import blender_scene
from dm_control.blender.mujoco_exporter import testing
import numpy as np


class AngleDistanceTest(parameterized.TestCase):

  @parameterized.parameters([
      [0, 0, 0],
      [90, 90, 0],
      [-90, -90, 0],
      [-90, 0, 90],
      [0, 90, 90],
      [-90, 90, 180],
      [-180, 180, 0],
      [-270, 90, 0],
      [-90, 270, 0],
      [-90, 360, 90],
  ])
  def test_angle_distance(self, lhs, rhs, result):
    self.assertAlmostEqual(blender_scene._angle_distance(lhs, rhs), result, 2)


class ObjectRefTest(absltest.TestCase):

  def test_hashing_and_comparison(self):
    mesh_a = blender_scene.ObjectRef.new_object(
        testing.build_mesh_object('mesh_a')
    )
    mesh_a_copy = blender_scene.ObjectRef.new_object(
        testing.build_mesh_object('mesh_a')
    )
    mesh_b = blender_scene.ObjectRef.new_object(
        testing.build_mesh_object('mesh_b')
    )

    self.assertEqual(hash(mesh_a), hash(mesh_a_copy))
    self.assertNotEqual(hash(mesh_a), hash(mesh_b))

    self.assertEqual(mesh_a, mesh_a_copy)
    self.assertNotEqual(mesh_a, mesh_b)

  def test_parent_of_root_object_is_noneref(self):
    obj_ref = blender_scene.ObjectRef.new_object(
        testing.build_mesh_object('mesh')
    )
    self.assertEqual(obj_ref.parent, blender_scene.NoneRef)

  def test_parent_of_object_parented_to_another_object(self):
    parent_obj = testing.build_mesh_object('parent_obj')
    child_obj = testing.build_mesh_object('child_obj', parent=parent_obj)

    parent_ref = blender_scene.ObjectRef.new_object(parent_obj)
    child_ref = blender_scene.ObjectRef.new_object(child_obj)
    self.assertEqual(child_ref.parent, parent_ref)

  def test_parent_of_object_parented_to_bone(self):
    bone = testing.build_bone('bone')
    armature = testing.build_armature('armature', bones=[bone])
    child_obj = testing.build_mesh_object(
        'child_obj', parent=armature, parent_bone=bone.name
    )

    parent_ref = blender_scene.ObjectRef.new_bone(armature, bone)
    child_ref = blender_scene.ObjectRef.new_object(child_obj)
    self.assertEqual(child_ref.parent, parent_ref)

  def test_parent_of_root_bone_is_armature(self):
    bone = testing.build_bone('bone')
    armature = testing.build_armature('armature', bones=[bone])

    armature_ref = blender_scene.ObjectRef.new_object(armature)
    bone_ref = blender_scene.ObjectRef.new_bone(armature, bone)
    self.assertEqual(bone_ref.parent, armature_ref)


class BlenderTreeMapperTest(parameterized.TestCase):

  @parameterized.parameters([
      [['obj1']],
      [['obj1', 'obj2']],
      [['obj2', 'obj1']],
      [['obj2', 'obj1', 'obj3']],
  ])
  def test_object_without_hierarchy(self, object_names):
    context = mock.MagicMock(spec=bpy.types.Context)
    context.scene.objects = [
        testing.build_mesh_object(name) for name in object_names
    ]

    objects = blender_scene.map_blender_tree(context, lambda x: x)
    self.assertEqual([o.native_obj.name for o in objects], object_names)

  @parameterized.parameters([
      # A hierarchy with 2 objects
      dict(
          parent_child_tuples=[('obj_1', 'obj_1_1')],
          expected_order=['obj_1', 'obj_1_1'],
      ),
      # Shallow hierarchy with 3 objects
      dict(
          parent_child_tuples=[('obj_1', 'obj_1_1'), ('obj_1', 'obj_1_2')],
          expected_order=['obj_1', 'obj_1_1', 'obj_1_2'],
      ),
      # Deep hierarchy with 3 objects
      dict(
          parent_child_tuples=[('obj_1', 'obj_1_1'), ('obj_1_1', 'obj_1_2')],
          expected_order=['obj_1', 'obj_1_1', 'obj_1_2'],
      ),
      # 2 hierarchies, 2 objects each
      dict(
          parent_child_tuples=[('obj_1', 'obj_1_1'), ('obj_2', 'obj_2_1')],
          expected_order=['obj_1', 'obj_1_1', 'obj_2', 'obj_2_1'],
      ),
  ])
  def test_object_with_hierarchy(self, parent_child_tuples, expected_order):
    objects = {}
    for parent_name, child_name in parent_child_tuples:
      parent = objects.get(parent_name, None)
      if not parent:
        parent = testing.build_mesh_object(parent_name)
        objects[parent_name] = parent

      child = testing.build_mesh_object(child_name, parent=parent)
      objects[child_name] = child

    context = mock.MagicMock(spec=bpy.types.Context)
    context.scene.objects = [o for o in objects.values()]

    object_names = blender_scene.map_blender_tree(context, lambda x: x.name)
    self.assertEqual(object_names, expected_order)

  def test_boneless_armature(self):
    armature = testing.build_armature('armature')

    context = mock.MagicMock(spec=bpy.types.Context)
    context.scene.objects = [armature]

    objects = blender_scene.map_blender_tree(context, lambda x: x)
    self.assertLen(objects, 1)

  def test_armature_with_bones(self):
    root = testing.build_bone('root')
    r_arm = testing.build_bone('r_arm', parent=root)
    l_arm = testing.build_bone('l_arm', parent=root)
    r_finger = testing.build_bone('r_finger', parent=r_arm)
    armature = testing.build_armature(
        'armature', bones=[r_finger, l_arm, root, r_arm]
    )
    expected_order = [
        'armature',
        'root_armature',
        'r_arm_armature',
        'r_finger_armature',
        'l_arm_armature',
    ]

    context = mock.MagicMock(spec=bpy.types.Context)
    # Randomize the order of objects to test the sorting mechanism.
    context.scene.objects = [armature]

    object_names = blender_scene.map_blender_tree(context, lambda x: x.name)
    self.assertEqual(object_names, expected_order)


class DegreesOfFreedomTest(parameterized.TestCase):

  def test_ik_with_unlimited_dofs_adds_all_dofs(self):
    constraint = testing.build_rotation_constraint(
        use_limit_x=False, use_limit_y=False, use_limit_z=False
    )
    bone = testing.build_bone('bone', constraint=constraint)
    armature = testing.build_armature('bone', bones=[bone])
    bone_ref = blender_scene.ObjectRef.new_bone(armature, bone)

    dofs = bone_ref.get_rotation_dofs()
    self.assertLen(dofs, 3)
    self.assertEqual(dofs[0].axis, mathutils.Vector((1, 0, 0)))
    self.assertEqual(dofs[1].axis, mathutils.Vector((0, 1, 0)))
    self.assertEqual(dofs[2].axis, mathutils.Vector((0, 0, 1)))
    self.assertFalse(dofs[0].limited)
    self.assertFalse(dofs[1].limited)
    self.assertFalse(dofs[2].limited)

  def test_ik_with_removed_dofs(self):
    constraint = testing.build_rotation_constraint(
        lock_x=False,
        lock_y=True,
        lock_z=False,
        use_limit_x=True,
        use_limit_z=False,
    )
    bone = testing.build_bone('bone', constraint=constraint)
    armature = testing.build_armature('bone', bones=[bone])
    bone_ref = blender_scene.ObjectRef.new_bone(armature, bone)

    dofs = bone_ref.get_rotation_dofs()
    self.assertLen(dofs, 2)
    self.assertEqual(dofs[0].axis, mathutils.Vector((1, 0, 0)))
    self.assertEqual(dofs[1].axis, mathutils.Vector((0, 0, 1)))
    self.assertTrue(dofs[0].limited)
    self.assertFalse(dofs[1].limited)

  def test_limiting_dof(self):
    constraint = testing.build_rotation_constraint(
        use_limit_x=True, min_x=math.radians(-15), max_x=math.radians(30)
    )
    bone = testing.build_bone('bone', constraint=constraint)
    armature = testing.build_armature('bone', bones=[bone])
    bone_ref = blender_scene.ObjectRef.new_bone(armature, bone)

    dofs = bone_ref.get_rotation_dofs()
    self.assertLen(dofs, 3)
    self.assertTrue(dofs[0].limited)
    self.assertFalse(dofs[1].limited)
    self.assertFalse(dofs[2].limited)
    np.testing.assert_array_almost_equal(dofs[0].limits, (-15, 30), 1e-2)


if __name__ == '__main__':
  absltest.main()
