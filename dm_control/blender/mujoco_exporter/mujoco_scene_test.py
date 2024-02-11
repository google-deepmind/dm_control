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

"""Tests for mujoco_scene.py."""

import collections
from unittest import mock
from xml.dom import minidom
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.blender.mujoco_exporter import blender_scene
from dm_control.blender.mujoco_exporter import mujoco_scene
from dm_control.blender.mujoco_exporter import testing

_DEFAULT_PARAMS = dict(armature_freejoint=False)


class ConverterTest(parameterized.TestCase):

  def test_convert_armature(self):
    armature = testing.build_armature('armature')
    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_root.createElement = mock.MagicMock(spec=minidom.Element)

    obj = blender_scene.ObjectRef(armature)
    created_element = mujoco_scene.export_to_xml(
        xml_root, [obj], **_DEFAULT_PARAMS
    )

    xml_root.createElement.assert_any_call('body')
    created_element.setAttribute.assert_any_call('name', 'armature')

  def test_convert_bone_without_constraints_creates_a_body(self):
    bone = testing.build_bone('bone')
    armature = testing.build_armature('armature', bones=[bone])

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_root.createElement.side_effect = mock.MagicMock(spec=minidom.Element)

    bone_obj = blender_scene.ObjectRef(armature, bone)
    armature_obj = blender_scene.ObjectRef(armature)
    created_element = mujoco_scene.export_to_xml(
        xml_root, [armature_obj, bone_obj], **_DEFAULT_PARAMS
    )

    xml_root.createElement.assert_any_call('body')
    created_element.setAttribute.assert_any_call('name', 'bone_armature')

  @parameterized.parameters([
      # Point light
      dict(
          args=dict(lin_att=1, quad_att=2, shadow=True, light_type='POINT'),
          out=dict(castshadow='true', attenuation='0 1 2', directional='false'),
      ),
      # Spot light
      dict(
          args=dict(lin_att=3, quad_att=4, shadow=True, light_type='SPOT'),
          out=dict(castshadow='true', attenuation='0 3 4', directional='true'),
      ),
      # Directional light
      dict(
          args=dict(lin_att=5, quad_att=6, shadow=True, light_type='SUN'),
          out=dict(castshadow='true', attenuation='0 5 6', directional='true'),
      ),
      # Shadows off
      dict(
          args=dict(lin_att=7, quad_att=8, shadow=False, light_type='SUN'),
          out=dict(castshadow='false', attenuation='0 7 8', directional='true'),
      ),
  ])
  def test_convert_light(self, args, out):
    light = testing.build_light('light', **args)

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_root.createElement = mock.MagicMock(spec=minidom.Element)

    obj = blender_scene.ObjectRef(light)
    created_element = mujoco_scene.export_to_xml(
        xml_root, [obj], **_DEFAULT_PARAMS
    )

    xml_root.createElement.assert_any_call('light')
    created_element.setAttribute.assert_any_call('name', 'light')
    for k, v in out.items():
      created_element.setAttribute.assert_any_call(k, v)

  def test_exported_mesh_references_mesh_asset_by_name(self):
    mesh = testing.build_mesh('mesh_asset_name')
    mesh_obj = testing.build_mesh_object('mesh_object_name', mesh=mesh)

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_root.createElement = mock.MagicMock(spec=minidom.Element)

    obj = blender_scene.ObjectRef(mesh_obj)
    created_element = mujoco_scene.export_to_xml(
        xml_root, [obj], **_DEFAULT_PARAMS
    )

    xml_root.createElement.assert_any_call('geom')
    created_element.setAttribute.assert_any_call('name', 'mesh_object_name')
    created_element.setAttribute.assert_any_call('mesh', 'mesh_asset_name')

  def test_mesh_with_no_material_keeps_its_name(self):
    mesh = testing.build_mesh('mesh')
    mesh_obj = testing.build_mesh_object('mesh_obj', mesh=mesh)
    obj = blender_scene.ObjectRef(mesh_obj)

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_elem = mock.MagicMock(spec=minidom.Element)
    xml_root.createElement.return_value = xml_elem
    mujoco_scene.export_to_xml(xml_root, [obj], **_DEFAULT_PARAMS)

    xml_elem.setAttribute.assert_any_call('name', 'mesh_obj')
    xml_elem.setAttribute.assert_any_call('mesh', 'mesh')

  def test_mesh_with_materials_builds_multiple_geoms(self):
    mesh = testing.build_mesh(
        'mesh',
        materials=[
            testing.build_material('mat_1'),
            testing.build_material('mat_2'),
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        vert_co=[
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
        ],
        mat_ids=[0, 1],
    )
    mesh_obj = testing.build_mesh_object('mesh_obj', mesh=mesh)

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_elem = mock.MagicMock(spec=minidom.Element)
    xml_root.createElement.return_value = xml_elem

    obj = blender_scene.ObjectRef(mesh_obj)
    mujoco_scene.export_to_xml(xml_root, [obj], **_DEFAULT_PARAMS)

    xml_elem.setAttribute.assert_any_call('name', 'mesh_obj_mat_1')
    xml_elem.setAttribute.assert_any_call('name', 'mesh_obj_mat_2')
    xml_elem.setAttribute.assert_any_call('mesh', 'mesh_mat_1')
    xml_elem.setAttribute.assert_any_call('mesh', 'mesh_mat_2')

  def test_empty_submeshes_are_not_exported(self):
    mesh = testing.build_mesh(
        'mesh',
        materials=[
            testing.build_material('mat_1'),
            testing.build_material('mat_2'),
        ],
        faces=[[0, 1, 2], [3, 4, 5]],
        vert_co=[
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6],
        ],
        mat_ids=[0, 0],
    )  # All faces reference material 0
    mesh_obj = testing.build_mesh_object('mesh_obj', mesh=mesh)

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_elem = mock.MagicMock(spec=minidom.Element)
    xml_root.createElement.return_value = xml_elem

    obj = blender_scene.ObjectRef(mesh_obj)
    mujoco_scene.export_to_xml(xml_root, [obj], **_DEFAULT_PARAMS)

    create_element_call_counts = collections.Counter(
        [m_[0][0] for m_ in xml_root.createElement.call_args_list]
    )
    self.assertEqual(create_element_call_counts['geom'], 1)
    xml_elem.setAttribute.assert_any_call('name', 'mesh_obj_mat_1')


if __name__ == '__main__':
  absltest.main()
