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

"""Tests for mujoco_assets.py."""

import collections
from unittest import mock
from xml.dom import minidom

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.blender.mujoco_exporter import blender_scene
from dm_control.blender.mujoco_exporter import mujoco_assets
from dm_control.blender.mujoco_exporter import testing

_DEFAULT_SETTINGS = dict(folder='', apply_mesh_modifiers=False)


class MujocoAssetsTest(parameterized.TestCase):

  def test_building_mesh_asset(self):
    mesh_object = testing.build_mesh_object('mesh')

    xml_root = mock.MagicMock(spec=minidom.Document)
    xml_root.createElement = mock.MagicMock(spec=minidom.Element)

    obj = blender_scene.ObjectRef(mesh_object)
    with mock.patch(mujoco_assets.__name__ + '.open', mock.mock_open()):
      element = mujoco_assets.export_to_xml(
          xml_root, [obj], **_DEFAULT_SETTINGS
      )

    xml_root.createElement.assert_any_call('mesh')
    element.setAttribute.assert_any_call('name', 'mesh')
    # Note there's no folder in the name.
    # The asset is assumed to be exported into the same folder as the .xml file.
    element.setAttribute.assert_any_call('file', 'mesh.msh')

  def test_exporting_mesh_asset(self):
    mesh = testing.build_mesh(name='mesh.001')
    mesh_object = testing.build_mesh_object('mesh', mesh=mesh)
    xml_root = mock.MagicMock(spec=minidom.Document)

    obj = blender_scene.ObjectRef(mesh_object)
    mock_open = mock.mock_open()
    with mock.patch(mujoco_assets.__name__ + '.open', mock_open):
      mujoco_assets.export_to_xml(xml_root, [obj], '/folder', False)
      mock_open.assert_called_once_with('/folder/mesh.001.msh', 'wb')

  def test_exporting_material_asset(self):
    material = testing.build_material(
        'mat', color=(0.1, 0.2, 0.3, 0.4), specular=0.5, metallic=0.6
    )
    xml_root = mock.MagicMock(spec=minidom.Document)

    result = mujoco_assets.material_asset_builder(xml_root, material)
    result.setAttribute.assert_any_call('name', 'mat')
    result.setAttribute.assert_any_call('specular', '0.5')
    result.setAttribute.assert_any_call('reflectance', '0.6')
    result.setAttribute.assert_any_call('rgba', '0.1 0.2 0.3 0.4')

  @parameterized.parameters([
      [0.0, 1.0],
      [0.25, 0.75],
      [0.5, 0.5],
      [1.0, 0.0],
      [5.0, 0.0],
      [-5.0, 1.0],
  ])
  def test_shininess_is_inverse_of_blender_mat_roughness(self, blender, mujoco):
    material = testing.build_material('mat', roughness=blender)
    xml_root = mock.MagicMock(spec=minidom.Document)

    result = mujoco_assets.material_asset_builder(xml_root, material)
    result.setAttribute.assert_any_call('shininess', str(mujoco))

  def test_export_unique_asset_instances(self):
    num_objects = 10
    shared_material = testing.build_material('mat_1')
    shared_mesh = testing.build_mesh('mesh.001', materials=[shared_material])

    objects = [
        blender_scene.ObjectRef(
            testing.build_mesh_object('mesh', mesh=shared_mesh)
        )
        for _ in range(num_objects)
    ]
    xml_root = mock.MagicMock(spec=minidom.Document)
    with mock.patch(mujoco_assets.__name__ + '.open', mock.mock_open()):
      mujoco_assets.export_to_xml(xml_root, objects, **_DEFAULT_SETTINGS)

    create_element_call_counts = collections.Counter(
        [m_[0][0] for m_ in xml_root.createElement.call_args_list]
    )
    self.assertEqual(create_element_call_counts['mesh'], 1)
    self.assertEqual(create_element_call_counts['material'], 1)

  def test_mesh_with_multiple_materials_split(self):
    mat_1 = testing.build_material('mat_1')
    mat_2 = testing.build_material('mat_2')
    mesh = testing.build_mesh(
        'mesh',
        materials=[mat_1, mat_2],
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

    mesh_obj = blender_scene.ObjectRef(
        testing.build_mesh_object('mesh', mesh=mesh)
    )

    with mock.patch(mujoco_assets.__name__ + '.open', mock.mock_open()):
      xml_root = mock.MagicMock(spec=minidom.Document)
      xml_elem = mock.MagicMock(spec=minidom.Element)
      xml_root.createElement.return_value = xml_elem
      mujoco_assets.export_to_xml(xml_root, [mesh_obj], **_DEFAULT_SETTINGS)

    # Verify that two mesh elements were created...
    create_element_call_counts = collections.Counter(
        [m_[0][0] for m_ in xml_root.createElement.call_args_list]
    )
    self.assertEqual(create_element_call_counts['mesh'], 2)

    # ...and that their names reflect the mesh-material pairing.
    xml_elem.setAttribute.assert_any_call('file', 'mesh_mat_1.msh')
    xml_elem.setAttribute.assert_any_call('file', 'mesh_mat_2.msh')
    xml_elem.setAttribute.assert_any_call('name', 'mesh_mat_1')
    xml_elem.setAttribute.assert_any_call('name', 'mesh_mat_2')


class MujocoMeshTests(parameterized.TestCase):

  def test_splitting_mesh_with_multiple_materials_into_submeshes(self):
    mat_0 = testing.build_material('mat_0')
    mat_1 = testing.build_material('mat_1')
    mesh = testing.build_mesh(
        'mesh',
        materials=[mat_0, mat_1],
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

    mesh_mat_0 = mujoco_assets.MujocoMesh(mesh, material_idx=0, two_sided=False)
    mesh_mat_1 = mujoco_assets.MujocoMesh(mesh, material_idx=1, two_sided=False)

    self.assertEqual(
        mesh_mat_0.vertices, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
    )
    self.assertEqual(
        mesh_mat_1.vertices, [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]
    )
    self.assertEqual(mesh_mat_0.faces, [0, 1, 2])
    self.assertEqual(mesh_mat_1.faces, [0, 1, 2])

  def test_one_sided_mesh(self):
    mesh = testing.build_mesh(
        'mesh',
        materials=[testing.build_material('mat')],
        faces=[[0, 1, 2]],
        vert_co=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        normals=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        mat_ids=[0],
    )

    mesh = mujoco_assets.MujocoMesh(mesh, material_idx=0, two_sided=False)
    self.assertEqual(mesh.faces, [0, 1, 2])
    self.assertEqual(
        mesh.vertices, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
    )
    self.assertEqual(
        mesh.normals, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
    )

  def test_two_sided_mesh(self):
    mesh = testing.build_mesh(
        'mesh',
        materials=[testing.build_material('mat')],
        faces=[[0, 1, 2]],
        vert_co=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        normals=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        mat_ids=[0],
    )

    mesh = mujoco_assets.MujocoMesh(mesh, material_idx=0, two_sided=True)
    self.assertEqual(mesh.faces, [0, 1, 2, 5, 4, 3])
    self.assertEqual(
        mesh.vertices,
        [.1, .1, .1, .2, .2, .2, .3, .3, .3,
         .1, .1, .1, .2, .2, .2, .3, .3, .3,])
    self.assertEqual(
        mesh.normals,
        [.1, .1, .1, .2, .2, .2, .3, .3, .3,
         -.1, -.1, -.1, -.2, -.2, -.2, -.3, -.3, -.3,])


if __name__ == '__main__':
  absltest.main()
