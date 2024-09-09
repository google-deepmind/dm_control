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

"""Mujoco asset exporters.

Note about Materials.
Material nodes are not supported, so please use basic Blender materials.
"""

import os
import struct
from typing import Sequence
from xml.dom import minidom

from dm_control.blender.fake_core import bpy
from dm_control.blender.mujoco_exporter import blender_scene

_MESH = 'MESH'


class MujocoMesh:
  """Mujoco representation of a mesh.

  Mujoco can't handle meshes that reference multiple materials. Therefore
  an instance of this class represents a submesh that uses a specific material.
  """

  def __init__(self, mesh: bpy.types.Mesh, material_idx: int, two_sided: bool):
    if mesh.uv_layers.active is None:
      raise ValueError(f'Mesh {mesh.name} does not have an active UV layer.')

    mesh.calc_loop_triangles()

    indices = []
    for face in mesh.loop_triangles:
      if face.material_index == material_idx:
        indices.extend([i for i in face.vertices])

    uv_layer = mesh.uv_layers.active.data
    # pylint: disable=g-complex-comprehension
    self.vertices = [c for i in indices for c in mesh.vertices[i].co]
    self.normals = [c for i in indices for c in mesh.vertices[i].normal]
    self.uvs = [c for i in indices for c in uv_layer[i].uv]
    # pylint: enable=g-complex-comprehension
    self.faces = list(range(len(self.vertices) // 3))

    if two_sided:
      # For two-sided meshes, duplicate the geometry with flipped normals
      # and reversed triangle winding order.
      base_vertex = len(self.vertices) // 3
      self.vertices += list(self.vertices)
      self.normals += [n * -1.0 for n in self.normals]
      self.uvs += list(self.uvs)
      self.faces += [i + base_vertex for i in reversed(self.faces)]

  def save(self, filepath: str) -> None:
    """Save the data in the MSH format."""
    nvertex = len(self.vertices)
    nnormal = len(self.normals)
    ntexcoord = len(self.uvs)
    nface = len(self.faces)
    assert nvertex % 3 == 0
    assert nnormal % 3 == 0
    assert ntexcoord % 2 == 0
    assert nface % 3 == 0

    fmt_msh = '4i{}f{}f{}f{}i'.format(nvertex, nnormal, ntexcoord, nface)
    with open(filepath, 'wb') as f:
      f.write(
          struct.pack(
              fmt_msh,
              nvertex // 3,
              nnormal // 3,
              ntexcoord // 2,
              nface // 3,
              *self.vertices,
              *self.normals,
              *self.uvs,
              *self.faces,
          )
      )


def mesh_asset_builder(
    doc: minidom.Document,
    mesh: bpy.types.Mesh,
    materials: Sequence[bpy.types.Material],
    folder: str,
) -> Sequence[minidom.Element]:
  """Exports a mesh associated with the object and creates an 'asset' node."""
  mat_names = blender_scene.map_materials(lambda m: m.name, materials)
  twosidedness = blender_scene.map_materials(
      lambda m: not m.use_backface_culling, materials
  )
  mat_names = mat_names or ['']
  twosidedness = twosidedness or [False]

  elements = []
  for mat_idx, (mat_name, twosided) in enumerate(zip(mat_names, twosidedness)):
    el_name = blender_scene.get_material_mesh_pair_name(mesh.name, mat_name)
    filename = '{}.msh'.format(el_name)
    filepath = os.path.join(folder, filename)

    if blender_scene.is_material_mesh_pair_valid(mesh, mat_idx):
      mesh_to_export = MujocoMesh(mesh, mat_idx, twosided)
      mesh_to_export.save(filepath)

      el = doc.createElement('mesh')
      el.setAttribute('name', el_name)
      el.setAttribute('file', filename)
      elements.append(el)
  return elements


def clip01(value):
  return min(1.0, max(0.0, value))


def material_asset_builder(
    doc: minidom.Document, mat: bpy.types.Material
) -> minidom.Element:
  """Builds a material asset node."""
  el = doc.createElement('material')
  el.setAttribute('name', mat.name)
  # el.setAttribute('texture', '')
  el.setAttribute('specular', str(clip01(mat.specular_intensity)))
  el.setAttribute('shininess', str(clip01(1.0 - mat.roughness)))
  el.setAttribute('reflectance', str(clip01(mat.metallic)))
  el.setAttribute('rgba', ' '.join([str(c) for c in mat.diffuse_color]))
  return el


def export_to_xml(
    doc: minidom.Document,
    objects: Sequence[blender_scene.ObjectRef],
    folder: str,
    apply_mesh_modifiers: bool,
) -> minidom.Element:
  """converts Blender scene objects to Mujoco assets."""
  asset_el = doc.createElement('asset')

  unique = set()
  for obj in objects:
    if obj.is_mesh and obj.mesh not in unique:
      # Use the base mesh object as a reference, because that's the reference
      # that will be shared between objects.
      unique.add(obj.mesh)
      if apply_mesh_modifiers:
        # Make sure to export the mesh with modifiers applied, and if that's not
        # possible (it may not be possible and None will be returned), default
        # to the base mesh.
        mesh_for_export = obj.get_modified_mesh() or obj.mesh
      else:
        mesh_for_export = obj.mesh
      for el in mesh_asset_builder(doc, mesh_for_export, obj.materials, folder):
        asset_el.appendChild(el)

    for material in obj.materials:
      if material not in unique:
        unique.add(material)
        asset_el.appendChild(material_asset_builder(doc, material))

  return asset_el
