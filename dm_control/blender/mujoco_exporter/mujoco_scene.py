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

"""Mujoco scene element builders and utilities."""

from typing import Sequence
from xml.dom import minidom

from dm_control.blender.fake_core import mathutils
from dm_control.blender.mujoco_exporter import blender_scene

_ARMATURE = 'ARMATURE'
_MESH = 'MESH'
_VEC_ZERO = mathutils.Vector((0, 0, 0))
_OZ = mathutils.Vector((0, 0, 1))


def color_to_mjcf(color: mathutils.Color) -> str:
  return '{} {} {}'.format(color.r, color.g, color.b)


def vec_to_mjcf(vec: mathutils.Vector) -> str:
  return '{} {} {}'.format(vec.x, vec.y, vec.z)


def quat_to_mjcf(quat: mathutils.Quaternion) -> str:
  return '{} {} {} {}'.format(quat.w, quat.x, quat.y, quat.z)


def bool_to_mjcf(bool_val: bool):
  return 'true' if bool_val else 'false'


def body_builder(
    doc: minidom.Document, blender_obj: blender_scene.ObjectRef
) -> minidom.Element:
  """Builds a mujoco body element."""
  transform = blender_obj.get_local_transform()

  el = doc.createElement('body')
  el.setAttribute('name', blender_obj.name)
  el.setAttribute('pos', vec_to_mjcf(transform.pos))
  el.setAttribute('quat', quat_to_mjcf(transform.rot))
  return el


def light_builder(
    doc: minidom.Document, light_obj: blender_scene.ObjectRef
) -> minidom.Element:
  """Builds an mjcf element that describes a light."""
  assert light_obj.is_light
  light = light_obj.as_light()

  directional = bool_to_mjcf(light.type == 'SUN' or light.type == 'SPOT')
  attenuation = '0 {} {}'.format(
      light.linear_attenuation, light.quadratic_attenuation
  )
  transform = light_obj.get_local_transform()

  el = doc.createElement('light')
  el.setAttribute('name', light_obj.name)
  el.setAttribute('pos', vec_to_mjcf(transform.pos))
  el.setAttribute('dir', vec_to_mjcf(transform.rot @ _OZ))
  el.setAttribute('directional', directional)
  el.setAttribute('castshadow', bool_to_mjcf(light.use_shadow))  # pytype: disable=wrong-arg-types
  el.setAttribute('diffuse', color_to_mjcf(light.color))
  el.setAttribute('attenuation', attenuation)
  return el


def mesh_geom_builder(
    doc: minidom.Document, mesh_obj: blender_scene.ObjectRef
) -> Sequence[minidom.Element]:
  """Builds a mujoco node for a mesh geom."""
  mesh = mesh_obj.mesh
  transform = mesh_obj.get_local_transform()

  elements = []
  mat_names = blender_scene.map_materials(lambda m: m.name, mesh_obj.materials)
  # It might be the case that the mesh doesn't have any materials assigned.
  # We still want to export such geom, but we don't want to make a reference to
  # the material in the node.
  # So in that case we're using a fake material without a name
  mat_names = mat_names or ['']
  for mat_idx, mat_name in enumerate(mat_names):
    if not blender_scene.is_material_mesh_pair_valid(mesh, mat_idx):
      continue

    obj_name = blender_scene.get_material_mesh_pair_name(
        mesh_obj.name, mat_name
    )
    mesh_name = blender_scene.get_material_mesh_pair_name(mesh.name, mat_name)

    el = doc.createElement('geom')
    el.setAttribute('name', obj_name)
    el.setAttribute('mesh', mesh_name)
    el.setAttribute('pos', vec_to_mjcf(transform.pos))
    el.setAttribute('quat', quat_to_mjcf(transform.rot))
    el.setAttribute('type', 'mesh')
    if mat_name:
      el.setAttribute('material', mat_name)

    elements.append(el)

  return elements


def joint_builder(
    doc: minidom.Document,
    dof: blender_scene.Dof,
    dof_type: str,
) -> minidom.Element:
  """Builds a mujoco hinge definition."""
  el = doc.createElement('joint')
  el.setAttribute('name', dof.name)
  el.setAttribute('type', dof_type)
  el.setAttribute('limited', bool_to_mjcf(dof.limited))
  el.setAttribute('pos', vec_to_mjcf(_VEC_ZERO))
  el.setAttribute('axis', vec_to_mjcf(dof.axis))
  el.setAttribute('range', '{} {}'.format(dof.limits[0], dof.limits[1]))
  return el


def export_to_xml(
    doc: minidom.Document,
    objects: Sequence[blender_scene.ObjectRef],
    armature_freejoint: bool,
) -> minidom.Element:
  """Converts Blender scene objects to Mujoco scene tree nodes."""
  root = doc.createElement('worldbody')
  parent_elements = {blender_scene.NoneRef: root}

  for obj in objects:
    # Build a subtree corresponding to this Blender object.
    element = None
    if obj.is_armature:
      element = body_builder(doc, obj)
      if armature_freejoint:
        element.appendChild(doc.createElement('freejoint'))
    elif obj.is_mesh:
      if not obj.parent.is_none and not obj.parent.is_bone:
        raise RuntimeError(
            'Mesh "{}" is parented to an object "{}", which is not a bone. '
            'Only mesh->bone parenting is supported at the moment'.format(
                obj.name, obj.parent.name
            )
        )
      geom_elements = mesh_geom_builder(doc, obj)
      if len(geom_elements) > 1:
        # If there's more than one geom, introduce a body to aggregate them
        # under a single element. That element will then be associated with this
        # blender object should other blender elements be parented to it.
        element = doc.createElement('body')
        for geom_el in geom_elements:
          element.appendChild(geom_el)
      elif len(geom_elements) == 1:
        # Since there's only one geom, consider it the main element
        element = geom_elements[0]
    elif obj.is_light:
      element = light_builder(doc, obj)
    elif obj.is_bone:
      element = body_builder(doc, obj)
      for dof in obj.get_rotation_dofs():
        element.appendChild(joint_builder(doc, dof, 'hinge'))

    # Inject it into the scene tree.
    if element:
      parent = obj.parent
      if parent:
        parent_el = parent_elements[parent]
        parent_el.appendChild(element)
        parent_elements[obj] = element
      else:
        root.appendChild(element)

  return root
