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

"""Blender 3.3 plugin for exporting models to MuJoCo native format."""

import contextlib
import os
from xml.dom import minidom

import bpy
from bpy_extras.io_utils import ExportHelper

from . import blender_scene
from . import mujoco_assets
from . import mujoco_scene


bl_info = {
    'name': 'Export MuJoCo',
    'author': 'Piotr Trochim',
    'version': (2, 0),
    'blender': (3, 3, 1),
    'location': 'File > Export > MuJoCo',
    'warning': '',
    'description': 'Export articulated MuJoCo model',
    'doc_url': '',
    'category': 'Import-Export',
}


@contextlib.contextmanager
def context_settings_cacher(context: bpy.types.Context):
  """Preserves the pose of exported objects and the scene mode."""
  # Cache the mode
  prev_mode = context.mode

  # Set the Object mode required by the exporter
  bpy.ops.object.mode_set(mode='OBJECT')

  # Set the armatures in their neutral pose
  pose_positions = []
  for o in context.scene.objects:
    if o.type == 'ARMATURE':
      pose_positions.append((o, o.data.pose_position))
      o.data.pose_position = 'REST'
  context.view_layer.update()

  try:
    yield
  finally:
    # Restore the poses.
    for armature, pose_position in pose_positions:
      armature.data.pose_position = pose_position
    context.view_layer.update()

    # Restore the mode
    bpy.ops.object.mode_set(mode=prev_mode)


def apply_scale():
  bpy.ops.object.select_all(action='SELECT')
  bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
  bpy.ops.object.select_all(action='DESELECT')


class ExportMjcf(bpy.types.Operator, ExportHelper):
  """Export to MJCF file format."""

  bl_idname = 'export_scene.mjcf'
  bl_label = 'Export MJCF'

  filename_ext = '.xml'
  filter_glob = bpy.props.StringProperty(default='*.xml', options={'HIDDEN'})

  # Export settings
  armature_freejoint: bpy.props.BoolProperty(
      name='Armature freejoint',
      description='Add a freejoint to armature body',
      default=False,
  )
  apply_mesh_modifiers: bpy.props.BoolProperty(
      name='Apply modifiers',
      description='Apply mesh modifiers',
      default=False,
  )

  def _export_mjcf(self, context: bpy.types.Context) -> str:
    """Converts a Blender scene to Mujoco XML format."""
    # Create a new XML document
    xml_doc = minidom.Document()
    mujoco = xml_doc.createElement('mujoco')
    xml_doc.appendChild(mujoco)

    # Create a list of blender objects, arranged according to their hierarchy,
    # where parents precede the children.
    blender_objects = blender_scene.map_blender_tree(
        context, lambda o: o if o.is_visible else None
    )
    # Remove None entries that correspond to invisible objects.
    blender_objects = [o for o in blender_objects if o]

    export_settings = self.as_keywords()

    # Build the scene tree
    worldbody_el = mujoco_scene.export_to_xml(
        doc=xml_doc,
        objects=blender_objects,
        armature_freejoint=export_settings['armature_freejoint'],
    )
    mujoco.appendChild(worldbody_el)

    # Build the asset tree
    asset_el = mujoco_assets.export_to_xml(
        doc=xml_doc,
        objects=blender_objects,
        folder=os.path.dirname(export_settings['filepath']),
        apply_mesh_modifiers=export_settings['apply_mesh_modifiers'],
    )
    mujoco.appendChild(asset_el)

    # Add compiler options that would allow to export small feature meshes.
    compiler_el = xml_doc.createElement('compiler')
    mujoco.appendChild(compiler_el)
    compiler_el.setAttribute('boundmass', '1e-3')
    compiler_el.setAttribute('boundinertia', '1e-9')
    # TODO(b/266818670): support assets export into subdirectory
    # compiler_el.setAttribute('meshdir', 'assets')

    # Write the XML to a file.
    with open(export_settings['filepath'], 'w') as file:
      file.write(xml_doc.toprettyxml(indent='  '))

  def execute(self, context: bpy.types.Context):
    """Export the scene."""
    with context_settings_cacher(context):
      apply_scale()
      self._export_mjcf(context)

    return {'FINISHED'}


def menu_func_export(self, context: bpy.types.Context):
  del context
  self.layout.operator(ExportMjcf.bl_idname, text='MuJoCo (.xml)')


def register():
  bpy.utils.register_class(ExportMjcf)
  bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
  bpy.utils.unregister_class(ExportMjcf)
  bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)


if __name__ == '__main__':
  register()
