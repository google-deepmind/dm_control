# Copyright 2017 The dm_control Authors.
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

"""Make dog model."""

import os

import numpy as np
from absl import app
from absl import flags
from lxml import etree

import add_torque_actuators
import build_back_legs
import build_front_legs
import build_neck
import build_tail
import build_torso
import create_skin
from dm_control import mjcf
from dm_control.utils import io as resources

flags.DEFINE_boolean(
  'make_skin', True, 'Whether to make a new dog_skin.skn')
flags.DEFINE_boolean(
  'use_tendons', False, 'Whether to add tendons to the model.')
flags.DEFINE_float(
  'lumbar_dofs_per_vertebra', 1.5,
  'Number of degrees of freedom per vertebra in lumbar spine.')
flags.DEFINE_float(
  'cervical_dofs_per_vertebra', 1.5,
  'Number of degrees of freedom vertebra in cervical spine.')
flags.DEFINE_float(
  'caudal_dofs_per_vertebra', 1,
  'Number of degrees of freedom vertebra in caudal spine.')

FLAGS = flags.FLAGS

BASE_MODEL = 'dog_base.xml'
ASSET_RELPATH = '../../../../suite/dog_assets'
ASSET_DIR = os.path.dirname(__file__) + '/' + ASSET_RELPATH
print(ASSET_DIR)


def exclude_contacts(model):
  """Exclude contacts from model.

  Args:
    model: model in which we want to exclude contacts.
  """
  physics = mjcf.Physics.from_mjcf_model(model)
  excluded_pairs = []
  for c in physics.data.contact:
    body1 = physics.model.id2name(physics.model.geom_bodyid[c.geom1], 'body')
    body2 = physics.model.id2name(physics.model.geom_bodyid[c.geom2], 'body')
    pair = body1 + ':' + body2
    if pair not in excluded_pairs:
      excluded_pairs.append(pair)
      model.contact.add('exclude', name=pair, body1=body1, body2=body2)
  # manual exclusions
  model.contact.add('exclude', name='C_1:jaw',
                    body1=model.find('body', 'C_1'),
                    body2=model.find('body', 'jaw'))
  model.contact.add(
    'exclude', name='torso:lower_arm_L',
    body1=model.find('body', 'torso'),
    body2='lower_arm_L')
  model.contact.add(
    'exclude', name='torso:lower_arm_R',
    body1=model.find('body', 'torso'),
    body2='lower_arm_R')
  model.contact.add(
    'exclude', name='C_4:scapula_R', body1='C_4', body2='scapula_R')
  model.contact.add(
    'exclude', name='C_4:scapula_L', body1='C_4', body2='scapula_L')
  model.contact.add(
    'exclude', name='C_5:upper_arm_R', body1='C_5', body2='upper_arm_R')
  model.contact.add(
    'exclude', name='C_5:upper_arm_L', body1='C_5', body2='upper_arm_L')
  model.contact.add(
    'exclude', name='C_6:upper_arm_R', body1='C_6', body2='upper_arm_R')
  model.contact.add(
    'exclude', name='C_6:upper_arm_L', body1='C_6', body2='upper_arm_L')
  model.contact.add(
    'exclude', name='C_7:upper_arm_R', body1='C_7', body2='upper_arm_R')
  model.contact.add(
    'exclude', name='C_7:upper_arm_L', body1='C_7', body2='upper_arm_L')
  model.contact.add(
    'exclude',
    name='upper_leg_L:upper_leg_R',
    body1='upper_leg_L',
    body2='upper_leg_R')
  for side in ['_L', '_R']:
    model.contact.add(
      'exclude',
      name='lower_leg' + side + ':pelvis',
      body1='lower_leg' + side,
      body2='pelvis')
    model.contact.add(
      'exclude',
      name='upper_leg' + side + ':foot' + side,
      body1='upper_leg' + side,
      body2='foot' + side)


def main(argv):
  del argv

  # Read flags.
  if FLAGS.is_parsed():
    use_tendons = FLAGS.use_tendons
    lumbar_dofs_per_vert = FLAGS.lumbar_dofs_per_vertebra
    cervical_dofs_per_vertebra = FLAGS.cervical_dofs_per_vertebra
    caudal_dofs_per_vertebra = FLAGS.caudal_dofs_per_vertebra
    make_skin = FLAGS.make_skin
  else:
    use_tendons = FLAGS['use_tendons'].default
    lumbar_dofs_per_vert = FLAGS['lumbar_dofs_per_vertebra'].default
    cervical_dofs_per_vertebra = FLAGS['cervical_dofs_per_vertebra'].default
    caudal_dofs_per_vertebra = FLAGS['caudal_dofs_per_vertebra'].default
    make_skin = FLAGS['make_skin'].default

  print('Load base model.')
  with open(BASE_MODEL, 'r') as f:
    model = mjcf.from_file(f)

  # Helper constants:
  side_sign = {'_L': np.array((1., -1., 1.)),
               '_R': np.array((1., 1., 1.))}
  primary_axis = {'_abduct': np.array((-1., 0., 0.)),
                  '_extend': np.array((0., 1., 0.)),
                  '_supinate': np.array((0., 0., -1.))}

  # Add meshes:
  print('Loading all meshes, getting positions and sizes.')
  meshdir = ASSET_DIR
  model.compiler.meshdir = meshdir
  texturedir = ASSET_DIR
  model.compiler.texturedir = texturedir
  site_meshes = []
  muscle_meshes = []
  bones = []
  for dirpath, _, filenames in resources.WalkResources(meshdir):
    prefix = 'extras/' if 'extras' in dirpath else ''
    for filename in filenames:
      if 'dog_skin.msh' in filename:
        skin_msh = model.asset.add('mesh', name='skin_msh',
                                   file=filename,
                                   scale=(1.25, 1.25, 1.25))
      name = filename[4:-4]
      name = name.replace('*', ':')
      if filename.startswith('BONE'):
        if 'Lingual' not in name:
          bones.append(name)
          model.asset.add('mesh', name=name, file=prefix + filename)

  # Put all bones in worldbody, get positions, remove bones:
  bone_geoms = []
  for bone in bones:
    geom = model.worldbody.add('geom', name=bone, mesh=bone, type='mesh',
                               contype=0, conaffinity=0, rgba=[1, .5, .5, .4])
    bone_geoms.append(geom)
  physics = mjcf.Physics.from_mjcf_model(model)
  bone_position = {}
  bone_size = {}
  for bone in bones:
    geom = model.find('geom', bone)
    bone_position[bone] = np.array(physics.bind(geom).xpos)
    bone_size[bone] = np.array(physics.bind(geom).rbound)
    geom.remove()

  # Torso
  print('Torso, lumbar spine, pelvis.')
  pelvic_bones, lumbar_joints = build_torso.create_torso(model,
                                                         bones, bone_position,
                                                         lumbar_dofs_per_vert,
                                                         side_sign,
                                                         parent=model.worldbody)

  print('Neck, skull, jaw.')
  # Cervical spine (neck) bodies:
  cervical_joints = build_neck.create_neck(model, bone_position,
                                           cervical_dofs_per_vertebra, bones,
                                           side_sign, bone_size,
                                           parent=model.find('body', 'torso'))

  print('Back legs.')
  nails, sole_sites = build_back_legs.create_back_legs(model,
                                                       primary_axis,
                                                       bone_position, bones,
                                                       side_sign, bone_size,
                                                       pelvic_bones,
                                                       parent=model.find(
                                                         'body',
                                                         'pelvis'))

  print('Shoulders, front legs.')
  palm_sites = build_front_legs.create_front_legs(nails,
                                                  model, primary_axis, bones,
                                                  side_sign,
                                                  parent=model.find('body',
                                                                    'torso'))

  print('Tail.')
  caudal_joints = build_tail.create_tail(caudal_dofs_per_vertebra,
                                         bone_size, model, bone_position,
                                         parent=model.find('body', 'pelvis'))

  print('Collision geoms, fixed tendons.')
  physics = mjcf.Physics.from_mjcf_model(model)

  print('Unify ribcage and jaw meshes.')
  for body in model.find_all('body'):
    body_meshes = [
      geom for geom in body.all_children() if geom.tag == 'geom' and
                                              hasattr(geom, 'mesh') and
                                              geom.mesh is not None
    ]
    if len(body_meshes) > 10:
      mergables = [('torso', 'Ribcage'), ('jaw', 'Jaw'),
                   ('skull', 'MergedSkull')]
      for bodyname, meshname in mergables:
        if body.name == bodyname:
          print('==== Merging ', bodyname)
          for mesh in body_meshes:
            print(mesh.name)
          body.add('inertial',
                   mass=physics.bind(body).mass,
                   pos=physics.bind(body).ipos,
                   quat=physics.bind(body).iquat,
                   diaginertia=physics.bind(body).inertia)

          for mesh in body_meshes:
            if 'eye' not in mesh.name:
              model.find('mesh', mesh.name).remove()
              mesh.remove()
          body.add(
            'geom',
            name=meshname,
            mesh=meshname,
            dclass='bone',
            pos=-bone_position[meshname])

  print('Add Actuators')
  actuated_joints = add_torque_actuators.add_motors(physics, model,
                                                    lumbar_joints,
                                                    cervical_joints,
                                                    caudal_joints)

  print('Excluding contacts.')
  exclude_contacts(model)

  if make_skin:
    create_skin.create(model, skin_msh)

  # Add skin from .skn
  print('Adding Skin.')
  skin_texture = model.asset.add('texture', name='skin',
                                 file='skin_texture.png', type='2d')
  skin_material = model.asset.add('material', name='skin', texture=skin_texture)
  skin = model.asset.add(
    'skin', name='skin', file='dog_skin.skn', material=skin_material)
  skin_msh.remove()

  print('Removing non-essential sites.')
  all_sites = model.find_all('site')
  for site in all_sites:
    if site.dclass is None:
      site.remove()

  physics = mjcf.Physics.from_mjcf_model(model)
  # sensors
  model.sensor.add('accelerometer', name='accelerometer',
                   site=model.find('site', 'head'))
  model.sensor.add('velocimeter', name='velocimeter',
                   site=model.find('site', 'head'))
  model.sensor.add('gyro', name='gyro', site=model.find('site', 'head'))
  model.sensor.add('subtreelinvel', name='torso_linvel',
                   body=model.find('body', 'torso'))
  model.sensor.add('subtreeangmom', name='torso_angmom',
                   body=model.find('body', 'torso'))
  for site in palm_sites + sole_sites:
    model.sensor.add('touch', name=site.name, site=site)
  anchors = [site for site in model.find_all('site') if 'anchor' in site.name]
  for site in anchors:
    model.sensor.add('force', name=site.name.replace('_anchor', ''), site=site)

  # Print stuff
  joint_acts = [model.find('actuator', j.name) for j in actuated_joints]
  print('{:20} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
    'name', 'mass', 'damping', 'stiffness', 'ratio', 'armature'))
  for i, j in enumerate(actuated_joints):
    dmp = physics.bind(j).damping[0]
    mass_eff = physics.bind(j).M0[0]
    dmp = physics.bind(j).damping[0]
    stf = physics.bind(joint_acts[i]).gainprm[0]
    arma = physics.bind(j).armature[0]
    print('{:20} {:10.4} {:10} {:10.4} {:10.4} {:10}'.format(
      j.name, mass_eff, dmp, stf, dmp / (2 * np.sqrt(mass_eff * stf)), arma))

  print('Finalising and saving model.')
  xml_string = model.to_xml_string('float', precision=4, zero_threshold=1e-7)
  root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

  print('Remove hashes from filenames')
  assets = list(root.find('asset').iter())
  for asset in assets:
    asset_filename = asset.get('file')
    if asset_filename is not None:
      name = asset_filename[:-4]
      extension = asset_filename[-4:]
      asset.set('file', name[:-41] + extension)

  print('Add <compiler meshdir/>, for locally-loadable model')
  compiler = etree.Element(
    'compiler', meshdir=ASSET_RELPATH, texturedir=ASSET_RELPATH)
  root.insert(0, compiler)

  print('Remove class="/"')
  default_elem = root.find('default')
  root.insert(6, default_elem[0])
  root.remove(default_elem)
  xml_string = etree.tostring(root, pretty_print=True)
  xml_string = xml_string.replace(b' class="/"', b'')

  print('Insert spaces between top level elements')
  lines = xml_string.splitlines()
  newlines = []
  for line in lines:
    newlines.append(line)
    if line.startswith(b'  <'):
      if line.startswith(b'  </') or line.endswith(b'/>'):
        newlines.append(b'')
  newlines.append(b'')
  xml_string = b'\n'.join(newlines)

  # Save to file.
  f = open('dog.xml', 'wb')
  f.write(xml_string)
  f.close()


if __name__ == '__main__':
  app.run(main)
