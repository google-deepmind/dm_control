# Copyright 2020 The dm_control Authors.
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
"""Utils for reference pose tasks."""

from dm_control import mjcf
from dm_control.utils import transformations as tr
import numpy as np


def add_walker(walker_fn, arena, name='walker', ghost=False, visible=True,
               position=(0, 0, 0)):
  """Create a walker."""
  walker = walker_fn(name=name)

  if ghost:
    # if the walker has a built-in tracking light remove it.
    light = walker.mjcf_model.find('light', 'tracking_light')
    if light:
      light.remove()

    # Remove the contacts.
    for geom in walker.mjcf_model.find_all('geom'):
      # alpha=0.999 ensures grey ghost reference.
      # for alpha=1.0 there is no visible difference between real walker and
      # ghost reference.
      geom.set_attributes(
          contype=0,
          conaffinity=0,
          rgba=(0.5, 0.5, 0.5, .999 if visible else 0.0))

    skin = walker.mjcf_model.find('skin', 'skin')
    if skin:
      if visible:
        skin.set_attributes(rgba=(0.5, 0.5, 0.5, 0.999))
      else:
        skin.set_attributes(rgba=(0.5, 0.5, 0.5, 0.))

  if position == (0, 0, 0):
    walker.create_root_joints(arena.attach(walker))
  else:
    spawn_site = arena.mjcf_model.worldbody.add('site', pos=position)
    walker.create_root_joints(arena.attach(walker, spawn_site))
    spawn_site.remove()

  return walker


def get_qpos_qvel_from_features(features):
  """Get qpos and qvel from logged features to set walker."""
  full_qpos = np.hstack([
      features['position'],
      features['quaternion'],
      features['joints'],
  ])
  full_qvel = np.hstack([
      features['velocity'],
      features['angular_velocity'],
      features['joints_velocity'],
  ])
  return full_qpos, full_qvel


def set_walker_from_features(physics, walker, features, offset=0):
  """Set the freejoint and walker's joints angles and velocities."""
  qpos, qvel = get_qpos_qvel_from_features(features)
  set_walker(physics, walker, qpos, qvel, offset=offset)


def set_walker(physics, walker, qpos, qvel, offset=0, null_xyz_and_yaw=False,
               position_shift=None, rotation_shift=None):
  """Set the freejoint and walker's joints angles and velocities."""
  qpos = np.array(qpos)
  if null_xyz_and_yaw:
    qpos[:3] = 0.
    euler = tr.quat_to_euler(qpos[3:7], ordering='ZYX')
    euler[0] = 0.
    quat = tr.euler_to_quat(euler, ordering='ZYX')
    qpos[3:7] = quat
  qpos[:3] += offset

  freejoint = mjcf.get_attachment_frame(walker.mjcf_model).freejoint

  physics.bind(freejoint).qpos = qpos[:7]
  physics.bind(freejoint).qvel = qvel[:6]

  physics.bind(walker.mocap_joints).qpos = qpos[7:]
  physics.bind(walker.mocap_joints).qvel = qvel[6:]
  if position_shift is not None or rotation_shift is not None:
    walker.shift_pose(physics, position=position_shift,
                      quaternion=rotation_shift, rotate_velocity=True)


def get_features(physics, walker):
  """Get walker features for reward functions."""
  walker_bodies = walker.mocap_tracking_bodies

  walker_features = {}
  root_pos, root_quat = walker.get_pose(physics)
  walker_features['position'] = np.array(root_pos)
  walker_features['quaternion'] = np.array(root_quat)
  joints = np.array(physics.bind(walker.mocap_joints).qpos)
  walker_features['joints'] = joints
  freejoint_frame = mjcf.get_attachment_frame(walker.mjcf_model)
  com = np.array(physics.bind(freejoint_frame).subtree_com)
  walker_features['center_of_mass'] = com
  end_effectors = np.array(
      walker.observables.end_effectors_pos(physics)[:]).reshape(-1, 3)
  walker_features['end_effectors'] = end_effectors
  if hasattr(walker.observables, 'appendages_pos'):
    appendages = np.array(
        walker.observables.appendages_pos(physics)[:]).reshape(-1, 3)
  else:
    appendages = np.array(end_effectors)
  walker_features['appendages'] = appendages
  xpos = np.array(physics.bind(walker_bodies).xpos)
  walker_features['body_positions'] = xpos
  xquat = np.array(physics.bind(walker_bodies).xquat)
  walker_features['body_quaternions'] = xquat
  root_vel, root_angvel = walker.get_velocity(physics)
  walker_features['velocity'] = np.array(root_vel)
  walker_features['angular_velocity'] = np.array(root_angvel)
  joints_vel = np.array(physics.bind(walker.mocap_joints).qvel)
  walker_features['joints_velocity'] = joints_vel
  return walker_features
