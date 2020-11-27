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

"""Procedurally generated Swimmer domain."""
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .03  # (Seconds)

SUITE = containers.TaggedTasks()


def get_model_and_assets(n_joints):
  """Returns a tuple containing the model XML string and a dict of assets.

  Args:
    n_joints: An integer specifying the number of joints in the swimmer.

  Returns:
    A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
    `{filename: contents_string}` pairs.
  """
  return _make_model(n_joints), common.ASSETS


@SUITE.add('benchmarking')
def swimmer6(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  """Returns a 6-link swimmer."""
  return _make_swimmer(6, time_limit, random=random,
                       environment_kwargs=environment_kwargs)


@SUITE.add('benchmarking')
def swimmer15(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns a 15-link swimmer."""
  return _make_swimmer(15, time_limit, random=random,
                       environment_kwargs=environment_kwargs)


def swimmer(n_links=3, time_limit=_DEFAULT_TIME_LIMIT,
            random=None, environment_kwargs=None):
  """Returns a swimmer with n links."""
  return _make_swimmer(n_links, time_limit, random=random,
                       environment_kwargs=environment_kwargs)


def _make_swimmer(n_joints, time_limit=_DEFAULT_TIME_LIMIT, random=None,
                  environment_kwargs=None):
  """Returns a swimmer control environment."""
  model_string, assets = get_model_and_assets(n_joints)
  physics = Physics.from_xml_string(model_string, assets=assets)
  task = Swimmer(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


def _make_model(n_bodies):
  """Generates an xml string defining a swimmer with `n_bodies` bodies."""
  if n_bodies < 3:
    raise ValueError('At least 3 bodies required. Received {}'.format(n_bodies))
  mjcf = etree.fromstring(common.read_model('swimmer.xml'))
  head_body = mjcf.find('./worldbody/body')
  actuator = etree.SubElement(mjcf, 'actuator')
  sensor = etree.SubElement(mjcf, 'sensor')

  parent = head_body
  for body_index in range(n_bodies - 1):
    site_name = 'site_{}'.format(body_index)
    child = _make_body(body_index=body_index)
    child.append(etree.Element('site', name=site_name))
    joint_name = 'joint_{}'.format(body_index)
    joint_limit = 360.0/n_bodies
    joint_range = '{} {}'.format(-joint_limit, joint_limit)
    child.append(etree.Element('joint', {'name': joint_name,
                                         'range': joint_range}))
    motor_name = 'motor_{}'.format(body_index)
    actuator.append(etree.Element('motor', name=motor_name, joint=joint_name))
    velocimeter_name = 'velocimeter_{}'.format(body_index)
    sensor.append(etree.Element('velocimeter', name=velocimeter_name,
                                site=site_name))
    gyro_name = 'gyro_{}'.format(body_index)
    sensor.append(etree.Element('gyro', name=gyro_name, site=site_name))
    parent.append(child)
    parent = child

  # Move tracking cameras further away from the swimmer according to its length.
  cameras = mjcf.findall('./worldbody/body/camera')
  scale = n_bodies / 6.0
  for cam in cameras:
    if cam.get('mode') == 'trackcom':
      old_pos = cam.get('pos').split(' ')
      new_pos = ' '.join([str(float(dim) * scale) for dim in old_pos])
      cam.set('pos', new_pos)

  return etree.tostring(mjcf, pretty_print=True)


def _make_body(body_index):
  """Generates an xml string defining a single physical body."""
  body_name = 'segment_{}'.format(body_index)
  visual_name = 'visual_{}'.format(body_index)
  inertial_name = 'inertial_{}'.format(body_index)
  body = etree.Element('body', name=body_name)
  body.set('pos', '0 .1 0')
  etree.SubElement(body, 'geom', {'class': 'visual', 'name': visual_name})
  etree.SubElement(body, 'geom', {'class': 'inertial', 'name': inertial_name})
  return body


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the swimmer domain."""

  def nose_to_target(self):
    """Returns a vector from nose to target in local coordinate of the head."""
    nose_to_target = (self.named.data.geom_xpos['target'] -
                      self.named.data.geom_xpos['nose'])
    head_orientation = self.named.data.xmat['head'].reshape(3, 3)
    return nose_to_target.dot(head_orientation)[:2]

  def nose_to_target_dist(self):
    """Returns the distance from the nose to the target."""
    return np.linalg.norm(self.nose_to_target())

  def body_velocities(self):
    """Returns local body velocities: x,y linear, z rotational."""
    xvel_local = self.data.sensordata[12:].reshape((-1, 6))
    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    return xvel_local[:, vx_vy_wz].ravel()

  def joints(self):
    """Returns all internal joint angles (excluding root joints)."""
    return self.data.qpos[3:].copy()


class Swimmer(base.Task):
  """A swimmer `Task` to reach the target or just swim."""

  def __init__(self, random=None):
    """Initializes an instance of `Swimmer`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Initializes the swimmer orientation to [-pi, pi) and the relative joint
    angle of each joint uniformly within its range.

    Args:
      physics: An instance of `Physics`.
    """
    # Random joint angles:
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    # Random target position.
    close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = .3 if close_target else 2
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    physics.named.model.light_pos['target_light', 'x'] = xpos
    physics.named.model.light_pos['target_light', 'y'] = ypos

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of joint angles, body velocities and target."""
    obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    obs['to_target'] = physics.nose_to_target()
    obs['body_velocities'] = physics.body_velocities()
    return obs

  def get_reward(self, physics):
    """Returns a smooth reward."""
    target_size = physics.named.model.geom_size['target', 0]
    return rewards.tolerance(physics.nose_to_target_dist(),
                             bounds=(0, target_size),
                             margin=5*target_size,
                             sigmoid='long_tail')
