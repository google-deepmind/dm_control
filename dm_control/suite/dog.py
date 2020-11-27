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

"""Dog Domain."""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np

from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT = 15
_CONTROL_TIMESTEP = .015

# Angle (in degrees) of local z from global z below which upright reward is 1.
_MAX_UPRIGHT_ANGLE = 30
_MIN_UPRIGHT_COSINE = np.cos(np.deg2rad(_MAX_UPRIGHT_ANGLE))

# Standing reward is 1 for body-over-foot height that is at least this fraction
# of the height at the default pose.
_STAND_HEIGHT_FRACTION = 0.9

# Torques which enforce joint range limits should stay below this value.
_EXCESSIVE_LIMIT_TORQUES = 150

# Horizontal speed above which Move reward is 1.
_WALK_SPEED = 1
_TROT_SPEED = 3
_RUN_SPEED = 9

_HINGE_TYPE = mujoco.wrapper.mjbindings.enums.mjtJoint.mjJNT_HINGE
_LIMIT_TYPE = mujoco.wrapper.mjbindings.enums.mjtConstraint.mjCNSTR_LIMIT_JOINT

SUITE = containers.TaggedTasks()

_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'dog_assets')


def make_model(floor_size, remove_ball):
  """Sets floor size, removes ball and walls (Stand and Move tasks)."""
  xml_string = common.read_model('dog.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  # set floor size.
  floor = xml_tools.find_element(mjcf, 'geom', 'floor')
  floor.attrib['size'] = str(floor_size) + ' ' + str(floor_size) + ' .1'

  if remove_ball:
    # Remove ball, target and walls.
    ball = xml_tools.find_element(mjcf, 'body', 'ball')
    ball.getparent().remove(ball)
    target = xml_tools.find_element(mjcf, 'geom', 'target')
    target.getparent().remove(target)
    ball_cam = xml_tools.find_element(mjcf, 'camera', 'ball')
    ball_cam.getparent().remove(ball_cam)
    head_cam = xml_tools.find_element(mjcf, 'camera', 'head')
    head_cam.getparent().remove(head_cam)
    for wall_name in ['px', 'nx', 'py', 'ny']:
      wall = xml_tools.find_element(mjcf, 'geom', 'wall_' + wall_name)
      wall.getparent().remove(wall)

  return etree.tostring(mjcf, pretty_print=True)


def get_model_and_assets(floor_size=10, remove_ball=True):
  """Returns a tuple containing the model XML string and a dict of assets."""
  assets = common.ASSETS.copy()
  _, _, filenames = next(resources.WalkResources(_ASSET_DIR))
  for filename in filenames:
    assets[filename] = resources.GetResource(os.path.join(_ASSET_DIR, filename))
  return make_model(floor_size, remove_ball), assets


@SUITE.add('no_reward_visualization')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  floor_size = _WALK_SPEED * _DEFAULT_TIME_LIMIT
  physics = Physics.from_xml_string(*get_model_and_assets(floor_size))
  task = Stand(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


@SUITE.add('no_reward_visualization')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  move_speed = _WALK_SPEED
  floor_size = move_speed * _DEFAULT_TIME_LIMIT
  physics = Physics.from_xml_string(*get_model_and_assets(floor_size))
  task = Move(move_speed=move_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


@SUITE.add('no_reward_visualization')
def trot(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Trot task."""
  move_speed = _TROT_SPEED
  floor_size = move_speed * _DEFAULT_TIME_LIMIT
  physics = Physics.from_xml_string(*get_model_and_assets(floor_size))
  task = Move(move_speed=move_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


@SUITE.add('no_reward_visualization')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  move_speed = _RUN_SPEED
  floor_size = move_speed * _DEFAULT_TIME_LIMIT
  physics = Physics.from_xml_string(*get_model_and_assets(floor_size))
  task = Move(move_speed=move_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


@SUITE.add('no_reward_visualization', 'hard')
def fetch(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Fetch task."""
  physics = Physics.from_xml_string(*get_model_and_assets(remove_ball=False))
  task = Fetch(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Dog domain."""

  def torso_pelvis_height(self):
    """Returns the height of the torso."""
    return self.named.data.xpos[['torso', 'pelvis'], 'z']

  def z_projection(self):
    """Returns rotation-invariant projection of local frames to the world z."""
    return np.vstack((self.named.data.xmat['skull', ['zx', 'zy', 'zz']],
                      self.named.data.xmat['torso', ['zx', 'zy', 'zz']],
                      self.named.data.xmat['pelvis', ['zx', 'zy', 'zz']]))

  def upright(self):
    """Returns projection from local z-axes to the z-axis of world."""
    return self.z_projection()[:, 2]

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_linvel']

  def torso_com_velocity(self):
    """Returns the velocity of the center-of-mass in the torso frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3).copy()
    return self.center_of_mass_velocity().dot(torso_frame)

  def com_forward_velocity(self):
    """Returns the com velocity in the torso's forward direction."""
    return self.torso_com_velocity()[0]

  def joint_angles(self):
    """Returns the configuration of all hinge joints (skipping free joints)."""
    hinge_joints = self.model.jnt_type == _HINGE_TYPE
    qpos_index = self.model.jnt_qposadr[hinge_joints]
    return self.data.qpos[qpos_index].copy()

  def joint_velocities(self):
    """Returns the velocity of all hinge joints (skipping free joints)."""
    hinge_joints = self.model.jnt_type == _HINGE_TYPE
    qvel_index = self.model.jnt_dofadr[hinge_joints]
    return self.data.qvel[qvel_index].copy()

  def inertial_sensors(self):
    """Returns inertial sensor readings."""
    return self.named.data.sensordata[['accelerometer', 'velocimeter', 'gyro']]

  def touch_sensors(self):
    """Returns touch readings."""
    return self.named.data.sensordata[['palm_L', 'palm_R', 'sole_L', 'sole_R']]

  def foot_forces(self):
    """Returns touch readings."""
    return self.named.data.sensordata[['foot_L', 'foot_R', 'hand_L', 'hand_R']]

  def ball_in_head_frame(self):
    """Returns the ball position and velocity in the frame of the head."""
    head_frame = self.named.data.site_xmat['head'].reshape(3, 3)
    head_pos = self.named.data.site_xpos['head']
    ball_pos = self.named.data.geom_xpos['ball']
    head_to_ball = ball_pos - head_pos
    head_vel, _ = self.data.object_velocity('head', 'site')
    ball_vel, _ = self.data.object_velocity('ball', 'geom')
    head_to_ball_vel = ball_vel - head_vel
    return np.hstack((head_to_ball.dot(head_frame),
                      head_to_ball_vel.dot(head_frame)))

  def target_in_head_frame(self):
    """Returns the target position in the frame of the head."""
    head_frame = self.named.data.site_xmat['head'].reshape(3, 3)
    head_pos = self.named.data.site_xpos['head']
    target_pos = self.named.data.geom_xpos['target']
    head_to_target = target_pos - head_pos
    return head_to_target.dot(head_frame)

  def ball_to_mouth_distance(self):
    """Returns the distance from the ball to the mouth."""
    ball_pos = self.named.data.geom_xpos['ball']
    upper_bite_pos = self.named.data.site_xpos['upper_bite']
    lower_bite_pos = self.named.data.site_xpos['lower_bite']
    upper_dist = np.linalg.norm(ball_pos - upper_bite_pos)
    lower_dist = np.linalg.norm(ball_pos - lower_bite_pos)
    return 0.5*(upper_dist + lower_dist)

  def ball_to_target_distance(self):
    """Returns the distance from the ball to the target."""
    ball_pos, target_pos = self.named.data.geom_xpos[['ball', 'target']]
    return np.linalg.norm(ball_pos - target_pos)


class Stand(base.Task):
  """A dog stand task generating upright posture."""

  def __init__(self, random=None, observe_reward_factors=False):
    """Initializes an instance of `Stand`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      observe_reward_factors: Boolean, whether the factorised reward is a
        key in the observation dict returned to the agent.
    """
    self._observe_reward_factors = observe_reward_factors
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Randomizes initial root velocities and actuator states.

    Args:
      physics: An instance of `Physics`.

    """
    physics.reset()

    # Measure stand heights from default pose, above which stand reward is 1.
    self._stand_height = physics.torso_pelvis_height() * _STAND_HEIGHT_FRACTION

    # Measure body weight.
    body_mass = physics.named.model.body_subtreemass['torso']
    self._body_weight = -physics.model.opt.gravity[2] * body_mass

    # Randomize horizontal orientation.
    azimuth = self.random.uniform(0, 2*np.pi)
    orientation = np.array((np.cos(azimuth/2), 0, 0, np.sin(azimuth/2)))
    physics.named.data.qpos['root'][3:] = orientation

    # Randomize root velocities in horizontal plane.
    physics.data.qvel[0] = 2 * self.random.randn()
    physics.data.qvel[1] = 2 * self.random.randn()
    physics.data.qvel[5] = 2 * self.random.randn()

    # Randomize actuator states.
    assert physics.model.nu == physics.model.na
    for actuator_id in range(physics.model.nu):
      ctrlrange = physics.model.actuator_ctrlrange[actuator_id]
      physics.data.act[actuator_id] = self.random.uniform(*ctrlrange)

  def get_observation_components(self, physics):
    """Returns the observations for the Stand task."""
    obs = collections.OrderedDict()
    obs['joint_angles'] = physics.joint_angles()
    obs['joint_velocites'] = physics.joint_velocities()
    obs['torso_pelvis_height'] = physics.torso_pelvis_height()
    obs['z_projection'] = physics.z_projection().flatten()
    obs['torso_com_velocity'] = physics.torso_com_velocity()
    obs['inertial_sensors'] = physics.inertial_sensors()
    obs['foot_forces'] = physics.foot_forces()
    obs['touch_sensors'] = physics.touch_sensors()
    obs['actuator_state'] = physics.data.act.copy()
    return obs

  def get_observation(self, physics):
    """Returns the observation, possibly adding reward factors."""
    obs = self.get_observation_components(physics)
    if self._observe_reward_factors:
      obs['reward_factors'] = self.get_reward_factors(physics)
    return obs

  def get_reward_factors(self, physics):
    """Returns the factorized reward."""
    # Keep the torso  at standing height.
    torso = rewards.tolerance(physics.torso_pelvis_height()[0],
                              bounds=(self._stand_height[0], float('inf')),
                              margin=self._stand_height[0])
    # Keep the pelvis at standing height.
    pelvis = rewards.tolerance(physics.torso_pelvis_height()[1],
                               bounds=(self._stand_height[1], float('inf')),
                               margin=self._stand_height[1])
    # Keep head, torso and pelvis upright.
    upright = rewards.tolerance(physics.upright(),
                                bounds=(_MIN_UPRIGHT_COSINE, float('inf')),
                                sigmoid='linear',
                                margin=_MIN_UPRIGHT_COSINE+1,
                                value_at_margin=0)

    # Reward for foot touch forces up to bodyweight.
    touch = rewards.tolerance(physics.touch_sensors().sum(),
                              bounds=(self._body_weight, float('inf')),
                              margin=self._body_weight,
                              sigmoid='linear',
                              value_at_margin=0.9)

    return np.hstack((torso, pelvis, upright, touch))

  def get_reward(self, physics):
    """Returns the reward, product of reward factors."""
    return np.product(self.get_reward_factors(physics))


class Move(Stand):
  """A dog move task for generating locomotion."""

  def __init__(self, move_speed, random, observe_reward_factors=False):
    """Initializes an instance of `Move`.

    Args:
      move_speed: A float. Specifies a target horizontal velocity.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      observe_reward_factors: Boolean, whether the factorised reward is a
        component of the observation dict.
    """
    self._move_speed = move_speed
    super().__init__(random, observe_reward_factors)

  def get_reward_factors(self, physics):
    """Returns the factorized reward."""
    standing = super().get_reward_factors(physics)

    speed_margin = max(1.0, self._move_speed)
    forward = rewards.tolerance(physics.com_forward_velocity(),
                                bounds=(self._move_speed, 2*self._move_speed),
                                margin=speed_margin,
                                value_at_margin=0,
                                sigmoid='linear')
    forward = (4*forward + 1) / 5

    return np.hstack((standing, forward))


class Fetch(Stand):
  """A dog fetch task to fetch a thrown ball."""

  def __init__(self, random, observe_reward_factors=False):
    """Initializes an instance of `Move`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      observe_reward_factors: Boolean, whether the factorised reward is a
        component of the observation dict.
    """
    super().__init__(random, observe_reward_factors)

  def initialize_episode(self, physics):
    super().initialize_episode(physics)

    # Set initial ball state: flying towards the center at an upward angle.
    radius = 0.75 * physics.named.model.geom_size['floor', 0]
    azimuth = self.random.uniform(0, 2*np.pi)
    position = (radius*np.sin(azimuth), radius*np.cos(azimuth), 0.05)
    physics.named.data.qpos['ball_root'][:3] = position
    vertical_height = self.random.uniform(0, 3)
    # Equating kinetic and potential energy: mv^2/2 = m*g*h -> v = sqrt(2gh)
    gravity = -physics.model.opt.gravity[2]
    vertical_velocity = np.sqrt(2 * gravity * vertical_height)
    horizontal_speed = self.random.uniform(0, 5)
    # Pointing towards the center, with some noise.
    direction = np.array((-np.sin(azimuth) + 0.05*self.random.randn(),
                          -np.cos(azimuth) + 0.05*self.random.randn()))
    horizontal_velocity = horizontal_speed * direction
    velocity = np.hstack((horizontal_velocity, vertical_velocity))
    physics.named.data.qvel['ball_root'][:3] = velocity

  def get_observation_components(self, physics):
    """Returns the common observations for the Stand task."""
    obs = super().get_observation_components(physics)
    obs['ball_state'] = physics.ball_in_head_frame()
    obs['target_position'] = physics.target_in_head_frame()
    return obs

  def get_reward_factors(self, physics):
    """Returns a reward to the agent."""
    standing = super().get_reward_factors(physics)

    # Reward for bringing mouth close to ball.
    bite_radius = physics.named.model.site_size['upper_bite', 0]
    bite_margin = 2
    reach_ball = rewards.tolerance(physics.ball_to_mouth_distance(),
                                   bounds=(0, bite_radius),
                                   sigmoid='reciprocal',
                                   margin=bite_margin)
    reach_ball = (6*reach_ball + 1) / 7

    # Reward for bringing the ball close to the target.
    target_radius = physics.named.model.geom_size['target', 0]
    bring_margin = physics.named.model.geom_size['floor', 0]
    ball_near_target = rewards.tolerance(
        physics.ball_to_target_distance(),
        bounds=(0, target_radius),
        sigmoid='reciprocal',
        margin=bring_margin)
    fetch_ball = (ball_near_target + 1) / 2

    # Let go of the ball if it's been fetched.
    if physics.ball_to_target_distance() < 2*target_radius:
      reach_ball = 1

    return np.hstack((standing, reach_ball, fetch_ball))
