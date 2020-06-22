# Copyright 2019 The dm_control Authors.
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

"""A task where the goal is to place a movable prop on top of a fixed prop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards
import numpy as np
import six


_PlaceWorkspace = collections.namedtuple(
    '_PlaceWorkspace', ['prop_bbox', 'target_bbox', 'tcp_bbox', 'arm_offset'])

_TARGET_RADIUS = 0.05
_PEDESTAL_RADIUS = 0.07

# Ensures that the prop does not collide with the table during initialization.
_PROP_Z_OFFSET = 1e-6

_WORKSPACE = _PlaceWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET),
        upper=(0.1, 0.1, _PROP_Z_OFFSET)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PEDESTAL_RADIUS + 0.1),
        upper=(0.1, 0.1, 0.4)),
    target_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PEDESTAL_RADIUS),
        upper=(0.1, 0.1, _PEDESTAL_RADIUS + 0.1)),
    arm_offset=robots.ARM_OFFSET)


class SphereCradle(composer.Entity):
  """A concave shape for easy placement."""
  _SPHERE_COUNT = 3

  def _build(self):
    self._mjcf_root = mjcf.element.RootElement(model='cradle')
    sphere_radius = _PEDESTAL_RADIUS * 0.7
    for ang in np.linspace(0, 2*np.pi, num=self._SPHERE_COUNT, endpoint=False):
      pos = 0.7 * sphere_radius * np.array([np.sin(ang), np.cos(ang), -1])
      self._mjcf_root.worldbody.add(
          'geom', type='sphere', size=[sphere_radius], condim=6, pos=pos)

  @property
  def mjcf_model(self):
    return self._mjcf_root


class Pedestal(composer.Entity):
  """A narrow pillar to elevate the target."""
  _HEIGHT = 0.2

  def _build(self, cradle, target_radius):
    self._mjcf_root = mjcf.element.RootElement(model='pedestal')

    self._mjcf_root.worldbody.add(
        'geom', type='capsule', size=[_PEDESTAL_RADIUS],
        fromto=[0, 0, -_PEDESTAL_RADIUS,
                0, 0, -(self._HEIGHT + _PEDESTAL_RADIUS)])
    attachment_site = self._mjcf_root.worldbody.add(
        'site', type='sphere', size=(0.003,), group=constants.TASK_SITE_GROUP)
    self.attach(cradle, attachment_site)
    self._target_site = workspaces.add_target_site(
        body=self.mjcf_model.worldbody,
        radius=target_radius, rgba=constants.RED)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def target_site(self):
    return self._target_site

  def _build_observables(self):
    return PedestalObservables(self)


class PedestalObservables(composer.Observables):
  """Observables for the `Pedestal` prop."""

  @define.observable
  def position(self):
    return observable.MJCFFeature('xpos', self._entity.target_site)


class Place(composer.Task):
  """Place the prop on top of another fixed prop held up by a pedestal."""

  def __init__(self, arena, arm, hand, prop, obs_settings, workspace,
               control_timestep, cradle):
    """Initializes a new `Place` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      prop: `composer.Entity` instance.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: A `_PlaceWorkspace` instance.
      control_timestep: Float specifying the control timestep in seconds.
      cradle: `composer.Entity` onto which the `prop` must be placed.
    """
    self._arena = arena
    self._arm = arm
    self._hand = hand
    self._arm.attach(self._hand)
    self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
    self.control_timestep = control_timestep

    # Add custom camera observable.
    self._task_observables = cameras.add_camera_observables(
        arena, obs_settings, cameras.FRONT_CLOSE)

    self._tcp_initializer = initializers.ToolCenterPointInitializer(
        self._hand, self._arm,
        position=distributions.Uniform(*workspace.tcp_bbox),
        quaternion=workspaces.DOWN_QUATERNION)

    self._prop = prop
    self._prop_frame = self._arena.add_free_entity(prop)
    self._pedestal = Pedestal(cradle=cradle, target_radius=_TARGET_RADIUS)
    self._arena.attach(self._pedestal)

    for obs in six.itervalues(self._pedestal.observables.as_dict()):
      obs.configure(**obs_settings.prop_pose._asdict())

    self._prop_placer = initializers.PropPlacer(
        props=[prop],
        position=distributions.Uniform(*workspace.prop_bbox),
        quaternion=workspaces.uniform_z_rotation,
        settle_physics=True,
        max_attempts_per_prop=50)

    self._pedestal_placer = initializers.PropPlacer(
        props=[self._pedestal],
        position=distributions.Uniform(*workspace.target_bbox),
        settle_physics=False)

    # Add sites for visual debugging.
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.tcp_bbox.lower,
        upper=workspace.tcp_bbox.upper,
        rgba=constants.GREEN, name='tcp_spawn_area')
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.prop_bbox.lower,
        upper=workspace.prop_bbox.upper,
        rgba=constants.BLUE, name='prop_spawn_area')
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.target_bbox.lower,
        upper=workspace.target_bbox.upper,
        rgba=constants.CYAN, name='pedestal_spawn_area')

  @property
  def root_entity(self):
    return self._arena

  @property
  def arm(self):
    return self._arm

  @property
  def hand(self):
    return self._hand

  @property
  def task_observables(self):
    return self._task_observables

  def initialize_episode(self, physics, random_state):
    self._pedestal_placer(physics, random_state,
                          ignore_contacts_with_entities=[self._prop])
    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._tcp_initializer(physics, random_state)
    self._prop_placer(physics, random_state)

  def get_reward(self, physics):
    target = physics.bind(self._pedestal.target_site).xpos
    obj = physics.bind(self._prop_frame).xpos
    tcp = physics.bind(self._hand.tool_center_point).xpos

    tcp_to_obj = np.linalg.norm(obj - tcp)
    grasp = rewards.tolerance(tcp_to_obj,
                              bounds=(0, _TARGET_RADIUS),
                              margin=_TARGET_RADIUS,
                              sigmoid='long_tail')

    obj_to_target = np.linalg.norm(obj - target)
    in_place = rewards.tolerance(obj_to_target,
                                 bounds=(0, _TARGET_RADIUS),
                                 margin=_TARGET_RADIUS,
                                 sigmoid='long_tail')

    tcp_to_target = np.linalg.norm(tcp - target)
    hand_away = rewards.tolerance(tcp_to_target,
                                  bounds=(4*_TARGET_RADIUS, np.inf),
                                  margin=3*_TARGET_RADIUS,
                                  sigmoid='long_tail')
    in_place_weight = 10.
    grasp_or_hand_away = grasp * (1 - in_place) + hand_away * in_place
    return (
        grasp_or_hand_away + in_place_weight * in_place) / (1 + in_place_weight)


def _place(obs_settings, cradle_prop_name):
  """Configure and instantiate a Place task.

  Args:
    obs_settings: `observations.ObservationSettings` instance.
    cradle_prop_name: The name of the prop onto which the Duplo brick must be
      placed. Must be either 'duplo' or 'cradle'.

  Returns:
    An instance of `Place`.

  Raises:
    ValueError: If `prop_name` is neither 'duplo' nor 'cradle'.
  """
  arena = arenas.Standard()
  arm = robots.make_arm(obs_settings=obs_settings)
  hand = robots.make_hand(obs_settings=obs_settings)

  prop = props.Duplo(
      observable_options=observations.make_options(
          obs_settings, observations.FREEPROP_OBSERVABLES))
  if cradle_prop_name == 'duplo':
    cradle = props.Duplo()
  elif cradle_prop_name == 'cradle':
    cradle = SphereCradle()
  else:
    raise ValueError(
        '`cradle_prop_name` must be either \'duplo\' or \'cradle\'.')

  task = Place(arena=arena, arm=arm, hand=hand, prop=prop,
               obs_settings=obs_settings,
               workspace=_WORKSPACE,
               control_timestep=constants.CONTROL_TIMESTEP,
               cradle=cradle)
  return task


@registry.add(tags.FEATURES)
def place_brick_features():
  return _place(obs_settings=observations.PERFECT_FEATURES,
                cradle_prop_name='duplo')


@registry.add(tags.VISION)
def place_brick_vision():
  return _place(obs_settings=observations.VISION, cradle_prop_name='duplo')


@registry.add(tags.FEATURES)
def place_cradle_features():
  return _place(obs_settings=observations.PERFECT_FEATURES,
                cradle_prop_name='cradle')


@registry.add(tags.VISION)
def place_cradle_vision():
  return _place(obs_settings=observations.VISION, cradle_prop_name='cradle')
