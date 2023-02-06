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

"""Tasks where the goal is to elevate a prop."""

import collections
import itertools

from dm_control import composer
from dm_control.composer import initializers
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


_LiftWorkspace = collections.namedtuple(
    '_LiftWorkspace', ['prop_bbox', 'tcp_bbox', 'arm_offset'])

_DUPLO_WORKSPACE = _LiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.0),
        upper=(0.1, 0.1, 0.0)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2),
        upper=(0.1, 0.1, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_BOX_SIZE = 0.09
_BOX_MASS = 1.3
_BOX_WORKSPACE = _LiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _BOX_SIZE),
        upper=(0.1, 0.1, _BOX_SIZE)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2),
        upper=(0.1, 0.1, 0.4)),
    arm_offset=robots.ARM_OFFSET)

_DISTANCE_TO_LIFT = 0.3


class _VertexSitesMixin:
  """Mixin class that adds sites corresponding to the vertices of a box."""

  def _add_vertex_sites(self, box_geom_or_site):
    """Add sites corresponding to the vertices of a box geom or site."""
    offsets = (
        (-half_length, half_length) for half_length in box_geom_or_site.size)
    site_positions = np.vstack(list(itertools.product(*offsets)))
    if box_geom_or_site.pos is not None:
      site_positions += box_geom_or_site.pos
    self._vertices = []
    for i, pos in enumerate(site_positions):
      site = box_geom_or_site.parent.add(
          'site', name='vertex_' + str(i), pos=pos, type='sphere', size=[0.002],
          rgba=constants.RED, group=constants.TASK_SITE_GROUP)
      self._vertices.append(site)

  @property
  def vertices(self):
    return self._vertices


class _BoxWithVertexSites(props.Primitive, _VertexSitesMixin):
  """Subclass of `Box` with sites marking the vertices of the box geom."""

  def _build(self, *args, **kwargs):
    super()._build(*args, geom_type='box', **kwargs)
    self._add_vertex_sites(self.geom)


class _DuploWithVertexSites(props.Duplo, _VertexSitesMixin):
  """Subclass of `Duplo` with sites marking the vertices of its sensor site."""

  def _build(self, *args, **kwargs):
    super()._build(*args, **kwargs)
    self._add_vertex_sites(self.mjcf_model.find('site', 'bounding_box'))


class Lift(composer.Task):
  """A task where the goal is to elevate a prop."""

  def __init__(
      self, arena, arm, hand, prop, obs_settings, workspace, control_timestep):
    """Initializes a new `Lift` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      prop: `composer.Entity` instance.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: `_LiftWorkspace` specifying the placement of the prop and TCP.
      control_timestep: Float specifying the control timestep in seconds.
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
    self._arena.add_free_entity(prop)
    self._prop_placer = initializers.PropPlacer(
        props=[prop],
        position=distributions.Uniform(*workspace.prop_bbox),
        quaternion=workspaces.uniform_z_rotation,
        ignore_collisions=True,
        settle_physics=True)

    # Add sites for visualizing bounding boxes and target height.
    self._target_height_site = workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=(-1, -1, 0), upper=(1, 1, 0),
        rgba=constants.RED, name='target_height')
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.tcp_bbox.lower, upper=workspace.tcp_bbox.upper,
        rgba=constants.GREEN, name='tcp_spawn_area')
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.prop_bbox.lower, upper=workspace.prop_bbox.upper,
        rgba=constants.BLUE, name='prop_spawn_area')

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

  def _get_height_of_lowest_vertex(self, physics):
    return min(physics.bind(self._prop.vertices).xpos[:, 2])

  def get_reward(self, physics):
    prop_height = self._get_height_of_lowest_vertex(physics)
    return rewards.tolerance(prop_height,
                             bounds=(self._target_height, np.inf),
                             margin=_DISTANCE_TO_LIFT,
                             value_at_margin=0,
                             sigmoid='linear')

  def initialize_episode(self, physics, random_state):
    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._prop_placer(physics, random_state)
    self._tcp_initializer(physics, random_state)
    # Compute the target height based on the initial height of the prop's
    # center of mass after settling.
    initial_prop_height = self._get_height_of_lowest_vertex(physics)
    self._target_height = _DISTANCE_TO_LIFT + initial_prop_height
    physics.bind(self._target_height_site).pos[2] = self._target_height


def _lift(obs_settings, prop_name):
  """Configure and instantiate a Lift task.

  Args:
    obs_settings: `observations.ObservationSettings` instance.
    prop_name: The name of the prop to be lifted. Must be either 'duplo' or
      'box'.

  Returns:
    An instance of `lift.Lift`.

  Raises:
    ValueError: If `prop_name` is neither 'duplo' nor 'box'.
  """
  arena = arenas.Standard()
  arm = robots.make_arm(obs_settings=obs_settings)
  hand = robots.make_hand(obs_settings=obs_settings)

  if prop_name == 'duplo':
    workspace = _DUPLO_WORKSPACE
    prop = _DuploWithVertexSites(
        observable_options=observations.make_options(
            obs_settings, observations.FREEPROP_OBSERVABLES))
  elif prop_name == 'box':
    workspace = _BOX_WORKSPACE
    # NB: The box is intentionally too large to pick up with a pinch grip.
    prop = _BoxWithVertexSites(
        size=[_BOX_SIZE] * 3,
        observable_options=observations.make_options(
            obs_settings, observations.FREEPROP_OBSERVABLES))
    prop.geom.mass = _BOX_MASS
  else:
    raise ValueError('`prop_name` must be either \'duplo\' or \'box\'.')
  task = Lift(arena=arena, arm=arm, hand=hand, prop=prop, workspace=workspace,
              obs_settings=obs_settings,
              control_timestep=constants.CONTROL_TIMESTEP)
  return task


@registry.add(tags.FEATURES)
def lift_brick_features():
  return _lift(obs_settings=observations.PERFECT_FEATURES, prop_name='duplo')


@registry.add(tags.VISION)
def lift_brick_vision():
  return _lift(obs_settings=observations.VISION, prop_name='duplo')


@registry.add(tags.FEATURES)
def lift_large_box_features():
  return _lift(obs_settings=observations.PERFECT_FEATURES, prop_name='box')


@registry.add(tags.VISION)
def lift_large_box_vision():
  return _lift(obs_settings=observations.VISION, prop_name='box')
