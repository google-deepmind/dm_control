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
"""Escape locomotion tasks."""


from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable as base_observable
from dm_control.rl import control
from dm_control.utils import rewards
import numpy as np

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0


class Escape(composer.Task):
  """A task solved by escaping a starting area (e.g. bowl-shaped terrain)."""

  def __init__(self,
               walker,
               arena,
               walker_spawn_position=(0, 0, 0),
               walker_spawn_rotation=None,
               physics_timestep=0.005,
               control_timestep=0.025):
    """Initializes this task.

    Args:
      walker: an instance of `locomotion.walkers.base.Walker`.
      arena: an instance of `locomotion.arenas`.
      walker_spawn_position: a sequence of 3 numbers, or a `composer.Variation`
        instance that generates such sequences, specifying the position at
        which the walker is spawned at the beginning of an episode.
      walker_spawn_rotation: a number, or a `composer.Variation` instance that
        generates a number, specifying the yaw angle offset (in radians) that is
        applied to the walker at the beginning of an episode.
      physics_timestep: a number specifying the timestep (in seconds) of the
        physics simulation.
      control_timestep: a number specifying the timestep (in seconds) at which
        the agent applies its control inputs (in seconds).
    """

    self._arena = arena
    self._walker = walker
    self._walker.create_root_joints(self._arena.attach(self._walker))
    self._walker_spawn_position = walker_spawn_position
    self._walker_spawn_rotation = walker_spawn_rotation

    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.sensors_touch)
    enabled_observables.append(self._walker.observables.egocentric_camera)
    for observable in enabled_observables:
      observable.enabled = True

    if 'CMUHumanoid' in str(type(self._walker)):
      core_body = 'walker/root'
      self._reward_body = 'walker/root'
    elif 'Rat' in str(type(self._walker)):
      core_body = 'walker/torso'
      self._reward_body = 'walker/head'
    else:
      raise ValueError('Expects Rat or CMUHumanoid.')

    def _origin(physics):
      """Returns origin position in the torso frame."""
      torso_frame = physics.named.data.xmat[core_body].reshape(3, 3)
      torso_pos = physics.named.data.xpos[core_body]
      return -torso_pos.dot(torso_frame)

    self._walker.observables.add_observable(
        'origin', base_observable.Generic(_origin))

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

  @property
  def root_entity(self):
    return self._arena

  def initialize_episode_mjcf(self, random_state):
    if hasattr(self._arena, 'regenerate'):
      self._arena.regenerate(random_state)
    self._arena.mjcf_model.visual.map.znear = 0.00025
    self._arena.mjcf_model.visual.map.zfar = 50.

  def initialize_episode(self, physics, random_state):
    super(Escape, self).initialize_episode(physics, random_state)

    # Initial configuration.
    orientation = random_state.randn(4)
    orientation /= np.linalg.norm(orientation)
    _find_non_contacting_height(physics, self._walker, orientation)

  def get_reward(self, physics):
    # Escape reward term.
    terrain_size = physics.model.hfield_size[_HEIGHTFIELD_ID, 0]
    escape_reward = rewards.tolerance(
        np.asarray(np.linalg.norm(
            physics.named.data.site_xpos[self._reward_body])),
        bounds=(terrain_size, float('inf')),
        margin=terrain_size,
        value_at_margin=0,
        sigmoid='linear')
    upright_reward = _upright_reward(physics, self._walker, deviation_angle=30)
    return upright_reward * escape_reward

  def get_discount(self, physics):
    return 1.


def _find_non_contacting_height(physics, walker, orientation,
                                x_pos=0.0, y_pos=0.0, maxiter=1000):
  """Find a height with no contacts given a body orientation.

  Args:
    physics: An instance of `Physics`.
    walker: the focal walker.
    orientation: A quaternion.
    x_pos: A float. Position along global x-axis.
    y_pos: A float. Position along global y-axis.
    maxiter: maximum number of iterations to try
  """
  z_pos = 0.0  # Start embedded in the floor.
  num_contacts = 1
  count = 1
  # Move up in 1cm increments until no contacts.
  while num_contacts > 0:
    try:
      with physics.reset_context():
        freejoint = mjcf.get_frame_freejoint(walker.mjcf_model)
        physics.bind(freejoint).qpos[:3] = x_pos, y_pos, z_pos
        physics.bind(freejoint).qpos[3:] = orientation
    except control.PhysicsError:
      # We may encounter a PhysicsError here due to filling the contact
      # buffer, in which case we simply increment the height and continue.
      pass
    num_contacts = physics.data.ncon
    z_pos += 0.01
    count += 1
    if count > maxiter:
      raise ValueError(
          'maxiter reached: possibly contacts in null pose of body.'
      )


def _upright_reward(physics, walker, deviation_angle=0):
  """Returns a reward proportional to how upright the torso is.

  Args:
    physics: an instance of `Physics`.
    walker: the focal walker.
    deviation_angle: A float, in degrees. The reward is 0 when the torso is
      exactly upside-down and 1 when the torso's z-axis is less than
      `deviation_angle` away from the global z-axis.
  """
  deviation = np.cos(np.deg2rad(deviation_angle))
  upright_torso = physics.bind(walker.root_body).xmat[-1]
  if hasattr(walker, 'pelvis_body'):
    upright_pelvis = physics.bind(walker.pelvis_body).xmat[-1]
    upright_zz = np.stack([upright_torso, upright_pelvis])
  else:
    upright_zz = upright_torso
  upright = rewards.tolerance(upright_zz,
                              bounds=(deviation, float('inf')),
                              sigmoid='linear',
                              margin=1 + deviation,
                              value_at_margin=0)
  return np.min(upright)
