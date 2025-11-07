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

"""Utilities used in tests, and for tuning the Duplo model."""


from dm_control import composer
from dm_control import mjcf
from scipy import optimize


def stack_bricks(top_brick, bottom_brick):
  """Stacks two Duplo bricks, returns the attachment frame of the top brick."""
  arena = composer.Arena()
  # Bottom brick is fixed in place, top brick has a freejoint.
  arena.attach(bottom_brick)
  attachment_frame = arena.add_free_entity(top_brick)
  # Attachment frame is positioned such that the top brick is on top of the
  # bottom brick.
  attachment_frame.pos = (0, 0, 0.0192)
  return arena, attachment_frame


def measure_separation_force(top_brick,
                             bottom_brick,
                             min_force=0.,
                             max_force=20.,
                             tolerance=0.01,
                             time_limit=0.5,
                             height_threshold=1e-3):
  """Utility for measuring the separation force for a pair of Duplo bricks.

  Args:
    top_brick: An instance of `Duplo` representing the top brick.
    bottom_brick: An instance of `Duplo` representing the bottom brick.
    min_force: A force that should be insufficient to separate the bricks (N).
    max_force: A force that should be sufficient to separate the bricks (N).
    tolerance: The desired precision of the solution (N).
    time_limit: The maximum simulation time (s) over which to apply force on
      each iteration. Increasing this value will result in smaller estimates
      of the separation force, since given sufficient time the bricks may slip
      apart gradually under a smaller force. This is due to MuJoCo's soft
      contact model (see http://mujoco.org/book/index.html#Soft).
    height_threshold: The distance (m) that the upper brick must move in the
      z-axis for the bricks to count as separated.

  Returns:
    A float, the measured separation force (N).
  """
  arena, attachment_frame = stack_bricks(top_brick, bottom_brick)
  physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
  bound_attachment_frame = physics.bind(attachment_frame)

  def func(force):
    """Returns +1 if the bricks separate under this force, and -1 otherwise."""
    with physics.model.disable('gravity'):
      # Reset the simulation.
      physics.reset()
      # Get the initial height.
      initial_height = bound_attachment_frame.xpos[2]
      # Apply an upward force to the attachment frame.
      bound_attachment_frame.xfrc_applied[2] = force
      # Advance the simulation until either the height threshold or time limit
      # is reached.
      while physics.time() < time_limit:
        physics.step()
        distance_lifted = bound_attachment_frame.xpos[2] - initial_height
        if distance_lifted > height_threshold:
          return 1.0
    return -1.0

  # Ensure that the min and max forces bracket the true separation force.
  while func(min_force) > 0:
    min_force *= 0.5
  while func(max_force) < 0:
    max_force *= 2

  return optimize.bisect(func, a=min_force, b=max_force, xtol=tolerance,
                         disp=True)
