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

"""Script for tuning Duplo stud sizes to give desired separation forces."""

import collections
import pprint
from absl import app
from absl import logging
from dm_control.entities.props import duplo
from dm_control.entities.props.duplo import utils
from scipy import optimize

# pylint: disable=protected-access,invalid-name
_StudSize = duplo._StudSize
ORIGINAL_STUD_SIZE_PARAMS = duplo._STUD_SIZE_PARAMS
# pylint: enable=protected-access,invalid-name

DESIRED_FORCES = _StudSize(minimum=6., lower_quartile=10., maximum=18.)

# The safety margin here is because the separation force isn't quite monotonic
# w.r.t. the stud radius. If we set the min and max radii according to the
# exact desired bounds on the separation force then we may occasionally sample
# stud radii that yield out-of-bounds forces.
SAFETY_MARGIN = 0.2


def get_separation_force_for_radius(radius, **duplo_kwargs):
  """Measures Duplo separation force as a function of stud radius."""

  top_brick = duplo.Duplo(**duplo_kwargs)
  bottom_brick = duplo.Duplo(**duplo_kwargs)

  # Set the radius of the studs on the bottom brick (this would normally be done
  # in `initialize_episode_mjcf`). Note: we also set the radius of the studs on
  # the top brick, since this has a (tiny!) effect on its mass.

  # pylint: disable=protected-access
  top_brick._active_stud_dclass.geom.size[0] = radius
  bottom_brick._active_stud_dclass.geom.size[0] = radius
  # pylint: enable=protected-access

  separation_force = utils.measure_separation_force(top_brick, bottom_brick)
  logging.debug('Stud radius: %f\tseparation force: %f N',
                radius, separation_force)
  return separation_force


class _KeepBracketingSolutions(object):
  """Wraps objective func, keeps closest solutions bracketing the target."""

  _solution = collections.namedtuple('_solution', ['x', 'residual'])

  def __init__(self, func):
    self._func = func
    self.below = self._solution(x=None, residual=-float('inf'))
    self.above = self._solution(x=None, residual=float('inf'))

  def __call__(self, x):
    residual = self._func(x)
    if self.below.residual < residual <= 0:
      self.below = self._solution(x=x, residual=residual)
    elif 0 < residual < self.above.residual:
      self.above = self._solution(x=x, residual=residual)
    return residual

  @property
  def closest(self):
    if abs(self.below.residual) < self.above.residual:
      return self.below
    else:
      return self.above


def tune_stud_radius(desired_force,
                     min_radius=0.0045,
                     max_radius=0.005,
                     desired_places=6,
                     side='closest',
                     **duplo_kwargs):
  """Find a stud size that gives the desired separation force."""

  @_KeepBracketingSolutions
  def func(radius):
    radius = round(radius, desired_places)  # Round radius for aesthetics (!)
    return (get_separation_force_for_radius(radius=radius, **duplo_kwargs)
            - desired_force)

  # Ensure that the min and max radii bracket the solution.
  while func(min_radius) > 0:
    min_radius = max(1e-3, min_radius - (max_radius - min_radius))
  while func(max_radius) < 0:
    max_radius += (max_radius - min_radius)

  tolerance = 10**-(desired_places)

  # Use bisection to refine the bounds on the optimal radius. Note: this assumes
  # that separation force is monotonic w.r.t. stud radius, but this isn't
  # exactly true in all cases.
  optimize.bisect(func, a=min_radius, b=max_radius, xtol=tolerance, disp=True)

  if side == 'below':
    solution = func.below
  elif side == 'above':
    solution = func.above
  else:
    solution = func.closest

  radius = round(solution.x, desired_places)
  force = get_separation_force_for_radius(radius, **duplo_kwargs)

  return radius, force


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tuned_stud_radii = {}
  tuned_separation_forces = {}

  for stud_params in sorted(ORIGINAL_STUD_SIZE_PARAMS):
    duplo_kwargs = stud_params._asdict()

    min_result = tune_stud_radius(
        desired_force=DESIRED_FORCES.minimum + SAFETY_MARGIN,
        variation=0.0, side='above', **duplo_kwargs)
    lq_result = tune_stud_radius(
        desired_force=DESIRED_FORCES.lower_quartile,
        variation=0.0, side='closest', **duplo_kwargs)
    max_result = tune_stud_radius(
        desired_force=DESIRED_FORCES.maximum - SAFETY_MARGIN,
        variation=0.0, side='below', **duplo_kwargs)

    radii, forces = zip(*(min_result, lq_result, max_result))

    logging.info('\nDuplo configuration: %s\nTuned radii: %s, forces: %s',
                 stud_params, radii, forces)
    tuned_stud_radii[stud_params] = _StudSize(*radii)
    tuned_separation_forces[stud_params] = _StudSize(*forces)

  logging.info('%s\nNew Duplo parameters:\n%s\nSeparation forces:\n%s',
               '-'*60,
               pprint.pformat(tuned_stud_radii),
               pprint.pformat(tuned_separation_forces))

if __name__ == '__main__':
  app.run(main)
