# Copyright 2018 The dm_control Authors.
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

"""A fake Physics class for unit testing observation framework."""

import contextlib

from dm_control.composer.observation import observable
from dm_control.rl import control
import numpy as np


class FakePhysics(control.Physics):
  """A fake Physics class for unit testing observation framework."""

  def __init__(self):
    self._step_counter = 0
    self._observables = {
        'twice': observable.Generic(FakePhysics.twice),
        'repeated': observable.Generic(FakePhysics.repeated, update_interval=5),
        'matrix': observable.Generic(FakePhysics.matrix, update_interval=3)
    }

  def step(self, sub_steps=1):
    self._step_counter += 1

  @property
  def observables(self):
    return self._observables

  def twice(self):
    return 2*self._step_counter

  def repeated(self):
    return [self._step_counter, self._step_counter]

  def sqrt(self):
    return np.sqrt(self._step_counter)

  def sqrt_plus_one(self):
    return np.sqrt(self._step_counter) + 1

  def matrix(self):
    return [[self._step_counter] * 3] * 2

  def time(self):
    return self._step_counter

  def timestep(self):
    return 1.0

  def set_control(self, ctrl):
    pass

  def reset(self):
    self._step_counter = 0

  def after_reset(self):
    pass

  @contextlib.contextmanager
  def suppress_physics_errors(self):
    yield
