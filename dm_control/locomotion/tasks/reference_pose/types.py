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
"""Types for reference pose tasks.
"""

from typing import Optional, Sequence, Text, Union

import dataclasses
import numpy as np


@dataclasses.dataclass
class ClipCollection:
  """Dataclass representing a collection of mocap reference clips."""
  ids: Sequence[Text]
  start_steps: Optional[Sequence[int]] = None
  end_steps: Optional[Sequence[int]] = None
  weights: Optional[Sequence[Union[int, float]]] = None

  def __post_init__(self):
    # validate clips collection and set length dependent defaults.
    num_clips = len(self.ids)

    if self.start_steps is None:
      # by default start at the beginning
      self.start_steps = (0,) * num_clips
    else:
      assert len(self.start_steps) == num_clips

    # without access to the actual clip we cannot specify an end_steps default
    if self.end_steps is not None:
      assert len(self.end_steps) == num_clips

    if self.weights is None:
      self.weights = (1.0,) * num_clips
    else:
      assert len(self.weights) == num_clips
      assert np.all(np.array(self.weights) >= 0.)

