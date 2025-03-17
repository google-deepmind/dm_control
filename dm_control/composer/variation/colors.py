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

"""Variations in colors.

Classes in this module allow users to specify a variations for each channel in
a variety of color spaces. The generated values are always RGBA arrays.
"""

import colorsys

from dm_control.composer.variation import base
from dm_control.composer.variation import variation_values
import numpy as np


class RgbVariation(base.Variation):
  """Represents a variation in the RGB color space.

  This class allows users to specify independent variations in the R, G, B, and
  alpha channels of a color, and generates the corresponding array of RGBA
  values.
  """

  def __init__(self, r, g, b, alpha=1.0):
    self._r, self._g, self._b = r, g, b
    self._alpha = alpha

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    return np.asarray(
        variation_values.evaluate([self._r, self._g, self._b, self._alpha],
                                  initial_value, current_value, random_state))

  def __eq__(self, other):
    if not isinstance(other, RgbVariation):
      return False
    return (
        self._r == other._r
        and self._g == other._g
        and self._b == other._b
        and self._alpha == other._alpha
    )


class HsvVariation(base.Variation):
  """Represents a variation in the HSV color space.

  This class allows users to specify independent variations in the H, S, V, and
  alpha channels of a color, and generates the corresponding array of RGBA
  values.
  """

  def __init__(self, h, s, v, alpha=1.0):
    self._h, self._s, self._v = h, s, v
    self._alpha = alpha

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    h, s, v, alpha = variation_values.evaluate(
        (self._h, self._s, self._v, self._alpha), initial_value, current_value,
        random_state)
    return np.asarray(list(colorsys.hsv_to_rgb(h, s, v)) + [alpha])

  def __eq__(self, other):
    if not isinstance(other, HsvVariation):
      return False
    return (
        self._h == other._h
        and self._s == other._s
        and self._v == other._v
        and self._alpha == other._alpha
    )

  def __repr__(self):
    return (
        f"HsvVariation(h={self._h}, s={self._s}, v={self._v}, "
        f"alpha={self._alpha})"
    )


class GrayVariation(HsvVariation):
  """Represents a variation in gray level.

  This class allows users to specify independent variations in the gray level
  and alpha channels of a color, and generates the corresponding array of RGBA
  values.
  """

  def __init__(self, gray_level, alpha=1.0):
    super().__init__(h=0.0, s=0.0, v=gray_level, alpha=alpha)

  def __repr__(self):
    return (
        f"GrayVariation(gray_level={self._v}, alpha={self._alpha})"
    )
