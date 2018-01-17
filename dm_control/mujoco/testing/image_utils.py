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

"""Utilities for testing rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Internal dependencies.
from dm_control import mujoco
from dm_control.mujoco.testing import assets
import numpy as np
from PIL import Image
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin


class ImagesNotClose(AssertionError):
  """Exception raised when two images are not equal."""

  def __init__(self, message, expected, actual):
    super(ImagesNotClose, self).__init__(message)
    self.expected = expected
    self.actual = actual


class _FrameSequence(object):
  """A sequence of pre-rendered frames used in integration tests."""

  _ASSETS_DIR = 'assets'
  _FRAMES_DIR = 'frames'
  _SUBDIR_TEMPLATE = '{name}_seed_{seed}_camera_{camera_id}_{width}x{height}'
  _FILENAME_TEMPLATE = 'frame_{frame_num:03}.png'

  def __init__(self,
               name,
               xml_string,
               camera_id=-1,
               height=128,
               width=128,
               num_frames=20,
               steps_per_frame=10,
               seed=0):
    """Initializes a new `_FrameSequence`.

    Args:
      name: A string containing the name to be used for the sequence.
      xml_string: An MJCF XML string containing the model to be rendered.
      camera_id: An integer or string specifying the camera ID to render.
      height: The height of the rendered frames, in pixels.
      width: The width of the rendered frames, in pixels.
      num_frames: The number of frames to render.
      steps_per_frame: The interval between frames, in simulation steps.
      seed: Integer or None, used to initialize the random number generator for
        generating actions.
    """
    self._name = name
    self._xml_string = xml_string
    self._camera_id = camera_id
    self._height = height
    self._width = width
    self._num_frames = num_frames
    self._steps_per_frame = steps_per_frame
    self._seed = seed

  def iter_render(self):
    """Returns an iterator that yields newly rendered frames as numpy arrays."""
    random_state = np.random.RandomState(self._seed)
    physics = mujoco.Physics.from_xml_string(self._xml_string)
    action_spec = mujoco.action_spec(physics)
    for _ in xrange(self._num_frames):
      for _ in xrange(self._steps_per_frame):
        actions = random_state.uniform(action_spec.minimum, action_spec.maximum)
        physics.set_control(actions)
        physics.step()
      yield physics.render(height=self._height,
                           width=self._width,
                           camera_id=self._camera_id)

  def iter_load(self):
    """Returns an iterator that yields saved frames as numpy arrays."""
    for directory, filename in self._iter_paths():
      path = os.path.join(directory, filename)
      yield _load_pixels(path)

  def save(self):
    """Saves a new set of golden output frames to disk."""
    for pixels, (relative_to_assets, filename) in zip(self.iter_render(),
                                                      self._iter_paths()):
      full_directory_path = os.path.join(self._ASSETS_DIR, relative_to_assets)
      if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)
      path = os.path.join(full_directory_path, filename)
      _save_pixels(pixels, path)

  def _iter_paths(self):
    subdir_name = self._SUBDIR_TEMPLATE.format(name=self._name,
                                               camera_id=self._camera_id,
                                               width=self._width,
                                               height=self._height,
                                               seed=self._seed)
    directory = os.path.join(self._FRAMES_DIR, subdir_name)
    for frame_num in xrange(self._num_frames):
      filename = self._FILENAME_TEMPLATE.format(frame_num=frame_num)
      yield directory, filename


cartpole = _FrameSequence('cartpole', assets.get_contents('cartpole.xml'),
                          width=320, height=240, camera_id=0,
                          steps_per_frame=5)

humanoid = _FrameSequence('humanoid', assets.get_contents('humanoid.xml'),
                          width=128, height=128, camera_id=-1)


SEQUENCES = {
    'cartpole': cartpole,
    'humanoid': humanoid,
}


def _save_pixels(pixels, path):
  image = Image.fromarray(pixels)
  image.save(path)


def _load_pixels(path):
  image_bytes = assets.get_contents(path)
  image = Image.open(six.BytesIO(image_bytes))
  return np.array(image)


def compute_rms(image1, image2):
  """Computes the RMS difference between two images."""
  abs_diff = np.abs(image1.astype(np.int16) - image2)
  values, counts = np.unique(abs_diff, return_counts=True)
  sum_of_squares = np.sum(counts * values.astype(np.int64) ** 2)
  return np.sqrt(float(sum_of_squares) / abs_diff.size)


def assert_images_close(expected, actual, tolerance=10.):
  """Tests whether two images are almost equal.

  Args:
    expected: A numpy array, the expected image.
    actual: A numpy array, the actual image.
    tolerance: A float specifying the maximum allowable RMS error between the
      expected and actual images.

  Raises:
    ImagesNotClose: If the images are not sufficiently close.
  """
  rms = compute_rms(expected, actual)
  if rms > tolerance:
    message = 'RMS error exceeds tolerance ({} > {})'.format(rms, tolerance)
    raise ImagesNotClose(message, expected=expected, actual=actual)


def save_images_on_failure(output_dir):
  """Decorator that saves debugging images if `ImagesNotClose` is raised.

  Args:
    output_dir: Path to the directory where the output images will be saved.

  Returns:
    A decorator function.
  """
  def decorator(test_method):
    """Decorator that saves debugging images if `ImagesNotClose` is raised."""
    method_name = test_method.__name__
    @functools.wraps(test_method)
    def decorated_method(*args, **kwargs):
      try:
        test_method(*args, **kwargs)
      except ImagesNotClose as e:
        difference = e.actual.astype(np.double) - e.expected
        difference = (0.5 * (difference + 255)).astype(np.uint8)
        _save_pixels(e.expected,
                     os.path.join(output_dir,
                                  '{}-expected.png'.format(method_name)))
        _save_pixels(e.actual,
                     os.path.join(output_dir,
                                  '{}-actual.png'.format(method_name)))
        _save_pixels(difference,
                     os.path.join(output_dir,
                                  '{}-difference.png'.format(method_name)))
        raise  # Reraise the exception with the original traceback.
    return decorated_method
  return decorator
