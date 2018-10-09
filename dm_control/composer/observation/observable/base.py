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

"""Classes representing observables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools

import numpy as np
import six

from dm_control.rl import specs


AGGREGATORS = {
    'min': functools.partial(np.min, axis=0),
    'max': functools.partial(np.max, axis=0),
    'mean': functools.partial(np.mean, axis=0),
    'median': functools.partial(np.median, axis=0),
    'sum': functools.partial(np.sum, axis=0),
}


def _get_aggregator(name_or_callable):
  """Returns aggregator from predefined set by name, else returns callable."""
  if name_or_callable is None:
    return None
  elif not callable(name_or_callable):
    try:
      return AGGREGATORS[name_or_callable]
    except KeyError:
      raise KeyError('Unrecognized aggregator name: {!r}. Valid names: {}.'
                     .format(name_or_callable, AGGREGATORS.keys()))
  else:
    return name_or_callable


@six.add_metaclass(abc.ABCMeta)
class Observable(object):
  """Abstract base class for an observable."""

  def __init__(self, update_interval, buffer_size, delay,
               aggregator, corruptor):
    self._update_interval = update_interval
    self._buffer_size = buffer_size
    self._delay = delay
    self._aggregator = _get_aggregator(aggregator)
    self._corruptor = corruptor
    self._enabled = False

  @property
  def update_interval(self):
    return self._update_interval

  @update_interval.setter
  def update_interval(self, value):
    self._update_interval = value

  @property
  def buffer_size(self):
    return self._buffer_size

  @buffer_size.setter
  def buffer_size(self, value):
    self._buffer_size = value

  @property
  def delay(self):
    return self._delay

  @delay.setter
  def delay(self, value):
    self._delay = value

  @property
  def aggregator(self):
    return self._aggregator

  @aggregator.setter
  def aggregator(self, value):
    self._aggregator = _get_aggregator(value)

  @property
  def corruptor(self):
    return self._corruptor

  @corruptor.setter
  def corruptor(self, value):
    self._corruptor = value

  @property
  def enabled(self):
    return self._enabled

  @enabled.setter
  def enabled(self, value):
    self._enabled = value

  @property
  def array_spec(self):
    """The `ArraySpec` which describes observation arrays from this observable.

      If this property is `None`, then the specification should be inferred by
      actually retrieving an observation from this observable.
    """
    return None

  @abc.abstractmethod
  def _callable(self, physics):
    pass

  def observation_callable(self, physics, random_state=None):
    """A callable which returns a (potentially corrupted) observation."""
    if self._corruptor:
      def _corrupted():
        return self._corruptor(self._callable(physics)(),
                               random_state=random_state)
      return _corrupted
    else:
      return self._callable(physics)

  def __call__(self, physics, random_state=None):
    """Convenience function to just call an observable."""
    return self.observation_callable(physics, random_state)()

  def configure(self, **kwargs):
    """Sets multiple attributes of this observable.

    Args:
      **kwargs: The keyword argument names correspond to the attributes
        being modified.
    Raises:
      AttributeError: If kwargs contained an attribute not in the observable.
    """
    for key, value in six.iteritems(kwargs):
      if not hasattr(self, key):
        raise AttributeError('Cannot add attribute %s in configure.' % key)
      self.__setattr__(key, value)


class Generic(Observable):
  """A generic observable defined via a callable."""

  def __init__(self, raw_observation_callable, update_interval=1,
               buffer_size=None, delay=None,
               aggregator=None, corruptor=None):
    """Initializes this observable.

    Args:
      raw_observation_callable: A callable which accepts a single argument of
        type `control.base.Physics` and returns the observation value.
      update_interval: (optional) An integer, number of simulation steps between
        successive updates to the value of this observable.
      buffer_size: (optional) The maximum size of the returned buffer.
        This option is only relevant when used in conjunction with an
        `observation.Updater`. If None, `observation.DEFAULT_BUFFER_SIZE` will
        be used.
      delay: (optional) Number of additional simulation steps that must be
        taken before an observation is returned. This option is only relevant
        when used in conjunction with an`observation.Updater`. If None,
        `observation.DEFAULT_DELAY` will be used.
      aggregator: (optional) Name of an item in `AGGREGATORS` or a callable that
        performs a reduction operation over the first dimension of the buffered
        observation before it is returned. A value of `None` means that no
        aggregation will be performed and the whole buffer will be returned.
      corruptor: (optional) A callable which takes a single observation as
        an argument, modifies it, and returns it. An example use case for this
        is to add random noise to the observation. When used in a
        `BufferedWrapper`, the corruptor is applied to the observation before
        it is added to the buffer. In particular, this means that the aggregator
        operates on corrupted observations.
    """
    self._raw_callable = raw_observation_callable
    super(Generic, self).__init__(
        update_interval, buffer_size, delay, aggregator, corruptor)

  def _callable(self, physics):
    return lambda: self._raw_callable(physics)


class MujocoFeature(Observable):
  """An observable corresponding to a named MuJoCo feature."""

  def __init__(self, kind, feature_name, update_interval=1,
               buffer_size=None, delay=None,
               aggregator=None, corruptor=None):
    """Initializes this observable.

    Args:
      kind: A string corresponding to a field name in MuJoCo's mjData struct.
      feature_name: A string, or list of strings, or a callable returning
        either, corresponding to the name(s) of an entity in the
        MuJoCo XML model.
      update_interval: (optional) An integer, number of simulation steps between
        successive updates to the value of this observable.
      buffer_size: (optional) The maximum size of the returned buffer.
        This option is only relevant when used in conjunction with an
        `observation.Updater`. If None, `observation.DEFAULT_BUFFER_SIZE` will
        be used.
      delay: (optional) Number of additional simulation steps that must be
        taken before an observation is returned. This option is only relevant
        when used in conjunction with an`observation.Updater`. If None,
        `observation.DEFAULT_DELAY` will be used.
      aggregator: (optional) Name of an item in `AGGREGATORS` or a callable that
        performs a reduction operation over the first dimension of the buffered
        observation before it is returned. A value of `None` means that no
        aggregation will be performed and the whole buffer will be returned.
      corruptor: (optional) A callable which takes a single observation as
        an argument, modifies it, and returns it. An example use case for this
        is to add random noise to the observation. When used in a
        `BufferedWrapper`, the corruptor is applied to the observation before
        it is added to the buffer. In particular, this means that the aggregator
        operates on corrupted observations.
    """
    self._kind = kind
    self._feature_name = feature_name
    super(MujocoFeature, self).__init__(
        update_interval, buffer_size, delay, aggregator, corruptor)

  def _callable(self, physics):
    named_indexer_for_kind = physics.named.data.__getattribute__(self._kind)
    if callable(self._feature_name):
      return lambda: named_indexer_for_kind[self._feature_name()]
    else:
      return lambda: named_indexer_for_kind[self._feature_name]


class MujocoCamera(Observable):
  """An observable corresponding to a MuJoCo camera."""

  def __init__(self, camera_name, height=240, width=320, update_interval=1,
               buffer_size=None, delay=None,
               aggregator=None, corruptor=None, depth=False):
    """Initializes this observable.

    Args:
      camera_name: A string corresponding to the name of a camera in the
        MuJoCo XML model.
      height: (optional) An integer, the height of the rendered image.
      width: (optional) An integer, the width of the rendered image.
      update_interval: (optional) An integer, number of simulation steps between
        successive updates to the value of this observable.
      buffer_size: (optional) The maximum size of the returned buffer.
        This option is only relevant when used in conjunction with an
        `observation.Updater`. If None, `observation.DEFAULT_BUFFER_SIZE` will
        be used.
      delay: (optional) Number of additional simulation steps that must be
        taken before an observation is returned. This option is only relevant
        when used in conjunction with an`observation.Updater`. If None,
        `observation.DEFAULT_DELAY` will be used.
      aggregator: (optional) Name of an item in `AGGREGATORS` or a callable that
        performs a reduction operation over the first dimension of the buffered
        observation before it is returned. A value of `None` means that no
        aggregation will be performed and the whole buffer will be returned.
      corruptor: (optional) A callable which takes a single observation as
        an argument, modifies it, and returns it. An example use case for this
        is to add random noise to the observation. When used in a
        `BufferedWrapper`, the corruptor is applied to the observation before
        it is added to the buffer. In particular, this means that the aggregator
        operates on corrupted observations.
      depth: (optional) A boolean. If `True`, renders a depth image (1-channel)
        instead of RGB (3-channel).
    """
    self._camera_name = camera_name
    self._height = height
    self._width = width

    self._n_channels = 1 if depth else 3
    self._dtype = np.float32 if depth else np.uint8
    self._depth = depth
    super(MujocoCamera, self).__init__(
        update_interval, buffer_size, delay, aggregator, corruptor)

  @property
  def height(self):
    return self._height

  @height.setter
  def height(self, value):
    self._height = value

  @property
  def width(self):
    return self._width

  @width.setter
  def width(self, value):
    self._width = value

  @property
  def array_spec(self):
    return specs.ArraySpec(
        shape=(self._height, self._width, self._n_channels), dtype=self._dtype)

  def _callable(self, physics):
    return lambda: physics.render(  # pylint: disable=g-long-lambda
        self._height, self._width, self._camera_name, depth=self._depth)
