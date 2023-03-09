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

"""The base empty arena that defines global settings for Composer."""

import os

from dm_control import mjcf
from dm_control.composer import entity as entity_module

_ARENA_XML_PATH = os.path.join(os.path.dirname(__file__), 'arena.xml')


class Arena(entity_module.Entity):
  """The base empty arena that defines global settings for Composer."""

  def __init__(self, *args, **kwargs):
    self._mjcf_root = None  # Declare that _mjcf_root exists to allay pytype.
    super().__init__(*args, **kwargs)

  # _build uses *args and **kwargs rather than named arguments, to get
  # around a signature-mismatch error from pytype in derived classes.

  def _build(self, *args, **kwargs) -> None:
    """Initializes this arena.

    The function takes two arguments through args, kwargs:
      name: A string, the name of this arena. If `None`, use the model name
        defined in the MJCF file.
      xml_path: An optional path to an XML file that will override the default
        composer arena MJCF.

    Args:
      *args: See above.
      **kwargs: See above.
    """
    if args:
      name = args[0]
    else:
      name = kwargs.get('name', None)
    if len(args) > 1:
      xml_path = args[1]
    else:
      xml_path = kwargs.get('xml_path', None)

    self._mjcf_root = mjcf.from_path(xml_path or _ARENA_XML_PATH)
    if name:
      self._mjcf_root.model = name

  def add_free_entity(self, entity):
    """Includes an entity in the arena as a free-moving body."""
    frame = self.attach(entity)
    frame.add('freejoint')
    return frame

  @property
  def mjcf_model(self):
    return self._mjcf_root
