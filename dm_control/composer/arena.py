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

  def _build(self, name=None):
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    self._mjcf_root = mjcf.from_path(_ARENA_XML_PATH)
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
