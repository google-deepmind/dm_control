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

"""Utility functions that operate on MJCF elements."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_freejoint(element):
  """Retrieves the free joint of a body. Returns `None` if there isn't one."""
  if element.tag != 'body':
    return None
  elif hasattr(element, 'freejoint') and element.freejoint is not None:
    return element.freejoint
  else:
    joints = element.find_all('joint', immediate_children_only=True)
    for joint in joints:
      if joint.type == 'free':
        return joint
    return None


def get_attachment_frame(mjcf_model):
  return mjcf_model.parent_model.find('attachment_frame', mjcf_model.model)


def get_frame_freejoint(mjcf_model):
  frame = get_attachment_frame(mjcf_model)
  return get_freejoint(frame)


def get_frame_joints(mjcf_model):
  """Retrieves all joints belonging to the attachment frame of an MJCF model."""
  frame = get_attachment_frame(mjcf_model)
  if frame:
    return frame.find_all('joint', immediate_children_only=True)
  else:
    return None
