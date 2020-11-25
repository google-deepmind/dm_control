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

_ACTUATOR_TAGS = ('general', 'motor', 'position',
                  'velocity', 'cylinder', 'muscle')


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


def commit_defaults(element, attributes=None):
  """Commits default values into attributes of the specified element.

  Args:
    element: A PyMJCF element.
    attributes: (optional) A list of strings specifying the attributes to be
      copied from defaults, or `None` if all attributes should be copied.
  """
  dclass = element.dclass
  parent = element.parent
  while dclass is None and parent != element.root:
    dclass = getattr(parent, 'childclass', None)
    parent = parent.parent
  if dclass is None:
    dclass = element.root.default

  while dclass != element.root:
    if element.tag in _ACTUATOR_TAGS:
      tags = _ACTUATOR_TAGS
    else:
      tags = (element.tag,)
    for tag in tags:
      default_element = getattr(dclass, tag)
      for name, value in default_element.get_attributes().items():
        if attributes is None or name in attributes:
          if hasattr(element, name) and getattr(element, name) is None:
            setattr(element, name, value)
    dclass = dclass.parent
