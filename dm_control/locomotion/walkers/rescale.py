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
"""Function to rescale the walkers."""


from dm_control import mjcf


def rescale_subtree(body, position_factor, size_factor):
  """Recursively rescales an entire subtree of an MJCF model."""
  for child in body.all_children():
    if getattr(child, 'fromto', None) is not None:
      new_pos = position_factor * 0.5 * (child.fromto[3:] + child.fromto[:3])
      new_size = size_factor * 0.5 * (child.fromto[3:] - child.fromto[:3])
      child.fromto[:3] = new_pos - new_size
      child.fromto[3:] = new_pos + new_size
    if getattr(child, 'pos', None) is not None:
      child.pos *= position_factor
    if getattr(child, 'size', None) is not None:
      child.size *= size_factor
    if child.tag == 'body' or child.tag == 'worldbody':
      rescale_subtree(child, position_factor, size_factor)


def rescale_humanoid(walker, position_factor, size_factor=None, mass=None):
  """Rescales a humanoid walker's lengths, sizes, and masses."""
  body = walker.mjcf_model.find('body', 'root')
  subtree_root = body.parent
  if size_factor is None:
    size_factor = position_factor
  rescale_subtree(subtree_root, position_factor, size_factor)

  if mass is not None:
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model.root_model)
    current_mass = physics.bind(walker.root_body).subtreemass
    mass_factor = mass / current_mass
    for body in walker.root_body.find_all('body'):
      inertial = getattr(body, 'inertial', None)
      if inertial:
        inertial.mass *= mass_factor
    for geom in walker.root_body.find_all('geom'):
      if geom.mass is not None:
        geom.mass *= mass_factor
      else:
        current_density = geom.density if geom.density is not None else 1000
        geom.density = current_density * mass_factor
