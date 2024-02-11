# Copyright 2023 The dm_control Authors.
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

"""Fake Blender bpy module."""

from __future__ import annotations  # postponed evaluation of annotations

from typing import Any, Collection, Sequence

from dm_control.blender.fake_core import mathutils

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring


class WindowManager:

  def progress_begin(self, start: int, end: int):
    pass

  def progress_update(self, steps_done: int):
    pass

  def progress_end(self):
    pass


class context:

  @property
  def window_manager(self) -> WindowManager:
    return WindowManager()

  @staticmethod
  def evaluated_depsgraph_get():
    pass


class types:

  class Constraint:

    @property
    def name(self):
      pass

    @property
    def owner_space(self):
      pass

  class Scene:
    pass

  class Object:

    @property
    def name(self) -> str:
      raise NotImplementedError()

    @property
    def parent(self) -> types.Object | None:
      pass

    @property
    def parent_bone(self) -> types.Bone | None:
      pass

    @property
    def data(self):
      pass

    @property
    def pose(self):
      pass

    @property
    def matrix_world(self) -> mathutils.Matrix:
      raise NotImplementedError()

    @matrix_world.setter
    def matrix_world(self, _) -> mathutils.Matrix:
      raise NotImplementedError()

    def select_set(self, _):
      pass

    def to_mesh(self):
      pass

    def evaluated_get(self, _) -> types.Object:
      pass

    @property
    def mode(self) -> str:
      return 'OBJECT'

    @property
    def type(self):
      pass

    def update_from_editmode(self):
      pass

  class Bone:

    @property
    def name(self) -> str:
      raise NotImplementedError()

    @property
    def parent(self) -> types.Bone | None:
      pass

    @property
    def matrix_local(self) -> mathutils.Matrix:
      raise NotImplementedError()

    @property
    def matrix(self) -> mathutils.Matrix:
      raise NotImplementedError()

  class bpy_struct:
    pass

  class Context:

    @property
    def scene(self) -> types.Scene:
      pass

  class Light:

    @property
    def type(self):
      pass

    @property
    def use_shadow(self):
      pass

    @property
    def color(self) -> mathutils.Color:
      raise NotImplementedError()

    @property
    def linear_attenuation(self):
      pass

    @property
    def quadratic_attenuation(self):
      pass

  class LimitRotationConstraint(Constraint):
    pass

  class LimitLocationConstraint(Constraint):
    pass

  class Material:

    @property
    def name(self) -> str:
      raise NotImplementedError()

    @property
    def specular_intensity(self):
      pass

    @property
    def metallic(self):
      pass

    @property
    def roughness(self) -> float:
      raise NotImplementedError()

    @property
    def diffuse_color(self) -> Sequence[float]:
      raise NotImplementedError()

  class Mesh:

    @property
    def name(self) -> str:
      raise NotImplementedError()

    def calc_loop_triangles(self):
      pass

    @property
    def uv_layers(self) -> Any:
      raise NotImplementedError()

    @property
    def loop_triangles(self) -> Collection[Any]:
      raise NotImplementedError()

    @property
    def vertices(self) -> Any:
      raise NotImplementedError()


class ops:

  class object:

    @staticmethod
    def select_all(action):
      pass

  class export_mesh:

    @classmethod
    def stl(
        cls, filepath, use_selection, use_mesh_modifiers, axis_forward, axis_up
    ):
      pass
