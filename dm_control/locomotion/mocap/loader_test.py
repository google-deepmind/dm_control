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
"""Tests for loader."""

import os

from absl.testing import absltest
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.mocap import mocap_pb2
from dm_control.locomotion.mocap import trajectory

from google.protobuf import descriptor
from google.protobuf import text_format
from dm_control.utils import io as resources

TEXTPROTOS = [
    os.path.join(os.path.dirname(__file__), 'test_001.textproto'),
    os.path.join(os.path.dirname(__file__), 'test_002.textproto'),
]

HDF5 = os.path.join(os.path.dirname(__file__), 'test_trajectories.h5')


class HDF5TrajectoryLoaderTest(absltest.TestCase):

  def assert_proto_equal(self, x, y, msg=''):
    self.assertEqual(type(x), type(y), msg=msg)
    for field in x.DESCRIPTOR.fields:
      x_field = getattr(x, field.name)
      y_field = getattr(y, field.name)
      if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        if field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
          for i, (x_child, y_child) in enumerate(zip(x_field, y_field)):
            self.assert_proto_equal(
                x_child, y_child,
                msg=os.path.join(msg, '{}[{}]'.format(field.name, i)))
        else:
          self.assertEqual(list(x_field), list(y_field),
                           msg=os.path.join(msg, field.name))
      else:
        if field.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
          self.assert_proto_equal(
              x_field, y_field, msg=os.path.join(msg, field.name))
        else:
          self.assertEqual(x_field, y_field, msg=os.path.join(msg, field.name))

  def test_hdf5_agrees_with_textprotos(self):

    hdf5_loader = loader.HDF5TrajectoryLoader(
        resources.GetResourceFilename(HDF5))

    for textproto_path in TEXTPROTOS:
      trajectory_textproto = resources.GetResource(textproto_path)
      trajectory_from_textproto = mocap_pb2.FittedTrajectory()
      text_format.Parse(trajectory_textproto, trajectory_from_textproto)

      trajectory_identifier = (
          trajectory_from_textproto.identifier.encode('utf-8'))
      self.assert_proto_equal(
          hdf5_loader.get_trajectory(trajectory_identifier)._proto,
          trajectory.Trajectory(trajectory_from_textproto)._proto)


if __name__ == '__main__':
  absltest.main()
