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

"""Tests for mjcf observables."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer.observation.observable import mjcf as mjcf_observable
from dm_env import specs
import numpy as np

_MJCF = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <body name="body" pos="0 0 0">
      <joint name="my_hinge" type="hinge" pos="-.1 -.2 -.3" axis="1 -1 0"/>
      <geom name="my_box" type="box" size=".1 .2 .3" rgba="0 0 1 1"/>
      <geom name="small_sphere" type="sphere" size=".12" pos=".1 .2 .3"/>
    </body>
    <camera name="world" mode="targetbody" target="body" pos="1 1 1" />
  </worldbody>
</mujoco>
"""


class ObservableTest(parameterized.TestCase):

  def testMJCFFeature(self):
    mjcf_root = mjcf.from_xml_string(_MJCF)
    physics = mjcf.Physics.from_mjcf_model(mjcf_root)

    my_hinge = mjcf_root.find('joint', 'my_hinge')
    hinge_observable = mjcf_observable.MJCFFeature(
        kind='qpos', mjcf_element=my_hinge)
    hinge_observation = hinge_observable.observation_callable(physics)()
    np.testing.assert_array_equal(
        hinge_observation, physics.named.data.qpos[my_hinge.full_identifier])

    small_sphere = mjcf_root.find('geom', 'small_sphere')
    sphere_observable = mjcf_observable.MJCFFeature(
        kind='xpos', mjcf_element=small_sphere, update_interval=5)
    sphere_observation = sphere_observable.observation_callable(physics)()
    self.assertEqual(sphere_observable.update_interval, 5)
    np.testing.assert_array_equal(
        sphere_observation, physics.named.data.geom_xpos[
            small_sphere.full_identifier])

    my_box = mjcf_root.find('geom', 'my_box')
    list_observable = mjcf_observable.MJCFFeature(
        kind='xpos', mjcf_element=[my_box, small_sphere])
    list_observation = (
        list_observable.observation_callable(physics)())
    np.testing.assert_array_equal(
        list_observation,
        physics.named.data.geom_xpos[[my_box.full_identifier,
                                      small_sphere.full_identifier]])

    with self.assertRaisesRegex(ValueError, 'expected an `mjcf.Element`'):
      mjcf_observable.MJCFFeature('qpos', 'my_hinge')
    with self.assertRaisesRegex(ValueError, 'expected an `mjcf.Element`'):
      mjcf_observable.MJCFFeature('geom_xpos', [my_box, 'small_sphere'])

  def testMJCFFeatureIndex(self):
    mjcf_root = mjcf.from_xml_string(_MJCF)
    physics = mjcf.Physics.from_mjcf_model(mjcf_root)

    small_sphere = mjcf_root.find('geom', 'small_sphere')
    sphere_xmat = np.array(
        physics.named.data.geom_xmat[small_sphere.full_identifier])

    observable_xrow = mjcf_observable.MJCFFeature(
        'xmat', small_sphere, index=[1, 3, 5, 7])
    np.testing.assert_array_equal(
        observable_xrow.observation_callable(physics)(),
        sphere_xmat[[1, 3, 5, 7]])

    observable_yyzz = mjcf_observable.MJCFFeature('xmat', small_sphere)[2:6]
    np.testing.assert_array_equal(
        observable_yyzz.observation_callable(physics)(), sphere_xmat[2:6])

  def testMJCFCamera(self):
    mjcf_root = mjcf.from_xml_string(_MJCF)
    physics = mjcf.Physics.from_mjcf_model(mjcf_root)

    camera = mjcf_root.find('camera', 'world')
    camera_observable = mjcf_observable.MJCFCamera(
        mjcf_element=camera, height=480, width=640, update_interval=7)
    self.assertEqual(camera_observable.update_interval, 7)
    camera_observation = camera_observable.observation_callable(physics)()
    np.testing.assert_array_equal(
        camera_observation, physics.render(480, 640, 'world'))
    self.assertEqual(camera_observation.shape,
                     camera_observable.array_spec.shape)
    self.assertEqual(camera_observation.dtype,
                     camera_observable.array_spec.dtype)

    camera_observable.height = 300
    camera_observable.width = 400
    camera_observation = camera_observable.observation_callable(physics)()
    self.assertEqual(camera_observable.height, 300)
    self.assertEqual(camera_observable.width, 400)
    np.testing.assert_array_equal(
        camera_observation, physics.render(300, 400, 'world'))
    self.assertEqual(camera_observation.shape,
                     camera_observable.array_spec.shape)
    self.assertEqual(camera_observation.dtype,
                     camera_observable.array_spec.dtype)

    with self.assertRaisesRegex(ValueError, 'expected an `mjcf.Element`'):
      mjcf_observable.MJCFCamera('world')
    with self.assertRaisesRegex(ValueError, 'expected an `mjcf.Element`'):
      mjcf_observable.MJCFCamera([camera])
    with self.assertRaisesRegex(ValueError, 'expected a <camera>'):
      mjcf_observable.MJCFCamera(mjcf_root.find('body', 'body'))

  @parameterized.parameters(
      dict(camera_type='rgb', channels=3, dtype=np.uint8,
           minimum=0, maximum=255),
      dict(camera_type='depth', channels=1, dtype=np.float32,
           minimum=0., maximum=np.inf),
      dict(camera_type='segmentation', channels=2, dtype=np.int32,
           minimum=-1, maximum=np.iinfo(np.int32).max),
  )
  def testMJCFCameraSpecs(self, camera_type, channels, dtype, minimum, maximum):
    width = 640
    height = 480
    shape = (height, width, channels)
    expected_spec = specs.BoundedArray(
        shape=shape, dtype=dtype, minimum=minimum, maximum=maximum)
    mjcf_root = mjcf.from_xml_string(_MJCF)
    camera = mjcf_root.find('camera', 'world')
    observable_kwargs = {} if camera_type == 'rgb' else {camera_type: True}
    camera_observable = mjcf_observable.MJCFCamera(
        mjcf_element=camera, height=height, width=width, update_interval=7,
        **observable_kwargs)
    self.assertEqual(camera_observable.array_spec, expected_spec)

  def testMJCFSegCamera(self):
    mjcf_root = mjcf.from_xml_string(_MJCF)
    physics = mjcf.Physics.from_mjcf_model(mjcf_root)
    camera = mjcf_root.find('camera', 'world')
    camera_observable = mjcf_observable.MJCFCamera(
        mjcf_element=camera, height=480, width=640, update_interval=7,
        segmentation=True)
    self.assertEqual(camera_observable.update_interval, 7)
    camera_observation = camera_observable.observation_callable(physics)()
    np.testing.assert_array_equal(
        camera_observation,
        physics.render(480, 640, 'world', segmentation=True))
    camera_observable.array_spec.validate(camera_observation)

  def testErrorIfSegmentationAndDepthBothEnabled(self):
    camera = mjcf.from_xml_string(_MJCF).find('camera', 'world')
    with self.assertRaisesWithLiteralMatch(
        ValueError, mjcf_observable._BOTH_SEGMENTATION_AND_DEPTH_ENABLED):
      mjcf_observable.MJCFCamera(mjcf_element=camera, segmentation=True,
                                 depth=True)

  def testMJCFCameraConfigureOptions(self):
    mjcf_root = mjcf.from_xml_string(_MJCF)
    camera = mjcf_root.find('camera', 'world')
    # Start with an RGB camera.
    camera_observable = mjcf_observable.MJCFCamera(
        mjcf_element=camera, height=480, width=640, update_interval=7,
        segmentation=False, depth=False)
    # Check attributes are configured correctly.
    self.assertEqual(camera_observable.depth, False)
    self.assertEqual(camera_observable.segmentation, False)
    self.assertEqual(camera_observable.dtype, np.uint8)
    self.assertEqual(camera_observable.n_channels, 3)
    # We can't configure the camera to be both depth and segmentation.
    with self.assertRaisesWithLiteralMatch(
        ValueError, mjcf_observable._BOTH_SEGMENTATION_AND_DEPTH_ENABLED):
      camera_observable.configure(depth=True, segmentation=True)
    # But we should be able to set them both to False.
    camera_observable.configure(depth=False, segmentation=False)
    self.assertEqual(camera_observable.depth, False)
    self.assertEqual(camera_observable.segmentation, False)
    self.assertEqual(camera_observable.dtype, np.uint8)
    self.assertEqual(camera_observable.n_channels, 3)
    # When we set one to True, the other should be False automatically.
    camera_observable.configure(depth=True)
    self.assertEqual(camera_observable.depth, True)
    self.assertEqual(camera_observable.segmentation, False)
    self.assertEqual(camera_observable.dtype, np.float32)
    self.assertEqual(camera_observable.n_channels, 1)
    camera_observable.configure(segmentation=True)
    self.assertEqual(camera_observable.depth, False)
    self.assertEqual(camera_observable.segmentation, True)
    self.assertEqual(camera_observable.dtype, np.int32)
    self.assertEqual(camera_observable.n_channels, 2)
    # We should also be able to explicitly set the second to False.
    camera_observable.configure(depth=True, segmentation=False)
    self.assertEqual(camera_observable.depth, True)
    self.assertEqual(camera_observable.segmentation, False)
    self.assertEqual(camera_observable.dtype, np.float32)
    self.assertEqual(camera_observable.n_channels, 1)
    camera_observable.configure(depth=False, segmentation=True)
    self.assertEqual(camera_observable.depth, False)
    self.assertEqual(camera_observable.segmentation, True)
    self.assertEqual(camera_observable.dtype, np.int32)
    self.assertEqual(camera_observable.n_channels, 2)
    # Finally set both to False and make sure we recover RGB settings.
    camera_observable.configure(depth=False, segmentation=False)
    self.assertEqual(camera_observable.depth, False)
    self.assertEqual(camera_observable.segmentation, False)
    self.assertEqual(camera_observable.dtype, np.uint8)
    self.assertEqual(camera_observable.n_channels, 3)


if __name__ == '__main__':
  absltest.main()
