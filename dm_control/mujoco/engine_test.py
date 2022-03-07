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

"""Tests for `engine`."""

import copy
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mujoco import engine
from dm_control.mujoco import wrapper
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.rl import control
import mock
import mujoco
import numpy as np

MODEL_PATH = assets.get_path('cartpole.xml')
MODEL_WITH_ASSETS = assets.get_contents('model_with_assets.xml')
ASSETS = {
    'texture.png': assets.get_contents('deepmind.png'),
    'mesh.stl': assets.get_contents('cube.stl'),
    'included.xml': assets.get_contents('sphere.xml')
}


class MujocoEngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._physics = engine.Physics.from_xml_path(MODEL_PATH)

  def _assert_attributes_equal(self, actual_obj, expected_obj, attr_to_compare):
    for name in attr_to_compare:
      actual_value = getattr(actual_obj, name)
      expected_value = getattr(expected_obj, name)
      try:
        if isinstance(expected_value, np.ndarray):
          np.testing.assert_array_equal(actual_value, expected_value)
        else:
          self.assertEqual(actual_value, expected_value)
      except AssertionError as e:
        raise AssertionError(
            f"Attribute '{name}' differs from expected value.") from e

  @parameterized.parameters(0, 'cart', u'cart')
  def testCameraIndexing(self, camera_id):
    height, width = 480, 640
    _ = engine.Camera(
        self._physics, height, width, camera_id=camera_id)

  def testDepthRender(self):
    plane_and_box = """
    <mujoco>
      <worldbody>
        <geom type="plane" pos="0 0 0" size="2 2 .1"/>
        <geom type="box" size=".1 .1 .1" pos="0 0 .1"/>
        <camera name="top" pos="0 0 3"/>
      </worldbody>
    </mujoco>
    """
    physics = engine.Physics.from_xml_string(plane_and_box)
    pixels = physics.render(height=200, width=200, camera_id='top', depth=True)
    # Nearest pixels should be 2.8m away
    np.testing.assert_approx_equal(pixels.min(), 2.8, 3)
    # Furthest pixels should be 3m away (depth is orthographic)
    np.testing.assert_approx_equal(pixels.max(), 3.0, 3)

  @parameterized.parameters([True, False])
  def testSegmentationRender(self, enable_geom_frame_rendering):
    box_four_corners = """
    <mujoco>
      <visual>
        <scale framelength="2"/>
      </visual>
      <worldbody>
        <geom name="box0" type="box" size=".2 .2 .2" pos="-1 1 .1"/>
        <geom name="box1" type="box" size=".2 .2 .2" pos="1 1 .1"/>
        <site name="box2" type="box" size=".2 .2 .2" pos="1 -1 .1"/>
        <site name="box3" type="box" size=".2 .2 .2" pos="-1 -1 .1"/>
        <camera name="top" pos="0 0 3"/>
      </worldbody>
    </mujoco>
    """
    physics = engine.Physics.from_xml_string(box_four_corners)
    obj_type_geom = enums.mjtObj.mjOBJ_GEOM  # Geom object type
    obj_type_site = enums.mjtObj.mjOBJ_SITE  # Site object type
    obj_type_decor = enums.mjtObj.mjOBJ_UNKNOWN  # Decor object type
    scene_options = wrapper.MjvOption()
    if enable_geom_frame_rendering:
      scene_options.frame = mujoco.mjtFrame.mjFRAME_GEOM
    pixels = physics.render(height=200, width=200, camera_id='top',
                            segmentation=True, scene_option=scene_options)

    # The pixel indices below were chosen so that toggling the frame decors do
    # not affect the segmentation results.
    with self.subTest('Center pixels should have background label'):
      np.testing.assert_equal(pixels[95:105, 95:105, 0], -1)
      np.testing.assert_equal(pixels[95:105, 95:105, 1], -1)
    with self.subTest('Geoms have correct object type'):
      np.testing.assert_equal(pixels[15:25, 0:10, 1], obj_type_geom)
      np.testing.assert_equal(pixels[15:25, 190:200, 1], obj_type_geom)
    with self.subTest('Sites have correct object type'):
      np.testing.assert_equal(pixels[190:200, 190:200, 1], obj_type_site)
      np.testing.assert_equal(pixels[190:200, 0:10, 1], obj_type_site)
    with self.subTest('Geoms have correct object IDs'):
      np.testing.assert_equal(pixels[15:25, 0:10, 0],
                              physics.model.name2id('box0', obj_type_geom))
      np.testing.assert_equal(pixels[15:25, 190:200, 0],
                              physics.model.name2id('box1', obj_type_geom))
    with self.subTest('Sites have correct object IDs'):
      np.testing.assert_equal(pixels[190:200, 190:200, 0],
                              physics.model.name2id('box2', obj_type_site))
      np.testing.assert_equal(pixels[190:200, 0:10, 0],
                              physics.model.name2id('box3', obj_type_site))
    with self.subTest('Decor elements present if and only if geom frames are '
                      'enabled'):
      contains_decor = np.any(pixels[:, :, 1] == obj_type_decor)
      self.assertEqual(contains_decor, enable_geom_frame_rendering)

  def testTextOverlay(self):
    height, width = 480, 640
    overlay = engine.TextOverlay(title='Title', body='Body', style='big',
                                 position='bottom right')

    no_overlay = self._physics.render(height, width, camera_id=0)
    with_overlay = self._physics.render(height, width, camera_id=0,
                                        overlays=[overlay])
    self.assertFalse(np.all(no_overlay == with_overlay),
                     msg='Images are identical with and without text overlay.')

  def testSceneOption(self):
    height, width = 480, 640
    scene_option = wrapper.MjvOption()

    # Render geoms as semi-transparent.
    scene_option.flags[enums.mjtVisFlag.mjVIS_TRANSPARENT] = 1

    no_scene_option = self._physics.render(height, width, camera_id=0)
    with_scene_option = self._physics.render(height, width, camera_id=0,
                                             scene_option=scene_option)
    self.assertFalse(np.all(no_scene_option == with_scene_option),
                     msg='Images are identical with and without scene option.')

  def testRenderFlags(self):
    height, width = 480, 640
    cam = engine.Camera(self._physics, height, width, camera_id=0)
    cam.scene.flags[enums.mjtRndFlag.mjRND_WIREFRAME] = 1  # Enable wireframe
    enabled = cam.render().copy()
    cam.scene.flags[enums.mjtRndFlag.mjRND_WIREFRAME] = 0  # Disable wireframe
    disabled = cam.render().copy()
    self.assertFalse(
        np.all(disabled == enabled),
        msg='Images are identical regardless of whether wireframe is enabled.')

  @parameterized.parameters(((0.5, 0.5), (1, 3)),  # pole
                            ((0.5, 0.1), (0, 0)),  # ground
                            ((0.9, 0.9), (None, None)),  # sky
                           )
  def testCameraSelection(self, coordinates, expected_selection):
    height, width = 480, 640
    camera = engine.Camera(self._physics, height, width, camera_id=0)

    # Test for b/63380170: Enabling visualization of body frames adds
    # "non-model" geoms to the scene. This means that the indices of geoms
    # within `camera._scene.geoms` don't match the rows of `model.geom_bodyid`.
    camera.option.frame = enums.mjtFrame.mjFRAME_BODY

    selected = camera.select(coordinates)
    self.assertEqual(expected_selection, selected[:2])

  @parameterized.parameters(
      dict(camera_id='cam0', height=200, width=300),
      dict(camera_id=1, height=300, width=200),
      dict(camera_id=-1, height=400, width=400),
  )
  def testCameraMatrix(self, camera_id, height, width):
    """Tests the camera_matrix() method.

       Creates a model with two cameras and two small geoms. We render the scene
       with one of the cameras and check that the geom locations, projected into
       pixel space, are correct, using segmenation rendering.
       xyz2pixels() shows how the transformation is used. For a description
       of the camera matrix see https://en.wikipedia.org/wiki/Camera_matrix.

    Args:
      camera_id: One of the two cameras. Can be either integer or String.
      height: The height of the image (pixels).
      width: The width of the image (pixels).
    """

    def xyz2pixels(x, y, z, camera_matrix):
      """Transforms from world coordinates to pixel coordinates."""
      xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))
      return xs/s, ys/s

    two_geoms_and_two_cameras = """
    <mujoco>
      <visual>
        <global fovy="55"/>
      </visual>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <geom name="red" pos=".2 0 0" size=".005" rgba="1 0 0 1"/>
        <geom name="green" pos=".2 .2 .1" size=".005" rgba="0 1 0 1"/>
        <camera name="cam0" pos="1 .5 1" zaxis="1 .5 1" fovy="20"/>
        <camera name="cam1" pos=".1 .1 1" xyaxes="1 1 0 -1 0 0"/>
      </worldbody>
    </mujoco>
    """
    physics = engine.Physics.from_xml_string(two_geoms_and_two_cameras)
    camera = engine.Camera(physics, width=width, height=height,
                           camera_id=camera_id)
    camera_matrix = camera.matrix  # Get camera matrix.
    pixels = camera.render(segmentation=True)  # Render a segmentation frame.
    for geom_id in [0, 1]:
      # Compute the location of the geom in pixel space using the camera matrix.
      x, y = xyz2pixels(*physics.data.geom_xpos[geom_id], camera_matrix)
      row = int(round(y))
      column = int(round(x))
      # Compare segmentation values of nearest pixel to corresponding geom.
      [obj_id, obj_type] = pixels[row, column, :]
      self.assertEqual(obj_type, enums.mjtObj.mjOBJ_GEOM)
      self.assertEqual(obj_id, geom_id)

  def testMovableCameraSetGetPose(self):
    height, width = 240, 320

    camera = engine.MovableCamera(self._physics, height, width)
    image = camera.render().copy()

    pose = camera.get_pose()

    lookat_offset = np.array([0.01, 0.02, -0.03])

    # Would normally pass the new values directly to camera.set_pose instead of
    # using the namedtuple _replace method, but this makes the asserts at the
    # end of the test a little cleaner.
    new_pose = pose._replace(distance=pose.distance * 1.5,
                             lookat=pose.lookat + lookat_offset,
                             azimuth=pose.azimuth + -15,
                             elevation=pose.elevation - 10)

    camera.set_pose(*new_pose)

    self.assertEqual(new_pose.distance, camera.get_pose().distance)
    self.assertEqual(new_pose.azimuth, camera.get_pose().azimuth)
    self.assertEqual(new_pose.elevation, camera.get_pose().elevation)
    np.testing.assert_allclose(new_pose.lookat, camera.get_pose().lookat)

    self.assertFalse(np.all(image == camera.render()))

  def testRenderExceptions(self):
    max_width = self._physics.model.vis.global_.offwidth
    max_height = self._physics.model.vis.global_.offheight
    max_camid = self._physics.model.ncam - 1
    with self.assertRaisesRegex(ValueError, 'width'):
      self._physics.render(max_height, max_width + 1, camera_id=max_camid)
    with self.assertRaisesRegex(ValueError, 'height'):
      self._physics.render(max_height + 1, max_width, camera_id=max_camid)
    with self.assertRaisesRegex(ValueError, 'camera_id'):
      self._physics.render(max_height, max_width, camera_id=max_camid + 1)
    with self.assertRaisesRegex(ValueError, 'camera_id'):
      self._physics.render(max_height, max_width, camera_id=-2)

  def testPhysicsRenderMethod(self):
    height, width = 240, 320
    image = self._physics.render(height=height, width=width)
    self.assertEqual(image.shape, (height, width, 3))
    depth = self._physics.render(height=height, width=width, depth=True)
    self.assertEqual(depth.shape, (height, width))
    segmentation = self._physics.render(height=height, width=width,
                                        segmentation=True)
    self.assertEqual(segmentation.shape, (height, width, 2))

  def testExceptionIfBothDepthAndSegmentation(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, engine._BOTH_SEGMENTATION_AND_DEPTH_ENABLED):
      self._physics.render(depth=True, segmentation=True)

  def testRenderFlagOverridesAreNotPersistent(self):
    camera = engine.Camera(self._physics)
    first_rgb = camera.render().copy()
    camera.render(segmentation=True)
    second_rgb = camera.render().copy()
    np.testing.assert_array_equal(first_rgb, second_rgb)

  def testCustomRenderFlags(self):
    default = self._physics.render()
    wireframe_string_key = self._physics.render(
        render_flag_overrides=dict(wireframe=True))
    self.assertFalse((default == wireframe_string_key).all())
    wireframe_enum_key = self._physics.render(
        render_flag_overrides={enums.mjtRndFlag.mjRND_WIREFRAME: True})
    np.testing.assert_array_equal(wireframe_string_key, wireframe_enum_key)

  @parameterized.parameters(dict(depth=True), dict(segmentation=True))
  def testExceptionIfRenderFlagOverridesAndDepthOrSegmentation(self, **kwargs):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        engine._RENDER_FLAG_OVERRIDES_NOT_SUPPORTED_FOR_DEPTH_OR_SEGMENTATION):
      self._physics.render(render_flag_overrides=dict(wireframe=True), **kwargs)

  def testExceptionIfOverlaysAndDepthOrSegmentation(self):
    overlay = engine.TextOverlay()
    with self.assertRaisesWithLiteralMatch(
        ValueError, engine._OVERLAYS_NOT_SUPPORTED_FOR_DEPTH_OR_SEGMENTATION):
      self._physics.render(depth=True, overlays=[overlay])
    with self.assertRaisesWithLiteralMatch(
        ValueError, engine._OVERLAYS_NOT_SUPPORTED_FOR_DEPTH_OR_SEGMENTATION):
      self._physics.render(segmentation=True, overlays=[overlay])

  def testNamedViews(self):
    self.assertEqual((1,), self._physics.control().shape)
    self.assertEqual((2,), self._physics.position().shape)
    self.assertEqual((2,), self._physics.velocity().shape)
    self.assertEqual((0,), self._physics.activation().shape)
    self.assertEqual((4,), self._physics.state().shape)
    self.assertEqual(0., self._physics.time())
    self.assertEqual(0.01, self._physics.timestep())

  def testSetGetPhysicsState(self):
    physics_state = self._physics.get_state()
    self._physics.set_state(physics_state)

    new_physics_state = np.random.random_sample(physics_state.shape)
    self._physics.set_state(new_physics_state)

    np.testing.assert_allclose(new_physics_state,
                               self._physics.get_state())

  def testSetInvalidPhysicsState(self):
    badly_shaped_state = np.repeat(self._physics.get_state(), repeats=2)

    with self.assertRaises(ValueError):
      self._physics.set_state(badly_shaped_state)

  def testNamedIndexing(self):
    self.assertEqual((3,), self._physics.named.data.xpos['cart'].shape)
    self.assertEqual((2, 3),
                     self._physics.named.data.xpos[['cart', 'pole']].shape)

  def testReload(self):
    self._physics.reload_from_xml_path(MODEL_PATH)

  def testReset(self):
    self._physics.reset()
    self.assertEqual(self._physics.data.qpos[1], 0)
    keyframe_id = 0
    self._physics.reset(keyframe_id=keyframe_id)
    self.assertEqual(self._physics.data.qpos[1],
                     self._physics.model.key_qpos[keyframe_id, 1])
    out_of_range = [-1, 3]
    max_valid = self._physics.model.nkey - 1
    for actual in out_of_range:
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          engine._KEYFRAME_ID_OUT_OF_RANGE.format(
              max_valid=max_valid, actual=actual)):
        self._physics.reset(keyframe_id=actual)

  def testLoadAndReloadFromStringWithAssets(self):
    physics = engine.Physics.from_xml_string(
        MODEL_WITH_ASSETS, assets=ASSETS)
    physics.reload_from_xml_string(MODEL_WITH_ASSETS, assets=ASSETS)

  @parameterized.parameters(*enums.mjtWarning._fields[:-1])
  def testDivergenceException(self, warning_name):
    warning_enum = getattr(enums.mjtWarning, warning_name)
    with self.assertRaisesWithLiteralMatch(
        control.PhysicsError,
        engine._INVALID_PHYSICS_STATE.format(warning_names=warning_name)):
      with self._physics.check_invalid_state():
        self._physics.data.warning[warning_enum].number = 1
    # Existing warnings should not raise an exception.
    with self._physics.check_invalid_state():
      pass
    self._physics.reset()
    with self._physics.check_invalid_state():
      pass

  @parameterized.parameters(float('inf'), float('nan'), 1e15)
  def testBadQpos(self, bad_value):
    with self._physics.reset_context():
      self._physics.data.qpos[0] = bad_value
    with self.assertRaises(control.PhysicsError):
      with self._physics.check_invalid_state():
        mujoco.mj_checkPos(self._physics.model.ptr, self._physics.data.ptr)
    self._physics.reset()
    with self._physics.check_invalid_state():
      mujoco.mj_checkPos(self._physics.model.ptr, self._physics.data.ptr)

  def testNanControl(self):
    with self._physics.reset_context():
      pass

    # Apply the controls.
    with self.assertRaisesWithLiteralMatch(
        control.PhysicsError,
        engine._INVALID_PHYSICS_STATE.format(warning_names='mjWARN_BADCTRL')):
      with self._physics.check_invalid_state():
        self._physics.data.ctrl[0] = float('nan')
        self._physics.step()

  def testSuppressPhysicsError(self):
    bad_value = float('nan')
    message = engine._INVALID_PHYSICS_STATE.format(
        warning_names='mjWARN_BADCTRL')

    def assert_physics_error():
      self._physics.data.ctrl[0] = bad_value
      with self.assertRaisesWithLiteralMatch(control.PhysicsError, message):
        self._physics.forward()

    def assert_warning():
      self._physics.data.ctrl[0] = bad_value
      with mock.patch.object(engine.logging, 'warn') as mock_warn:
        self._physics.forward()
      mock_warn.assert_called_once_with(message)

    assert_physics_error()
    with self._physics.suppress_physics_errors():
      assert_warning()
      with self._physics.suppress_physics_errors():
        assert_warning()
      assert_warning()
    assert_physics_error()

  @parameterized.named_parameters(
      ('_copy', lambda x: x.copy()),
      ('_deepcopy', copy.deepcopy),
      ('_pickle_and_unpickle', lambda x: pickle.loads(pickle.dumps(x))),
  )
  def testCopyOrPicklePhysics(self, func):
    for _ in range(10):
      self._physics.step()
    physics2 = func(self._physics)
    self.assertNotEqual(physics2.model.ptr, self._physics.model.ptr)
    self.assertNotEqual(physics2.data.ptr, self._physics.data.ptr)
    model_attr_to_compare = ('nnames', 'njmax', 'body_pos', 'geom_quat')
    self._assert_attributes_equal(
        physics2.model, self._physics.model, model_attr_to_compare)
    data_attr_to_compare = ('time', 'energy', 'qpos', 'xpos')
    self._assert_attributes_equal(
        physics2.data, self._physics.data, data_attr_to_compare)
    for _ in range(10):
      self._physics.step()
      physics2.step()
    self._assert_attributes_equal(
        physics2.model, self._physics.model, model_attr_to_compare)
    self._assert_attributes_equal(
        physics2.data, self._physics.data, data_attr_to_compare)

  @parameterized.named_parameters(
      ('_copy', lambda x: x.copy()),
      ('_pickle_and_unpickle', lambda x: pickle.loads(pickle.dumps(x))),
  )
  def testSuppressErrorsAfterCopyOrPicklePhysics(self, func):
    # Regression test for a problem that used to exist where
    # suppress_physics_errors couldn't be used on Physics objects that were
    # unpickled.
    physics2 = func(self._physics)
    with physics2.suppress_physics_errors():
      pass

  def testCopyDataOnly(self):
    physics2 = self._physics.copy(share_model=True)
    self.assertEqual(physics2.model.ptr, self._physics.model.ptr)
    self.assertNotEqual(physics2.data.ptr, self._physics.data.ptr)

  def testForwardDynamicsUpdatedAfterReset(self):
    gravity = -9.81
    self._physics.model.opt.gravity[2] = gravity
    with self._physics.reset_context():
      pass
    self.assertAlmostEqual(
        self._physics.named.data.sensordata['accelerometer'][2], -gravity)

  def testActuationNotAppliedInAfterReset(self):
    self._physics.data.ctrl[0] = 1.
    self._physics.after_reset()  # Calls `forward()` with actuation disabled.
    self.assertEqual(self._physics.data.actuator_force[0], 0.)
    self._physics.forward()  # Call `forward` directly with actuation enabled.
    self.assertEqual(self._physics.data.actuator_force[0], 1.)

  def testActionSpec(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="0.1"/>
          <joint type="hinge" name="hinge"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="hinge" ctrllimited="false"/>
        <motor joint="hinge" ctrllimited="true" ctrlrange="-1 2"/>
      </actuator>
    </mujoco>
    """
    physics = engine.Physics.from_xml_string(xml)
    spec = engine.action_spec(physics)
    self.assertEqual(float, spec.dtype)
    np.testing.assert_array_equal(spec.minimum, [-mujoco.mjMAXVAL, -1.0])
    np.testing.assert_array_equal(spec.maximum, [mujoco.mjMAXVAL, 2.0])

if __name__ == '__main__':
  absltest.main()
