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

"""Tests for core.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized

from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import core
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

import mock
import numpy as np
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin


HUMANOID_XML_PATH = assets.get_path("humanoid.xml")
MODEL_WITH_ASSETS = assets.get_contents("model_with_assets.xml")
ASSETS = {
    "texture.png": assets.get_contents("deepmind.png"),
    "mesh.stl": assets.get_contents("cube.stl"),
    "included.xml": assets.get_contents("sphere.xml")
}

SCALAR_TYPES = (int, float)
ARRAY_TYPES = (np.ndarray,)

OUT_DIR = absltest.get_default_test_tmpdir()
if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)  # Ensure that the output directory exists.


class CoreTest(parameterized.TestCase):

  def setUp(self):
    self.model = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
    self.data = core.MjData(self.model)

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
        self.fail("Attribute '{}' differs from expected value: {}"
                  .format(name, str(e)))

  def _assert_structs_equal(self, expected, actual):
    for name in set(dir(actual) + dir(expected)):
      if not name.startswith("_"):
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        self.assertEqual(
            expected_value,
            actual_value,
            msg="struct field '{}' has value {}, expected {}".format(
                name, actual_value, expected_value))

  def testLoadXML(self):
    with open(HUMANOID_XML_PATH, "r") as f:
      xml_string = f.read()
    model = core.MjModel.from_xml_string(xml_string)
    core.MjData(model)
    with self.assertRaises(TypeError):
      core.MjModel()
    with self.assertRaises(core.Error):
      core.MjModel.from_xml_path("/path/to/nonexistent/model/file.xml")

    xml_with_warning = """
        <mujoco>
          <size njmax='2'/>
            <worldbody>
              <body pos='0 0 0'>
                <geom type='box' size='.1 .1 .1'/>
              </body>
              <body pos='0 0 0'>
                <joint type='slide' axis='1 0 0'/>
                <geom type='box' size='.1 .1 .1'/>
              </body>
            </worldbody>
        </mujoco>"""
    with mock.patch.object(core, "logging") as mock_logging:
      core.MjModel.from_xml_string(xml_with_warning)
      mock_logging.warn.assert_called_once_with(
          "Error: Pre-allocated constraint buffer is full. "
          "Increase njmax above 2. Time = 0.0000.")

  def testLoadXMLWithAssetsFromString(self):
    core.MjModel.from_xml_string(MODEL_WITH_ASSETS, assets=ASSETS)
    with self.assertRaises(core.Error):
      # Should fail to load without the assets
      core.MjModel.from_xml_string(MODEL_WITH_ASSETS)

  def testSaveLastParsedModelToXML(self):
    save_xml_path = os.path.join(OUT_DIR, "tmp_humanoid.xml")

    not_last_parsed = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
    last_parsed = core.MjModel.from_xml_path(HUMANOID_XML_PATH)

    # Modify the model before saving it in order to confirm that the changes are
    # written to the XML.
    last_parsed.geom_pos.flat[:] = np.arange(last_parsed.geom_pos.size)

    core.save_last_parsed_model_to_xml(save_xml_path, check_model=last_parsed)

    loaded = core.MjModel.from_xml_path(save_xml_path)
    self._assert_attributes_equal(last_parsed, loaded, ["geom_pos"])
    core.MjData(loaded)

    # Test that `check_model` results in a ValueError if it is not the most
    # recently parsed model.
    with self.assertRaisesWithLiteralMatch(
        ValueError, core._NOT_LAST_PARSED_ERROR):
      core.save_last_parsed_model_to_xml(save_xml_path,
                                         check_model=not_last_parsed)

  def testBinaryIO(self):
    bin_path = os.path.join(OUT_DIR, "tmp_humanoid.mjb")
    self.model.save_binary(bin_path)
    core.MjModel.from_binary_path(bin_path)
    byte_string = self.model.to_bytes()
    core.MjModel.from_byte_string(byte_string)

  def testDimensions(self):
    self.assertEqual(self.data.qpos.shape[0], self.model.nq)
    self.assertEqual(self.data.qvel.shape[0], self.model.nv)
    self.assertEqual(self.model.body_pos.shape, (self.model.nbody, 3))

  def testStep(self):
    t0 = self.data.time
    mjlib.mj_step(self.model.ptr, self.data.ptr)
    self.assertEqual(self.data.time, t0 + self.model.opt.timestep)
    self.assert_(np.all(np.isfinite(self.data.qpos[:])))
    self.assert_(np.all(np.isfinite(self.data.qvel[:])))

  def testMultipleData(self):
    data2 = core.MjData(self.model)
    self.assertNotEqual(self.data.ptr, data2.ptr)
    t0 = self.data.time
    mjlib.mj_step(self.model.ptr, self.data.ptr)
    self.assertEqual(self.data.time, t0 + self.model.opt.timestep)
    self.assertEqual(data2.time, 0)

  def testMultipleModel(self):
    model2 = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
    self.assertNotEqual(self.model.ptr, model2.ptr)
    self.model.opt.timestep += 0.001
    self.assertEqual(self.model.opt.timestep, model2.opt.timestep + 0.001)

  def testModelName(self):
    self.assertEqual(self.model.name, "humanoid")

  @parameterized.named_parameters(
      ("_copy", lambda x: x.copy()),
      ("_pickle_unpickle", lambda x: cPickle.loads(cPickle.dumps(x))),)
  def testCopyOrPickleModel(self, func):
    timestep = 0.12345
    self.model.opt.timestep = timestep
    body_pos = self.model.body_pos + 1
    self.model.body_pos[:] = body_pos
    model2 = func(self.model)
    self.assertNotEqual(model2.ptr, self.model.ptr)
    self.assertEqual(model2.opt.timestep, timestep)
    np.testing.assert_array_equal(model2.body_pos, body_pos)

  @parameterized.named_parameters(
      ("_copy", lambda x: x.copy()),
      ("_pickle_unpickle", lambda x: cPickle.loads(cPickle.dumps(x))),)
  def testCopyOrPickleData(self, func):
    for _ in xrange(10):
      mjlib.mj_step(self.model.ptr, self.data.ptr)
    data2 = func(self.data)
    attr_to_compare = ("time", "energy", "qpos", "xpos")
    self.assertNotEqual(data2.ptr, self.data.ptr)
    self._assert_attributes_equal(data2, self.data, attr_to_compare)
    for _ in xrange(10):
      mjlib.mj_step(self.model.ptr, self.data.ptr)
      mjlib.mj_step(data2.model.ptr, data2.ptr)
    self._assert_attributes_equal(data2, self.data, attr_to_compare)

  @parameterized.named_parameters(
      ("_copy", lambda x: x.copy()),
      ("_pickle_unpickle", lambda x: cPickle.loads(cPickle.dumps(x))),)
  def testCopyOrPickleStructs(self, func):
    for _ in xrange(10):
      mjlib.mj_step(self.model.ptr, self.data.ptr)
    data2 = func(self.data)
    self.assertNotEqual(data2.ptr, self.data.ptr)
    for name in ["warning", "timer", "solver"]:
      self._assert_structs_equal(getattr(self.data, name), getattr(data2, name))
    for _ in xrange(10):
      mjlib.mj_step(self.model.ptr, self.data.ptr)
      mjlib.mj_step(data2.model.ptr, data2.ptr)
    for expected, actual in zip(self.data.timer, data2.timer):
      self._assert_structs_equal(expected, actual)

  @parameterized.parameters(
      ("right_foot", "body", 6),
      ("right_foot", enums.mjtObj.mjOBJ_BODY, 6),
      ("left_knee", "joint", 11),
      ("left_knee", enums.mjtObj.mjOBJ_JOINT, 11))
  def testNamesIds(self, name, object_type, object_id):
    output_id = self.model.name2id(name, object_type)
    self.assertEqual(object_id, output_id)
    output_name = self.model.id2name(object_id, object_type)
    self.assertEqual(name, output_name)

  def testNamesIdsExceptions(self):
    with self.assertRaisesRegexp(core.Error, "does not exist"):
      self.model.name2id("nonexistent_body_name", "body")
    with self.assertRaisesRegexp(core.Error, "is not a valid object type"):
      self.model.name2id("right_foot", "nonexistent_type_name")

  def testNamelessObject(self):
    # The model in humanoid.xml contains a single nameless camera.
    name = self.model.id2name(0, "camera")
    self.assertEqual("", name)

  def testWarningCallback(self):
    self.data.qpos[0] = np.inf
    with mock.patch.object(core, "logging") as mock_logging:
      mjlib.mj_step(self.model.ptr, self.data.ptr)
    mock_logging.warn.assert_called_once_with(
        "Nan, Inf or huge value in QPOS at DOF 0. The simulation is unstable. "
        "Time = 0.0000.")

  def testErrorCallback(self):
    with mock.patch.object(core, "logging") as mock_logging:
      mjlib.mj_activate(b"nonexistent_activation_key")
    mock_logging.fatal.assert_called_once_with(
        "Could not open activation key file nonexistent_activation_key")

  def testSingleCallbackContext(self):

    callback_was_called = [False]

    def callback(unused_model, unused_data):
      callback_was_called[0] = True

    mjlib.mj_step(self.model.ptr, self.data.ptr)
    self.assertFalse(callback_was_called[0])

    class DummyError(RuntimeError):
      pass

    try:
      with core.callback_context("mjcb_passive", callback):

        # Stepping invokes the `mjcb_passive` callback.
        mjlib.mj_step(self.model.ptr, self.data.ptr)
        self.assertTrue(callback_was_called[0])

        # Exceptions should not prevent `mjcb_passive` from being reset.
        raise DummyError("Simulated exception.")

    except DummyError:
      pass

    # `mjcb_passive` should have been reset to None.
    callback_was_called[0] = False
    mjlib.mj_step(self.model.ptr, self.data.ptr)
    self.assertFalse(callback_was_called[0])

  def testNestedCallbackContexts(self):

    last_called = [None]
    outer_called = "outer called"
    inner_called = "inner called"

    def outer(unused_model, unused_data):
      last_called[0] = outer_called

    def inner(unused_model, unused_data):
      last_called[0] = inner_called

    with core.callback_context("mjcb_passive", outer):

      # This should execute `outer` a few times.
      mjlib.mj_step(self.model.ptr, self.data.ptr)
      self.assertEqual(last_called[0], outer_called)

      with core.callback_context("mjcb_passive", inner):

        # This should execute `inner` a few times.
        mjlib.mj_step(self.model.ptr, self.data.ptr)
        self.assertEqual(last_called[0], inner_called)

      # When we exit the inner context, the `mjcb_passive` callback should be
      # reset to `outer`.
      mjlib.mj_step(self.model.ptr, self.data.ptr)
      self.assertEqual(last_called[0], outer_called)

    # When we exit the outer context, the `mjcb_passive` callback should be
    # reset to None, and stepping should not affect `last_called`.
    last_called[0] = None
    mjlib.mj_step(self.model.ptr, self.data.ptr)
    self.assertIsNone(last_called[0])

  def testDisableFlags(self):
    xml_string = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"/>
        <body name="cube" pos="0 0 0.1">
          <geom type="box" size="0.1 0.1 0.1" mass="1"/>
          <site name="cube_site" type="box" size="0.1 0.1 0.1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
      <sensor>
        <touch name="touch_sensor" site="cube_site"/>
      </sensor>
    </mujoco>
    """
    model = core.MjModel.from_xml_string(xml_string)
    data = core.MjData(model)
    for _ in xrange(100):  # Let the simulation settle for a while.
      mjlib.mj_step(model.ptr, data.ptr)

    # With gravity and contact enabled, the cube should be stationary and the
    # touch sensor should give a reading of ~9.81 N.
    self.assertAlmostEqual(data.qvel[0], 0, places=4)
    self.assertAlmostEqual(data.sensordata[0], 9.81, places=2)

    # If we disable both contacts and gravity then the cube should remain
    # stationary and the touch sensor should read zero.
    with model.disable("contact", "gravity"):
      mjlib.mj_step(model.ptr, data.ptr)
    self.assertAlmostEqual(data.qvel[0], 0, places=4)
    self.assertEqual(data.sensordata[0], 0)

    # If we disable contacts but not gravity then the cube should fall through
    # the floor.
    with model.disable(enums.mjtDisableBit.mjDSBL_CONTACT):
      for _ in xrange(10):
        mjlib.mj_step(model.ptr, data.ptr)
    self.assertLess(data.qvel[0], -0.1)

  def testDisableFlagsExceptions(self):
    with self.assertRaisesRegexp(ValueError, "not a valid flag name"):
      with self.model.disable("invalid_flag_name"):
        pass
    with self.assertRaisesRegexp(ValueError,
                                 "not a value in `enums.mjtDisableBit`"):
      with self.model.disable(-99):
        pass


def _get_attributes_test_params():
  model = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
  data = core.MjData(model)
  # Get the names of the non-private attributes of model and data through
  # introspection. These are passed as parameters to each of the test methods
  # in AttributesTest.
  array_args = []
  scalar_args = []
  skipped_args = []
  for parent_name, parent_obj in zip(("model", "data"), (model, data)):
    for attr_name in dir(parent_obj):
      if not attr_name.startswith("_"):  # Skip 'private' attributes
        args = (parent_name, attr_name)
        attr = getattr(parent_obj, attr_name)
        if isinstance(attr, ARRAY_TYPES):
          array_args.append(args)
        elif isinstance(attr, SCALAR_TYPES):
          scalar_args.append(args)
        elif callable(attr):
          # Methods etc. should be covered specifically in CoreTest.
          continue
        else:
          skipped_args.append(args)
  return array_args, scalar_args, skipped_args


_array_args, _scalar_args, _skipped_args = _get_attributes_test_params()


class AttributesTest(parameterized.TestCase):
  """Generic tests covering attributes of MjModel and MjData."""

  # Iterates over ('parent_name', 'attr_name') tuples
  @parameterized.parameters(*_array_args)
  def testReadWriteArray(self, parent_name, attr_name):
    attr = getattr(getattr(self, parent_name), attr_name)
    if not isinstance(attr, ARRAY_TYPES):
      raise TypeError("{}.{} has incorrect type {!r} - must be one of {!r}."
                      .format(parent_name, attr_name, type(attr), ARRAY_TYPES))
    # Check that we can read the contents of the array
    old_contents = attr[:]
    # Don't write to integer arrays since these might contain pointers.
    if not np.issubdtype(old_contents.dtype, int):
      # Write unique values to the array, check that we can read them back.
      new_contents = np.arange(old_contents.size, dtype=old_contents.dtype)
      new_contents.shape = old_contents.shape
      attr[:] = new_contents
      np.testing.assert_array_equal(new_contents, attr[:])
      self._take_steps()  # Take a few steps, check that we don't get segfaults.

  @parameterized.parameters(*_scalar_args)
  def testReadWriteScalar(self, parent_name, attr_name):
    parent_obj = getattr(self, parent_name)
    # Check that we can read the value.
    attr = getattr(parent_obj, attr_name)
    if not isinstance(attr, SCALAR_TYPES):
      raise TypeError("{}.{} has incorrect type {!r} - must be one of {!r}."
                      .format(parent_name, attr_name, type(attr), SCALAR_TYPES))
    # Don't write to integers since these might be pointers.
    if not isinstance(attr, int):
      # Set the value of this attribute, check that we can read it back.
      new_value = type(attr)(99)
      setattr(parent_obj, attr_name, new_value)
      self.assertEqual(new_value, getattr(parent_obj, attr_name))
      self._take_steps()  # Take a few steps, check that we don't get segfaults.

  @parameterized.parameters(*_skipped_args)
  @absltest.unittest.skip("No tests defined for attributes of this type.")
  def testSkipped(self, *unused_args):
    # This is a do-nothing test that indicates where we currently lack coverage.
    pass

  def setUp(self):
    self.model = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
    self.data = core.MjData(self.model)

  def _take_steps(self, n=5):
    for _ in xrange(n):
      mjlib.mj_step(self.model.ptr, self.data.ptr)


if __name__ == "__main__":
  absltest.main()
