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

import gc
import os
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import _render
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import core
import mujoco
import numpy as np

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
    super().setUp()
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

  def testLoadXML(self):
    with open(HUMANOID_XML_PATH, "r") as f:
      xml_string = f.read()
    model = core.MjModel.from_xml_string(xml_string)
    core.MjData(model)
    with self.assertRaises(TypeError):
      core.MjModel()
    with self.assertRaises(ValueError):
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

    # This model should compile successfully, but raise a warning on the first
    # simulation step.
    model = core.MjModel.from_xml_string(xml_with_warning)
    data = core.MjData(model)
    mujoco.mj_step(model.ptr, data.ptr)

  def testLoadXMLWithAssetsFromString(self):
    core.MjModel.from_xml_string(MODEL_WITH_ASSETS, assets=ASSETS)
    with self.assertRaises(ValueError):
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
    mujoco.mj_step(self.model.ptr, self.data.ptr)
    self.assertEqual(self.data.time, t0 + self.model.opt.timestep)
    self.assertTrue(np.all(np.isfinite(self.data.qpos[:])))
    self.assertTrue(np.all(np.isfinite(self.data.qvel[:])))

  def testMultipleData(self):
    data2 = core.MjData(self.model)
    self.assertNotEqual(self.data.ptr, data2.ptr)
    t0 = self.data.time
    mujoco.mj_step(self.model.ptr, self.data.ptr)
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
      ("_pickle_unpickle", lambda x: pickle.loads(pickle.dumps(x))),)
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
      ("_pickle_unpickle", lambda x: pickle.loads(pickle.dumps(x))),)
  def testCopyOrPickleData(self, func):
    for _ in range(10):
      mujoco.mj_step(self.model.ptr, self.data.ptr)
    data2 = func(self.data)
    attr_to_compare = ("time", "energy", "qpos", "xpos")
    self.assertNotEqual(data2.ptr, self.data.ptr)
    self._assert_attributes_equal(data2, self.data, attr_to_compare)
    for _ in range(10):
      mujoco.mj_step(self.model.ptr, self.data.ptr)
      mujoco.mj_step(data2.model.ptr, data2.ptr)
    self._assert_attributes_equal(data2, self.data, attr_to_compare)

  @parameterized.named_parameters(
      ("_copy", lambda x: x.copy()),
      ("_pickle_unpickle", lambda x: pickle.loads(pickle.dumps(x))),)
  def testCopyOrPickleStructs(self, func):
    for _ in range(10):
      mujoco.mj_step(self.model.ptr, self.data.ptr)
    data2 = func(self.data)
    self.assertNotEqual(data2.ptr, self.data.ptr)
    attr_to_compare = ("warning", "timer", "solver")
    self._assert_attributes_equal(self.data, data2, attr_to_compare)
    for _ in range(10):
      mujoco.mj_step(self.model.ptr, self.data.ptr)
      mujoco.mj_step(data2.model.ptr, data2.ptr)
    self._assert_attributes_equal(self.data, data2, attr_to_compare)

  @parameterized.parameters(
      ("right_foot", "body", 6),
      ("right_foot", mujoco.mjtObj.mjOBJ_BODY, 6),
      ("left_knee", "joint", 11),
      ("left_knee", mujoco.mjtObj.mjOBJ_JOINT, 11),
  )
  def testNamesIds(self, name, object_type, object_id):
    output_id = self.model.name2id(name, object_type)
    self.assertEqual(object_id, output_id)
    output_name = self.model.id2name(object_id, object_type)
    self.assertEqual(name, output_name)

  def testNamesIdsExceptions(self):
    with self.assertRaisesRegex(core.Error, "does not exist"):
      self.model.name2id("nonexistent_body_name", "body")
    with self.assertRaisesRegex(core.Error, "is not a valid object type"):
      self.model.name2id("right_foot", "nonexistent_type_name")

  def testNamelessObject(self):
    # The model in humanoid.xml contains a single nameless camera.
    name = self.model.id2name(0, "camera")
    self.assertEqual("", name)

  def testSingleCallbackContext(self):

    callback_was_called = [False]

    def callback(unused_model, unused_data):
      callback_was_called[0] = True

    mujoco.mj_step(self.model.ptr, self.data.ptr)
    self.assertFalse(callback_was_called[0])

    class DummyError(RuntimeError):
      pass

    try:
      with core.callback_context("mjcb_passive", callback):

        # Stepping invokes the `mjcb_passive` callback.
        mujoco.mj_step(self.model.ptr, self.data.ptr)
        self.assertTrue(callback_was_called[0])

        # Exceptions should not prevent `mjcb_passive` from being reset.
        raise DummyError("Simulated exception.")

    except DummyError:
      pass

    # `mjcb_passive` should have been reset to None.
    callback_was_called[0] = False
    mujoco.mj_step(self.model.ptr, self.data.ptr)
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
      mujoco.mj_step(self.model.ptr, self.data.ptr)
      self.assertEqual(last_called[0], outer_called)

      with core.callback_context("mjcb_passive", inner):

        # This should execute `inner` a few times.
        mujoco.mj_step(self.model.ptr, self.data.ptr)
        self.assertEqual(last_called[0], inner_called)

      # When we exit the inner context, the `mjcb_passive` callback should be
      # reset to `outer`.
      mujoco.mj_step(self.model.ptr, self.data.ptr)
      self.assertEqual(last_called[0], outer_called)

    # When we exit the outer context, the `mjcb_passive` callback should be
    # reset to None, and stepping should not affect `last_called`.
    last_called[0] = None
    mujoco.mj_step(self.model.ptr, self.data.ptr)
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
    for _ in range(100):  # Let the simulation settle for a while.
      mujoco.mj_step(model.ptr, data.ptr)

    # With gravity and contact enabled, the cube should be stationary and the
    # touch sensor should give a reading of ~9.81 N.
    self.assertAlmostEqual(data.qvel[0], 0, places=4)
    self.assertAlmostEqual(data.sensordata[0], 9.81, places=2)

    # If we disable both contacts and gravity then the cube should remain
    # stationary and the touch sensor should read zero.
    with model.disable("contact", "gravity"):
      mujoco.mj_step(model.ptr, data.ptr)
    self.assertAlmostEqual(data.qvel[0], 0, places=4)
    self.assertEqual(data.sensordata[0], 0)

    # If we disable contacts but not gravity then the cube should fall through
    # the floor.
    with model.disable(mujoco.mjtDisableBit.mjDSBL_CONTACT):
      for _ in range(10):
        mujoco.mj_step(model.ptr, data.ptr)
    self.assertLess(data.qvel[0], -0.1)

  def testDisableFlagsExceptions(self):
    with self.assertRaises(ValueError):
      with self.model.disable("invalid_flag_name"):
        pass
    with self.assertRaises(ValueError):
      with self.model.disable(-99):
        pass

  @parameterized.parameters(
      # The tip is .5 meters from the cart so we expect its horizontal velocity
      # to be 1m/s + .5m*1rad/s = 1.5m/s.
      dict(
          qpos=[0., 0.],  # Pole pointing upwards.
          qvel=[1., 1.],
          expected_linvel=[1.5, 0., 0.],
          expected_angvel=[0., 1., 0.],
      ),
      # For the same velocities but with the pole pointing down, we expect the
      # velocities to cancel, making the global tip velocity now equal to
      # 1m/s - 0.5m*1rad/s = 0.5m/s.
      dict(
          qpos=[0., np.pi],  # Pole pointing downwards.
          qvel=[1., 1.],
          expected_linvel=[0.5, 0., 0.],
          expected_angvel=[0., 1., 0.],
      ),
      # In the site's local frame, which is now flipped w.r.t the world, the
      # velocity is in the negative x direction.
      dict(
          qpos=[0., np.pi],  # Pole pointing downwards.
          qvel=[1., 1.],
          expected_linvel=[-0.5, 0., 0.],
          expected_angvel=[0., 1., 0.],
          local=True,
      ),
  )
  def testObjectVelocity(
      self, qpos, qvel, expected_linvel, expected_angvel, local=False):
    cartpole = """
    <mujoco>
      <worldbody>
        <body name='cart'>
          <joint type='slide' axis='1 0 0'/>
          <geom name='cart' type='box' size='0.2 0.2 0.2'/>
          <body name='pole'>
            <joint name='hinge' type='hinge' axis='0 1 0'/>
            <geom name='mass' pos='0 0 .5' size='0.04'/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = core.MjModel.from_xml_string(cartpole)
    data = core.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_step1(model.ptr, data.ptr)
    linvel, angvel = data.object_velocity("mass", "geom", local_frame=local)
    np.testing.assert_array_almost_equal(linvel, expected_linvel)
    np.testing.assert_array_almost_equal(angvel, expected_angvel)

  def testContactForce(self):
    box_on_floor = """
    <mujoco>
      <worldbody>
        <geom name='floor' type='plane' size='1 1 1'/>
        <body name='box' pos='0 0 .1'>
          <freejoint/>
          <geom name='box' type='box' size='.1 .1 .1'/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = core.MjModel.from_xml_string(box_on_floor)
    data = core.MjData(model)
    # Settle for 500 timesteps (1 second):
    for _ in range(500):
      mujoco.mj_step(model.ptr, data.ptr)
    normal_force = 0.
    for contact_id in range(data.ncon):
      force = data.contact_force(contact_id)
      normal_force += force[0, 0]
    box_id = 1
    box_weight = -model.opt.gravity[2]*model.body_mass[box_id]
    self.assertAlmostEqual(normal_force, box_weight)
    # Test raising of out-of-range errors:
    bad_ids = [-1, data.ncon]
    for bad_id in bad_ids:
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          core._CONTACT_ID_OUT_OF_RANGE.format(
              max_valid=data.ncon - 1, actual=bad_id)):
        data.contact_force(bad_id)

  @parameterized.parameters(
      dict(
          condim=3,  # Only sliding friction.
          expected_torques=[False, False, False],  # No torques.
      ),
      dict(
          condim=4,  # Sliding and torsional friction.
          expected_torques=[True, False, False],  # Only torsional torque.
      ),
      dict(
          condim=6,  # Sliding, torsional and rolling.
          expected_torques=[True, True, True],  # All torques are nonzero.
      ),
  )
  def testContactTorque(self, condim, expected_torques):
    ball_on_floor = """
    <mujoco>
      <worldbody>
        <geom name='floor' type='plane' size='1 1 1'/>
        <body name='ball' pos='0 0 .1'>
          <freejoint/>
          <geom name='ball' size='.1' friction='1 .1 .1'/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = core.MjModel.from_xml_string(ball_on_floor)
    data = core.MjData(model)
    model.geom_condim[:] = condim
    data.qvel[3:] = np.array((1., 1., 1.))
    # Settle for 10 timesteps (20 milliseconds):
    for _ in range(10):
      mujoco.mj_step(model.ptr, data.ptr)
    contact_id = 0  # This model has only one contact.
    _, torque = data.contact_force(contact_id)
    nonzero_torques = torque != 0
    np.testing.assert_array_equal(nonzero_torques, np.array((expected_torques)))

  def testFreeMjrContext(self):
    for _ in range(5):
      renderer = _render.Renderer(640, 480)
      mjr_context = core.MjrContext(self.model, renderer)
      # Explicit freeing should not break any automatic GC triggered later.
      del mjr_context
      renderer.free()
      del renderer
      gc.collect()

  def testSceneGeomsAttribute(self):
    scene = core.MjvScene(model=self.model)
    self.assertEqual(scene.ngeom, 0)
    self.assertEmpty(scene.geoms)
    geom_types = (
        mujoco.mjtObj.mjOBJ_BODY,
        mujoco.mjtObj.mjOBJ_GEOM,
        mujoco.mjtObj.mjOBJ_SITE,
    )
    for geom_type in geom_types:
      scene.ngeom += 1
      scene.geoms[scene.ngeom - 1].objtype = geom_type
    self.assertLen(scene.geoms, len(geom_types))
    self.assertEqual(tuple(g.objtype for g in scene.geoms), geom_types)

  def testInvalidFontScale(self):
    invalid_font_scale = 99
    with self.assertRaises(ValueError):
      core.MjrContext(model=self.model,
                      gl_context=None,  # Don't need a context for this test.
                      font_scale=invalid_font_scale)


def _get_attributes_test_params():
  model = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
  data = core.MjData(model)
  # Get the names of the non-private attributes of model and data through
  # introspection. These are passed as parameters to each of the test methods
  # in AttributesTest.
  array_args = []
  scalar_args = []
  skipped_args = []
  for parent_name, parent_obj in zip(("model", "data"),
                                     (model._model, data._data)):
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
    _ = attr[:]

    # Write unique values into the array and read them back.
    self._write_unique_values(attr_name, attr)
    self._take_steps()  # Take a few steps, check that we don't get segfaults.

  def _write_unique_values(self, attr_name, target_array):
    # If the target array is structured, recursively write unique values into
    # each subfield.
    if target_array.dtype.fields is not None:
      for field_name in target_array.dtype.fields:
        self._write_unique_values(attr_name, target_array[field_name])
    # Don't write to integer arrays since these might contain pointers. Also
    # don't write directly into the stack.
    elif (attr_name != "stack"
          and not np.issubdtype(target_array.dtype, np.integer)):
      new_contents = np.arange(target_array.size, dtype=target_array.dtype)
      new_contents.shape = target_array.shape
      target_array[:] = new_contents
      np.testing.assert_array_equal(new_contents, target_array[:])

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
    super().setUp()
    self.model = core.MjModel.from_xml_path(HUMANOID_XML_PATH)
    self.data = core.MjData(self.model)

  def _take_steps(self, n=5):
    for _ in range(n):
      mujoco.mj_step(self.model.ptr, self.data.ptr)


if __name__ == "__main__":
  absltest.main()
