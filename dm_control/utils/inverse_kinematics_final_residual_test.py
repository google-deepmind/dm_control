"""IK result residuals and convergence describe the returned physical pose."""
import math

from absl.testing import absltest
from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
import numpy as np


_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01" mass="1"/>
      <site name="tip" size="0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


class FinalResidualTest(absltest.TestCase):

  def _result(self, target_pos=None, angle=None, inplace=False, **kwargs):
    physics = mujoco.Physics.from_xml_string(_XML)
    target_quat = None if angle is None else np.array([
        math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
    result = ik.qpos_from_site_pose(
        physics, 'tip', target_pos=target_pos, target_quat=target_quat,
        max_steps=1, tol=1e-10, inplace=inplace, **kwargs)
    if not inplace:
      np.testing.assert_array_equal(physics.data.qpos, np.zeros(4))
    physics.data.qpos[:] = result.qpos
    physics.forward()
    # Independent geometry: translation distance and signed planar angle.
    # Do not reuse the solver's quaternion residual implementation.
    actual_error = 0.
    if target_pos is not None:
      actual_error += np.linalg.norm(
          np.asarray(target_pos) - physics.named.data.site_xpos['tip'])
    if angle is not None:
      matrix = physics.named.data.site_xmat['tip'].reshape(3, 3)
      actual_angle = math.atan2(matrix[1, 0], matrix[0, 0])
      difference = math.atan2(math.sin(angle - actual_angle),
                              math.cos(angle - actual_angle))
      actual_error += abs(difference) * kwargs.get('rot_weight', 1.)
    self.assertAlmostEqual(float(result.err_norm), actual_error, places=12)
    self.assertEqual(bool(result.success), actual_error < 1e-10)
    return result, physics

  def test_last_permitted_translation_update_can_converge(self):
    for inplace in (False, True):
      with self.subTest(inplace=inplace):
        result, _ = self._result(target_pos=[.01, -.02, .03], inplace=inplace)
        np.testing.assert_allclose(result.qpos[:3], [.01, -.02, .03], atol=1e-12)
        self.assertTrue(result.success)

  def test_last_permitted_rotation_update_can_converge(self):
    for inplace in (False, True):
      with self.subTest(inplace=inplace):
        result, _ = self._result(angle=.04, inplace=inplace)
        self.assertAlmostEqual(result.qpos[3], .04, places=12)
        self.assertTrue(result.success)

  def test_combined_pose_uses_final_weighted_error(self):
    result, _ = self._result(target_pos=[.01, -.02, .03], angle=.04,
                             rot_weight=.7)
    self.assertTrue(result.success)

  def test_clipped_update_reports_remaining_error_without_success(self):
    result, _ = self._result(target_pos=[.02, 0., 0.], max_update_norm=.005)
    self.assertAlmostEqual(result.err_norm, .015, places=12)
    self.assertFalse(result.success)

  def test_regularized_update_reports_remaining_error_without_success(self):
    result, _ = self._result(target_pos=[.4, 0., 0.])
    self.assertAlmostEqual(result.qpos[0], .4 / 1.03, places=12)
    self.assertAlmostEqual(result.err_norm, .4 - .4 / 1.03, places=12)
    self.assertFalse(result.success)

  def test_already_converged_configuration_is_unchanged(self):
    result, _ = self._result(target_pos=[0., 0., 0.], angle=0.)
    np.testing.assert_array_equal(result.qpos, np.zeros(4))
    self.assertEqual(result.err_norm, 0.)
    self.assertTrue(result.success)

  def test_progress_termination_reports_unmodified_configuration(self):
    result, _ = self._result(target_pos=[.02, 0., 0.], progress_thresh=.5)
    np.testing.assert_array_equal(result.qpos, np.zeros(4))
    self.assertAlmostEqual(result.err_norm, .02, places=12)
    self.assertFalse(result.success)


if __name__ == '__main__':
  absltest.main()
