"""Tests for the Dog model."""

import io
from absl import app
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.locomotion.walkers import dog
from unittest import mock
import sys
import os
# Add the directory containing build_dog.py to sys.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'assets/dog_v2')))

import build_dog  # nopep8


class DogTest(parameterized.TestCase):

  @parameterized.parameters([
      dog.Dog,
      dog.DogMuscleActuated,
  ])
  def test_can_compile_and_step_simulation(self, walker_type):
    walker = walker_type()
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    for _ in range(100):
      physics.step()

  @parameterized.parameters([
      ("torque_model", ['build_dog.py', '--make_skin=True', '--lengthrange_from_joints=False',
                        '--use_muscles=False', '--add_markers=True', '--muscle_strength_scale=-1',
                        '--muscle_dynamics=Sigmoid', '--make_muscles_skin=True',
                        '--lumbar_dofs_per_vertebra=3']),
      ("muscle_model", ['build_dog.py', '--make_skin=True', '--lengthrange_from_joints=True',
                        '--use_muscles=True', '--add_markers=True', '--muscle_strength_scale=-1',
                        '--muscle_dynamics=Sigmoid', '--make_muscles_skin=True',
                        '--lumbar_dofs_per_vertebra=3'])
  ])
  def test_dog_model_creation(self, name, testargs):
    with mock.patch('sys.argv', testargs), mock.patch('sys.exit') as mock_exit:
      # Store the original stdout
      original_stdout = sys.stdout

      try:
        # Redirect stdout to capture prints
        captured_output = io.StringIO()
        sys.stdout = captured_output
        app.run(build_dog.main)
        self.assertTrue(mock_exit.called)
      finally:
        # Restore the original stdout
        sys.stdout = original_stdout


if __name__ == '__main__':
  absltest.main()
