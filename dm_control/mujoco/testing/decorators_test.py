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

"""Tests of the decorators module."""

from absl.testing import absltest
from dm_control.mujoco.testing import decorators
import mock


class RunThreadedTest(absltest.TestCase):

  @mock.patch(decorators.__name__ + ".threading")
  def test_number_of_threads(self, mock_threading):
    num_threads = 5

    mock_threads = [mock.MagicMock() for _ in range(num_threads)]
    for thread in mock_threads:
      thread.start = mock.MagicMock()
      thread.join = mock.MagicMock()

    mock_threading.Thread = mock.MagicMock(side_effect=mock_threads)

    test_decorator = decorators.run_threaded(num_threads=num_threads)
    tested_method = mock.MagicMock()
    tested_method.__name__ = "foo"
    test_runner = test_decorator(tested_method)
    test_runner(self)

    for thread in mock_threads:
      thread.start.assert_called_once()
      thread.join.assert_called_once()

  def test_number_of_iterations(self):
    calls_per_thread = 5

    tested_method = mock.MagicMock()
    tested_method.__name__ = "foo"
    test_decorator = decorators.run_threaded(
        num_threads=1, calls_per_thread=calls_per_thread)
    test_runner = test_decorator(tested_method)
    test_runner(self)

    self.assertEqual(calls_per_thread, tested_method.call_count)

  @mock.patch(decorators.__name__ + ".threading")
  def test_using_the_main_thread(self, mock_threading):
    mock_thread = mock.MagicMock()
    mock_thread.start = mock.MagicMock()
    mock_thread.join = mock.MagicMock()
    mock_threading.current_thread = mock.MagicMock(return_value=mock_thread)

    test_decorator = decorators.run_threaded(num_threads=None,
                                             calls_per_thread=1)
    tested_method = mock.MagicMock()
    tested_method.__name__ = "foo"
    test_runner = test_decorator(tested_method)
    test_runner(self)

    tested_method.assert_called_once()
    mock_thread.start.assert_not_called()
    mock_thread.join.assert_not_called()


if __name__ == "__main__":
  absltest.main()
