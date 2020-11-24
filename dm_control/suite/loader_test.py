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

"""Tests for the dm_control.suite loader."""

from absl.testing import absltest
from dm_control import suite
from dm_control.rl import control


class LoaderTest(absltest.TestCase):

  def test_load_without_kwargs(self):
    env = suite.load('cartpole', 'swingup')
    self.assertIsInstance(env, control.Environment)

  def test_load_with_kwargs(self):
    env = suite.load('cartpole', 'swingup',
                     task_kwargs={'time_limit': 40, 'random': 99})
    self.assertIsInstance(env, control.Environment)


class LoaderConstantsTest(absltest.TestCase):

  def testSuiteConstants(self):
    self.assertNotEmpty(suite.BENCHMARKING)
    self.assertNotEmpty(suite.EASY)
    self.assertNotEmpty(suite.HARD)
    self.assertNotEmpty(suite.EXTRA)


if __name__ == '__main__':
  absltest.main()
