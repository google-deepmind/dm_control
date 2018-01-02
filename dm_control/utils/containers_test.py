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

"""Tests for control.utils.containers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest

from dm_control.utils import containers


class TaskTest(absltest.TestCase):

  def test_factory_registered(self):
    tasks = containers.Tasks()

    @tasks.add
    def test_factory1():  # pylint: disable=unused-variable
      return 'executed 1'

    @tasks.add
    def test_factory2():  # pylint: disable=unused-variable
      return 'executed 2'

    with self.assertRaises(ValueError):
      @tasks.add
      def test_factory1():  # pylint: disable=function-redefined
        return

    self.assertEqual(2, len(tasks))
    self.assertEqual(set(['test_factory1', 'test_factory2']),
                     set(tasks.keys()))
    self.assertEqual('executed 1', tasks['test_factory1']())
    self.assertEqual('executed 2', tasks['test_factory2']())

  def test_procedural_names(self):
    tasks = containers.Tasks()
    names = set(('easy', 'normal', 'hard'))
    for name in names:
      tasks.add(lambda: None, name=name)
    self.assertEqual(len(names), len(tasks))
    self.assertSetEqual(names, set(tasks.keys()))

  def test_iteration_order(self):
    tasks = containers.Tasks()
    expected_order = ['first', 'second', 'third', 'fourth']
    for name in expected_order:
      tasks.add(lambda: None, name=name)
    actual_order = list(tasks)
    self.assertEqual(expected_order, actual_order)


class TaggedTaskTest(absltest.TestCase):

  def test_registration(self):
    tasks = containers.TaggedTasks()

    @tasks.add()
    def test_factory1():  # pylint: disable=unused-variable
      return 'executed 1'

    @tasks.add('basic', 'stable')
    def test_factory2():  # pylint: disable=unused-variable
      return 'executed 2'

    @tasks.add('expert', 'stable')
    def test_factory3():  # pylint: disable=unused-variable
      return 'executed 3'

    @tasks.add('expert', 'unstable')
    def test_factory4():  # pylint: disable=unused-variable
      return 'executed 4'

    self.assertEqual(4, len(tasks))
    self.assertEqual(set(['basic', 'expert', 'stable', 'unstable']),
                     set(tasks.tags()))

    self.assertEqual(1, len(tasks.tagged('basic')))
    self.assertEqual(2, len(tasks.tagged('expert')))
    self.assertEqual(2, len(tasks.tagged('stable')))
    self.assertEqual(1, len(tasks.tagged('unstable')))

    self.assertEqual('executed 2', tasks['test_factory2']())

    self.assertEqual('executed 3', tasks.tagged('expert')['test_factory3']())

    self.assertNotIn('test_factory4', tasks.tagged('stable'))

  def test_iteration_order(self):
    tasks = containers.TaggedTasks()

    @tasks.add()
    def first():  # pylint: disable=unused-variable
      pass

    @tasks.add()
    def second():  # pylint: disable=unused-variable
      pass

    @tasks.add()
    def third():  # pylint: disable=unused-variable
      pass

    @tasks.add()
    def fourth():  # pylint: disable=unused-variable
      pass

    expected_order = ['first', 'second', 'third', 'fourth']
    actual_order = list(tasks)
    self.assertEqual(expected_order, actual_order)

if __name__ == '__main__':
  absltest.main()
