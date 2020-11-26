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

"""Tests for dm_control.utils.containers."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.utils import containers


class TaggedTaskTest(parameterized.TestCase):

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

    self.assertLen(tasks, 4)
    self.assertEqual(set(['basic', 'expert', 'stable', 'unstable']),
                     set(tasks.tags()))

    self.assertLen(tasks.tagged('basic'), 1)
    self.assertLen(tasks.tagged('expert'), 2)
    self.assertLen(tasks.tagged('stable'), 2)
    self.assertLen(tasks.tagged('unstable'), 1)

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

  def test_override_behavior(self):
    tasks = containers.TaggedTasks(allow_overriding_keys=False)

    @tasks.add()
    def some_func():
      pass

    expected_message = containers._NAME_ALREADY_EXISTS.format(name='some_func')
    with self.assertRaisesWithLiteralMatch(ValueError, expected_message):
      tasks.add()(some_func)

    tasks.allow_overriding_keys = True
    tasks.add()(some_func)  # Override should now succeed.

  @parameterized.parameters(
      {'query': ['a'], 'expected_keys': frozenset(['f1', 'f2', 'f3'])},
      {'query': ['b', 'c'], 'expected_keys': frozenset(['f2'])},
      {'query': ['c'], 'expected_keys': frozenset(['f2', 'f3'])},
      {'query': ['b', 'd'], 'expected_keys': frozenset()},
      {'query': ['e'], 'expected_keys': frozenset()},
      {'query': [], 'expected_keys': frozenset()})
  def test_query_tag_intersection(self, query, expected_keys):
    tasks = containers.TaggedTasks()

    # pylint: disable=unused-variable
    @tasks.add('a', 'b')
    def f1():
      pass

    @tasks.add('a', 'b', 'c')
    def f2():
      pass

    @tasks.add('a', 'c', 'd')
    def f3():
      pass
    # pylint: enable=unused-variable

    result = tasks.tagged(*query)
    self.assertSetEqual(frozenset(result.keys()), expected_keys)


if __name__ == '__main__':
  absltest.main()
