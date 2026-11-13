# Copyright 2026 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for replacing task factories and their tags."""

from absl.testing import absltest
from dm_control.utils import containers


class TaggedTasksOverrideTest(absltest.TestCase):

  def test_override_removes_stale_tag_and_old_factory(self):
    tasks = containers.TaggedTasks(allow_overriding_keys=True)

    @tasks.add('old', 'shared')
    def example():
      return 'original'

    def replacement():
      return 'replacement'
    replacement.__name__ = 'example'
    tasks.add('new', 'shared')(replacement)

    self.assertIs(tasks['example'], replacement)
    self.assertEmpty(tasks.tagged('old'))
    self.assertIs(tasks.tagged('new')['example'], replacement)
    self.assertIs(tasks.tagged('shared')['example'], replacement)
    self.assertNotIn('old', tasks.tags())
    self.assertCountEqual(tasks.tags(), ['new', 'shared'])

  def test_override_without_tags_removes_all_tag_memberships(self):
    tasks = containers.TaggedTasks(allow_overriding_keys=True)

    @tasks.add('first', 'second')
    def example():
      return 1

    def replacement():
      return 2

    replacement.__name__ = 'example'
    tasks.add()(replacement)

    self.assertEqual(tasks['example'](), 2)
    self.assertEmpty(tasks.tags())
    self.assertEmpty(tasks.tagged('first'))
    self.assertEmpty(tasks.tagged('second'))

  def test_override_does_not_remove_tags_of_other_tasks(self):
    tasks = containers.TaggedTasks(allow_overriding_keys=True)

    @tasks.add('shared', 'old')
    def first():
      return 1

    @tasks.add('shared')
    def second():
      return 2

    def replacement():
      return 3

    replacement.__name__ = 'first'
    tasks.add('new')(replacement)

    self.assertCountEqual(tasks.tagged('shared'), ['second'])
    self.assertCountEqual(tasks.tagged('new'), ['first'])
    self.assertCountEqual(tasks.tags(), ['new', 'shared'])
    self.assertEqual(tasks['first'](), 3)
    self.assertEqual(tasks['second'](), 2)

  def test_override_keeps_original_registration_position(self):
    tasks = containers.TaggedTasks(allow_overriding_keys=True)

    @tasks.add('first')
    def alpha():
      return 1

    @tasks.add('second')
    def beta():
      return 2

    def replacement():
      return 3

    replacement.__name__ = 'alpha'
    tasks.add('replacement')(replacement)

    self.assertEqual(list(tasks), ['alpha', 'beta'])
    self.assertEqual(tasks['alpha'](), 3)


if __name__ == '__main__':
  absltest.main()
