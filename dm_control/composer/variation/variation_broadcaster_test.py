# Copyright 2024 The dm_control Authors.
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

from absl.testing import absltest
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.composer.variation import variation_broadcaster
import numpy as np


class VariationBroadcasterTest(absltest.TestCase):

  def test_can_generate_values(self):
    random_state = np.random.RandomState(2348)
    expected_values = [random_state.uniform(0, 1) for _ in range(5)]

    random_state = np.random.RandomState(2348)
    broadcaster = variation_broadcaster.VariationBroadcaster(
        distributions.Uniform(0, 1)
    )
    proxy_1 = broadcaster.get_proxy()
    proxy_2 = broadcaster.get_proxy()
    proxy_3 = broadcaster.get_proxy()

    self.assertEqual(
        variation.evaluate(proxy_1, random_state=random_state),
        expected_values[0],
    )
    self.assertEqual(
        variation.evaluate(proxy_2, random_state=random_state),
        expected_values[0],
    )
    self.assertEqual(
        variation.evaluate(proxy_3, random_state=random_state),
        expected_values[0],
    )

    self.assertEqual(
        variation.evaluate(proxy_1, random_state=random_state),
        expected_values[1],
    )
    self.assertEqual(
        variation.evaluate(proxy_1, random_state=random_state),
        expected_values[2],
    )

    self.assertEqual(
        variation.evaluate(proxy_2, random_state=random_state),
        expected_values[1],
    )
    self.assertEqual(
        variation.evaluate(proxy_3, random_state=random_state),
        expected_values[1],
    )
    self.assertEqual(
        variation.evaluate(proxy_3, random_state=random_state),
        expected_values[2],
    )

    self.assertEqual(
        variation.evaluate(proxy_3, random_state=random_state),
        expected_values[3],
    )
    self.assertEqual(
        variation.evaluate(proxy_1, random_state=random_state),
        expected_values[3],
    )
    self.assertEqual(
        variation.evaluate(proxy_2, random_state=random_state),
        expected_values[2],
    )

    self.assertEqual(
        variation.evaluate(proxy_1, random_state=random_state),
        expected_values[4],
    )
    self.assertEqual(
        variation.evaluate(proxy_2, random_state=random_state),
        expected_values[3],
    )
    self.assertEqual(
        variation.evaluate(proxy_2, random_state=random_state),
        expected_values[4],
    )
    self.assertEqual(
        variation.evaluate(proxy_3, random_state=random_state),
        expected_values[4],
    )


if __name__ == '__main__':
  absltest.main()
