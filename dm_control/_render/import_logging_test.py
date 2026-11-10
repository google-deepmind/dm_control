# Copyright 2026 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Regression tests for application logging when importing dm_control."""

import subprocess
import sys

from absl.testing import absltest


class ImportLoggingTest(absltest.TestCase):

  def test_mjcf_import_leaves_root_logger_unconfigured(self):
    script = """
import logging
assert not logging.root.handlers
from dm_control import mjcf
assert not logging.root.handlers, logging.root.handlers
logging.basicConfig(level=logging.INFO, format='APP LOG: %(message)s')
logging.info('configured after import')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False,
        capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("APP LOG: configured after import", result.stderr)

  def test_mjcf_import_preserves_existing_handler(self):
    script = """
import logging
import sys
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('BEFORE: %(message)s'))
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
from dm_control import mjcf
assert logging.root.handlers == [handler], logging.root.handlers
logging.info('handler still configured')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False,
        capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("BEFORE: handler still configured", result.stderr)


if __name__ == "__main__":
  absltest.main()
