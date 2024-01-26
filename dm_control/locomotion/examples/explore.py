# Copyright 2019 The dm_control Authors.
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

"""Simple script to launch viewer with an example environment."""

from absl import app

from dm_control.locomotion.examples import basic_cmu_2019
from dm_control import viewer


def main(unused_argv):
  viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_gaps)

if __name__ == '__main__':
  app.run(main)
