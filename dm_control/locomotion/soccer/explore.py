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

"""Interactive viewer for MuJoCo soccer enviornmnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import app
from dm_control import viewer
from dm_control.locomotion import soccer


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  viewer.launch(environment_loader=functools.partial(soccer.load, team_size=2))


if __name__ == '__main__':
  app.run(main)
