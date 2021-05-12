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

import functools
from absl import app
from absl import flags
from dm_control import viewer
from dm_control.locomotion import soccer

FLAGS = flags.FLAGS

flags.DEFINE_enum("walker_type", "BOXHEAD", ["BOXHEAD", "ANT", "HUMANOID"],
                  "The type of walker to explore with.")
flags.DEFINE_bool(
    "enable_field_box", True,
    "If `True`, enable physical bounding box enclosing the ball"
    " (but not the players).")
flags.DEFINE_bool("disable_walker_contacts", False,
                  "If `True`, disable walker-walker contacts.")
flags.DEFINE_bool(
    "terminate_on_goal", False,
    "If `True`, the episode terminates upon a goal being scored.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  viewer.launch(
      environment_loader=functools.partial(
          soccer.load,
          team_size=2,
          walker_type=soccer.WalkerType[FLAGS.walker_type],
          disable_walker_contacts=FLAGS.disable_walker_contacts,
          enable_field_box=FLAGS.enable_field_box,
          keep_aspect_ratio=True,
          terminate_on_goal=FLAGS.terminate_on_goal))


if __name__ == "__main__":
  app.run(main)
