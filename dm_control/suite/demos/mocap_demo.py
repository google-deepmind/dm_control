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

"""Demonstration of amc parsing for CMU mocap database.

To run the demo, supply a path to a `.amc` file:

    python mocap_demo --filename='path/to/mocap.amc'

CMU motion capture clips are available at mocap.cs.cmu.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Internal dependencies.

from absl import app
from absl import flags

from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('filename', None, 'amc file to be converted.')
flags.DEFINE_integer('max_num_frames', 90,
                     'Maximum number of frames for plotting/playback')


def main(unused_argv):
  env = humanoid_CMU.stand()

  # Parse and convert specified clip.
  converted = parse_amc.convert(FLAGS.filename,
                                env.physics, env.control_timestep())

  max_frame = min(FLAGS.max_num_frames, converted.qpos.shape[1] - 1)

  width = 480
  height = 480
  video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

  for i in range(max_frame):
    p_i = converted.qpos[:, i]
    with env.physics.reset_context():
      env.physics.data.qpos[:] = p_i
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

  tic = time.time()
  for i in range(max_frame):
    if i == 0:
      img = plt.imshow(video[i])
    else:
      img.set_data(video[i])
    toc = time.time()
    clock_dt = toc - tic
    tic = time.time()
    # Real-time playback not always possible as clock_dt > .03
    plt.pause(np.max(0.01, .03 - clock_dt))  # Need min display time > 0.0.
    plt.draw()
  plt.waitforbuttonpress()


if __name__ == '__main__':
  flags.mark_flag_as_required('filename')
  app.run(main)
