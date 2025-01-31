# Copyright 2023 The dm_control Authors.
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


import numpy as np

from dm_control.locomotion.walkers.assets.dog_v2 import muscles as muscles_constants


def array_to_string(array):
    """Converts an array of numbers to a string representation.

    Args:
        array : The array of numbers to be converted.

    Returns:
        str: The string representation of the array.

    """
    return " ".join(["%8g" % num for num in array])


def slices2paths(mtu, slices, muscle_length):
    """Convert slices to paths.

    Args:
        mtu : The name of the muscle-tendon unit.
        slices : List of cross-section slices.
        muscle_length : The length of the muscle.

    Returns:
        list: List of converted paths.
    """
    paths = []
    for cross_section in slices:
        if cross_section is not None:
            path = cross_section.to_3D()

            v = np.array(path.vertices)
            x_min = np.min(v[:, 0])
            x_max = np.max(v[:, 0])

            z_min = np.min(v[:, 2])
            z_max = np.max(v[:, 2])

            y_min = np.min(v[:, 1])
            y_max = np.max(v[:, 1])

            if mtu in muscles_constants.LATERAL:
                test = [
                    path.centroid,
                    np.array([path.centroid[0], path.centroid[1], z_min]),
                    np.array([path.centroid[0], path.centroid[1], z_max]),
                    np.array([path.centroid[0], y_max, path.centroid[2]]),
                    np.array([path.centroid[0], y_min, path.centroid[2]]),
                ]
            else:
                test = [
                    path.centroid,
                    np.array([x_max, path.centroid[1], path.centroid[2]]),
                    np.array([x_min, path.centroid[1], path.centroid[2]]),
                    np.array([path.centroid[0], y_max, path.centroid[2]]),
                    np.array([path.centroid[0], y_min, path.centroid[2]]),
                ]
            paths.append(test)

    const = round(muscle_length * 100)
    if len(paths) > const:
        indices = np.linspace(start=0, stop=len(paths) - 1, num=const, dtype=int)

    tmp = []
    for idx in indices:
        tmp.append(paths[idx])

    paths = tmp
    return paths
