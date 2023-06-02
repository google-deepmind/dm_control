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

import os

from dm_control.mujoco.wrapper import util
from dm_control.suite import common
from dm_control.utils import io as resources
from dm_control.utils import xml_tools

import numpy as np

from scipy.spatial.transform import Rotation as Rot


from muscles import lateral, flexors_back, flexors_front, \
    extensors_back, extensors_front

muscle_legs = extensors_back + extensors_front + \
    flexors_back + flexors_front


def array_to_string(array):
    return ' '.join(['%8g' % num for num in array])


def calculate_transformation(element):
    def get_rotation(element):
        if 'quat' in element.keys():
            r = Rot.from_quat(element.get('quat'))
        elif 'euler' in element.keys():
            r = Rot.from_euler('xyz', element.get('euler'))
        elif 'axisangle' in element.keys():
            raise NotImplementedError
        elif 'xyaxes' in element.keys():
            raise NotImplementedError
        elif 'zaxis' in element.keys():
            raise NotImplementedError
        else:
            r = Rot.identity()
        return r.as_matrix()

    # Calculate all transformation matrices from root until this element
    all_transformations = []
    while element.keys():
        if "pos" in element.keys():
            pos = np.array(element.get('pos').split(), dtype=float)
            rot = get_rotation(element)
            all_transformations.append(
                np.vstack((np.hstack((rot, pos.reshape([-1, 1]))), np.array([0, 0, 0, 1])))
            )
        element = element.getparent()

    # Apply all transformations
    T = np.eye(4)
    for transformation in reversed(all_transformations):
        T = np.matmul(T, transformation)

    return T


def get_model_and_assets(model_name, include_names={}):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        './')
    print(models_dir)

    xml_path = os.path.join(models_dir, f'{model_name}.xml')

    model = resources.GetResource(xml_path)

    assets = common.ASSETS.copy()
    meshes_dir = os.path.join(models_dir, 'dog_assets')
    _, _, filenames = next(resources.WalkResources(meshes_dir))
    for filename in filenames:
        file_path = os.path.join(meshes_dir, filename)
        assets[filename] = resources.GetResource(file_path)

    for k, v in include_names.items():
        file_path = os.path.join(models_dir, f'{v}.xml')
        assets[k] = resources.GetResource(file_path)

    return model, assets


def export(mjcf_model, out_dir, save_meshes, meshdir, out_file_name=None):
    if out_file_name is None:
        out_file_name = mjcf_model.model + '.xml'
    elif not out_file_name.lower().endswith('.xml'):
        raise ValueError('If `out_file_name` is specified it must end with '
                         '\'.xml\': got {}'.format(out_file_name))

    assets = mjcf_model.get_assets()
    new_dict = {}
    for k, v in assets.items():
        new_k = save_meshes + k
        new_dict[new_k] = v

    xml_string = mjcf_model.to_xml_string()
    index = xml_string.find('compiler') + len("compiler")
    if bool(meshdir):
        xml_string = xml_string[:index] + " meshdir=\"" + \
            meshdir + '\" ' + xml_string[index:]

    xml_string = xml_string.replace('general', 'muscle')

    # remove extra default class='/'
    start = xml_string.find('<default')
    end = xml_string.find('</default>')
    xml_string = xml_string.replace(xml_string[start: end + len('</default>')], '')

    new_dict[out_dir + out_file_name] = xml_string

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path, contents in new_dict.items():
        with open(path, 'wb') as f:
            f.write(util.to_binary_string(contents))


def slices2paths(mtu, slices, muscle_length):
    paths = []
    area = -1
    for cross_section in slices:
        if cross_section is not None:
            try:
                tmp_area = cross_section.area
            except Exception as e:
                tmp_area = -1
            
            if tmp_area > area:
                area = tmp_area

            path = cross_section.to_3D()

            v = np.array(path.vertices)
            x_min = np.min(v[:, 0])
            x_max = np.max(v[:, 0])

            z_min = np.min(v[:, 2])
            z_max = np.max(v[:, 2])

            y_min = np.min(v[:, 1])
            y_max = np.max(v[:, 1])

            if mtu in lateral:
                test = [path.centroid,
                np.array([path.centroid[0], path.centroid[1], z_min]),
                np.array([path.centroid[0], path.centroid[1], z_max]),
                np.array([path.centroid[0], y_max, path.centroid[2]]),
                np.array([path.centroid[0], y_min, path.centroid[2]])]
            else:
                test = [path.centroid,
                np.array([x_max, path.centroid[1], path.centroid[2]]),
                np.array([x_min, path.centroid[1], path.centroid[2]]),
                np.array([path.centroid[0], y_max, path.centroid[2]]),
                np.array([path.centroid[0], y_min, path.centroid[2]])]
            paths.append(test)

    const = round(muscle_length * 100)
    if len(paths) > const:
        indices = np.linspace(start=0,
            stop=len(paths) - 1,
            num=const,
            dtype=int)

    tmp = []
    for idx in indices:
        tmp.append(paths[idx])

    paths = tmp
    assert area != -1
    return paths, area


def getClosestGeom(kd_tree, point, mjcf, mtu):
    dist = 100000000000
    closest_geom_name = 0
    for bone, tree in kd_tree.items():
        dist_new, i = tree.query(np.array([point]), k=1)
        body = xml_tools.find_element(mjcf, 'geom', bone).getparent()
        if mtu in muscle_legs:
            if body.attrib["name"] not in ["upper_leg_L",
                    "upper_leg_R", "lower_leg_L", "lower_leg_R",
                    "foot_L", "foot_R", "toe_L", "toe_R",
                    "scapula_L", "scapula_R", "upper_arm_L",
                    "upper_arm_R", "lower_arm_L", "lower_arm_R",
                    "hand_L", "hand_R", "finger_L", "finger_R",
                    "pelvis"]:
                continue
        else:
            if body.attrib["name"] in ["upper_leg_L",
                    "upper_leg_R", "lower_leg_L", "lower_leg_R",
                    "foot_L", "foot_R", "toe_L", "toe_R",
                    "lower_arm_L", "lower_arm_R",
                    "hand_L", "hand_R", "finger_L", "finger_R"]:
                continue

        if dist_new < dist:
            dist = dist_new
            closest_geom_name = bone
    return closest_geom_name
