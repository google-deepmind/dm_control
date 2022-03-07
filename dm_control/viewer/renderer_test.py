# Copyright 2018 The dm_control Authors.
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
"""Tests of the renderer module."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import types
from dm_control.viewer import renderer
import mock
import numpy as np


renderer.mujoco = mock.MagicMock()

_SCREEN_SIZE = types.MJRRECT(0, 0, 320, 240)


class BaseRendererTest(absltest.TestCase):

  class MockRenderer(renderer.BaseRenderer):
    pass

  class MockRenderComponent(renderer.Component):

    counter = 0

    def __init__(self):
      self._call_order = -1

    def render(self, context, viewport):
      self._call_order = BaseRendererTest.MockRenderComponent.counter
      BaseRendererTest.MockRenderComponent.counter += 1

    @property
    def call_order(self):
      return self._call_order

  def setUp(self):
    super().setUp()
    self.renderer = BaseRendererTest.MockRenderer()
    self.context = mock.MagicMock()
    self.viewport = mock.MagicMock()

  def test_rendering_components(self):
    regular_component = BaseRendererTest.MockRenderComponent()
    screen_capture_component = BaseRendererTest.MockRenderComponent()
    self.renderer.components += regular_component
    self.renderer.screen_capture_components += screen_capture_component
    self.renderer._render_components(self.context, self.viewport)
    self.assertEqual(0, regular_component.call_order)
    self.assertEqual(1, screen_capture_component.call_order)


@absltest.skip('b/222664582')
class OffScreenRendererTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = mock.MagicMock()
    self.model.vis.global_.offwidth = _SCREEN_SIZE.width
    self.model.vis.global_.offheight = _SCREEN_SIZE.height

    self.surface = mock.MagicMock()
    self.renderer = renderer.OffScreenRenderer(self.model, self.surface)
    self.renderer._mujoco_context = mock.MagicMock()

    self.viewport = mock.MagicMock()
    self.scene = mock.MagicMock()

    self.viewport.width = 3
    self.viewport.height = 3
    self.viewport.dimensions = np.array([3, 3])

  def test_render_context_initialization(self):
    self.renderer._mujoco_context = None
    self.renderer.render(self.viewport, self.scene)
    self.assertIsNotNone(self.renderer._mujoco_context)

  def test_resizing_pixel_buffer_to_viewport_size(self):
    self.renderer.render(self.viewport, self.scene)
    self.assertEqual((self.viewport.width, self.viewport.height, 3),
                     self.renderer._rgb_buffer.shape)

  def test_rendering_components(self):
    regular_component = mock.MagicMock()
    screen_capture_components = mock.MagicMock()
    self.renderer.components += [regular_component]
    self.renderer.screen_capture_components += [screen_capture_components]
    self.renderer._render_on_gl_thread(self.viewport, self.scene)
    regular_component.render.assert_called_once()
    screen_capture_components.render.assert_called_once()


@absltest.skip('b/222664582')
class PerturbationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = mock.MagicMock()
    self.data = mock.MagicMock()
    self.scene = mock.MagicMock()
    self.valid_pos = np.array([1, 2, 3])

    self.body_id = 0
    self.data.xpos = [np.array([0, 1, 2])]
    self.data.xmat = [np.identity(3)]

    self.perturbation = renderer.Perturbation(
        self.body_id, self.model, self.data, self.scene)

    renderer.mujoco.reset_mock()

  def test_start_params_validation(self):
    self.perturbation.start_move(None, self.valid_pos)
    self.assertEqual(0, renderer.mujoco.mjv_initPerturb.call_count)
    self.assertEqual(enums.mjtMouse.mjMOUSE_NONE, self.perturbation._action)

    self.perturbation.start_move(enums.mjtMouse.mjMOUSE_MOVE_V, None)
    self.assertEqual(0, renderer.mujoco.mjv_initPerturb.call_count)
    self.assertEqual(enums.mjtMouse.mjMOUSE_NONE, self.perturbation._action)

  def test_starting_an_operation(self):
    self.perturbation.start_move(enums.mjtMouse.mjMOUSE_MOVE_V, self.valid_pos)
    renderer.mujoco.mjv_initPerturb.assert_called_once()
    self.assertEqual(enums.mjtMouse.mjMOUSE_MOVE_V, self.perturbation._action)

  def test_starting_translation(self):
    self.perturbation.start_move(enums.mjtMouse.mjMOUSE_MOVE_V, self.valid_pos)
    self.assertEqual(
        enums.mjtPertBit.mjPERT_TRANSLATE, self.perturbation._perturb.active)

  def test_starting_rotation(self):
    self.perturbation.start_move(enums.mjtMouse.mjMOUSE_ROTATE_V,
                                 self.valid_pos)
    self.assertEqual(
        enums.mjtPertBit.mjPERT_ROTATE, self.perturbation._perturb.active)

  def test_starting_grip_transform(self):
    self.perturbation.start_move(enums.mjtMouse.mjMOUSE_MOVE_V, self.valid_pos)
    np.testing.assert_array_equal(
        [1, 1, 1], self.perturbation._perturb.localpos)

  def test_ticking_operation(self):
    self.perturbation._action = enums.mjtMouse.mjMOUSE_MOVE_V
    self.perturbation.tick_move([.1, .2])
    renderer.mujoco.mjv_movePerturb.assert_called_once()
    action, dx, dy = renderer.mujoco.mjv_movePerturb.call_args[0][2:5]
    self.assertEqual(self.perturbation._action, action)
    self.assertEqual(.1, dx)
    self.assertEqual(.2, dy)

  def test_ticking_stopped_operation_yields_no_results(self):
    self.perturbation._action = None
    self.perturbation.tick_move([.1, .2])
    self.assertEqual(0, renderer.mujoco.mjv_movePerturb.call_count)

    self.perturbation._action = enums.mjtMouse.mjMOUSE_NONE
    self.perturbation.tick_move([.1, .2])
    self.assertEqual(0, renderer.mujoco.mjv_movePerturb.call_count)

  def test_stopping_operation(self):
    self.perturbation._action = enums.mjtMouse.mjMOUSE_MOVE_V
    self.perturbation._perturb.active = enums.mjtPertBit.mjPERT_TRANSLATE
    self.perturbation.end_move()
    self.assertEqual(enums.mjtMouse.mjMOUSE_NONE, self.perturbation._action)
    self.assertEqual(0, self.perturbation._perturb.active)

  def test_applying_operation_results_while_not_paused(self):
    with self.perturbation.apply(False):
      renderer.mujoco.mjv_applyPerturbPose.assert_called_once()
      self.assertEqual(0, renderer.mujoco.mjv_applyPerturbPose.call_args[0][3])
      renderer.mujoco.mjv_applyPerturbForce.assert_called_once()

  def test_applying_operation_results_while_paused(self):
    with self.perturbation.apply(True):
      renderer.mujoco.mjv_applyPerturbPose.assert_called_once()
      self.assertEqual(1, renderer.mujoco.mjv_applyPerturbPose.call_args[0][3])
      self.assertEqual(0, renderer.mujoco.mjv_applyPerturbForce.call_count)

  def test_clearing_applied_forces_after_appling_operation(self):
    self.data.xfrc_applied = np.zeros(1)
    with self.perturbation.apply(True):
      # At this point the simulation will calculate forces to apply and assign
      # them to a proper MjvData structure field, as we're doing below.
      self.data.xfrc_applied[self.body_id] = 1

    # While exiting, the context clears that information.
    self.assertEqual(0, self.data.xfrc_applied[self.body_id])


@absltest.skip('b/222664582')
class RenderSettingsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.settings = renderer.RenderSettings()
    self.scene = wrapper.MjvScene()

  def test_applying_settings(self):
    self.settings._stereo_mode = 5
    self.settings._render_flags[:] = np.arange(len(self.settings._render_flags))
    self.settings.apply_settings(self.scene)
    self.assertEqual(self.settings._stereo_mode, self.scene.stereo)
    np.testing.assert_array_equal(self.settings._render_flags, self.scene.flags)

  def test_toggle_rendering_flag(self):
    self.settings._render_flags[0] = 1
    self.settings.toggle_rendering_flag(0)
    self.assertEqual(0, self.settings._render_flags[0])
    self.settings.toggle_rendering_flag(0)
    self.assertEqual(1, self.settings._render_flags[0])

  def test_toggle_visualization_flag(self):
    self.settings._visualization_options.flags[0] = 1
    self.settings.toggle_visualization_flag(0)
    self.assertEqual(0, self.settings._visualization_options.flags[0])
    self.settings.toggle_visualization_flag(0)
    self.assertEqual(1, self.settings._visualization_options.flags[0])

  def test_toggle_geom_group(self):
    self.settings._visualization_options.geomgroup[0] = 1
    self.settings.toggle_geom_group(0)
    self.assertEqual(0, self.settings._visualization_options.geomgroup[0])
    self.settings.toggle_geom_group(0)
    self.assertEqual(1, self.settings._visualization_options.geomgroup[0])

  def test_toggle_site_group(self):
    self.settings._visualization_options.sitegroup[0] = 1
    self.settings.toggle_site_group(0)
    self.assertEqual(0, self.settings._visualization_options.sitegroup[0])
    self.settings.toggle_site_group(0)
    self.assertEqual(1, self.settings._visualization_options.sitegroup[0])

  def test_toggle_stereo_buffering(self):
    self.settings.toggle_stereo_buffering()
    self.assertEqual(enums.mjtStereo.mjSTEREO_QUADBUFFERED,
                     self.settings._stereo_mode)
    self.settings.toggle_stereo_buffering()
    self.assertEqual(enums.mjtStereo.mjSTEREO_NONE,
                     self.settings._stereo_mode)

  def test_cycling_forward_through_render_modes(self):
    self.settings._visualization_options.frame = 0
    self.settings.select_next_rendering_mode()
    self.assertEqual(1, self.settings._visualization_options.frame)

    self.settings._visualization_options.frame = enums.mjtFrame.mjNFRAME - 1
    self.settings.select_next_rendering_mode()
    self.assertEqual(0, self.settings._visualization_options.frame)

  def test_cycling_backward_through_render_modes(self):
    self.settings._visualization_options.frame = 0
    self.settings.select_prev_rendering_mode()
    self.assertEqual(enums.mjtFrame.mjNFRAME - 1,
                     self.settings._visualization_options.frame)

    self.settings._visualization_options.frame = 1
    self.settings.select_prev_rendering_mode()
    self.assertEqual(0, self.settings._visualization_options.frame)

  def test_cycling_forward_through_labeling_modes(self):
    self.settings._visualization_options.label = 0
    self.settings.select_next_labeling_mode()
    self.assertEqual(1, self.settings._visualization_options.label)

    self.settings._visualization_options.label = enums.mjtLabel.mjNLABEL - 1
    self.settings.select_next_labeling_mode()
    self.assertEqual(0, self.settings._visualization_options.label)

  def test_cycling_backward_through_labeling_modes(self):
    self.settings._visualization_options.label = 0
    self.settings.select_prev_labeling_mode()
    self.assertEqual(enums.mjtLabel.mjNLABEL - 1,
                     self.settings._visualization_options.label)

    self.settings._visualization_options.label = 1
    self.settings.select_prev_labeling_mode()
    self.assertEqual(0, self.settings._visualization_options.label)


@absltest.skip('b/222664582')
class SceneCameraTest(parameterized.TestCase):

  @mock.patch.object(renderer.wrapper.core,
                     '_estimate_max_renderable_geoms',
                     return_value=1000)
  @mock.patch.object(renderer.wrapper.core.mujoco, 'MjvScene')
  def setUp(self, mock_make_scene, _):
    super().setUp()
    self.model = mock.MagicMock()
    self.data = mock.MagicMock()
    self.options = mock.MagicMock()
    self.camera = renderer.SceneCamera(self.model, self.data, self.options)
    mock_make_scene.assert_called_once()

  def test_freelook_mode(self):
    self.camera.set_freelook_mode()
    self.assertEqual(-1, self.camera._camera.trackbodyid)
    self.assertEqual(-1, self.camera._camera.fixedcamid)
    self.assertEqual(enums.mjtCamera.mjCAMERA_FREE, self.camera._camera.type_)
    self.assertEqual('Free', self.camera.name)

  def test_tracking_mode(self):
    body_id = 5
    self.camera.set_tracking_mode(body_id)
    self.assertEqual(body_id, self.camera._camera.trackbodyid)
    self.assertEqual(-1, self.camera._camera.fixedcamid)
    self.assertEqual(enums.mjtCamera.mjCAMERA_TRACKING,
                     self.camera._camera.type_)

    self.model.id2name = mock.MagicMock(return_value='body_name')
    self.assertEqual('Tracking body "body_name"', self.camera.name)

  def test_fixed_mode(self):
    camera_id = 5
    self.camera.set_fixed_mode(camera_id)
    self.assertEqual(-1, self.camera._camera.trackbodyid)
    self.assertEqual(camera_id, self.camera._camera.fixedcamid)
    self.assertEqual(enums.mjtCamera.mjCAMERA_FIXED,
                     self.camera._camera.type_)

    self.model.id2name = mock.MagicMock(return_value='camera_name')
    self.assertEqual('camera_name', self.camera.name)

  def test_look_at(self):
    target_pos = [10, 20, 30]
    distance = 5.
    self.camera.look_at(target_pos, distance)
    np.testing.assert_array_equal(target_pos, self.camera._camera.lookat)
    np.testing.assert_array_equal(distance, self.camera._camera.distance)

  def test_moving_camera(self):
    action = enums.mjtMouse.mjMOUSE_MOVE_V
    offset = [0.1, -0.2]
    with mock.patch(renderer.__name__ + '.mujoco') as mock_mujoco:
      self.camera.move(action, offset)
      mock_mujoco.mjv_moveCamera.assert_called_once()

  def test_zoom_to_scene(self):
    scene_center = np.array([1, 2, 3])
    scene_extents = np.array([10, 20, 30])

    self.camera.look_at = mock.MagicMock()
    self.model.stat = mock.MagicMock()
    self.model.stat.center = scene_center
    self.model.stat.extent = scene_extents

    self.camera.zoom_to_scene()
    self.camera.look_at.assert_called_once()
    np.testing.assert_array_equal(
        scene_center, self.camera.look_at.call_args[0][0])
    np.testing.assert_array_equal(
        scene_extents * 1.5, self.camera.look_at.call_args[0][1])

  def test_camera_transform(self):
    self.camera._scene.camera[0].up[:] = [0, 1, 0]
    self.camera._scene.camera[0].forward[:] = [0, 0, 1]
    self.camera._scene.camera[0].pos[:] = [5, 0, 0]
    self.camera._scene.camera[1].pos[:] = [10, 0, 0]

    rotation_mtx, position = self.camera.transform
    np.testing.assert_array_equal([-1, 0, 0], rotation_mtx[0])
    np.testing.assert_array_equal([0, 1, 0], rotation_mtx[1])
    np.testing.assert_array_equal([0, 0, 1], rotation_mtx[2])
    np.testing.assert_array_equal([7.5, 0, 0], position)

  @parameterized.parameters(
      (0, 0, False),
      (0, 1, False),
      (1, 0, False),
      (2, 1, False),
      (1, 2, True))
  def test_is_camera_initialized(self, frustum_near, frustum_far, result):
    gl_camera = mock.MagicMock()
    self.camera._scene = mock.MagicMock()
    self.camera._scene.camera = [gl_camera]

    gl_camera.frustum_near = frustum_near
    gl_camera.frustum_far = frustum_far
    self.assertEqual(result, self.camera.is_initialized)


@absltest.skip('b/222664582')
class RaycastsTest(absltest.TestCase):

  @mock.patch.object(renderer.wrapper.core,
                     '_estimate_max_renderable_geoms',
                     return_value=1000)
  @mock.patch.object(renderer.wrapper.core.mujoco, 'MjvScene')
  def setUp(self, mock_make_scene, _):
    super().setUp()
    self.model = mock.MagicMock()
    self.data = mock.MagicMock()
    self.options = mock.MagicMock()

    self.viewport = mock.MagicMock()
    self.camera = renderer.SceneCamera(self.model, self.data, self.options)
    mock_make_scene.assert_called_once()
    self.initialize_camera(True)

  def initialize_camera(self, enable):
    gl_camera = mock.MagicMock()
    self.camera._scene = mock.MagicMock()
    self.camera._scene.camera = [gl_camera]
    gl_camera.frustum_near = 1 if enable else 0
    gl_camera.frustum_far = 2 if enable else 0

  def test_raycast_mapping_geom_to_body_id(self):
    def build_mjv_select(mock_body_id, mock_geom_id, mock_position):
      def mock_select(
          m, d, vopt, aspectratio, relx, rely, scn, selpnt, geomid, skinid):
        del m, d, vopt, aspectratio, relx, rely, scn, skinid  # Unused.
        selpnt[:] = mock_position
        geomid[:] = mock_geom_id
        return mock_body_id
      return mock_select

    geom_id = 0
    body_id = 5
    world_pos = [1, 2, 3]
    self.model.geom_bodyid = np.zeros(10)
    self.model.geom_bodyid[geom_id] = body_id
    mock_select = build_mjv_select(body_id, geom_id, world_pos)

    with mock.patch(renderer.__name__ + '.mujoco') as mock_mujoco:
      mock_mujoco.mjv_select = mock.MagicMock(side_effect=mock_select)
      hit_body_id, hit_world_pos = self.camera.raycast(self.viewport, [0, 0])
      self.assertEqual(hit_body_id, body_id)
      np.testing.assert_array_equal(hit_world_pos, world_pos)

  def test_raycast_hitting_empty_space(self):
    def mock_select(
        m, d, vopt, aspectratio, relx, rely, scn, selpnt, geomid, skinid):
      del (m, d, vopt, aspectratio, relx, rely, scn, selpnt, geomid,
           skinid)  # Unused.
      mock_body_id = -1  # Nothing selected.
      return mock_body_id

    with mock.patch(renderer.__name__ + '.mujoco') as mock_mujoco:
      mock_mujoco.mjv_select = mock.MagicMock(side_effect=mock_select)
      hit_body_id, hit_world_pos = self.camera.raycast(self.viewport, [0, 0])
      self.assertEqual(-1, hit_body_id)
      self.assertIsNone(hit_world_pos)

  def test_raycast_maps_coordinates_to_viewport_space(self):
    def build_mjv_select(expected_aspect_ratio, expected_viewport_pos):
      def mock_select(
          m, d, vopt, aspectratio, relx, rely, scn, selpnt, geomid, skinid):
        del m, d, vopt, scn, selpnt, geomid, skinid  # Unused.
        self.assertEqual(expected_aspect_ratio, aspectratio)
        np.testing.assert_array_equal(expected_viewport_pos, [relx, rely])
        mock_body_id = 0
        return mock_body_id
      return mock_select

    viewport_pos = [.5, .5]
    self.viewport.screen_to_inverse_viewport.return_value = viewport_pos
    mock_select = build_mjv_select(self.viewport.aspect_ratio, viewport_pos)

    with mock.patch(renderer.__name__ + '.mujoco') as mock_mujoco:
      mock_mujoco.mjv_select = mock.MagicMock(side_effect=mock_select)
      self.camera.raycast(self.viewport, [50, 25])

  def test_raycasts_disabled_when_camera_is_not_initialized(self):
    self.initialize_camera(False)
    hit_body_id, hit_world_pos = self.camera.raycast(self.viewport, [0, 0])
    self.assertEqual(-1, hit_body_id)
    self.assertIsNone(hit_world_pos)


class ViewportTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.viewport = renderer.Viewport()
    self.viewport.set_size(100, 100)

  @parameterized.parameters(
      ([0, 0], [0., 0.]),
      ([100, 0], [1., 0.]),
      ([0, 100], [0., 1.]),
      ([50, 50], [.5, .5]))
  def test_screen_to_viewport(self, screen_coords, viewport_coords):
    np.testing.assert_array_equal(
        viewport_coords, self.viewport.screen_to_viewport(screen_coords))

  @parameterized.parameters(
      ([0, 0], [0., 1.]),
      ([100, 0], [1., 1.]),
      ([0, 100], [0., 0.]),
      ([50, 50], [.5, .5]))
  def test_screen_to_inverse_viewport(self, screen_coords, viewport_coords):
    np.testing.assert_array_equal(
        viewport_coords,
        self.viewport.screen_to_inverse_viewport(screen_coords))

  @parameterized.parameters(
      ([10, 10], 1.),
      ([30, 40], 3./4.))
  def test_aspect_ratio(self, screen_size, aspect_ratio):
    self.viewport.set_size(screen_size[0], screen_size[1])
    self.assertEqual(aspect_ratio, self.viewport.aspect_ratio)


if __name__ == '__main__':
  absltest.main()
