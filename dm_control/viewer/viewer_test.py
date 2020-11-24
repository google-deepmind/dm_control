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
"""Tests of the viewer.py module."""


from absl.testing import absltest
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.viewer import util
from dm_control.viewer import viewer
import mock


class ViewerTest(absltest.TestCase):

  def setUp(self):
    super(ViewerTest, self).setUp()
    self.viewport = mock.MagicMock()
    self.mouse = mock.MagicMock()
    self.keyboard = mock.MagicMock()
    self.viewer = viewer.Viewer(self.viewport, self.mouse, self.keyboard)

    self.viewer._render_settings = mock.MagicMock()
    self.physics = mock.MagicMock()
    self.renderer = mock.MagicMock()
    self.renderer.priority_components = util.QuietSet()

  def _extract_bind_call_args(self, bind_mock):
    call_args = []
    for calls in bind_mock.call_args_list:
      args = calls[0]
      if len(args) == 2:
        call_args.append(args[1])
    return call_args

  def test_initialize_creates_components(self):
    with mock.patch(viewer.__name__ + '.renderer'):
      self.viewer.initialize(self.physics, self.renderer, touchpad=False)
    self.assertIsNotNone(self.viewer._camera)
    self.assertIsNotNone(self.viewer._manipulator)
    self.assertIsNotNone(self.viewer._free_camera)
    self.assertIsNotNone(self.viewer._camera_select)
    self.assertEqual(self.renderer, self.viewer._renderer)

  def test_initialize_creates_touchpad_specific_input_mapping(self):
    self.viewer._input_map = mock.MagicMock()
    with mock.patch(viewer.__name__ + '.renderer'):
      self.viewer.initialize(self.physics, self.renderer, touchpad=True)
    call_args = self._extract_bind_call_args(self.viewer._input_map.bind)
    self.assertIn(viewer._MOVE_OBJECT_VERTICAL_TOUCHPAD, call_args)
    self.assertIn(viewer._MOVE_OBJECT_HORIZONTAL_TOUCHPAD, call_args)
    self.assertIn(viewer._ROTATE_OBJECT_TOUCHPAD, call_args)
    self.assertIn(viewer._PAN_CAMERA_VERTICAL_TOUCHPAD, call_args)
    self.assertIn(viewer._PAN_CAMERA_HORIZONTAL_TOUCHPAD, call_args)

  def test_initialize_create_mouse_specific_input_mapping(self):
    self.viewer._input_map = mock.MagicMock()
    with mock.patch(viewer.__name__ + '.renderer'):
      self.viewer.initialize(self.physics, self.renderer, touchpad=False)
    call_args = self._extract_bind_call_args(self.viewer._input_map.bind)
    self.assertIn(viewer._MOVE_OBJECT_VERTICAL_MOUSE, call_args)
    self.assertIn(viewer._MOVE_OBJECT_HORIZONTAL_MOUSE, call_args)
    self.assertIn(viewer._ROTATE_OBJECT_MOUSE, call_args)
    self.assertIn(viewer._PAN_CAMERA_VERTICAL_MOUSE, call_args)
    self.assertIn(viewer._PAN_CAMERA_HORIZONTAL_MOUSE, call_args)

  def test_initialization_flushes_old_input_map(self):
    self.viewer._input_map = mock.MagicMock()
    with mock.patch(viewer.__name__ + '.renderer'):
      self.viewer.initialize(self.physics, self.renderer, touchpad=False)
    self.viewer._input_map.clear_bindings.assert_called_once()

  def test_deinitialization_deletes_components(self):
    self.viewer._camera = mock.MagicMock()
    self.viewer._manipulator = mock.MagicMock()
    self.viewer._free_camera = mock.MagicMock()
    self.viewer._camera_select = mock.MagicMock()
    self.viewer._renderer = mock.MagicMock()
    self.viewer.deinitialize()
    self.assertIsNone(self.viewer._camera)
    self.assertIsNone(self.viewer._manipulator)
    self.assertIsNone(self.viewer._free_camera)
    self.assertIsNone(self.viewer._camera_select)
    self.assertIsNone(self.viewer._renderer)

  def test_deinitialization_flushes_old_input_map(self):
    self.viewer._input_map = mock.MagicMock()
    self.viewer.deinitialize()
    self.viewer._input_map.clear_bindings.assert_called_once()

  def test_rendering_uninitialized(self):
    self.viewer.render()  # nothing crashes

  def test_zoom_to_scene_uninitialized(self):
    self.viewer.zoom_to_scene()  # nothing crashes

  def test_rendering(self):
    self.viewer._camera = mock.MagicMock()
    self.viewer._renderer = mock.MagicMock()
    self.viewer.render()
    self.viewer._camera.render.assert_called_once_with(self.viewer.perturbation)
    self.viewer._renderer.render.assert_called_once()

  def test_applying_render_settings_before_rendering_a_scene(self):
    self.viewer._camera = mock.MagicMock()
    self.viewer._renderer = mock.MagicMock()
    self.viewer.render()
    self.viewer._render_settings.apply_settings.assert_called_once()

  def test_zoom_to_scene(self):
    self.viewer._camera = mock.MagicMock()
    self.viewer.zoom_to_scene()
    self.viewer._camera.zoom_to_scene.assert_called_once()

  def test_retrieving_perturbation(self):
    object_perturbation = mock.MagicMock()
    self.viewer._manipulator = mock.MagicMock()
    self.viewer._manipulator.perturbation = object_perturbation
    self.assertEqual(object_perturbation, self.viewer.perturbation)

  def test_retrieving_perturbation_without_manipulator(self):
    self.viewer._manipulator = None
    self.assertEqual(self.viewer._null_perturbation, self.viewer.perturbation)

  def test_retrieving_perturbation_without_selected_object(self):
    self.viewer._manipulator = mock.MagicMock()
    self.viewer._manipulator.perturbation = None
    self.assertEqual(self.viewer._null_perturbation, self.viewer.perturbation)


class CameraSelectorTest(absltest.TestCase):

  def setUp(self):
    super(CameraSelectorTest, self).setUp()
    self.camera = mock.MagicMock()
    self.model = mock.MagicMock()
    self.free_camera = mock.MagicMock()
    self.model.ncam = 2

    options = {
        'camera': self.camera,
        'model': self.model,
        'free_camera': self.free_camera
    }

    self.controller = viewer.CameraSelector(**options)

  def test_activating_freelook_camera_by_default(self):
    self.assertEqual(self.controller._free_ctrl, self.controller._active_ctrl)

  def test_cycling_forward_through_cameras(self):
    self.controller.select_next()
    self.assertIsNone(self.controller._active_ctrl)
    self.controller._free_ctrl.deactivate.assert_called_once()
    self.controller._free_ctrl.reset_mock()
    self.controller._camera.set_fixed_mode.assert_called_once_with(0)
    self.controller._camera.reset_mock()

    self.controller.select_next()
    self.assertIsNone(self.controller._active_ctrl)
    self.controller._camera.set_fixed_mode.assert_called_once_with(1)
    self.controller._camera.reset_mock()

    self.controller.select_next()
    self.assertEqual(self.controller._free_ctrl, self.controller._active_ctrl)
    self.controller._free_ctrl.activate.assert_called_once()

  def test_cycling_backwards_through_cameras(self):
    self.controller.select_previous()
    self.assertIsNone(self.controller._active_ctrl)
    self.controller._free_ctrl.deactivate.assert_called_once()
    self.controller._free_ctrl.reset_mock()
    self.controller._camera.set_fixed_mode.assert_called_once_with(1)
    self.controller._camera.reset_mock()

    self.controller.select_previous()
    self.assertIsNone(self.controller._active_ctrl)
    self.controller._camera.set_fixed_mode.assert_called_once_with(0)
    self.controller._camera.reset_mock()

    self.controller.select_previous()
    self.assertEqual(self.controller._free_ctrl, self.controller._active_ctrl)
    self.controller._free_ctrl.activate.assert_called_once()

  def test_controller_activation(self):
    old_controller = mock.MagicMock()
    new_controller = mock.MagicMock()
    self.controller._active_ctrl = old_controller
    self.controller._activate(new_controller)
    old_controller.deactivate.assert_called_once()
    new_controller.activate.assert_called_once()

  def test_controller_activation_not_repeated_for_already_active_one(self):
    controller = mock.MagicMock()
    self.controller._active_ctrl = controller
    self.controller._activate(controller)
    self.assertEqual(0, controller.deactivate.call_count)
    self.assertEqual(0, controller.activate.call_count)


class FreeCameraControllerTest(absltest.TestCase):

  def setUp(self):
    super(FreeCameraControllerTest, self).setUp()
    self.viewport = mock.MagicMock()
    self.camera = mock.MagicMock()
    self.mouse = mock.MagicMock()
    self.selection_service = mock.MagicMock()

    options = {
        'camera': self.camera,
        'viewport': self.viewport,
        'pointer': self.mouse,
        'selection_service': self.selection_service
    }

    self.controller = viewer.FreeCameraController(**options)
    self.controller._action = mock.MagicMock()

  def test_activation_while_not_in_tracking_mode(self):
    self.controller._tracked_body_idx = -1
    self.controller.activate()
    self.camera.set_freelook_mode.assert_called_once()

  def test_activation_while_in_tracking_mode(self):
    self.controller._tracked_body_idx = 1
    self.controller.activate()
    self.camera.set_tracking_mode.assert_called_once_with(1)

  def test_activation_and_deactivation_flag(self):
    self.controller.activate()
    self.assertTrue(self.controller._active)
    self.controller.deactivate()
    self.assertFalse(self.controller._active)

  def test_vertical_panning_camera_with_active_controller(self):
    self.controller._active = True
    self.controller.set_pan_vertical_mode(True)
    self.controller._action.begin.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_MOVE_V)
    self.controller.set_pan_vertical_mode(False)
    self.controller._action.end.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_MOVE_V)

  def test_vertical_panning_camera_with_inactive_controller(self):
    self.controller._active = False
    self.controller.set_pan_vertical_mode(True)
    self.assertEqual(0, self.controller._action.begin.call_count)
    self.controller.set_pan_vertical_mode(False)
    self.assertEqual(0, self.controller._action.end.call_count)

  def test_horizontal_panning_camera_with_active_controller(self):
    self.controller._active = True
    self.controller.set_pan_horizontal_mode(True)
    self.controller._action.begin.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_MOVE_H)
    self.controller.set_pan_horizontal_mode(False)
    self.controller._action.end.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_MOVE_H)

  def test_horizontal_panning_camera_with_inactive_controller(self):
    self.controller._active = False
    self.controller.set_pan_horizontal_mode(True)
    self.assertEqual(0, self.controller._action.begin.call_count)
    self.controller.set_pan_horizontal_mode(False)
    self.assertEqual(0, self.controller._action.end.call_count)

  def test_rotating_camera_with_active_controller(self):
    self.controller._active = True
    self.controller.set_rotate_mode(True)
    self.controller._action.begin.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_ROTATE_H)
    self.controller.set_rotate_mode(False)
    self.controller._action.end.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_ROTATE_H)

  def test_rotating_camera_with_inactive_controller(self):
    self.controller._active = False
    self.controller.set_rotate_mode(True)
    self.assertEqual(0, self.controller._action.begin.call_count)
    self.controller.set_rotate_mode(False)
    self.assertEqual(0, self.controller._action.end.call_count)

  def test_centering_with_active_controller(self):
    self.controller._active = True
    self.camera.raycast.return_value = 1, 2
    self.controller.center()
    self.camera.raycast.assert_called_once()

  def test_centering_with_inactive_controller(self):
    self.controller._active = False
    self.controller.center()
    self.assertEqual(0, self.camera.raycast.call_count)

  def test_moving_mouse_moves_camera(self):
    position = [100, 200]
    translation = [1, 0]
    viewport_space_translation = [2, 0]
    action = 1
    self.viewport.screen_to_viewport.return_value = viewport_space_translation
    self.controller._action.in_progress = True
    self.controller._action.watermark = action

    self.controller.on_move(position, translation)
    self.viewport.screen_to_viewport.assert_called_once_with(translation)
    self.camera.move.assert_called_once_with(action, viewport_space_translation)

  def test_mouse_move_doesnt_work_without_an_action_selected(self):
    self.controller._action.in_progress = False
    self.controller.on_move([], [])
    self.assertEqual(0, self.camera.move.call_count)

  def test_zoom_with_active_controller(self):
    self.controller._active = True
    expected_zoom_vector = [0, -0.05]
    self.controller.zoom(1.)
    self.camera.move.assert_called_once_with(
        enums.mjtMouse.mjMOUSE_ZOOM, expected_zoom_vector)

  def test_zoom_with_inactive_controller(self):
    self.controller._active = False
    self.controller.zoom(1.)
    self.assertEqual(0, self.camera.move.call_count)

  def test_tracking_with_active_controller(self):
    self.controller._active = True
    selected_body_id = 5
    self.selection_service.selected_body_id = selected_body_id
    self.controller._tracked_body_idx = -1

    self.controller.track()
    self.assertEqual(self.controller._tracked_body_idx, selected_body_id)
    self.camera.set_tracking_mode.assert_called_once_with(selected_body_id)

  def test_tracking_with_inactive_controller(self):
    self.controller._active = False
    selected_body_id = 5
    self.selection_service.selected_body_id = selected_body_id
    self.controller.track()
    self.assertEqual(self.controller._tracked_body_idx, -1)
    self.assertEqual(0, self.camera.set_tracking_mode.call_count)

  def test_free_look_mode_with_active_controller(self):
    self.controller._active = True
    self.controller._tracked_body_idx = 5
    self.controller.free_look()
    self.assertEqual(self.controller._tracked_body_idx, -1)
    self.camera.set_freelook_mode.assert_called_once()

  def test_free_look_mode_with_inactive_controller(self):
    self.controller._active = False
    self.controller._tracked_body_idx = 5
    self.controller.free_look()
    self.assertEqual(self.controller._tracked_body_idx, 5)
    self.assertEqual(0, self.camera.set_freelook_mode.call_count)


class ManipulationControllerTest(absltest.TestCase):

  def setUp(self):
    super(ManipulationControllerTest, self).setUp()
    self.viewport = mock.MagicMock()
    self.camera = mock.MagicMock()
    self.mouse = mock.MagicMock()

    options = {
        'camera': self.camera,
        'viewport': self.viewport,
        'pointer': self.mouse,
    }

    self.controller = viewer.ManipulationController(**options)

    self.body_id = 1
    self.click_pos_on_body = [1, 2, 3]
    self.camera.raycast.return_value = (self.body_id, self.click_pos_on_body)

  def test_selecting_a_body(self):
    self.camera.raycast.return_value = (self.body_id, self.click_pos_on_body)
    self.controller.select()
    self.assertIsNotNone(self.controller._perturb)

  def test_selecting_empty_space_cancels_selection(self):
    self.camera.raycast.return_value = (-1, None)
    self.controller.select()
    self.assertIsNone(self.controller._perturb)

  def test_vertical_movement_operation(self):
    self.controller._perturb = mock.MagicMock()

    self.controller.set_move_vertical_mode(True)
    self.controller._perturb.start_move.assert_called_once()
    self.assertEqual(enums.mjtMouse.mjMOUSE_MOVE_V,
                     self.controller._perturb.start_move.call_args[0][0])

    self.controller.set_move_vertical_mode(False)
    self.controller._perturb.end_move.assert_called_once()

  def test_horzontal_movement_operation(self):
    self.controller._perturb = mock.MagicMock()

    self.controller.set_move_horizontal_mode(True)
    self.controller._perturb.start_move.assert_called_once()
    self.assertEqual(enums.mjtMouse.mjMOUSE_MOVE_H,
                     self.controller._perturb.start_move.call_args[0][0])

    self.controller.set_move_horizontal_mode(False)
    self.controller._perturb.end_move.assert_called_once()

  def test_rotation_operation(self):
    self.controller._perturb = mock.MagicMock()

    self.controller.set_rotate_mode(True)
    self.controller._perturb.start_move.assert_called_once()
    self.assertEqual(enums.mjtMouse.mjMOUSE_ROTATE_H,
                     self.controller._perturb.start_move.call_args[0][0])

    self.controller.set_rotate_mode(False)
    self.controller._perturb.end_move.assert_called_once()

  def test_every_action_generates_a_fresh_grab_pos(self):
    some_action = 0
    self.controller._perturb = mock.MagicMock()
    self.controller._update_action(some_action)
    self.camera.raycast.assert_called_once()

  def test_actions_not_started_without_object_selected(self):
    some_action = 0
    self.controller._perturb = None
    self.controller._update_action(some_action)
    self.assertEqual(0, self.camera.raycast.call_count)

  def test_on_move_requires_an_action_to_be_started_first(self):
    self.controller._perturb = mock.MagicMock()
    self.controller._action = mock.MagicMock()
    self.controller._action.in_progress = False

    self.controller.on_move([], [])
    self.assertEqual(0, self.controller._perturb.tick_move.call_count)

  def test_dragging_selected_object_moves_it(self):
    screen_pos = [1, 2]
    screen_translation = [3, 4]
    viewport_offset = [5, 6]
    self.controller._perturb = mock.MagicMock()
    self.controller._action = mock.MagicMock()
    self.controller._action.in_progress = True
    self.viewport.screen_to_viewport.return_value = viewport_offset

    self.controller.on_move(screen_pos, screen_translation)
    self.viewport.screen_to_viewport.assert_called_once_with(screen_translation)
    self.controller._perturb.tick_move.assert_called_once_with(viewport_offset)

  def test_operations_require_object_to_be_selected(self):
    self.controller._perturb = None

    # No exceptions should be raised.
    self.controller.set_move_vertical_mode(True)
    self.controller.set_move_vertical_mode(False)
    self.controller.set_move_horizontal_mode(True)
    self.controller.set_move_horizontal_mode(False)
    self.controller.set_rotate_mode(True)
    self.controller.set_rotate_mode(False)
    self.controller.on_move([1, 2], [3, 4])


if __name__ == '__main__':
  absltest.main()
