
import abc
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np

_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'assets/dog_v2/dog{model_version}.xml')
_WALKER_INVIS_GROUP = 2
_WALKER_GEOM_GROUP = 5


class _DogBase(legacy_base.Walker, metaclass=abc.ABCMeta):
  """The abstract base class for walkers for the dog model."""

  def _build(self,
             name='walker',
             initializer=None):
    self._mjcf_root = mjcf.from_path(self._xml_path)
    if name:
      self._mjcf_root.model = name

    super()._build(initializer=initializer)

  def _build_observables(self):
    return DogObservables(self)

  @property
  @abc.abstractmethod
  def _xml_path(self):
    raise NotImplementedError

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @composer.cached_property
  def actuators(self):
    return tuple(self._mjcf_root.find_all('actuator'))

  @composer.cached_property
  def observable_joints(self):
    return tuple(self.joints)

  @composer.cached_property
  def joints(self):
    return tuple(self._mjcf_root.find_all('joint'))

  @composer.cached_property
  def root_body(self):
    return self._mjcf_root.find('body', 'torso')

  @composer.cached_property
  def head(self):
    return self._mjcf_root.find('body', 'skull')

  @composer.cached_property
  def ground_contact_geoms(self):
    return tuple(self._mjcf_root.find('body', 'foot_L').find_all('geom') +
                 self._mjcf_root.find('body', 'foot_R').find_all('geom') +
                 self._mjcf_root.find('body', 'hand_L').find_all('geom') +
                 self._mjcf_root.find('body', 'hand_R').find_all('geom'))

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'x-axis')

  @composer.cached_property
  def end_effectors(self):
    return (self._mjcf_root.find('body', 'foot_L'),
            self._mjcf_root.find('body', 'foot_R'),
            self._mjcf_root.find('body', 'hand_L'),
            self._mjcf_root.find('body', 'hand_R'))

  @composer.cached_property
  def bodies(self):
    return tuple(self._mjcf_root.find_all('body'))

  @composer.cached_property
  def mocap_tracking_sites(self):
    """Collection of markers for mocap tracking."""
    return tuple(
        b for b in self._mjcf_root.find_all('site') if "marker" not in b.name)


class Dog(_DogBase):
  """A torque actuated dog walker."""

  @property
  def _xml_path(self):
    return _XML_PATH.format(model_version='')


class DogMuscleActuated(_DogBase):
  """A muscle actuated dog walker."""

  @property
  def _xml_path(self):
    return _XML_PATH.format(model_version='_muscles_-1_Sigmoid')


class DogObservables(legacy_base.WalkerObservables):
  """Observables for the Dog."""

  @composer.observable
  def egocentric_camera(self):
    options = mj_wrapper.MjvOption()

    # Don't render this walker's geoms.
    options.geomgroup[_WALKER_INVIS_GROUP] = 0
    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=64, height=64, scene_option=options)

  @composer.observable
  def head_height(self):
    return observable.MJCFFeature('xpos', self._entity.head)[2]

  @composer.observable
  def actuator_activation(self):
    return observable.MJCFFeature('act',
                                  self._entity.mjcf_model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` with the head's position appended."""
    def relative_pos_in_egocentric_frame(physics):
      end_effectors_with_head = (
          self._entity.end_effectors + (self._entity.head,))
      end_effector = physics.bind(end_effectors_with_head).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)
    return observable.Generic(relative_pos_in_egocentric_frame)

  @property
  def proprioception(self):
    return [
        self.joints_pos,
        self.joints_vel,
        self.body_height,
        self.actuator_activation,
        self.appendages_pos,
        self.world_zaxis
    ] + self._collect_from_attachments('proprioception')
