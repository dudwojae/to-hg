import numpy as np

from absl import flags
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env import environment

from collections import namedtuple


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale'])

NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale  # 17

is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
  # HACK: we should infer the point type automatically
  is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def stack_ndarray_dicts(lst, axis=0):
    # issue https://github.com/deepmind/pysc2/issues/273
    res = {}

    for k in lst[0].keys():
        for i, d in enumerate(lst):
            if i == 0:
                res[k] = np.expand_dims(d[k], axis=axis)
            else:
                res[k] = np.concatenate([res[k], np.expand_dims(d[k], axis=axis)], axis=axis)

    return res

'''
screen features: height_map, unit_hit_points, unit_hit_points_ration,
unit_energy, unit_energy_ratio, unit_shields, unit_shields_ratio,
unit_density, unit_density_aa, buff_duration, build_progress
'''

FLAT_FEATURES = [FlatFeature(features.FeatureType.SCALAR, 0, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 1, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 2, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 3, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 4, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 5, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 6, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 7, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 8, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 9, 1.),
                 FlatFeature(features.FeatureType.SCALAR, 10, 1.)]


class TimeStepPreprocessor(object):
    """
    Process pysc2.env.environment.Timestep observations.
    """
    def __init__(self):
        self.num_screen_ft = len(features.SCREEN_FEATURES)  # 17
        self.num_minimap_ft = len(features.MINIMAP_FEATURES)  # 7
        self.num_player_ft = len(FLAT_FEATURES)  # 11
        self.num_actions = len(actions.FUNCTIONS)

    def get_state(self, timestep):
        """
        Argutment:
            timestep: a single pysc2.env.environment.TimeStep observation.
        Returns:
             dictionary, with screen & minimap & player features.
        """
        """Get static channel dimensions of network inputs."""

        state = {
            'screen': self.get_feature_screen(timestep),
            'minimap': self.get_feature_minimap(timestep),
            'player': self.get_feature_player(timestep),
            'available_actions': self.get_available_actions(timestep),
            'reward': self.get_step_reward(timestep),
            'step_type': self.get_step_type(timestep)
        }

        return state

    @staticmethod  # self X , independent functions
    def get_feature_screen(timestep):
        assert isinstance(timestep, environment.TimeStep)
        return timestep.observation['feature_screen']

    @staticmethod
    def get_feature_minimap(timestep):
        assert isinstance(timestep, environment.TimeStep)
        return timestep.observation['feature_minimap']

    @staticmethod
    def get_feature_player(timestep):
        assert isinstance(timestep, environment.TimeStep)
        return timestep.observation['player']

    @staticmethod
    def get_step_type(timestep):
        assert isinstance(timestep, environment.TimeStep)
        return timestep.step_type

    @staticmethod
    def get_step_reward(timestep):
        assert isinstance(timestep, environment.TimeStep)
        return timestep.reward

    @staticmethod
    def get_available_actions(timestep, num_actions=len(actions.FUNCTIONS)):
        assert isinstance(timestep, environment.TimeStep)
        available_actions = np.zeros((num_actions,), dtype=np.float32)
        available_actions[timestep.observation['available_actions']] = 1.
        return available_actions
