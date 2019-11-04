import numpy as np
from utils import arglist
from pysc2.lib import features
from pysc2.lib import actions
from collections import namedtuple

FlatFeature = namedtuple('FlatFeatures', ['type', 'index', 'scale'])

NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale  # 17

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

is_spatial_action = {}  # x, y
for name, arg_type in actions.TYPES._asdict().items():  # HACK: we should infer the point type automatically
    # example: name= screen 0 / arg_type= screen [0, 0]
    is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']


def stack_ndarray_dicts(lst, axis=0):
    # issue https://github.com/deepmind/pysc2/issues/273
    res = {}

    for k in lst[0].keys():  # screen, minimap, flat, available_actions
        for i, d in enumerate(lst):
            if i == 0:
                res[k] = np.expand_dims(d[k], axis=axis)
            else:
                res[k] = np.concatenate([res[k], np.expand_dims(d[k], axis=axis)], axis=axis)

    return res


class Preprocess():
    """Compute network inputs from pysc2 observations.
      See https://github.com/deepmind/pysc2/blob/master/docs/environment.md
      for the semantics of the available observations.
      """

    def __init__(self):
        self.num_screen_channels = len(features.SCREEN_FEATURES)  # 17
        self.num_minimap_channels = len(features.MINIMAP_FEATURES)  # 7
        self.num_flat_channels = len(FLAT_FEATURES)  # 11
        self.available_actions_channels = arglist.NUM_ACTIONS  # 549

    def get_input_channels(self):
        """Get static channel dimensions of network inputs."""
        return {
            'screen': self.num_screen_channels,
            'minimap': self.num_minimap_channels,
            'player': self.num_flat_channels,
            'available_actions': self.available_actions_channels}

    def preprocess_obs(self, obs_list):
        return stack_ndarray_dicts(
            [self._preprocess_obs(o.observation) for o in obs_list])  # o: state (env.reset())

    def _preprocess_obs(self, obs):
        """Comput screen, minimap and flat network inputs from raw observations"""
        available_actions = np.zeros(arglist.NUM_ACTIONS, dtype=np.float32)
        available_actions[obs['available_actions']] = 1.

        screen = self._preprocess_spatial(obs['feature_screen'])
        minimap = self._preprocess_spatial(obs['feature_minimap'])

        # TODO available_actions, control groups, cargo, multi select, build queue
        flat = np.concatenate([obs['player']])

        return {
            'screen': screen,
            'minimap': minimap,
            'player': flat,
            'available_actions': available_actions}

    def _preprocess_spatial(self, spatial):
        return spatial

    def _onehot1d(self, x):
        y = np.zeros((self.num_flat_channels, ), dtype='float32')
        y[x] = 1.
        return y
