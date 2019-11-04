import numpy as np
import torch
from pysc2.lib import actions
from pysc2.lib import features

DEVICE = torch.device('cuda:0')

SEED = 123
FEAT2DSIZE = 32
NUM_ACTIONS = len(actions.FUNCTIONS)
NUM_PLAYERS = features.SCREEN_FEATURES.player_id.scale  # 17
NUM_ENVS = 2
MAX_WINDOWS = 1
N_STEPS = 16

EPS = np.finfo(np.float32).eps.item()

action_shape = {'categorical': (NUM_ACTIONS,),
                'screen1': (1, FEAT2DSIZE, FEAT2DSIZE),
                'screen2': (1, FEAT2DSIZE, FEAT2DSIZE)}

observation_shape = {'minimap': (7, FEAT2DSIZE, FEAT2DSIZE),
                     'screen': (17, FEAT2DSIZE, FEAT2DSIZE),
                     'nonspatial': (NUM_ACTIONS,)}


# A2C parameters
class A2C:
    LEARNINGRATE = 1e-3
    BatchSize = 32
    entropy_weight = 1e-4
    value_loss_weight = 1.0
    max_gradient_norm = 500.0
    GAMMA = 0.95
