"""
    A base agent to write custom scripted agents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from pysc2.lib import actions
from utils.preprocessors import stack_ndarray_dicts

class BaseAgent(object):
    """
    A customized base agent to write custom scripted agents.
    It can also act as a passive agent that does nothing but no ops.
    """
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def step(self, state):
        self.steps += 1
        self.reward += state['reward']
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def select_action(self, timestep):
        return self.step(timestep)

    @staticmethod
    def mask_unavailable_actions(available_actions, model_action):
        """Mask out unavailable actions & compute masked softmax."""
        masked_action = model_action * available_actions
        # masked_action = torch.nn.functional.softmax(masked_action, dim=1)
        masked_action = masked_action / masked_action.sum(dim=1, keepdim=True)
        return masked_action  # shape of (B=1, num_actions)

    @staticmethod
    def mask_unused_arguments(action_tuple):
        """
        Replace sampled argument id with -1, for all arguments not used.
        Batch-level implementation.
        """
        fn_id, arg_ids = action_tuple
        for i in range(fn_id.shape[0]):
            a_0 = fn_id[i]
            unused_types = set(actions.TYPES) - set(actions.FUNCTIONS._func_list[a_0].args)

            for arg_type in unused_types:
                arg_ids[arg_type][i] = -1

        return (fn_id, arg_ids)

    @staticmethod
    def actions_to_pysc2(actions_tuple, size_tuple):
        # FIXME: deprecate
        """
        Convert actions to pysc2 format.
        Batch-level implementation.
        """
        if isinstance(size_tuple, int):  # size_tuple int True or False
            height = width = size_tuple

        elif isinstance(size_tuple, tuple):
            height, width = size_tuple

        else:
            raise AttributeError

        fn_id, arg_ids = actions_tuple
        actions_list = []
        for i in range(fn_id.shape[0]):
            a_0 = fn_id[i]
            a_1 = []
            for arg_type in actions.FUNCTIONS._func_list[a_0].args:
                arg_id = arg_ids[arg_type][i]

                if arg_type.name in ['screen', 'screen2', 'minimap']:
                    arg = [arg_id % width, arg_id // height]

                else:
                    arg = [arg_id]

                a_1.append(arg)

            pysc2_action = actions.FunctionCall(a_0, a_1)
            actions_list.append(pysc2_action)

        return actions_list

    @staticmethod
    def flatten_first_dims(x):
        new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
        return x.reshape(*new_shape)

    @staticmethod
    def flatten_first_dims_dict(x):
        return {k: BaseAgent.flatten_first_dims(v) for k, v in x.items()}

    @staticmethod
    def stack_and_flatten_actions(lst, axis=0):
        fn_id_list, arg_dict_list = zip(*lst)
        fn_id = np.stack(fn_id_list, axis=axis)
        fn_id = BaseAgent.flatten_first_dims(fn_id)
        arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
        arg_ids = BaseAgent.flatten_first_dims_dict(arg_ids)

        return (fn_id, arg_ids)
