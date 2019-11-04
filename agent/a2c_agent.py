"""
    A2C Agent. A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C).
        Original paper: https://arxiv.org/abs/1602.01783
        OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from pysc2.env import environment

from pysc2.lib.actions import FunctionCall
from pysc2.lib.actions import FUNCTIONS as ACTION_FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES

from agent import base_agent

from utils import arglist


class A2CAgent(base_agent.BaseAgent):
    """Implementation of pysc2 agent to be trained with A2C."""
    def __init__(self, network):
        super(A2CAgent, self).__init__()

        assert isinstance(network, nn.Module)
        self.device = arglist.DEVICE
        # self.dtype = torch.FloatTensor
        # self.atype = torch.LongTensor

        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), arglist.A2C.LEARNINGRATE)

        # self.network.eval()
        # self.use_gpu = args.use_gpu

        # if self.use_gpu and torch.cuda.is_available():
        #     self.network.cuda()
        # self.size = args.size

    def get_value(self, obs):
        _, value_estimate = self._step(obs)
        return value_estimate

    def _make_var(self, input_dict):
        new_dict = {}
        for k, v in input_dict.items():
            new_dict[k] = torch.tensor(v).to(self.device)

        return new_dict

    def _step(self, obs):
        """Add function docstring."""
        # assert isinstance(timestep, environment.TimeStep)

        obs_var = self._make_var(obs)

        policy, value = self.network(obs_var['screen'], obs_var['minimap'], obs_var['player'])
        available_actions = obs_var['available_actions']
        samples = self._sample_action(available_actions, policy)  # tuple
        assert len(samples) == 2

        return samples, value.cpu().data.numpy()  # Return as numpy arrays

    def _sample_action(self, available_actions, policy):
        """Sample from available actions according to policy."""
        model_action, model_action_args = policy
        masked_action = self.mask_unavailable_actions(available_actions, model_action)

        while True:
            sampled_action = Categorical(probs=masked_action).sample()  # torch.size([])
            if (available_actions.gather(1, sampled_action.unsqueeze(1)) == 1).all():
                sampled_action = sampled_action.data.cpu().numpy()
                break

        sampled_action_args = {}
        for arg_type, action_arg in model_action_args.items():
            sampled_action_arg = Categorical(probs=action_arg).sample()
            sampled_action_args[arg_type] = sampled_action_arg.data.cpu().numpy()

        return sampled_action, sampled_action_args

    def _make_actions(self, actions):
        n_id, arg_ids = actions

        assert isinstance(n_id, np.ndarray)
        assert isinstance(arg_ids, dict)

        n_id = torch.from_numpy(n_id)
        fn_id_var = n_id.long().to(self.device)

        args_var = {}
        for k, v in arg_ids.items():
            v = torch.from_numpy(v)
            args_var[k] = v.long().to(self.device)

        return fn_id_var, args_var

    def _compute_policy_log_probs(self, available_actions, policy, actions):

        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))

        def compute_log_probs(probs, labels):
            new_labels = labels.clone()
            new_labels[new_labels < 0] = 0
            selected_probs = probs.gather(1, new_labels.unsqueeze(1))
            out = logclip(selected_probs)

            return out.view(-1)

        fn_id, arg_ids = actions
        fn_pi, arg_pis = policy
        masked_actions = self.mask_unavailable_actions(available_actions, fn_pi)
        fn_log_prob = compute_log_probs(masked_actions, fn_id)

        log_prob = fn_log_prob
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]
            arg_log_prob = compute_log_probs(arg_pi, arg_id)

            arg_id_masked = arg_id.clone()
            arg_id_masked[arg_id_masked != -1] = 1
            arg_id_masked[arg_id_masked == -1] = 0
            arg_log_prob = arg_log_prob * arg_id_masked.float()
            log_prob = log_prob + arg_log_prob

        return log_prob

    def _compute_policy_entropy(self, available_actions, policy, actions):

        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))

        def compute_entropy(probs):
            return -(logclip(probs) * probs).sum(-1)

        _, arg_ids = actions
        fn_pi, arg_pis = policy
        masked_actions = self.mask_unavailable_actions(available_actions, fn_pi)

        entropy = compute_entropy(masked_actions).mean()
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]

            batch_mask = arg_id.clone()
            batch_mask[batch_mask != -1] = 1
            batch_mask[batch_mask == -1] = 0

            if (batch_mask == 0).all():
                arg_entropy = (compute_entropy(arg_pi) * 0.0).sum()

            else:
                arg_entropy = (compute_entropy(arg_pi) * batch_mask.float()).sum() \
                              / batch_mask.float().sum()

            entropy = entropy + arg_entropy

        return entropy

    def optimize(self, obs, actions, returns, advs):
        obs_var = self._make_var(obs)
        returns = torch.tensor(returns).to(self.device)
        advs = torch.tensor(advs).to(self.device)

        policy, value = self.network(obs_var['screen'], obs_var['minimap'], obs_var['player'])
        # print(policy, value)
        actions = self._make_actions(actions)
        available_actions = obs_var['available_actions']

        log_probs = self._compute_policy_log_probs(available_actions, policy, actions)

        policy_loss = -(advs * log_probs).mean()
        value_loss = (returns - value).pow(2).mean()

        entropy = self._compute_policy_entropy(available_actions, policy, actions)
        loss = policy_loss + value_loss * arglist.A2C.value_loss_weight \
            - entropy * arglist.A2C.entropy_weight

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.network.parameters(), arglist.A2C.max_gradient_norm)
        self.optimizer.step()

        return loss
