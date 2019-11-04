import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.functional as F
from torch.distributions import Categorical

import os
import numpy as np

from networks.acnetworks_newchallenge import FullyConv
from utils.preprocessors import Preprocessor
from utils import arglist


class A2CAgent(object):
    """A2C agent.

    Run buile(...) first, then init() or load(...).
    """
    def __init__(self, args):

        self.dtype = torch.FloatTensor
        self.atype = torch.LongTensor
        self.network = FullyConv(args)

        if args.cuda:
            self.dtype = torch.cuda.FloatTensor
            self.atype = torch.cuda.LongTensor
            self.network.cuda()

        self.optimizer = optim.Adam(self.network.get_trainable_params(), lr=arglist.A2C.LEARNINGRATE)

    def get_obs_feed(self, obs):  # _make_var in StarCraft bot
        new_dict = {}
        for k, v in obs.items():
            new_dict[k] = Variable(self.dtype(v))

        return new_dict

    def get_actions_feed(self, actions):
        n_id, arg_ids = actions
        args_var = {}
        fn_id_var = Variable(self.atype(n_id))
        for k, v in arg_ids.items():
            args_var[k] = Variable(self.atype(v))

        return fn_id_var, args_var

    def mask_unavailable_actions(self, available_actions, fn_pi):
        fn_pi = fn_pi * available_actions
        fn_pi = fn_pi / fn_pi.sum(1, keepdim=True)

        return fn_pi

    def sample_actions(self, available_actions, policy):
        # Sample actions
        # Avoid the case where the sampled action is NOT available

        def sample(probs):
            dist = Categorical(probs=probs)

            return dist.sample()

        fn_pi, arg_pis = policy
        fn_pi = self.mask_unavailable_actions(available_actions, fn_pi)
        while True:
            fn_samples = sample(fn_pi)
            if (available_actions.gather(1, fn_samples.unsqueeze(1)) == 1).all():
                fn_samples = fn_samples.data.cpu().numpy()
                break

        arg_samples = dict()
        for arg_type, arg_pi in arg_pis.items():
            arg_samples[arg_type] = sample(arg_pi)

        return fn_samples, arg_samples

    def step(self, obs):
        """
        Args:
          obs: dict of preprocessed observation arrays, with num_batch elements
            in the first dimensions.

        Returns:
          actions: arrays (see `compute_total_log_probs`)
          values: array of shape [num_batch] containing value estimates.
        """
        feed_dict = self.get_obs_feed(obs)
        policy, value = self.network.forward(feed_dict['screen'], feed_dict['minimap'], feed_dict['flat'])

        available_actions = feed_dict['available_actions']
        samples = self.sample_actions(available_actions, policy)

        return samples, value.cpu.data.numpy()

    def get_value(self, obs):
        _, value_estimate = self.step(obs)

        return value_estimate

    def train(self, obs, actions, returns, advs, summary=False):
        """
        Args:
          obs: dict of preprocessed observation arrays, with num_batch elements
            in the first dimensions.
          actions: see `compute_total_log_probs`.
          returns: array of shape [num_batch].
          advs: array of shape [num_batch].
          summary: Whether to return a summary.
        Returns:
          summary: (agent_step, loss, Summary) or None.
        """
        obs_var = self.get_obs_feed(obs)
        returns_var = Variable(self.dtype(returns))
        advs_var = Variable(self.dtype(advs))
        actions_var = self.get_actions_feed(actions)
        available_actions = obs_var['available_actions']
        policy, value = self.network.forward(obs_var['screen'],
                                             obs_var['minimap'], obs_var['flat'])

        log_probs = self.compute_policy_log_probs(available_actions, policy, actions_var)
        policy_loss = -(advs_var * log_probs).mean()
        value_loss = (returns_var - value).pow(2).mean()

        entropy = self.compute_policy_entropy(available_actions, policy, actions_var)
        loss = policy_loss + value_loss * arglist.A2C.value_loss_weight \
               - entropy * arglist.A2C.entropy_weight

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.network.parameters(), arglist.A2C.max_gradient_norm)
        self.optimizer.step()

        return None

    def compute_policy_log_probs(self, available_actions, policy, actions_var):
        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))

        def compute_log_probs(probs, labels):
            new_labels = labels.clone()
            new_labels[new_labels < 0] = 0
            selected_probs = probs.gather(1, new_labels.unsqueeze(1))
            out = logclip(selected_probs)

            # Log of 0 will be 0
            # out[selected_probs == 0] = 0
            return out.view(-1)

        fn_id, arg_ids = actions_var
        fn_pi, arg_pis = policy
        fn_pi = self.mask_unavailable_actions(available_actions, fn_pi)
        fn_log_prob = compute_log_probs(fn_pi, fn_id)

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

    def compute_policy_entropy(self, available_actions, policy, actions_var):
        def logclip(x):
            return torch.log(torch.clamp(x, 1e-12, 1.0))

        def compute_entropy(probs):
            return -(logclip(probs) * probs).sum(-1)

        _, arg_ids = actions_var
        fn_pi, arg_pis = policy
        fn_pi = self.mask_unavailable_actions(available_actions, fn_pi)

        entropy = compute_entropy(fn_pi).mean()
        for arg_type in arg_ids.keys():
            arg_id = arg_ids[arg_type]
            arg_pi = arg_pis[arg_type]

            batch_mask = arg_id.clone()
            batch_mask[batch_mask != -1] = 1
            batch_mask[batch_mask == -1] = 0

            if (batch_mask == 0).all():
                arg_entropy = (compute_entropy(arg_pi) * 0.0).sum()

            else:
                arg_entropy = (compute_entropy(arg_pi) * batch_mask.float()).sum() / batch_mask.float().sum()

            entropy = entropy + arg_entropy

        return entropy

