"""
    Run & train A2C agent.
"""

import os
import sys
import time
import shutil
from functools import partial
import numpy as np

from absl import app
from absl import flags
from pysc2.env import sc2_env
from pysc2.env import environment

from utils.environment import SubprocVenEnv, make_sc2env, SingleEnv
from utils.preprocessors import stack_ndarray_dicts
from utils import arglist

from agent.a2c_agent import A2CAgent


class MiniGame:
    def __init__(self, map_name, envs, learner, preprocess, nb_episodes=50000):
        self.map_name = map_name
        self.nb_max_steps = 2000
        self.nb_episodes = nb_episodes

        # env_args = dict(map_name=self.map_name,
        #             step_mul=8,
        #             game_steps_per_episode=0)
        #
        # vis_env_args = env_args.copy()
        # vis_env_args['visualize'] = False
        #
        # num_vis = min(arglist.NUM_ENVS, arglist.MAX_WINDOWS)
        # env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
        # num_no_vis = arglist.NUM_ENVS - num_vis
        # if num_no_vis > 0:
        #     env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)

        # self.envs = SubprocVenEnv(env_fns)
        self.envs = envs
        self.n_steps = arglist.N_STEPS
        self.learner = learner
        self.discount = arglist.A2C.GAMMA
        self.preprocess = preprocess

    def reset(self):
        obs_raw = self.envs.reset()
        self.last_obs = self.preprocess.preprocess_obs(obs_raw)

    def run_a2c(self, is_training=True):
        reward_cumulative = 0
        # for i_episode in range(self.nb_episodes):
        #
        # obs_raw = self.envs.reset()
        # self.last_obs = self.preprocess.preprocess_obs(obs_raw)

        shapes = (self.n_steps, self.envs.n_envs)
        values = np.zeros(shapes, dtype=np.float32)
        rewards = np.zeros(shapes, dtype=np.float32)
        dones = np.zeros(shapes, dtype=np.float32)

        all_obs = []
        all_actions = []
        all_scores = []

        last_obs = self.last_obs

        for n in range(self.n_steps):
            actions, value_estimate = self.learner._step(last_obs)
            actions = self.learner.mask_unused_arguments(actions)

            size = last_obs['screen'].shape[2:]

            values[n, :] = value_estimate
            all_obs.append(last_obs)
            all_actions.append(actions)

            pysc2_actions = self.learner.actions_to_pysc2(actions, size)
            obs_raw = self.envs.step(pysc2_actions)
            last_obs = self.preprocess.preprocess_obs(obs_raw)

            rewards[n, :] = [t.reward for t in obs_raw]
            dones[n, :] = [t.last() for t in obs_raw]

            for t in obs_raw:
                if t.last():
                    cum_reward = t.observation['score_cumulative']
                    reward_cumulative += cum_reward
                    # reward_cumulative.append(cum_reward[0])

        self.last_obs = last_obs

        next_values = self.learner.get_value(last_obs)

        returns, advs = MiniGame.compute_returns_advantages(rewards, dones, values,
                                                            next_values, self.discount)

        actions = self.learner.stack_and_flatten_actions(all_actions)
        obs = self.learner.flatten_first_dims_dict(stack_ndarray_dicts(all_obs))
        returns = self.learner.flatten_first_dims(returns)
        advs = self.learner.flatten_first_dims(advs)

        loss = self.learner.optimize(obs, actions, returns, advs)

        return reward_cumulative, loss

    @staticmethod
    def compute_returns_advantages(rewards, dones, values, next_values, discount):
        """Compute returns and advantages from received rewards and value estimates.
        Args:
            rewards: array of shape [n_steps, n_env] containing received rewards.
            dones: array of shape [n_steps, n_env] indicating whether an episode is
              finished after a time step.
            values: array of shape [n_steps, n_env] containing estimated values.
            next_values: array of shape [n_env] containing estimated values after the
              last step for each environment.
            discount: scalar discount for future rewards.
        Returns:
            returns: array of shape [n_steps, n_env]
            advs: array of shape [n_steps, n_env]
        """
        returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]]).astype('float32')

        returns[-1, :] = next_values
        for t in reversed(range(rewards.shape[0])):
            future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
            returns[t, :] = rewards[t, :] + future_rewards

        returns = returns[:-1, :]  # np.ndarray
        advs = returns - values    # np.ndarray

        return returns, advs
