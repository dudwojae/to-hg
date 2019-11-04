import sys
import os
import shutil
import argparse

from logger import Logger
from pysc2.env import sc2_env

from runs.a2c_minigame import MiniGame
from utils import arglist
from utils.preprocessors import Preprocess
from functools import partial
from utils.environment import SubprocVenEnv, make_sc2env, SingleEnv
import torch
from absl import flags

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(arglist.SEED)

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

env_names = ["DefeatZerglingsAndBanelings", "MoveToBeacon", "CollectMineralShards", "DefeatRoaches",
             "FindAndDefeatZerglings",
             "BuildMarines", "CollectMineralsAndGas"]

rl_algo = 'a2c'


def main():
    for map_name in env_names:
        if rl_algo == 'a2c':
            from agent.a2c_agent import A2CAgent
            from networks.a2c_network_tmp import A2CNet

            env_args = dict(map_name=map_name,
                            step_mul=8,
                            game_steps_per_episode=0)

            # vis_env_args = env_args.copy()
            env_args['visualize'] = False

            num_vis = min(arglist.NUM_ENVS, arglist.MAX_WINDOWS)
            env_fns = [partial(make_sc2env, **env_args)] * num_vis
            num_no_vis = arglist.NUM_ENVS - num_vis
            if num_no_vis > 0:
                env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)

            envs = SubprocVenEnv(env_fns)

            network = A2CNet()
            learner = A2CAgent(network)

            preprocess = Preprocess()
            game = MiniGame(map_name, envs, learner, preprocess, nb_episodes=10000)
            game.reset()

            current_epoch = 0
            try:
                while True:
                    result, loss = game.run_a2c(is_training=True)

                    current_epoch += 1
                    print(current_epoch)
                    if 0 <= -1 <= current_epoch:
                        break

            except KeyboardInterrupt:
                pass

            envs.close()

        else:
            raise NotImplementedError()


if __name__ == "__main__":
    main()