import numpy as np

import torch
import torch.nn as nn

from absl import flags
from pysc2.lib import features
from pysc2.lib import actions

from utils import preprocessors, arglist
from utils.layers import Flatten, weight_init_fn
FLAGS = flags.FLAGS
# flags.DEFINE_integer('image_size', 64, 'Height & width of inputs.')
flags.DEFINE_bool('use_gpu', True, 'Use gpu if True, else use cpu.')


class A2CNet(nn.Module):
    def __init__(self):
        super(A2CNet, self).__init__()

        self.screen_specs = features.SCREEN_FEATURES  # Categorical, Scalar
        self.minimap_specs = features.MINIMAP_FEATURES  # Categorical, Scalar
        self.player_specs = preprocessors.FLAT_FEATURES

        # Initialize embedding layers (for FeatureType.CATEGORICAL features)
        self.screen_embeddings, self.num_screens = \
            self._initialize_embeddings(self.screen_specs, self._embed_spatial_fn)
        self.minimap_embeddings, self.num_minimaps = \
            self._initialize_embeddings(self.minimap_specs, self._embed_spatial_fn)
        self.player_embeddings, self.num_player = \
            self._initialize_embeddings(self.player_specs, self._embed_player_fn)

        # Initialize layers (screen, minimap, player)
        self.screen_conv = nn.DataParallel(nn.Sequential(
            nn.DataParallel(nn.Conv2d(self.num_screens, 16, 5, stride=1, padding=2)),
            nn.ReLU(),
            nn.DataParallel(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.ReLU()))

        self.minimap_conv = nn.DataParallel(nn.Sequential(
            nn.DataParallel(nn.Conv2d(self.num_minimaps, 16, 5, stride=1, padding=2)),
            nn.ReLU(),
            nn.DataParallel(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.ReLU()))

        self.player_conv = nn.DataParallel(nn.Sequential(
            nn.DataParallel(nn.Linear(75 * arglist.FEAT2DSIZE * arglist.FEAT2DSIZE, 256)),
            nn.ReLU()))

        self.value_op = nn.DataParallel(nn.Linear(256, 1))
        self.action_op = nn.DataParallel(nn.Sequential(
            nn.Linear(256, arglist.NUM_ACTIONS),
            nn.Softmax(dim=1)))

        self.non_spatial_outputs = self._initialize_non_spatial_actions(256)
        self.spatial_outputs = self._initialize_spatial_actions(75)

        self.apply(weight_init_fn)
        self.train()

    def forward(self, screen_input, minimap_input, player_input):
        """
            screen_input:
            minimap_input:
            flat_input:
        """
        _, _, resolution, _ = screen_input.size()  # B, C, H, W

        screen_emb = self.embed_op(
            x=screen_input,
            specs=self.screen_specs,
            embeddings=self.screen_embeddings,
            one_hot_fn=self.make_one_hot_3d
        )

        minimap_emb = self.embed_op(
            x=minimap_input,
            specs=self.minimap_specs,
            embeddings=self.minimap_embeddings,
            one_hot_fn=self.make_one_hot_3d
        )

        player_emb = self.embed_op(
            x=player_input,
            specs=self.player_specs,
            embeddings=self.player_embeddings,
            one_hot_fn=self.make_one_hot_1d
        )

        screen_out = self.screen_conv(screen_emb)
        minimap_out = self.minimap_conv(minimap_emb)
        player_out = player_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, resolution, resolution)

        state_out = torch.cat((screen_out, minimap_out, player_out), dim=1)
        state_flattened = state_out.view(state_out.size(0), -1)

        fc = self.player_conv(state_flattened)
        value = self.value_op(fc).view(-1)
        action = self.action_op(fc)

        action_args = {}
        for name, arg_type in actions.TYPES._asdict().items():
            if name in ['screen', 'screen2', 'minimap']:
                action_arg = self.spatial_outputs.get(arg_type.id)(state_out)

            else:
                action_arg = self.non_spatial_outputs.get(arg_type.id)(fc)
            action_args[arg_type] = action_arg

        policy = (action, action_args)  # main action, action arguments

        return policy, value

    def _embed_spatial_fn(self, in_, out_):
        '''1 * 1 2D convolution for spatial features.'''
        return nn.DataParallel(nn.Conv2d(in_, out_, 1, stride=1, padding=0))

    def _embed_player_fn(self, in_, out_):
        '''Linear transform for player features'''
        return nn.DataParallel(nn.Linear(in_, out_))

    def _initialize_embeddings(self, specs, embed_fn):
        """
        Intialize embedding operations.
        Arguments:
            specs: list of features.
            embed_fn: embedding function.
        Returns:
            embeddings: dict, with embeddings (key: spec.index, value: lookup table)
            out_size_total: int, total number of channels after embedding operation.
        """
        embeddings = {}
        out_size_total = 0
        for spec in specs:
            if spec.type == features.FeatureType.CATEGORICAL:
                out_size = np.round(np.log2(spec.scale)).astype(np.int32).item()
                out_size = np.max((out_size, 1))
                out_size_total += out_size
                embeddings[spec.index] = nn.DataParallel(nn.Sequential(
                    embed_fn(spec.scale, out_size),
                    nn.ReLU()))

            elif spec.type == features.FeatureType.SCALAR:
                out_size_total += 1

            else:
                raise NotImplementedError

        return embeddings, out_size_total

    def embed_op(self, x, specs, embeddings, one_hot_fn):
        '''Perform embedding operation.'''
        n_ch = x.size()[1]  # 2, 17, 64, 64
        assert n_ch == len(specs)
        assert n_ch >= len(embeddings)

        outs = []
        for feat, spec in zip(torch.chunk(x, n_ch, dim=1), specs):
            if spec.type == features.FeatureType.CATEGORICAL:
                indices = one_hot_fn(
                    labels=feat,
                    num_classes=spec.scale)

                out = embeddings.get(spec.index)(indices.float())
                outs.append(out)

            elif spec.type == features.FeatureType.SCALAR:
                out = self.log_transform(feat.float(), spec.scale)
                outs.append(out)

            else:
                raise NotImplementedError

        return torch.cat(outs, dim=1)

    def _initialize_non_spatial_actions(self, in_features=256):
        '''Initialize non-spatial action operations'''

        out = {}
        for name, arg_type in actions.TYPES._asdict().items():
            if name not in ['screen', 'screen2', 'minimap']:
                out[arg_type.id] = nn.DataParallel(nn.Sequential(
                    nn.Linear(in_features, arg_type.sizes[0]),
                    nn.Softmax(dim=1)))

        return out

    def _initialize_spatial_actions(self, in_channels):
        '''Initialize spatial action operations'''

        out = {}
        for name, arg_type in actions.TYPES._asdict().items():
            if name in ['screen', 'screen2', 'minimap']:
                out[arg_type.id] = nn.DataParallel(nn.Sequential(
                    nn.Conv2d(in_channels, 1, 1, stride=1, padding=0),
                    Flatten(),
                    nn.Softmax(dim=1)))

        return out

    @staticmethod
    def make_one_hot_1d(labels, num_classes):
        """..."""
        desired_shape = (labels.size(0), num_classes)
        labels = labels.contiguous().view(-1, 1)
        target = torch.zeros(*desired_shape, requires_grad=False).to(arglist.DEVICE)
        target = target.scatter_(1, labels.long(), 1)
        # if use_gpu:
        #     target = target.cuda()

        return target

    @staticmethod
    def make_one_hot_3d(labels, num_classes):
        """..."""
        desired_shape = (labels.size(0), num_classes, labels.size(2), labels.size(3))
        target = torch.zeros(*desired_shape, requires_grad=False).to(arglist.DEVICE)
        target = target.scatter_(1, labels.long(), 1)
        # if use_gpu:
        #     target = target.cuda()

        return target

    @staticmethod
    def log_transform(input_, scale):
        """Apply log transformation."""
        out = torch.log(8 * input_ / scale + 1)
        return out
