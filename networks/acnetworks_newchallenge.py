import torch
import torch.nn as nn

from pysc2.lib import actions
from pysc2.lib import features

import numpy as np

from utils.layers import Flatten, weight_init_fn, make_one_hot_1d, make_one_hot_2d, conv2d_init, linear_init
from utils.preprocessors import is_spatial_action, FLAT_FEATURES
from utils import arglist


class FullyConv(torch.nn.Module):
    """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

    Both, NHWC and NCHW data formats are supported for the network
    computations. Inputs and outputs are always in NHWC.
    """

    def __init__(self, args):
        super(FullyConv, self).__init__()

        # spatial features
        self.screen_specs = features.SCREEN_FEATURES  # 17
        self.minimap_specs = features.MINIMAP_FEATURES  # 7

        # non-spatial features
        self.flat_specs = FLAT_FEATURES  # 11
        # self.dtype = torch.FloatTensor
        # self.atype = torch.LongTensor

        # if args.cuda:
        #     self.dtype = torch.cuda.FloatTensor
        #     self.atype = torch.cuda.LongTensor

        self.embed_screen = self._init_embed_obs(self.screen_specs, self._embed_spatial)
        self.embed_minimap = self._init_embed_obs(self.minimap_specs, self._embed_spatial)
        self.embed_flat = self._init_embed_obs(self.flat_specs, self._embed_flat)

        self.screen_out = nn.DataParallel(nn.Sequential(
            nn.DataParallel(nn.Conv2d(20, 16, 5, stride=1, padding=2)),
            nn.ReLU(),
            nn.DataParallel(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.ReLU()))

        self.minimap_out = nn.DataParallel(nn.Sequential(
            nn.DataParallel(nn.Conv2d(6, 16, 5, stride=1, padding=2)),
            nn.ReLU(),
            nn.DataParallel(nn.Conv2d(16, 32, 3, stride=1, padding=1)),
            nn.ReLU()))

        self.fc = nn.DataParallel(nn.Sequential(
            nn.DataParallel(nn.Linear(75 * arglist.FEAT2DSIZE * arglist.FEAT2DSIZE, 256)),
            nn.ReLU()))

        self.value = nn.DataParallel(nn.Linear(in_features=256, out_features=1))
        self.fn_out = self._non_spatial_outputs(256, arglist.NUM_ACTIONS)
        self.non_spatial_outputs = self._init_non_spatial()
        self.spatial_outputs = self._init_spatial()

    # def cuda(self):
    #     for k, v in self.embed_screen.items():
    #         v.cuda()
    #     for k, v in self.embed_minimap.items():
    #         v.cuda()
    #     for k, v in self.embed_flat.items():
    #         v.cuda()
    #     self.screen_out.cuda()
    #     self.minimap_out.cuda()
    #     self.fc.cuda()
    #     self.value.cuda()
    #     self.fn_out.cuda()
    #     for k, v in self.non_spatial_outputs.items():
    #         v.cuda()
    #     for k, v in self.spatial_outputs.items():
    #         v.cuda()
    #
    # def get_trainable_params(self, with_id=False):
    #     params = []
    #     ids = {}
    #     for k, v in self.embed_screen.items():
    #         ids['embed_screen:' + str(k)] = v
    #         params.extend(list(v.parameters()))
    #     for k, v in self.embed_minimap.items():
    #         ids['embed_minimap:' + str(k)] = v
    #         params.extend(list(v.parameters()))
    #     for k, v in self.embed_flat.items():
    #         ids['embed_flat:' + str(k)] = v
    #         params.extend(list(v.parameters()))
    #     ids['screen_out:0'] = self.screen_out
    #     params.extend(list(self.screen_out.parameters()))
    #     ids['minimap_out:0'] = self.minimap_out
    #     params.extend(list(self.minimap_out.parameters()))
    #     ids['fc:0'] = self.fc
    #     params.extend(list(self.fc.parameters()))
    #     ids['value:0'] = self.value
    #     params.extend(list(self.value.parameters()))
    #     ids['fn_out:0'] = self.fn_out
    #     params.extend(list(self.fn_out.parameters()))
    #     for k, v in self.non_spatial_outputs.items():
    #         ids['non_spatial_outputs:' + str(k)] = v
    #         params.extend(list(v.parameters()))
    #     for k, v in self.spatial_outputs.items():
    #         ids['spatial_outputs:' + str(k)] = v
    #         params.extend(list(v.parameters()))
    #     if not with_id:
    #         return params
    #     else:
    #         return ids

    def _init_non_spatial(self):
        out = {}
        for arg_type in actions.TYPES:
            if not is_spatial_action[arg_type]:
                out[arg_type.id] = self._non_spatial_outputs(256, arg_type.sizes[0])
        return out

    def _init_spatial(self):
        out = {}
        for arg_type in actions.TYPES:
            if is_spatial_action[arg_type]:
                out[arg_type.id] = self._spatial_outputs(75)
        return out

    def _init_embed_obs(self, spec, embed_fn):
        """
            Define network architectures
            Each input channel is processed by a Sequential network
        """
        out_sequence = {}
        for s in spec:
            if s.type == features.FeatureType.CATEGORICAL:
                dims = dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                sequence = nn.DataParallel(nn.Sequential(
                            embed_fn(s.scale, dims),
                            nn.ReLU(True)))
                out_sequence[s.index] = sequence
        return out_sequence

    def embed_obs(self, x, spec, networks, one_hot):
        feats = torch.chunk(x, len(spec), dim=1)

        out_list = []
        for s in spec:
            f = feats[s.index]
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                indices = one_hot(f, self.dtype, C=s.scale)
                out = networks[s.index](indices.float())

            elif s.type == features.FeatureType.SCALAR:
                out = self._log_transform(f, s.scale)

            else:
                raise NotImplementedError

            out_list.append(out)
        # Channel dimension is 1
        return torch.cat(out_list, 1)

    def _log_transform(self, x, scale):
        return torch.log(8 * x / scale + 1)

    def _embed_spatial(self, in_, out_):
        return conv2d_init(in_, out_, kernel_size=1, stride=1, padding=0)  # NCHW

    def _embed_flat(self, in_, out_):
        return linear_init(in_, out_)

    def _non_spatial_outputs(self, in_, out_):
        return nn.DataParallel(nn.Sequential(nn.Linear(in_, out_),
                                             nn.Softmax(dim=1)))

    def _spatial_outputs(self, in_):
        return nn.DataParallel(nn.Sequential(
            nn.Conv2d(in_, 1, 1, stride=1),
            Flatten(),
            nn.Softmax(dim=1)))

    def forward(self, screen_input, minimap_input, flat_input):
        _, _, resolution, _ = screen_input.size()

        screen_emb = self.embed_obs(screen_input, self.screen_specs,
                                    self.embed_screen, make_one_hot_2d)

        minimap_emb = self.embed_obs(minimap_input, self.minimap_specs,
                                     self.embed_minimap, make_one_hot_2d)

        flat_emb = self.embed_obs(flat_input, self.flat_specs,
                                  self.embed_flat, make_one_hot_1d)

        screen_out = self.screen_out(screen_emb)
        minimap_out = self.minimap_out(minimap_emb)
        broadcast_out = flat_emb.unsqueeze(2).unsqueeze(3).\
            expand(flat_emb.size(0), flat_emb.size(1), resolution, resolution)

        state_out = torch.cat([screen_out.float(), minimap_out.float(), broadcast_out.float()], dim=1)
        flat_out = state_out.view(state_out.size(0), -1)
        fc = self.fc(flat_out)

        value = self.value(fc).view(-1)
        fn_out = self.fn_out(fc)

        args_out = dict()
        for arg_type in actions.TYPES:
            if is_spatial_action[arg_type]:
                arg_out = self.spatial_outputs[arg_type.id](state_out)

            else:
                arg_out = self.non_spatial_outputs[arg_type.id](fc)
            args_out[arg_type] = arg_out
        policy = (fn_out, args_out)

        return policy, value





