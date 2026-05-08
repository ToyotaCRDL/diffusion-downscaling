import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .vae_wrapper import precip_converter


def positionalencoding2d(channel, height, width):
    if channel % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(channel))
    pe = torch.zeros(channel, height, width)
    # Each dimension use half of channel
    half_c = int(channel / 2)
    div_term = torch.exp(torch.arange(0.0, half_c, 2) * -(math.log(10000.0) / half_c))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:half_c:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:half_c:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[half_c::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[half_c + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.unsqueeze(0)


class ConditionEncoder(ModelMixin, ConfigMixin):

    ignore_for_config = ["null_condition"]

    @register_to_config
    def __init__(
        self,
        LR_list: list = ["temperature", "precipitation", "pressure"],
        LR_size: list = [8, 8],
        out_channels: int = 32,
        null_condition: float | str = 0.0,
        learnable_pos_emb: bool = False,
        token_type: str = "channel",  # pixel or channel
        concat_to_latent: list | None = None,
        cfg_threshold: float = 0.0,
    ):
        super().__init__()
        self.LR_list = LR_list
        self.cfg_threshold = cfg_threshold
        self.proj = nn.Conv2d(len(LR_list), out_channels, kernel_size=3, padding=1)
        self.precip_converter = precip_converter()

        assert token_type in ["pixel", "channel"]
        self.token_type = token_type
        if self.token_type == "pixel":
            if not learnable_pos_emb:
                self.register_buffer("pos_emb", positionalencoding2d(out_channels, LR_size[0], LR_size[1]))
            else:
                self.pos_emb = nn.Parameter(torch.zeros(out_channels, LR_size[0], LR_size[1]))

        if isinstance(null_condition, str):
            with open(null_condition, "rb") as fin:
                null_condition = pickle.load(fin)
            assert set(LR_list) <= set(null_condition.keys())
            null_condition = torch.cat([torch.tensor(null_condition[LR_name]) for LR_name in LR_list])
            assert np.all([x == y for x, y in zip(null_condition.shape[1:], LR_size)])
        else:
            null_condition = torch.ones([len(LR_list)] + LR_size).mul_(null_condition)
        self.register_buffer("null_condition", null_condition)

        self.concat_to_latent = concat_to_latent is not None
        if self.concat_to_latent:
            self.reshape_to_latent = nn.Upsample(size=concat_to_latent, mode="bilinear")

    def concat_LR(self, x: dict):
        LR_data = []
        for LR_item in self.LR_list:
            x_ = x[LR_item]
            LR_data.append(x_)
        LR_data = torch.cat(LR_data, dim=1)
        return LR_data

    def forward(
        self,
        x: dict,
        geo_data: torch.FloatTensor | None,
        mode: str = "train",
        do_cfg: bool = False,
    ):
        LR_data = self.concat_LR(x)

        num_batches = LR_data.shape[0]

        if mode == "train" and self.cfg_threshold > 0:
            null_indicator = torch.rand(num_batches) < self.cfg_threshold
            LR_data[null_indicator] = self.null_condition

        if mode == "pipeline" and do_cfg:
            null_data = self.null_condition.unsqueeze(0)
            null_data = null_data.repeat(num_batches, 1, 1, 1)
            LR_data = torch.cat([null_data, LR_data])

        encoder_hidden_states = self.proj(LR_data)
        if self.token_type == "pixel":
            encoder_hidden_states = encoder_hidden_states + self.pos_emb
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 3, 1)
            encoder_hidden_states = encoder_hidden_states.view(
                encoder_hidden_states.shape[0],
                -1,
                encoder_hidden_states.shape[-1],
            )
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], -1
            )

        if self.concat_to_latent:
            concat_to_latent = self.reshape_to_latent(LR_data)
            if geo_data is not None:
                concat_geo = self.reshape_to_latent(geo_data)
                if mode == "pipeline" and do_cfg:
                    concat_geo = torch.cat([concat_geo, concat_geo])  # TODO: null for geo
                concat_to_latent = torch.cat([concat_to_latent, concat_geo], dim=1)
        else:
            concat_to_latent = None

        return encoder_hidden_states, concat_to_latent
