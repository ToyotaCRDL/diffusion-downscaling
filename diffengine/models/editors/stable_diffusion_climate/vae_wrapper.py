import pickle

import safetensors.torch  # noqa: F401
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from diffengine.registry import MODELS


class precip_converter:

    def __init__(self, alpha=0.287851):
        # alpha is a numerical solution for "alpha * log(1 + xmax/alpha) = 1" with xmax=9
        # to satisfy transform(xmax) = 1
        self.alpha = alpha

    def transform(self, x):
        x = self.alpha * torch.log(1 + x / self.alpha)
        return x

    def inverse_transform(self, x):
        x = self.alpha * (torch.exp(x / self.alpha) - 1)
        return x


class VaeWrapper(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        vae_cfg: dict,
        HR_list: list = ["temperature", "precipitation"],
        geo_list: list | None = ["mask", "topo"],
        input_dim: list = [3, 400, 400],
        original_size: list = [400, 400],
        HR_scaling_factor: list = [1.0, 1.0],
        emb_scaling_factor: list | None = None,
        geo_with_vae: bool = False,
    ):
        super().__init__()
        self.HR_list = HR_list
        self.geo_list = geo_list
        self.input_dim = input_dim
        self.vae = MODELS.build(vae_cfg)
        if input_dim[1] == original_size[0] and input_dim[2] == original_size[1]:
            self.reshape_to_vae_input = nn.Identity()
            self.reshape_to_original_size = nn.Identity()
        else:
            self.reshape_to_vae_input = nn.Upsample(size=input_dim[1:], mode="bilinear")
            self.reshape_to_original_size = nn.Upsample(size=original_size, mode="bilinear")
        self.geo_with_vae = geo_with_vae
        self.HR_scaling_factor = HR_scaling_factor
        self.precip_converter = precip_converter()

        for idx, HR_name in enumerate(self.HR_list):
            self.register_buffer(f"{HR_name}_emb_mean", torch.tensor(0.0))
            std_ = torch.tensor(1.0)
            if emb_scaling_factor is not None:
                std_.mul_(emb_scaling_factor[idx])
            self.register_buffer(f"{HR_name}_emb_std", std_)


    def encode(self, x: dict):

        latents = []
        for idx, HR_name in enumerate(self.HR_list):
            x_ = x[HR_name]
            with torch.no_grad():
                x_ = x_ * self.HR_scaling_factor[idx]

                if HR_name == "precipitation":
                    assert self.HR_scaling_factor[idx] == 1
                    x_ = self.precip_converter.transform(x_)

                x_ = self.reshape_to_vae_input(x_)
                x_ = x_.repeat(1, self.input_dim[0], 1, 1)
                latents_ = self.vae.encode(x_).latents
            HR_mean = getattr(self, f"{HR_name}_emb_mean")
            HR_std = getattr(self, f"{HR_name}_emb_std")
            latents_ = (latents_ - HR_mean) / HR_std
            latents.append(latents_)

        latents = torch.cat(latents, dim=1)

        return latents

    def encode_geo_data(self, x: dict):

        assert set(self.geo_list) <= set(x.keys())

        geo_latents = []
        for geo_item in self.geo_list:
            x_ = x[geo_item]
            if self.geo_with_vae:
                with torch.no_grad():
                    x_ = self.reshape_to_vae_input(x_)
                    x_ = x_.repeat(1, self.input_dim[0], 1, 1)
                    geo_latents_ = self.vae.encode(x_).latents
            else:
                geo_latents_ = x_
            geo_latents.append(geo_latents_)

        geo_latents = torch.cat(geo_latents, dim=1)

        return geo_latents

    def decode(self, x: torch.FloatTensor):
        x_chunked = x.chunk(len(self.HR_list), dim=1)
        results = {}
        for idx, (latents_, HR_name) in enumerate(zip(x_chunked, self.HR_list)):
            HR_mean = getattr(self, f"{HR_name}_emb_mean")
            HR_std = getattr(self, f"{HR_name}_emb_std")
            latents_ = latents_ * HR_std + HR_mean
            res_ = self.vae.decode(latents_, force_not_quantize=True).sample
            res_ = res_.mean(dim=1, keepdim=True)
            res_ = self.reshape_to_original_size(res_)

            if HR_name == "precipitation":
                res_ = self.precip_converter.inverse_transform(res_)

            res_ = res_ / self.HR_scaling_factor[idx]
            results[HR_name] = res_

        return results
