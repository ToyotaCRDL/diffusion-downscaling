import inspect
from typing import List, Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

from .condition_encoder import ConditionEncoder
from .vae_wrapper import VaeWrapper

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """From StableDiffusionPipeline"""
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """From StableDiffusionPipeline"""
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionClimatePipeline(DiffusionPipeline, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        vae: VaeWrapper,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        cond_encoder: ConditionEncoder,
        LR_list: list = ["temperature", "precipitation", "pressure"],
        HR_list: list = ["temperature", "precipitation"],
    ):
        super().__init__()

        self.LR_list = LR_list
        self.HR_list = HR_list

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            cond_encoder=cond_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.vae.config.block_out_channels) - 1)
        self.register_to_config()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # from StableDiffusionPipeline
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # from StableDiffusionPipeline
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        LR_images: dict,
        geo_data: dict | None = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        num_images_per_lrinfo: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_latent: Optional[bool] = False,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.
        """

        device = self._execution_device

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        batch_size = LR_images[self.cond_encoder.LR_list[0]].shape[0]

        if geo_data is not None:
            geo_latents = self.vae.encode_geo_data(geo_data)
        else:
            geo_latents = None
        encoder_hidden_states, concat_to_latent = self.cond_encoder(
            LR_images,
            geo_latents,
            mode="pipeline",
            do_cfg=self.do_classifier_free_guidance,
        )

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if concat_to_latent is not None:
            num_channels_latents -= concat_to_latent.shape[1]
        latents = self.prepare_latents(
            batch_size * num_images_per_lrinfo,
            num_channels_latents,
            height,
            width,
            encoder_hidden_states.dtype,
            device,
            generator,
            latents,
        )

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_lrinfo)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # Denoising loop
        self._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # concat conditions to the unet input
            if concat_to_latent is not None:
                latent_model_input = torch.cat([latent_model_input, concat_to_latent], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=timestep_cond,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_null, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_null + self.guidance_scale * (noise_pred_cond - noise_pred_null)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        if not output_latent:
            results = self.vae.decode(latents)
        else:
            results = {"latent": latents}

        # Offload all models
        self.maybe_free_model_hooks()

        return results
