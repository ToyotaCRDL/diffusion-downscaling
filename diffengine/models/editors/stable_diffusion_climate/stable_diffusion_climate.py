from copy import deepcopy
from typing import Optional, Union

import torch

from mmengine import print_log
from mmengine.model import BaseModel
from torch import nn

from diffengine.registry import MODELS

from .pipeline_stable_diffusion_climate import StableDiffusionClimatePipeline


@MODELS.register_module()
class StableDiffusionClimate(BaseModel):
    """Stable Diffusion for Climate dataset."""

    def __init__(
        self,
        scheduler: dict,
        vae: dict,
        unet: dict,
        cond_encoder: dict,
        loss: dict | None = None,
        unet_lora_config: dict | None = None,
        prior_loss_weight: float = 1.0,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        gradient_checkpointing: bool = False,
        enable_xformers: bool = False,
        HR_list: list = ["temperature", "precipitation"],
        HR_scale_factor: list = [0.3303, 0.2866],
        LR_list: list = ["temperature", "precipitation", "pressure"],
        pipeline_scheduler: dict | None = None,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {}
        if loss is None:
            loss = {}
        if noise_generator is None:
            noise_generator = {}
        if timesteps_generator is None:
            timesteps_generator = {}
        super().__init__()

        self.unet_lora_config = deepcopy(unet_lora_config)
        self.prior_loss_weight = prior_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers
        self.HR_list = HR_list
        self.HR_scale_factor = HR_scale_factor
        self.LR_list = LR_list

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss, default_args={"type": "L2Loss", "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        self.scheduler = MODELS.build(scheduler)
        self.vae = MODELS.build(vae)
        self.unet = MODELS.build(unet)
        self.cond_encoder = MODELS.build(cond_encoder)
        self.noise_generator = MODELS.build(noise_generator, default_args={"type": "WhiteNoise"})
        self.timesteps_generator = MODELS.build(timesteps_generator, default_args={"type": "TimeSteps"})
        self.prepare_model()
        self.set_lora()
        self.set_xformers()
        self.pipeline_scheduler = pipeline_scheduler

        # self.infer()

    def init_weights(self):
        """To skip initialization log"""
        pass

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.unet_lora_config is not None:
            from diffengine.models.archs import create_peft_config
            from peft import get_peft_model

            unet_lora_config = create_peft_config(self.unet_lora_config)
            self.unet = get_peft_model(self.unet, unet_lora_config)
            self.unet.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available

            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.
        """
        return next(self.parameters()).device

    def set_pipeline(self):
        self.pipeline = StableDiffusionClimatePipeline(
            vae=self.vae,
            unet=self.unet,
            scheduler=self.scheduler,
            cond_encoder=self.cond_encoder,
            LR_list=self.LR_list,
            HR_list=self.HR_list,
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            self.pipeline.scheduler = self.pipeline.scheduler.from_config(
                self.pipeline.scheduler.config, **scheduler_args
            )
        self.pipeline.set_progress_bar_config(disable=True)

        # replace scheduler in pipeline
        if self.pipeline_scheduler is not None:
            pipeline_scheduler = MODELS.build(self.pipeline_scheduler)
            self.pipeline.scheduler = pipeline_scheduler.from_config(self.pipeline.scheduler.config)

    def save_pipeline(self, save_directory):
        if hasattr(self, "pipeline"):
            self.pipeline.save_pretrained(save_directory)
        else:
            self.set_pipeline()
            self.pipeline.save_pretrained(save_directory)
            self.del_pipeline

    def del_pipeline(self):
        del self.pipeline
        torch.cuda.empty_cache()

    @torch.no_grad()
    def val_step(self, data: Union[tuple, dict, list]) -> list:  # noqa
        """Val step."""
        assert self.pipeline is not None
        data = self.data_preprocessor(data)

        res = dict(zip(self.HR_list, [[] for _ in range(len(self.HR_list))]))
        for i in range(self.eval_pipeline_kwargs["num_ensemble"]):
            res_ = self.pipeline(
                LR_images=data["inputs"]["LR_images"],
                geo_data=data["inputs"]["geo_data"],
                generator=torch.Generator(device=self.device).manual_seed(i),
                **self.eval_pipeline_kwargs["pipeline_args"],
            )

            for HR_item in self.HR_list:
                res[HR_item].append(res_[HR_item].to("cpu"))

        for HR_item in self.HR_list:
            res[HR_item] = torch.stack(res[HR_item], dim=1)

        return res

    @torch.no_grad()
    def test_step(self, data: Union[tuple, dict, list]) -> list:  # noqa
        """Test step."""
        res = self.val_step(data)
        return res

    def loss(
        self,
        model_pred: torch.Tensor,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(prediction_type=self.prediction_type)

        if self.scheduler.config.prediction_type == "epsilon":
            if hasattr(self.scheduler, "get_target_noise"):
                noise = self.scheduler.get_target_noise(noise, timesteps)
            gt = noise
        elif "v_prediction" in self.scheduler.config.prediction_type:
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight,
            )
        else:
            loss = self.loss_module(model_pred.float(), gt.float(), weight=weight)
        loss_dict["loss"] = loss
        return loss_dict

    def _preprocess_model_input(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Preprocess model input."""
        if self.input_perturbation_gamma > 0:
            input_noise = noise + self.input_perturbation_gamma * torch.randn_like(noise)
        else:
            input_noise = noise
        return self.scheduler.add_noise(latents, input_noise, timesteps)

    def forward(self, inputs: dict, data_samples: Optional[list] = None, mode: str = "loss") -> dict:  # noqa
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.
        """
        assert mode == "loss"
        num_batches = len(inputs["HR_images"][self.HR_list[0]])

        latents = self.vae.encode(inputs["HR_images"])

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches, self.device)

        noisy_latents = self._preprocess_model_input(latents, noise, timesteps)

        if "geo_data" in inputs.keys():
            geo_latents = self.vae.encode_geo_data(inputs["geo_data"])
            encoder_hidden_states, concat_to_latent = self.cond_encoder(inputs["LR_images"], geo_latents)
        else:
            encoder_hidden_states, concat_to_latent = self.cond_encoder(inputs["LR_images"], None)

        if concat_to_latent is not None:
            noisy_latents = torch.cat([noisy_latents, concat_to_latent], dim=1)

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        return self.loss(model_pred, noise, latents, timesteps)
