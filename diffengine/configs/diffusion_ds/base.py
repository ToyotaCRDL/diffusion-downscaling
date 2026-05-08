from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from mmengine.config import ConfigDict as dict  # to enable recursive update of dict
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import TestLoop, ValLoop
from mmengine.visualization import LocalVisBackend
from torch.optim import AdamW
from torch.utils.data._utils.collate import default_collate

from diffengine.datasets import DownscalingDataset
from diffengine.datasets.transforms import (
    LoadDownscalingData,
    PackInputs,
)
from diffengine.engine.hooks import (
    RemoveVaeFromCheckpointHook,
    SetEvalPipelineHook,
)
from diffengine.models.editors import (
    ConditionEncoder,
    StableDiffusionClimate,
    VaeWrapper,
)

# Settings
HR_list = [
    "temperature",
    "precipitation",
    "tmax",
    "tmin",
    "gsr",
]
HR_scaling_factor = [1.0, 1.0, 1.0, 1.0, 1.0]  # scaling factors multiplied before VAE encoding
emb_scaling_factor = [0.364, 0.295, 0.371, 0.360, 0.322]  # scaling factors multiplied to VAE embeddings
LR_list = [
    "temperature",
    "precipitation",
    "pressure",
    "dlr",
    "dsr",
    "rh2",
    "tmax",
    "tmin",
    "wind",
]
geo_list = ["mask", "topo"]  # If None, no geo_data is used
cond_emb_dim = 64  # channels of projected LR conditions
concat_to_latent = [50, 50]  # Specify latent HxW. If None, conditions are not concat to latents
# CFG settings
cfg_threshold = 0.0
guidance_scale = 1.0

# Adjust parameters based on settings
cross_attention_dim = 64
unet_in_channels = 4 * len(HR_list)
if concat_to_latent is not None:
    unet_in_channels += len(LR_list)
    if geo_list is not None:
        unet_in_channels += len(geo_list)

train_pipeline = [
    dict(type=LoadDownscalingData),
    dict(
        type=PackInputs,
        input_keys=["HR_images", "LR_images", "geo_data", "filename"],
        skip_to_tensor_key=["HR_images", "LR_images", "geo_data", "filename"],
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    collate_fn=default_collate,
    dataset=dict(
        type=DownscalingDataset,
        data_prefix="data/downscaling/pt",
        split="train",
        HR_list=HR_list,
        LR_list=LR_list,
        geo_list=geo_list,
        pipeline=train_pipeline,
    ),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

custom_hooks = [
    dict(type=RemoveVaeFromCheckpointHook),
]


model = dict(
    type=StableDiffusionClimate,
    HR_list=HR_list,
    LR_list=LR_list,
    scheduler=dict(
        type=DDPMScheduler,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        clip_sample=False,
        num_train_timesteps=1000,
        steps_offset=1,
        trained_betas=None,
        prediction_type="v_prediction",
        rescale_betas_zero_snr=False,
    ),
    vae=dict(
        type=VaeWrapper,
        HR_list=HR_list,
        geo_list=geo_list,
        input_dim=[3, 400, 400],
        HR_scaling_factor=HR_scaling_factor,
        emb_scaling_factor=emb_scaling_factor,
        vae_cfg=dict(
            type=VQModel.from_pretrained,
            pretrained_model_name_or_path="kandinsky-community/kandinsky-3",
            subfolder="movq",
            variant="fp16",
        ),
    ),
    unet=dict(
        type=UNet2DConditionModel,
        act_fn="silu",
        attention_head_dim=8,
        block_out_channels=[160, 320, 640, 640],
        center_input_sample=False,
        cross_attention_dim=cross_attention_dim,
        down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
        downsample_padding=1,
        flip_sin_to_cos=True,
        freq_shift=0,
        in_channels=unet_in_channels,
        layers_per_block=2,
        mid_block_scale_factor=1,
        norm_eps=1e-05,
        norm_num_groups=32,
        out_channels=4 * len(HR_list),
        sample_size=50,  # 400/8
        up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    ),
    cond_encoder=dict(
        type=ConditionEncoder,
        LR_list=LR_list,
        LR_size=[8, 8],
        out_channels=cond_emb_dim,
        learnable_pos_emb=True,
        concat_to_latent=concat_to_latent,
        cfg_threshold=cfg_threshold,
    ),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype="float16",
    optimizer=dict(type=AdamW, lr=1e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0),
)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type=CheckpointHook,
        interval=10,
        max_keep_ckpts=3,
    )
)

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type=LocalVisBackend),
    ],
)

default_scope = "diffengine"

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=4),
    dist_cfg=dict(backend="nccl"),
)

load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
