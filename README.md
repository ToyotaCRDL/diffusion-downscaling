# Generative Climate Downscaling

This repository provides code, configuration, and utility scripts for generative
climate downscaling with diffusion models. The main training pipeline is defined
in [`diffengine/configs/diffusion_ds/base.py`](diffengine/configs/diffusion_ds/base.py)
and is designed to generate high-resolution climate fields from low-resolution
climate variables while preserving multivariate dependencies.

The training configuration expects preprocessed climate downscaling tensors under
`data/downscaling/pt`. To make the repository self-contained for installation and
pipeline checks, we include `make_toy_downscaling_data.py`, which creates a tiny
synthetic dataset with the same directory layout and tensor shapes as the
expected data.

The synthetic dataset is intended only for smoke testing the environment,
dataloader, model construction, checkpointing, and a short end-to-end training
run. It is not scientific training data. For full experiments, replace the
synthetic data with the corresponding preprocessed downscaling tensor dataset at
the same path and use the experiment settings described in the configuration
file.

## Requirements

Python 3.10.12 is recommended.

Install the minimal Python environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
mkdir -p .cache/huggingface
export HF_HOME="$PWD/.cache/huggingface"
```

The first run downloads the following pretrained component from Hugging Face:

- `kandinsky-community/kandinsky-3` MOVQ/VQ model

Use a CUDA GPU for practical training. The smoke command below reduces the
dataloader and training length, but it still builds the model from the target
config.

## Prepare Synthetic Data

```bash
python make_toy_downscaling_data.py
```

This creates the following local data directory:

```text
data/downscaling/pt/
├── mask.dat
├── topo.dat
├── train/
├── validation/
└── test/
```

Each climate variable is stored as a `.dat` tensor file:

- HR variables: `[1, 400, 400]`
- LR variables: `[1, 8, 8]`
- Geo variables: `[1, 400, 400]`

## Dataset Layout Expected by the Config

The configuration reads the following variables:

- HR: `temperature`, `precipitation`, `tmax`, `tmin`, `gsr`
- LR: `temperature`, `precipitation`, `pressure`, `dlr`, `dsr`, `rh2`,
  `tmax`, `tmin`, `wind`
- Geo: `mask`, `topo`

Every HR and LR variable directory must contain matching `.dat` file names.

## Training

Run a short one-epoch training job:

```bash
python diffengine/tools/train.py diffengine/configs/diffusion_ds/base.py \
  --work-dir work_dirs/res_diffusion_ds \
  --cfg-options \
  train_dataloader.batch_size=1 \
  train_dataloader.num_workers=0 \
  train_cfg.max_epochs=1 \
  default_hooks.checkpoint.interval=1
```

For a full experiment, replace the synthetic data with the preprocessed
downscaling tensor dataset at the same path and restore the desired batch size,
worker count, epoch count, and checkpoint interval.

## Inference from a Checkpoint

See [`inference_from_checkpoint.ipynb`](inference_from_checkpoint.ipynb) for
the full checkpoint loading and one-sample generation procedure.

## Acknowledgements

This codebase is built on top of
[DiffEngine](https://github.com/okotaku/diffengine), together with its
underlying [MMEngine](https://github.com/open-mmlab/mmengine) training
infrastructure, and [Diffusers](https://github.com/huggingface/diffusers) from
Hugging Face. We gratefully acknowledge these projects and their contributors.
