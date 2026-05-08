"""Create a tiny climate downscaling tensor dataset for smoke runs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

HR_LIST = ["temperature", "precipitation", "tmax", "tmin", "gsr"]
LR_LIST = [
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
GEO_LIST = ["mask", "topo"]
SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}


def _smooth_field(generator: torch.Generator, size: tuple[int, int]) -> torch.Tensor:
    """Generate a smooth single-channel field in [0, 1]."""
    coarse = torch.rand((1, 1, 25, 25), generator=generator)
    field = F.interpolate(coarse, size=size, mode="bicubic", align_corners=False)
    field = field.squeeze(0).clamp(0.0, 1.0)
    return field.to(torch.float32)


def _make_hr(variable: str, sample_idx: int, generator: torch.Generator) -> torch.Tensor:
    field = _smooth_field(generator, (400, 400))
    offset = 0.03 * sample_idx
    if variable == "precipitation":
        return (field * 0.5 + offset).clamp_min(0.0)
    if variable == "gsr":
        return (field * 0.8 + 0.1 + offset).clamp(0.0, 1.0)
    return (field * 2.0 - 1.0 + offset).to(torch.float32)


def _make_lr(hr: torch.Tensor) -> torch.Tensor:
    lr = F.interpolate(hr.unsqueeze(0), size=(8, 8), mode="bilinear", align_corners=False)
    return lr.squeeze(0).to(torch.float32)


def _write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.contiguous(), path)


def make_dataset(root: Path, train: int, val: int, test: int, seed: int) -> None:
    if root.exists():
        shutil.rmtree(root)

    generator = torch.Generator().manual_seed(seed)
    counts = {"train": train, "val": val, "test": test}

    mask = torch.ones((1, 400, 400), dtype=torch.float32)
    yy = torch.linspace(0.0, 1.0, 400).view(1, 400, 1)
    xx = torch.linspace(0.0, 1.0, 400).view(1, 1, 400)
    topo = (0.6 * yy + 0.4 * xx).to(torch.float32)
    _write_tensor(root / "mask.dat", mask)
    _write_tensor(root / "topo.dat", topo)

    for split, count in counts.items():
        split_dir = SPLIT_DIRS[split]
        for sample_idx in range(count):
            filename = f"{sample_idx:06d}.dat"
            hr_cache: dict[str, torch.Tensor] = {}
            for variable in HR_LIST:
                hr = _make_hr(variable, sample_idx, generator)
                hr_cache[variable] = hr
                _write_tensor(root / split_dir / variable / "HR" / filename, hr)

            for variable in LR_LIST:
                source = hr_cache.get(variable)
                if source is None:
                    source = _smooth_field(generator, (400, 400))
                _write_tensor(root / split_dir / variable / "LR" / filename, _make_lr(source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/downscaling/pt"),
    )
    parser.add_argument("--train", type=int, default=2, help="Number of training samples.")
    parser.add_argument("--val", type=int, default=1, help="Number of validation samples.")
    parser.add_argument("--test", type=int, default=1, help="Number of test samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_dataset(args.root, args.train, args.val, args.test, args.seed)
    print(f"Created toy dataset at {args.root}")


if __name__ == "__main__":
    main()
