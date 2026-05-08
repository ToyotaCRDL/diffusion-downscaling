import os.path as osp
from importlib import import_module

import torch

from diffengine.datasets.transforms.base import BaseTransform
from diffengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDownscalingData(BaseTransform):

    def transform(self, results: dict) -> dict:
        HR_images = {}
        for k, v in results["HR_path"].items():
            HR_images[k] = torch.load(osp.join(v, results["filename"]))
        LR_images = {}
        for k, v in results["LR_path"].items():
            LR_images[k] = torch.load(osp.join(v, results["filename"]))
        results["HR_images"] = HR_images
        results["LR_images"] = LR_images
        return results


@TRANSFORMS.register_module()
class TorchVisonTransformWrapper:
    """Minimal torchvision transform wrapper used by climate configs."""

    def __init__(self, transform, *args, keys: list[str] | None = None, **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        self.keys = keys
        if isinstance(transform, str):
            module_, attr_ = transform.rsplit(".", 1)
            transform = getattr(import_module(module_), attr_)
        self.t = transform(*args, **kwargs)

    def transform(self, results: dict) -> dict:
        for key in self.keys:
            if key in results:
                results[key] = self.t(results[key])
        return results
