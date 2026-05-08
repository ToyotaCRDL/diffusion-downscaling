import os
import os.path as osp

import torch

from diffengine.registry import DATASETS

from .base_dataset import BaseDataset


@DATASETS.register_module()
class DownscalingDataset(BaseDataset):

    def __init__(
        self,
        *args,
        data_prefix: str = "data",
        split: str = "train",
        HR_list: list = ["temperature", "precipitation"],
        LR_list: list = ["temperature", "precipitation", "pressure"],
        geo_list: list | None = ["mask", "topo"],
        **kwargs,
    ):
        self.split = split
        self.HR_list = HR_list
        self.LR_list = LR_list
        self.geo_list = geo_list
        super().__init__(*args, data_prefix=data_prefix, **kwargs)

    def load_annotations(self):
        if self.split == "train":
            data_prefix = osp.join(self.data_prefix, "train")
        elif self.split == "val":
            data_prefix = osp.join(self.data_prefix, "validation")
        elif self.split == "test":
            data_prefix = osp.join(self.data_prefix, "test")
        else:
            raise ValueError(f"split {self.split} is not available.")

        HR_path = {}
        for metric in self.HR_list:
            HR_path[metric] = osp.join(data_prefix, metric, "HR")
        LR_path = {}
        for metric in self.LR_list:
            LR_path[metric] = osp.join(data_prefix, metric, "LR")

        if self.geo_list is not None:
            geo_data = {}
            for geo_item in self.geo_list:
                geo_data[geo_item] = torch.load(osp.join(self.data_prefix, f"{geo_item}.dat"))
                if len(geo_data[geo_item].shape) == 2:
                    geo_data[geo_item] = geo_data[geo_item].unsqueeze(0)
        else:
            geo_data = None

        dir_path = HR_path[self.HR_list[0]]
        # file_list = [x for x in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, x))]
        file_list = [entry.name for entry in os.scandir(dir_path) if entry.is_file() and entry.name.endswith(".dat")]
        file_list.sort()

        data_infos = []
        for data_idx, filename in enumerate(file_list):
            info = {"LR_path": LR_path}
            info["HR_path"] = HR_path
            info["filename"] = filename
            info["data_idx"] = data_idx
            info["geo_data"] = geo_data
            data_infos.append(info)

        return data_infos
