# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from os import PathLike

from mmengine.dataset.base_dataset import Compose
from torch.utils.data import Dataset


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(
        self,
        data_prefix,
        pipeline: Sequence = (),
    ):
        super(BaseDataset, self).__init__()
        self.data_prefix = expanduser(data_prefix)
        self.data_infos = self.load_annotations()
        self.pipeline = Compose(pipeline)

    @abstractmethod
    def load_annotations(self):
        pass

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_infos)
